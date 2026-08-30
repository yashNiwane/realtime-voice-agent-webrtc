"""
Ollama Streaming LLM Engine for Real-Time Conversation.

Features:
- Non-blocking streaming token generation via Ollama API.
- Explicit disabling of reasoning/thinking tokens (`think: false`).
- Stream-level `<think>...</think>` tag stripping filter.
- Multi-step tool calling loop with automatic tool execution and follow-up response streaming.
- Conversation session context manager with sliding memory window.
"""

import asyncio
import json
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx
from loguru import logger

from server.config import config, LLMConfig
from server.tools import TOOLS_SCHEMA, execute_tool_call


class ThinkTagFilter:
    """
    Streaming text filter that strips `<think>...</think>` tags and their enclosed reasoning tokens.
    Handles partial tags across chunk boundaries seamlessly.
    """

    def __init__(self):
        self.in_think_block: bool = False
        self.buffer: str = ""

    def process(self, chunk: str) -> str:
        """
        Process an incoming text chunk and return only clean user-facing text.
        """
        if not chunk:
            return ""

        self.buffer += chunk
        output = []

        while self.buffer:
            if self.in_think_block:
                end_idx = self.buffer.find("</think>")
                if end_idx != -1:
                    # Exited think block
                    self.in_think_block = False
                    self.buffer = self.buffer[end_idx + len("</think>") :]
                else:
                    # Still inside think block; retain last few chars in case of partial '</think>'
                    if len(self.buffer) > 8:
                        self.buffer = self.buffer[-8:]
                    break
            else:
                start_idx = self.buffer.find("<think>")
                if start_idx != -1:
                    # Emits text before '<think>'
                    output.append(self.buffer[:start_idx])
                    self.in_think_block = True
                    self.buffer = self.buffer[start_idx + len("<think>") :]
                else:
                    # Check for partial '<think' prefix at the end of buffer
                    partial_match = False
                    for i in range(1, len("<think>")):
                        if self.buffer.endswith("<think>"[:i]):
                            output.append(self.buffer[:-i])
                            self.buffer = self.buffer[-i:]
                            partial_match = True
                            break
                    if not partial_match:
                        output.append(self.buffer)
                        self.buffer = ""
                    break

        return "".join(output)

    def flush(self) -> str:
        """Flush any remaining non-think buffer content."""
        if not self.in_think_block and self.buffer:
            res = self.buffer
            self.buffer = ""
            return res
        self.buffer = ""
        return ""


class LLMEvent:
    """Event emitted during LLM streaming."""
    def __init__(
        self,
        event_type: str,  # "token", "tool_call", "done", "error"
        content: str = "",
        tool_data: Optional[Dict[str, Any]] = None,
        latency_ms: float = 0.0,
    ):
        self.type = event_type
        self.content = content
        self.tool_data = tool_data
        self.latency_ms = latency_ms


class OllamaLLMEngine:
    """
    High-performance streaming interface for Ollama with tool calling and conversational memory.
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        cfg = llm_config or config.llm
        self.base_url = (base_url or cfg.base_url).rstrip("/")
        self.model = model or cfg.model
        self.system_prompt = system_prompt or cfg.system_prompt
        self.temperature = temperature if temperature is not None else cfg.temperature
        self.max_tokens = max_tokens or cfg.max_tokens
        self.enable_thinking = cfg.enable_thinking

        # Determine endpoint URL
        if not self.base_url.endswith("/v1"):
            self.chat_endpoint = f"{self.base_url}/v1/chat/completions"
        else:
            self.chat_endpoint = f"{self.base_url}/chat/completions"

        logger.info(
            f"🧠 Ollama LLM Engine ready: model='{self.model}' @ endpoint='{self.chat_endpoint}', "
            f"thinking={self.enable_thinking}"
        )

    def create_session_messages(self) -> List[Dict[str, Any]]:
        """Create fresh message history with system prompt."""
        return [{"role": "system", "content": self.system_prompt}]

    async def stream_response(
        self,
        user_text: str,
        messages: List[Dict[str, Any]],
        max_tool_iterations: int = 3,
    ) -> AsyncGenerator[LLMEvent, None]:
        """
        Stream response tokens from Ollama, execute tools if invoked, and yield LLM events.

        Args:
            user_text: Spoken user query transcript.
            messages: Stateful conversation history list to mutate in-place.
            max_tool_iterations: Limit on chained function calls.

        Yields:
            LLMEvent instances for tokens, tool execution notifications, and final completion.
        """
        t0 = time.perf_counter()
        messages.append({"role": "user", "content": user_text})

        # Trim conversation history if too long to maintain low latency
        if len(messages) > 15:
            # Keep system prompt + recent 10 messages
            system_msg = messages[0]
            messages[:] = [system_msg] + messages[-10:]

        iteration = 0
        total_tokens_emitted = 0
        full_assistant_reply = ""

        async with httpx.AsyncClient(timeout=30.0) as client:
            while iteration < max_tool_iterations:
                iteration += 1
                think_filter = ThinkTagFilter()

                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "tools": TOOLS_SCHEMA,
                    "options": {
                        "think": self.enable_thinking,
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                }

                try:
                    tool_calls_accumulator: Dict[int, Dict[str, Any]] = {}
                    current_turn_content = ""

                    async with client.stream(
                        "POST", self.chat_endpoint, json=payload, headers={"Content-Type": "application/json"}
                    ) as response:
                        if response.status_code != 200:
                            err_body = await response.aread()
                            err_msg = f"Ollama HTTP {response.status_code}: {err_body.decode(errors='ignore')}"
                            logger.error(err_msg)
                            yield LLMEvent("error", content=err_msg)
                            return

                        async for line in response.aiter_lines():
                            line = line.strip()
                            if not line or not line.startswith("data:"):
                                continue

                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break

                            try:
                                chunk_json = json.loads(data_str)
                            except Exception:
                                continue

                            choices = chunk_json.get("choices", [])
                            if not choices:
                                continue

                            choice = choices[0]
                            delta = choice.get("delta", {})

                            # 1. Accumulate streaming tool calls
                            if "tool_calls" in delta and delta["tool_calls"]:
                                for tc in delta["tool_calls"]:
                                    tc_idx = tc.get("index", 0)
                                    if tc_idx not in tool_calls_accumulator:
                                        tool_calls_accumulator[tc_idx] = {
                                            "id": tc.get("id", f"call_{int(time.time()*1000)}"),
                                            "type": "function",
                                            "function": {
                                                "name": tc.get("function", {}).get("name", ""),
                                                "arguments": tc.get("function", {}).get("arguments", ""),
                                            },
                                        }
                                    else:
                                        fn = tc.get("function", {})
                                        if "name" in fn and fn["name"]:
                                            tool_calls_accumulator[tc_idx]["function"]["name"] += fn["name"]
                                        if "arguments" in fn and fn["arguments"]:
                                            tool_calls_accumulator[tc_idx]["function"]["arguments"] += fn["arguments"]

                            # 2. Process text content tokens
                            token_piece = delta.get("content", "")
                            if token_piece:
                                clean_token = think_filter.process(token_piece)
                                if clean_token:
                                    current_turn_content += clean_token
                                    full_assistant_reply += clean_token
                                    total_tokens_emitted += 1
                                    yield LLMEvent("token", content=clean_token)

                    # Flush any remaining buffer in think filter
                    remaining = think_filter.flush()
                    if remaining:
                        current_turn_content += remaining
                        full_assistant_reply += remaining
                        yield LLMEvent("token", content=remaining)

                    # Check if tool calls were triggered
                    if tool_calls_accumulator:
                        # Append assistant tool call message to history
                        tool_calls_list = list(tool_calls_accumulator.values())
                        messages.append({
                            "role": "assistant",
                            "content": current_turn_content or None,
                            "tool_calls": tool_calls_list,
                        })

                        # Execute each tool
                        for tc_item in tool_calls_list:
                            tc_id = tc_item["id"]
                            func_name = tc_item["function"]["name"]
                            raw_args = tc_item["function"]["arguments"]

                            logger.info(f"⚡ [LLM TOOL INVOCATION] Calling tool '{func_name}' with args: {raw_args}")
                            tool_result = await execute_tool_call(func_name, raw_args)

                            # Emit tool event for WebRTC DataChannel telemetry UI
                            yield LLMEvent(
                                "tool_call",
                                content=tool_result,
                                tool_data={
                                    "id": tc_id,
                                    "name": func_name,
                                    "arguments": raw_args,
                                    "result": tool_result,
                                },
                            )

                            # Append tool response message to history
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "name": func_name,
                                "content": tool_result,
                            })

                        # Continue loop to stream follow-up LLM response with tool results
                        continue

                    # If no tool calls were requested, this turn is complete
                    if current_turn_content:
                        messages.append({"role": "assistant", "content": current_turn_content})

                    break

                except Exception as e:
                    logger.exception(f"LLM streaming exception: {e}")
                    yield LLMEvent("error", content=str(e))
                    break

        elapsed_ms = (time.perf_counter() - t0) * 1000
        yield LLMEvent("done", content=full_assistant_reply, latency_ms=elapsed_ms)

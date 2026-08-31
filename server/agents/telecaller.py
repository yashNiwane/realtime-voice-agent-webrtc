"""
Telecaller Voice Agent: Manages conversational flow, lead state, and sidecar tool execution.
"""

import json
import re
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union
from loguru import logger

from server.agents.analyst import analyze_lead_async, get_sales_coaching_async
from server.agents.prompts import TELECALLER_SYSTEM_PROMPT

MAX_TRANSCRIPT_TURNS = 200


class VoiceTelecallerSession:
    """
    Stateful conversational sales telecaller session.
    Tracks transcript, auto-extracts lead parameters with regex heuristics,
    and executes specialized financial sales tools.
    """

    def __init__(self, agent_name: str = "Ananya", company_name: str = "Apex Bank"):
        self.agent_name = agent_name
        self.company_name = company_name
        self.transcript: List[Dict[str, str]] = []
        self.lead: Dict[str, Any] = {}

    def record_turn(self, role: str, content: str) -> None:
        """Append user or assistant message to session transcript."""
        if content and content.strip():
            self.transcript.append({"role": role, "content": content.strip()})
            if len(self.transcript) > MAX_TRANSCRIPT_TURNS:
                self.transcript = self.transcript[-MAX_TRANSCRIPT_TURNS:]
            if role == "user":
                self.extract_lead_from_transcript()

    def get_transcript_text(self, turns: Optional[int] = None) -> str:
        """Get formatted JSON transcript."""
        items = self.transcript if turns is None else self.transcript[-turns:]
        return json.dumps(items, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Lead Auto-Extraction (Fast regex pattern matching)
    # ------------------------------------------------------------------
    _NAME_PATTERNS = [
        re.compile(r"mera\s+naam\s+(.{2,30}?)\s+(?:hai|he|ho)", re.I),
        re.compile(r"my\s+name\s+is\s+(.{2,30})", re.I),
        re.compile(r"i(?:'m| am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.I),
    ]
    _CITY_PATTERNS = [
        re.compile(r"main\s+(.{2,30}?)\s+(?:se\s+hu|mein\s+rehta|mein\s+rahti)", re.I),
        re.compile(r"(?:rehta|rahti|rehti)\s+(?:hu|hoon|ho)\s+(?:hu\s+)?(.{2,30})", re.I),
        re.compile(r"(?:from|in|living in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.I),
    ]
    _REQ_PATTERNS = [
        re.compile(r"mujhe\s+(.{5,60}?)\s+chahiye", re.I),
        re.compile(r"(?:looking\s+for|need|want|interested\s+in)\s+(.{5,60})", re.I),
    ]
    _INCOME_PATTERNS = [
        re.compile(r"(\d[\d,\.]*\s*(?:lakh|lac|k|thousand|rupees|rs|inr|₹|pm|per month))", re.I),
        re.compile(r"salary\s*(?:is|hai)?\s*(\d[\d,\.]*)", re.I),
    ]
    _PHONE_PATTERNS = [
        re.compile(r"(?:\+91[\-\s]?)?([6-9]\d{9})"),
    ]

    def extract_lead_from_transcript(self) -> None:
        """Scan user messages to automatically populate lead profile fields."""
        user_text = " ".join(
            t["content"] for t in self.transcript if t["role"] == "user"
        )
        if not user_text.strip():
            return

        field_map = {
            "name": self._NAME_PATTERNS,
            "city": self._CITY_PATTERNS,
            "requirement": self._REQ_PATTERNS,
            "monthly_income": self._INCOME_PATTERNS,
            "phone": self._PHONE_PATTERNS,
        }
        for field, patterns in field_map.items():
            if field in self.lead and self.lead[field]:
                continue
            for pat in patterns:
                m = pat.search(user_text)
                if m:
                    val = m.group(1).strip().rstrip(".,!?")
                    if len(val) >= 2:
                        self.lead[field] = val
                        break

    # ------------------------------------------------------------------
    # Tool Handlers
    # ------------------------------------------------------------------
    async def update_lead_info(self, field: str, value: str) -> Dict[str, Any]:
        """Record customer lead attribute."""
        clean_field = field.strip().lower().replace(" ", "_")
        clean_value = str(value).strip()
        self.lead[clean_field] = clean_value
        logger.info(f"📋 [LEAD FIELD UPDATED] {clean_field} = '{clean_value}'")
        return {
            "status": "recorded",
            "field": clean_field,
            "value": clean_value,
            "lead_profile": dict(self.lead),
        }

    async def get_lead_profile(self) -> Dict[str, Any]:
        """Retrieve collected lead profile."""
        return {"lead": dict(self.lead), "turns_recorded": len(self.transcript)}

    async def analyze_conversation(self, llm_engine: Any = None) -> Dict[str, Any]:
        """Request structured assessment from background lead analyst agent."""
        return await analyze_lead_async(self.transcript, self.lead, llm_engine=llm_engine)

    async def ask_sales_coach(self, situation: str, llm_engine: Any = None) -> str:
        """Request immediate tactical whisper advice from sales coach agent."""
        return await get_sales_coaching_async(situation, self.transcript, self.lead, llm_engine=llm_engine)

    async def end_call(self, reason: str = "call completed") -> Dict[str, Any]:
        """Finalize lead and prepare warm closing."""
        self.extract_lead_from_transcript()
        return {
            "status": "call_ended",
            "reason": reason,
            "final_lead_profile": dict(self.lead),
        }


# Pipecat TelecallerAgent Worker (if pipecat is installed)
try:
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.frames.frames import LLMMessagesAppendFrame
    from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.services.llm_service import FunctionCallParams, FunctionCallResultProperties
    from pipecat.workers.llm import LLMWorker, tool

    class TelecallerAgent(LLMWorker):
        """Pipecat voice agent orchestrating full audio pipeline with sidecar delegation."""

        def __init__(self, *, name: str, transport: Any, stt: Any, llm: Any, tts: Any):
            self.session = VoiceTelecallerSession()
            self.transcript = self.session.transcript
            self.lead = self.session.lead

            self._context = LLMContext()
            aggregators = LLMContextAggregatorPair(
                self._context,
                user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
            )
            user_agg = aggregators.user()
            assistant_agg = aggregators.assistant()

            pipeline = Pipeline(
                [
                    transport.input(),
                    stt,
                    user_agg,
                    llm,
                    tts,
                    transport.output(),
                    assistant_agg,
                ]
            )

            super().__init__(name, llm=llm, pipeline=pipeline, active=True)

            latency_observer = UserBotLatencyObserver()

            @latency_observer.event_handler("on_first_bot_speech_latency")
            async def _on_first_speech_latency(seconds: float):
                logger.info(f"⏱️ LATENCY | user stopped -> bot speaking: {seconds * 1000:.0f} ms")

            self._observer.add_observer(latency_observer)

            @user_agg.event_handler("on_user_turn_stopped")
            async def _on_user_turn(aggregator, message=None):
                content = getattr(message, "content", None)
                if content:
                    self.session.record_turn("user", content)

            @assistant_agg.event_handler("on_assistant_turn_stopped")
            async def _on_assistant_turn(aggregator, message=None):
                content = getattr(message, "content", None)
                if content:
                    self.session.record_turn("assistant", content)

        @tool
        async def update_lead_info(self, params: FunctionCallParams, field: str, value: str):
            """Record customer lead info."""
            res = await self.session.update_lead_info(field, value)
            await params.result_callback(res)

        @tool
        async def get_lead_profile(self, params: FunctionCallParams):
            """Get all lead details collected so far."""
            res = await self.session.get_lead_profile()
            await params.result_callback(res)

        @tool(cancel_on_interruption=False, timeout_secs=60)
        async def analyze_conversation(self, params: FunctionCallParams):
            """Query background lead analyst agent."""
            payload = {"transcript": self.session.transcript, "lead": dict(self.session.lead)}
            async with self.job("lead_analyst", payload=payload, timeout=50) as j:
                pass
            await params.result_callback(j.response or {"error": "analyst unavailable"})

        @tool(cancel_on_interruption=False, timeout_secs=60)
        async def ask_sales_coach(self, params: FunctionCallParams, situation: str):
            """Query background sales coach for tactical objection handling advice."""
            payload = {
                "situation": situation,
                "recent_transcript": self.session.get_transcript_text(turns=10),
                "lead": dict(self.session.lead),
            }
            async with self.job("sales_coach", payload=payload, timeout=50) as j:
                pass
            await params.result_callback(j.response or {"error": "coach unavailable"})

        @tool
        async def end_call(self, params: FunctionCallParams, reason: str):
            """Politely wrap up the call."""
            await self.session.end_call(reason)
            await params.result_callback(
                None, properties=FunctionCallResultProperties(run_llm=False)
            )
            await self._pipeline.queue_frame(
                LLMMessagesAppendFrame(
                    messages=[
                        {
                            "role": "developer",
                            "content": f"Wrap up now: say a warm one-line goodbye in Hindi/Hinglish (reason: {reason}).",
                        }
                    ],
                    run_llm=True,
                )
            )
            await self.flush_pipeline(timeout=3.0)

except ImportError:
    class TelecallerAgent:
        pass

"""
Realtime Voice Agent Orchestration Pipeline (Hugging Face Realtime & FastRTC Architecture).

Architecture:
  [ Mic Audio ] 
        │
        ▼ (Silero VAD - 32ms frames)
  [ Speech Utterance ] ───(SPEECH_START: Barge-in Interrupt)───► Cancel Active Generation & Flush Audio
        │
        ▼ (Qwen3-ASR FP16 GPU <200ms)
  [ User Transcript ]
        │
        ▼ (Llama-cpp Gemma 4 E2B GPU >100 tok/s)
  [ Token Stream ]
        │
        ▼ (Adaptive StreamTextChunker: 3-word early first chunk + natural clause grouping)
  [ Text Phrases ]
        │
        ▼ (Kokoro-82M / Edge-TTS GPU <30ms)
  [ 48kHz Audio Chunks ]
        │
        ▼ (Jitter-Free Paced Frame Output)
  [ Client Ear ]
"""

import asyncio
import re
import time
from typing import Any, AsyncGenerator, Callable, Coroutine, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from server.asr_engine import Qwen3ASREngine
from server.config import config
from server.llm_engine import LLMEvent
from server.tts_engine import MultiEngineTTSManager


class SentenceTextChunker:
    """
    High-Fidelity Sentence-Based Stream Text Chunker for Conversational TTS.
    
    Principles:
    1. Chunks strictly by complete grammatical sentences (terminated by '.', '!', '?', '।', '॥', '\n')
       to preserve natural phonetics, intonation contours, and human-like emotional speech.
    2. Protects numbers, percentages, currency, and abbreviations (e.g. '₹5,000', '10.5%', 'Rs.', 'Mr.', 'Dr.', 'vs.')
       from accidental premature splitting.
    3. Handles long run-on sentences (> 14 words) gracefully by allowing natural clause boundaries (',', ';', '—')
       only when necessary to maintain streaming fluidity without sacrificing prosody.
    """

    ABBREVIATIONS = {
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.",
        "i.e.", "e.g.", "a.m.", "p.m.", "rs.", "approx.", "dept.", "no.",
        "ltd.", "inc.", "corp.", "co.", "st.", "ave.", "jan.", "feb.", "mar.",
        "apr.", "jun.", "jul.", "aug.", "sep.", "oct.", "nov.", "dec."
    }

    def __init__(self, min_clause_words: int = 14):
        self.buffer: str = ""
        self.min_clause_words: int = min_clause_words
        self.sentence_delimiters = (".", "!", "?", "।", "॥", "\n")
        self.clause_delimiters = (",", ";", ":", "—", "-")

    def _is_abbreviation(self, text: str, dot_pos: int) -> bool:
        """Check if a dot at dot_pos is part of an abbreviation or decimal number."""
        if dot_pos > 0 and dot_pos < len(text) - 1:
            if text[dot_pos - 1].isdigit() and text[dot_pos + 1].isdigit():
                return True

        preceding = text[: dot_pos + 1].split()[-1].lower() if text[: dot_pos + 1].split() else ""
        if preceding in self.ABBREVIATIONS:
            return True

        if len(preceding) == 2 and preceding[0].isalpha() and preceding[1] == ".":
            return True

        return False

    def push_token(self, token: str) -> Optional[str]:
        """
        Push incoming token and return completed sentence if boundary reached, else None.
        """
        self.buffer += token
        buf = self.buffer

        # Find potential sentence boundaries
        for i, char in enumerate(buf):
            if char in self.sentence_delimiters:
                if char == "." and self._is_abbreviation(buf, i):
                    continue

                is_at_end = (i == len(buf) - 1)
                has_trailing_space = (i < len(buf) - 1 and buf[i + 1] in (" ", "\t", "\n", '"', "”", "'", "’"))

                if has_trailing_space or char == "\n" or (is_at_end and char in ("?", "!", "।", "॥")):
                    split_idx = i + 1
                    while split_idx < len(buf) and buf[split_idx] in ('"', '”', "'", "’", " "):
                        split_idx += 1

                    sentence = buf[:split_idx].strip()
                    remaining = buf[split_idx:].lstrip()

                    if sentence and len(sentence.split()) > 0:
                        self.buffer = remaining
                        return sentence

        # Long sentence clause fallback (> min_clause_words words)
        words = buf.strip().split()
        if len(words) >= self.min_clause_words:
            for d in self.clause_delimiters:
                pos = buf.rfind(d)
                if pos > 0 and len(buf[:pos].split()) >= 6:
                    sentence = buf[:pos + 1].strip()
                    remaining = buf[pos + 1:].lstrip()
                    if sentence:
                        self.buffer = remaining
                        return sentence

        return None

    def flush(self) -> Optional[str]:
        """Flush remaining text at the end of generation."""
        clean_buf = self.buffer.strip()
        self.buffer = ""
        if clean_buf:
            return clean_buf
        return None

    def reset(self):
        """Reset state for next turn."""
        self.buffer = ""


class RealtimeVoiceSession:
    """
    Encapsulates a full-duplex conversational voice turn with cancellation and telemetry.
    """

    def __init__(
        self,
        asr_engine: Qwen3ASREngine,
        llm_engine: Any,
        tts_manager: MultiEngineTTSManager,
        telemetry_callback: Optional[Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]] = None,
        audio_output_callback: Optional[Callable[[bytes, np.ndarray, str, float], Coroutine[Any, Any, None]]] = None,
        preferred_lang: str = "Hindi",
        preferred_tts: str = "kokoro",
    ):
        self.asr = asr_engine
        self.llm = llm_engine
        self.tts = tts_manager
        self.telemetry_cb = telemetry_callback
        self.audio_output_cb = audio_output_callback
        self.preferred_lang = preferred_lang
        self.preferred_tts = preferred_tts

        self.messages: List[Dict[str, Any]] = self.llm.create_session_messages()
        self._active_turn_task: Optional[asyncio.Task] = None
        self._is_interrupted: bool = False
        self.generation_id: int = 0

    async def _emit_telemetry(self, data: Dict[str, Any]):
        if self.telemetry_cb:
            try:
                await self.telemetry_cb(data)
            except Exception as e:
                logger.debug(f"Telemetry callback error: {e}")

    def interrupt(self):
        """
        Barge-in interrupt handler: Immediately halts active LLM/TTS generation,
        increments generation ID to discard in-flight audio, and sanitizes message history.
        """
        self._is_interrupted = True
        self.generation_id += 1

        if self._active_turn_task and not self._active_turn_task.done():
            self._active_turn_task.cancel()
            logger.info(f"🛑 [PIPELINE] Active turn cancelled by user barge-in (gen_id={self.generation_id}).")

        # Sanitize message history: if an interrupted turn left a trailing user message,
        # pop it so the LLM does not get confused by multiple consecutive user turns
        if self.messages and len(self.messages) > 1 and self.messages[-1].get("role") == "user":
            self.messages.pop()

    async def process_utterance(self, pcm_audio: np.ndarray, sample_rate: int = 16000):
        """
        Process user speech audio through the complete streaming cascaded pipeline.
        """
        # Cancel any ongoing turn
        self.interrupt()
        self._is_interrupted = False
        current_gen = self.generation_id

        self._active_turn_task = asyncio.create_task(
            self._run_streaming_pipeline(pcm_audio, sample_rate, current_gen)
        )
        try:
            await self._active_turn_task
        except asyncio.CancelledError:
            logger.debug(f"Turn task cancelled (gen_id={current_gen}).")

    async def _run_streaming_pipeline(self, pcm_audio: np.ndarray, sample_rate: int, gen_id: int):
        t0_turn = time.perf_counter()

        # =====================================================================
        # Stage 1: Fast GPU Speech-to-Text (ASR)
        # =====================================================================
        t0_asr = time.perf_counter()
        transcript, detected_lang, asr_latency = await self.asr.async_transcribe_pcm(
            pcm_audio, sample_rate
        )

        if not transcript or self._is_interrupted or gen_id != self.generation_id:
            return

        active_language = self.preferred_lang or detected_lang or "Hindi"
        logger.info(f"🎙️ [ASR] '{transcript}' ({active_language}, {asr_latency:.1f}ms) [gen={gen_id}]")

        await self._emit_telemetry({
            "type": "final_transcript",
            "text": transcript,
            "language": active_language,
            "latency_ms": round(asr_latency, 1),
            "generation_id": gen_id,
        })

        # =====================================================================
        # Stage 2 & 3: Streaming LLM + Sentence Text Chunker + Fast GPU TTS
        # =====================================================================
        chunker = SentenceTextChunker()
        t0_llm = time.perf_counter()
        first_token_time: Optional[float] = None
        first_audio_emitted = False
        full_llm_text = ""

        # Background TTS Worker Queue for Concurrent Synthesis
        tts_queue: asyncio.Queue[Tuple[str, int]] = asyncio.Queue()
        tts_results: asyncio.Queue[Tuple[int, bytes, np.ndarray, float, str]] = asyncio.Queue()
        tts_sequence_counter = 0

        async def _tts_worker():
            """Concurrent TTS worker consuming phrases as they are chunked."""
            while True:
                item = await tts_queue.get()
                if item is None:
                    break
                phrase_text, seq_num = item
                if self._is_interrupted or gen_id != self.generation_id:
                    tts_queue.task_done()
                    continue

                try:
                    pcm_bytes, pcm16_arr, tts_lat, engine_name = await self.tts.synthesize(
                        phrase_text,
                        language=active_language,
                        preferred_engine=self.preferred_tts,
                    )
                    if not self._is_interrupted and gen_id == self.generation_id:
                        if pcm_bytes and len(pcm16_arr) > 0:
                            await tts_results.put((seq_num, pcm_bytes, pcm16_arr, tts_lat, engine_name))
                        else:
                            await tts_results.put((seq_num, b"", np.zeros(0, dtype=np.int16), 0.0, "skip"))
                except Exception as e:
                    logger.error(f"TTS synthesis error on '{phrase_text}': {e}")
                    if not self._is_interrupted and gen_id == self.generation_id:
                        await tts_results.put((seq_num, b"", np.zeros(0, dtype=np.int16), 0.0, "error"))
                finally:
                    tts_queue.task_done()

        # Start background TTS synthesis worker
        tts_worker_task = asyncio.create_task(_tts_worker())

        async def _audio_dispatcher():
            """Dispatches generated audio chunks to client in sequential order."""
            nonlocal first_audio_emitted
            next_expected_seq = 0
            pending_chunks: Dict[int, Tuple[bytes, np.ndarray, float, str]] = {}

            while True:
                item = await tts_results.get()
                if item is None:
                    break
                seq_num, pcm_bytes, pcm16_arr, tts_lat, engine_name = item
                if self._is_interrupted or gen_id != self.generation_id:
                    tts_results.task_done()
                    continue

                pending_chunks[seq_num] = (pcm_bytes, pcm16_arr, tts_lat, engine_name)

                # Dispatch in order
                while next_expected_seq in pending_chunks:
                    c_bytes, c_arr, c_lat, c_eng = pending_chunks.pop(next_expected_seq)
                    next_expected_seq += 1

                    if not self._is_interrupted and gen_id == self.generation_id and len(c_arr) > 0:
                        if self.audio_output_cb:
                            await self.audio_output_cb(c_bytes, c_arr, c_eng, c_lat, gen_id)

                        if not first_audio_emitted:
                            first_audio_emitted = True
                            e2e_lat = (time.perf_counter() - t0_asr) * 1000
                            await self._emit_telemetry({
                                "type": "tts_start",
                                "engine": c_eng,
                                "tts_latency_ms": round(c_lat, 1),
                                "e2e_latency_ms": round(e2e_lat, 1),
                                "generation_id": gen_id,
                            })

                tts_results.task_done()

        audio_dispatcher_task = asyncio.create_task(_audio_dispatcher())

        try:
            # Stream LLM tokens
            async for event in self.llm.stream_response(transcript, self.messages):
                if self._is_interrupted or gen_id != self.generation_id:
                    break

                if event.type == "token":
                    token = event.content
                    if first_token_time is None:
                        first_token_time = (time.perf_counter() - t0_llm) * 1000
                        await self._emit_telemetry({
                            "type": "llm_ttft",
                            "ttft_ms": round(first_token_time, 1),
                            "generation_id": gen_id,
                        })

                    await self._emit_telemetry({
                        "type": "llm_token",
                        "token": token,
                        "generation_id": gen_id,
                    })
                    full_llm_text += token

                    # Check for completed clause
                    phrase = chunker.push_token(token)
                    if phrase and not self._is_interrupted and gen_id == self.generation_id:
                        await tts_queue.put((phrase, tts_sequence_counter))
                        tts_sequence_counter += 1

                elif event.type == "tool_call":
                    logger.info(f"🔧 [TOOL] {event.tool_data['name']} executed.")
                    await self._emit_telemetry({
                        "type": "tool_call",
                        "name": event.tool_data["name"],
                        "arguments": event.tool_data["arguments"],
                        "result": event.tool_data["result"],
                        "generation_id": gen_id,
                    })

                elif event.type == "done":
                    # Flush any trailing text chunk
                    trailing_phrase = chunker.flush()
                    if trailing_phrase and not self._is_interrupted and gen_id == self.generation_id:
                        await tts_queue.put((trailing_phrase, tts_sequence_counter))
                        tts_sequence_counter += 1

            if not self._is_interrupted and gen_id == self.generation_id:
                # Wait for all TTS jobs to complete
                await tts_queue.join()
                await tts_queue.put(None)
                await tts_worker_task

                # Wait for audio dispatcher to flush
                await tts_results.put(None)
                await audio_dispatcher_task

                total_turn_ms = (time.perf_counter() - t0_turn) * 1000
                await self._emit_telemetry({
                    "type": "llm_done",
                    "full_text": full_llm_text,
                    "total_turn_ms": round(total_turn_ms, 1),
                    "generation_id": gen_id,
                })

        finally:
            if not tts_worker_task.done():
                tts_worker_task.cancel()
            if not audio_dispatcher_task.done():
                audio_dispatcher_task.cancel()

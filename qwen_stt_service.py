"""
Realtime Speech-to-Text Service using Qwen3-ASR integrated with Pipecat AI framework.

Optimized for ultra-low latency (<50ms-100ms chunk processing) using:
- Dynamic PyTorch quantization (INT8) on CPU
- Non-blocking asynchronous inference worker
- Streaming ring buffer with VAD gating
- Interim & Final transcription frames
"""

import asyncio
import concurrent.futures
import time
from typing import AsyncGenerator, Optional

import numpy as np
import torch
from loguru import logger
from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import parse_asr_output

from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

_LANGUAGE_CODE_MAP = {
    "hindi": Language.HI,
    "english": Language.EN,
    "chinese": Language.ZH,
    "cantonese": Language.ZH,
    "spanish": Language.ES,
    "french": Language.FR,
    "german": Language.DE,
    "japanese": Language.JA,
    "russian": Language.RU,
    "arabic": Language.AR,
    "portuguese": Language.PT,
    "italian": Language.IT,
    "korean": Language.KO,
    "thai": Language.TH,
    "vietnamese": Language.VI,
    "turkish": Language.TR,
    "dutch": Language.NL,
    "polish": Language.PL,
    "swedish": Language.SV,
    "danish": Language.DA,
    "finnish": Language.FI,
}


def to_pipecat_language(lang_str: Optional[str]) -> Language:
    """Safely map language string to Pipecat Language enum."""
    if not lang_str:
        return Language.EN
    key = str(lang_str).lower().strip()
    if key in _LANGUAGE_CODE_MAP:
        return _LANGUAGE_CODE_MAP[key]
    try:
        return Language(key)
    except Exception:
        return Language.EN


def is_valid_speech_text(text: str) -> bool:
    """Filter out background noise artifacts, repetitive symbols, and silence hallucinations."""
    if not text:
        return False
    cleaned = text.strip()
    # Check for meaningful alphanumeric or Indic/Devanagari characters
    letters = [c for c in cleaned if c.isalnum() or '\u0900' <= c <= '\u097F']
    if len(letters) < 2:
        return False

    # Common ASR noise hallucinations on silence/background rumble
    noise_patterns = {
        "thank you", "thanks for watching", "subtitles by", "you", "yeah",
        "bye", "goodbye", "amara.org", "[blank_audio]", "(music)", "(applause)",
        "...", "---", "***"
    }
    if cleaned.lower() in noise_patterns:
        return False

    # Repetitive single character (e.g. "........" or "aaaaaaa")
    if len(set(letters)) == 1 and len(letters) > 4:
        return False

    return True


class Qwen3ASREngine:
    """Core inference engine for Qwen3-ASR model with CPU optimizations."""

    def __init__(
        self,
        model_path: str,
        language: Optional[str] = "English",
        use_quantization: bool = True,
        num_threads: int = 8,
        max_new_tokens: int = 48,
    ):
        self.model_path = model_path
        self.language = language
        self.use_quantization = use_quantization
        self.num_threads = num_threads
        self.max_new_tokens = max_new_tokens

        torch.set_num_threads(self.num_threads)
        logger.info(f"Loading Qwen3-ASR model from {model_path} (threads={num_threads})...")
        t0 = time.perf_counter()

        self.wrapper = Qwen3ASRModel.from_pretrained(
            model_path,
            max_new_tokens=self.max_new_tokens,
        )
        self.model = self.wrapper.model
        self.processor = self.wrapper.processor

        if self.use_quantization:
            logger.info("Applying dynamic int8 quantization to Linear layers...")
            try:
                self.model = torch.ao.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("Quantization applied successfully.")
            except Exception as e:
                logger.warning(f"Failed to apply dynamic quantization: {e}. Falling back to float32.")

        self.model.eval()
        self.prompt = self.wrapper._build_text_prompt(context="", force_language=self.language)
        load_time = (time.perf_counter() - t0) * 1000
        logger.info(f"Qwen3-ASR initialized in {load_time:.1f}ms")

        # Warm up engine
        self._warmup()

    def _warmup(self):
        """Warm up PyTorch JIT and caches with dummy audio."""
        try:
            dummy = np.zeros(8000, dtype=np.float32)  # 0.5s silence
            inputs = self.processor(text=[self.prompt], audio=[dummy], return_tensors="pt", padding=True)
            inputs = inputs.to(self.model.device if hasattr(self.model, "device") else "cpu")
            with torch.inference_mode():
                _ = self.model.generate(**inputs, max_new_tokens=4)
            logger.debug("Qwen3-ASR warmup completed.")
        except Exception as e:
            logger.warning(f"Qwen3-ASR warmup warning: {e}")

    @torch.inference_mode()
    def transcribe_pcm(self, pcm_data: np.ndarray, sr: int = 16000) -> tuple[str, str, float]:
        """
        Transcribe 1D float32 or int16 PCM numpy array.
        Returns (text, language, latency_ms).
        """
        t0 = time.perf_counter()
        if pcm_data.ndim > 1:
            pcm_data = pcm_data.flatten()

        if pcm_data.dtype == np.int16:
            pcm_data = (pcm_data.astype(np.float32) / 32768.0)
        elif pcm_data.dtype != np.float32:
            pcm_data = pcm_data.astype(np.float32)

        if len(pcm_data) < 400:  # < 25ms
            return "", self.language or "", 0.0

        inputs = self.processor(text=[self.prompt], audio=[pcm_data], return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device if hasattr(self.model, "device") else "cpu")

        text_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        decoded = self.processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        raw_out = decoded[0] if decoded else ""
        lang, text = parse_asr_output(raw_out, user_language=self.language)
        latency_ms = (time.perf_counter() - t0) * 1000

        final_text = text.strip()
        if not is_valid_speech_text(final_text):
            final_text = ""

        return final_text, lang or self.language or "", latency_ms


class Qwen3ASRSTTService(STTService):
    """
    Pipecat STT service wrapping Qwen3-ASR with streaming support,
    low-latency VAD segmentation, and interim transcription frames.
    """

    def __init__(
        self,
        *,
        model_path: str,
        language: Optional[str] = "English",
        use_quantization: bool = True,
        num_threads: int = 8,
        chunk_size_ms: int = 50,
        sample_rate: int = 16000,
        stream_interim_results: bool = True,
        interim_interval_ms: int = 250,
        settings: Optional[STTSettings] = None,
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            settings=settings or STTSettings(language=language or "en"),
            **kwargs,
        )
        self._model_path = model_path
        self._language = language
        self._use_quantization = use_quantization
        self._num_threads = num_threads
        self._chunk_size_ms = chunk_size_ms
        self._stream_interim_results = stream_interim_results
        self._interim_interval_ms = interim_interval_ms

        self._engine: Optional[Qwen3ASREngine] = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="qwen_asr_worker"
        )

        # Audio buffers
        self._audio_buffer = bytearray()
        self._speech_buffer = bytearray()
        self._is_speaking = False
        self._last_interim_time = 0.0
        self._interim_task: Optional[asyncio.Task] = None

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        # Initialize engine in thread pool to avoid blocking event loop
        loop = asyncio.get_running_loop()
        self._engine = await loop.run_in_executor(
            self._executor,
            lambda: Qwen3ASREngine(
                model_path=self._model_path,
                language=self._language,
                use_quantization=self._use_quantization,
                num_threads=self._num_threads,
            ),
        )

    async def cleanup(self):
        await super().cleanup()
        if self._interim_task and not self._interim_task.done():
            self._interim_task.cancel()
        self._executor.shutdown(wait=False)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Transcribe raw 16-bit mono PCM audio bytes."""
        if not audio or self._engine is None:
            return

        pcm = np.frombuffer(audio, dtype=np.int16)
        if len(pcm) == 0:
            return

        loop = asyncio.get_running_loop()
        text, lang, latency_ms = await loop.run_in_executor(
            self._executor, self._engine.transcribe_pcm, pcm, self.sample_rate
        )

        if text:
            logger.info(f"Qwen3-ASR Final Transcription [{latency_ms:.1f}ms]: '{text}'")
            yield TranscriptionFrame(
                text=text,
                user_id=self._user_id,
                timestamp=time_now_iso8601(),
                language=to_pipecat_language(lang),
                finalized=True,
            )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming audio frames and VAD speaking state."""
        await super().process_frame(frame, direction)

        if isinstance(frame, (VADUserStartedSpeakingFrame, UserStartedSpeakingFrame)):
            self._is_speaking = True
            self._speech_buffer.clear()
            # Include recent pre-speech buffer to avoid clipping first syllable
            if len(self._audio_buffer) > 0:
                self._speech_buffer.extend(self._audio_buffer[-int(self.sample_rate * 2 * 0.25):])  # 250ms pre-roll
            self._last_interim_time = time.monotonic()

        elif isinstance(frame, (VADUserStoppedSpeakingFrame, UserStoppedSpeakingFrame)):
            self._is_speaking = False
            if len(self._speech_buffer) > 0:
                audio_to_transcribe = bytes(self._speech_buffer)
                self._speech_buffer.clear()
                await self.process_generator(self.run_stt(audio_to_transcribe))

        elif isinstance(frame, AudioRawFrame):
            # Maintain sliding history buffer (500ms)
            max_buf_len = int(self.sample_rate * 2 * 0.5)
            self._audio_buffer.extend(frame.audio)
            if len(self._audio_buffer) > max_buf_len:
                self._audio_buffer = self._audio_buffer[-max_buf_len:]

            if self._is_speaking:
                self._speech_buffer.extend(frame.audio)

                # Periodic interim transcription during active speech
                now = time.monotonic()
                if (
                    self._stream_interim_results
                    and (now - self._last_interim_time) * 1000 >= self._interim_interval_ms
                    and len(self._speech_buffer) >= int(self.sample_rate * 2 * 0.4)
                ):
                    self._last_interim_time = now
                    # Trigger async interim decode without blocking pipeline
                    if self._interim_task is None or self._interim_task.done():
                        snapshot = bytes(self._speech_buffer)
                        self._interim_task = asyncio.create_task(self._process_interim(snapshot))

    async def _process_interim(self, audio_bytes: bytes):
        """Run non-blocking interim transcription on speech snapshot."""
        try:
            if not self._engine or len(audio_bytes) == 0:
                return
            pcm = np.frombuffer(audio_bytes, dtype=np.int16)
            loop = asyncio.get_running_loop()
            text, lang, lat_ms = await loop.run_in_executor(
                self._executor, self._engine.transcribe_pcm, pcm, self.sample_rate
            )
            if text and self._is_speaking:
                await self.push_frame(
                    InterimTranscriptionFrame(
                        text=text,
                        user_id=self._user_id,
                        timestamp=time_now_iso8601(),
                        language=to_pipecat_language(lang),
                    )
                )
        except Exception as e:
            logger.debug(f"Interim transcription error: {e}")

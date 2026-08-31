"""
Multi-Engine Text-to-Speech (TTS) Manager.

Supported Engines:
1. Kokoro-82M: Ultra-fast 82M open-weight neural TTS (<30ms on GPU, 24kHz HD studio output).
2. Edge-TTS: Free Microsoft Neural Cloud TTS (hi-IN-SwaraNeural / en-IN-NeerjaNeural).
3. VITS Hindi: Local offline neural TTS model (facebook/mms-tts-hin).
4. Cartesia TTS: Ultra-low latency Sonic-3 cloud API.

All audio streams are converted and resampled to standardized 48,000 Hz 16-bit mono PCM
for seamless, glitch-free WebRTC audio playback.
"""

import asyncio
import io
import os
import re
import time
from typing import Any, Dict, Optional, Tuple
import edge_tts
import httpx
import numpy as np
import scipy.signal
import soundfile as sf
import torch
from loguru import logger
from transformers import AutoTokenizer, VitsModel

from server.config import config, TTSConfig


def clean_tts_text(text: str) -> str:
    """
    Sanitize LLM output text by stripping markdown, think tags, and symbols for natural speech synthesis.
    Preserves all Devanagari script, Latin text, currency symbols, percentages, and punctuation.
    """
    if not text:
        return ""
    # Strip thinking / thought tags if any remain
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<thought>.*?</thought>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    # Strip markdown bold, italics, headers, code blocks
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"`.*?`", "", cleaned)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
    cleaned = re.sub(r"_([^_]+)_", r"\1", cleaned)
    cleaned = re.sub(r"^#+\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^[-*•]\s*", "", cleaned, flags=re.MULTILINE)
    # Strip bullet numbering prefixes like "1. ", "2) "
    cleaned = re.sub(r"^\d+[\.\)]\s+", "", cleaned, flags=re.MULTILINE)
    # Normalize currency symbol for phonemizers
    cleaned = re.sub(r"₹\s*(\d+)", r"Rs. \1", cleaned)
    cleaned = re.sub(r"&\s*", " and ", cleaned)
    # Preserve Devanagari (\u0900-\u097F), Latin letters, numbers, and spoken punctuation
    cleaned = re.sub(r"[^\w\s.,!?;:।॥'\-—%₹$€\"“”‘’\u0900-\u097F\u200C\u200D]", " ", cleaned, flags=re.UNICODE)
    # Strip standalone hanging punctuation
    cleaned = re.sub(r"^\s*[,;:\-—]\s*", "", cleaned)
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def resample_to_48k_pcm16(
    audio_data: np.ndarray, source_sr: int
) -> Tuple[bytes, np.ndarray]:
    """
    Resample 1D float32 audio waveform to 48,000 Hz 16-bit signed PCM.
    """
    if audio_data.ndim > 1:
        audio_data = audio_data.flatten()

    # Ensure float in range [-1.0, 1.0]
    if audio_data.dtype == np.int16:
        audio_float = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype != np.float32:
        audio_float = audio_data.astype(np.float32)
    else:
        audio_float = audio_data

    # Resample if sample rate differs from 48000 Hz
    if source_sr != 48000:
        if source_sr == 24000:
            resampled = scipy.signal.resample_poly(audio_float, up=2, down=1)
        elif source_sr == 16000:
            resampled = scipy.signal.resample_poly(audio_float, up=3, down=1)
        else:
            num_target_samples = int(round(len(audio_float) * 48000 / source_sr))
            resampled = np.interp(
                np.linspace(0, len(audio_float), num_target_samples, endpoint=False),
                np.arange(len(audio_float)),
                audio_float,
            )
    else:
        resampled = audio_float

    resampled_clipped = np.clip(resampled, -1.0, 1.0)
    pcm_int16 = (resampled_clipped * 32767.0).astype(np.int16)
    pcm_bytes = pcm_int16.tobytes()

    return pcm_bytes, pcm_int16


class MultiEngineTTSManager:
    """
    Unified manager orchestrating Kokoro-82M, Edge-TTS, VITS, and Cartesia with automatic fallback.
    """

    def __init__(self, tts_config: Optional[TTSConfig] = None):
        self.cfg = tts_config or config.tts
        self.default_engine = self.cfg.default_engine
        self.target_sample_rate = self.cfg.sample_rate

        # Kokoro state
        self._kokoro_pipelines: Dict[str, Any] = {}
        self._kokoro_lock = asyncio.Lock()

        # VITS state
        self._vits_model: Optional[VitsModel] = None
        self._vits_tokenizer: Optional[AutoTokenizer] = None
        self._vits_lock = asyncio.Lock()
        self._vits_device = torch.device(
            "cuda"
            if self.cfg.vits_device.startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )

        logger.info(
            f"🔊 TTS Manager initialized (default_engine='{self.default_engine}', "
            f"device={self._vits_device}, output_sr={self.target_sample_rate}Hz)"
        )

    def _get_kokoro_pipeline(self, lang_code: str = "h"):
        """Lazy load Kokoro KPipeline for Hindi ('h') or English ('a')."""
        if lang_code in self._kokoro_pipelines:
            return self._kokoro_pipelines[lang_code]
        try:
            from kokoro import KPipeline
            logger.info(f"Loading Kokoro-82M TTS Pipeline (lang='{lang_code}', device={self._vits_device})...")
            pipeline = KPipeline(lang_code=lang_code, device=str(self._vits_device))
            self._kokoro_pipelines[lang_code] = pipeline
            return pipeline
        except Exception as e:
            logger.error(f"Failed to load Kokoro-82M pipeline: {e}")
            raise e

    async def _synthesize_kokoro(
        self, text: str, language: str
    ) -> Tuple[bytes, np.ndarray, float, str]:
        """Synthesize ultra-fast neural audio with Kokoro-82M and resample to 48kHz PCM."""
        t0 = time.perf_counter()
        clean = clean_tts_text(text)
        if not clean:
            return b"", np.zeros(0, dtype=np.int16), 0.0, "none"

        # Determine language & voice:
        lang_lower = (language or "").lower().strip()
        num_devanagari = len(re.findall(r"[\u0900-\u097F]", clean))
        num_latin = len(re.findall(r"[a-zA-Z]", clean))

        is_hindi_req = ("hi" in lang_lower or "hindi" in lang_lower)
        if is_hindi_req or num_devanagari > 0:
            lang_code = "h"
            voice = self.cfg.kokoro_voice_hi
        else:
            lang_code = "a"
            voice = self.cfg.kokoro_voice_en

        def _sync_kokoro():
            pipeline = self._get_kokoro_pipeline(lang_code)
            generator = pipeline(clean, voice=voice, speed=self.cfg.kokoro_speed, split_pattern=r"\n+")
            audio_chunks = []
            for _, _, audio in generator:
                if audio is not None and len(audio) > 0:
                    audio_chunks.append(audio)
            if audio_chunks:
                return np.concatenate(audio_chunks)
            return np.zeros(0, dtype=np.float32)

        loop = asyncio.get_running_loop()
        waveform = await loop.run_in_executor(None, _sync_kokoro)
        if len(waveform) == 0:
            raise ValueError(f"Kokoro returned empty audio for text: '{clean}'")

        pcm_bytes, pcm_int16 = resample_to_48k_pcm16(waveform, 24000)
        latency_ms = (time.perf_counter() - t0) * 1000
        return pcm_bytes, pcm_int16, latency_ms, f"Kokoro-82M ({voice})"

    def _get_vits_engine(self) -> Tuple[VitsModel, AutoTokenizer]:
        """Lazy load and warm up local VITS Hindi model."""
        if self._vits_model is not None and self._vits_tokenizer is not None:
            return self._vits_model, self._vits_tokenizer

        model_id = self.cfg.vits_model_id
        logger.info(f"Loading local VITS Hindi TTS model ({model_id}) onto {self._vits_device}...")
        t0 = time.perf_counter()

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = VitsModel.from_pretrained(model_id)
        model = model.to(self._vits_device)
        model.eval()

        try:
            inputs = tokenizer("नमस्ते", return_tensors="pt").to(self._vits_device)
            with torch.no_grad():
                _ = model(**inputs)
        except Exception as e:
            logger.debug(f"VITS warmup note: {e}")

        self._vits_model = model
        self._vits_tokenizer = tokenizer
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"✨ VITS Hindi model loaded and warmed in {elapsed_ms:.1f}ms")
        return self._vits_model, self._vits_tokenizer

    def _synthesize_vits_local(self, text: str) -> Tuple[bytes, np.ndarray, float]:
        t0 = time.perf_counter()
        clean = clean_tts_text(text)
        if not clean:
            return b"", np.zeros(0, dtype=np.int16), 0.0

        model, tokenizer = self._get_vits_engine()

        inputs = tokenizer(clean, return_tensors="pt").to(self._vits_device)
        with torch.no_grad():
            outputs = model(**inputs)
            waveform = outputs.waveform[0].cpu().numpy()

        native_sr = getattr(model.config, "sampling_rate", 16000)
        pcm_bytes, pcm_int16 = resample_to_48k_pcm16(waveform, native_sr)
        latency_ms = (time.perf_counter() - t0) * 1000

        return pcm_bytes, pcm_int16, latency_ms

    async def _synthesize_edge_tts(
        self, text: str, language: str
    ) -> Tuple[bytes, np.ndarray, float, str]:
        """Synthesize high-fidelity Indian female voice using Microsoft Edge Neural TTS."""
        t0 = time.perf_counter()
        clean = clean_tts_text(text)
        if not clean:
            return b"", np.zeros(0, dtype=np.int16), 0.0, "none"

        num_devanagari = len(re.findall(r"[\u0900-\u097F]", clean))
        num_latin = len(re.findall(r"[a-zA-Z]", clean))

        # Select natural Indian female voice based on language and script
        if num_devanagari > num_latin or (language and "hi" in language.lower()):
            voice = self.cfg.edge_voice_hi  # hi-IN-SwaraNeural (Female)
        else:
            voice = self.cfg.edge_voice_en  # en-IN-NeerjaNeural (Female)

        communicate = edge_tts.Communicate(clean, voice=voice)
        audio_buffer = bytearray()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.extend(chunk["data"])

        if not audio_buffer:
            raise ValueError(f"Edge-TTS returned empty audio stream for voice '{voice}'")

        audio_data, source_sr = sf.read(io.BytesIO(audio_buffer))
        pcm_bytes, pcm_int16 = resample_to_48k_pcm16(audio_data, source_sr)
        latency_ms = (time.perf_counter() - t0) * 1000

        return pcm_bytes, pcm_int16, latency_ms, f"Edge-TTS Female ({voice.split('-')[-1].replace('Neural', '')})"

    async def _synthesize_cartesia(
        self, text: str, language: str
    ) -> Tuple[bytes, np.ndarray, float, str]:
        t0 = time.perf_counter()
        clean = clean_tts_text(text)
        if not clean:
            return b"", np.zeros(0, dtype=np.int16), 0.0, "none"

        if not self.cfg.cartesia_api_key:
            raise ValueError("Cartesia API key not configured")

        headers = {
            "X-API-Key": self.cfg.cartesia_api_key,
            "Cartesia-Version": "2026-03-01",
            "Content-Type": "application/json",
        }
        lang_lower = language.lower().strip()
        cartesia_lang = "hi" if ("hi" in lang_lower or "hindi" in lang_lower) else "en"

        payload = {
            "model_id": self.cfg.cartesia_model,
            "transcript": clean,
            "voice": {"mode": "id", "id": self.cfg.cartesia_voice_id},
            "output_format": {
                "container": "wav",
                "sample_rate": 48000,
                "encoding": "pcm_s16le",
            },
            "language": cartesia_lang,
        }

        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                "https://api.cartesia.ai/tts/bytes",
                headers=headers,
                json=payload,
            )
            if resp.status_code == 200 and resp.content:
                audio_data, source_sr = sf.read(io.BytesIO(resp.content))
                pcm_bytes, pcm_int16 = resample_to_48k_pcm16(audio_data, source_sr)
                latency_ms = (time.perf_counter() - t0) * 1000
                return pcm_bytes, pcm_int16, latency_ms, "Cartesia (Sonic-3)"
            elif resp.status_code == 402:
                raise PermissionError("Cartesia quota exceeded (HTTP 402)")
            else:
                raise RuntimeError(f"Cartesia error HTTP {resp.status_code}: {resp.text}")

    async def synthesize(
        self,
        text: str,
        language: str = "Hindi",
        preferred_engine: Optional[str] = None,
    ) -> Tuple[bytes, np.ndarray, float, str]:
        clean_text = clean_tts_text(text)
        if not clean_text:
            empty_arr = np.zeros(0, dtype=np.int16)
            return b"", empty_arr, 0.0, "none"

        engine = (preferred_engine or self.default_engine).lower().strip()
        lang_lower = language.lower().strip()
        is_hindi = "hi" in lang_lower or "hindi" in lang_lower

        # 1. Kokoro-82M (Realtime Fast GPU Engine - Default Primary)
        if engine in ("kokoro", "kokoro-82m", "auto", "default"):
            try:
                pcm_bytes, pcm_int16, lat_ms, engine_name = await self._synthesize_kokoro(
                    clean_text, language
                )
                logger.info(f"🔊 Kokoro-82M TTS generated ({engine_name}) in {lat_ms:.1f}ms")
                return pcm_bytes, pcm_int16, lat_ms, engine_name
            except Exception as e:
                logger.warning(f"Kokoro-82M primary synthesis note: {e}. Cascading to Edge-TTS fallback...")

        # 2. Edge-TTS Neural Female Voice (High Quality HD Studio Voice - Primary or Fallback)
        if engine in ("edge", "edge_tts") or engine in ("kokoro", "kokoro-82m", "auto", "default"):
            try:
                pcm_bytes, pcm_int16, lat_ms, engine_name = await self._synthesize_edge_tts(
                    clean_text, language
                )
                logger.info(f"🔊 Edge-TTS generated ({engine_name}) in {lat_ms:.1f}ms")
                return pcm_bytes, pcm_int16, lat_ms, engine_name
            except Exception as e:
                logger.warning(f"Edge-TTS failed: {e}. Falling back to alternative...")

        # 3. Cartesia Sonic-3 Engine
        if engine == "cartesia":
            try:
                pcm_bytes, pcm_int16, lat_ms, engine_name = await self._synthesize_cartesia(
                    clean_text, language
                )
                logger.info(f"🔊 Cartesia Sonic-3 TTS generated in {lat_ms:.1f}ms")
                return pcm_bytes, pcm_int16, lat_ms, engine_name
            except Exception as e:
                logger.warning(f"Cartesia Sonic-3 failed: {e}. Falling back to Edge-TTS...")
                try:
                    return await self._synthesize_edge_tts(clean_text, language)
                except Exception:
                    pass

        # 4. VITS Local Offline Engine (Devanagari Hindi only)
        if engine in ("vits", "local_vits"):
            num_dev = len(re.findall(r"[\u0900-\u097F]", clean_text))
            if num_dev > 0:
                try:
                    loop = asyncio.get_running_loop()
                    async with self._vits_lock:
                        pcm_bytes, pcm_int16, lat_ms = await loop.run_in_executor(
                            None, self._synthesize_vits_local, clean_text
                        )
                    logger.info(f"🔊 VITS Local Offline TTS generated in {lat_ms:.1f}ms")
                    return pcm_bytes, pcm_int16, lat_ms, "VITS Neural (Offline Local)"
                except Exception as e:
                    logger.warning(f"VITS Local TTS failed: {e}. Falling back to Edge-TTS...")
            # If text has Latin characters or VITS failed, use Edge-TTS
            try:
                return await self._synthesize_edge_tts(clean_text, language)
            except Exception as e:
                logger.error(f"All TTS synthesis options failed: {e}")

        empty_arr = np.zeros(0, dtype=np.int16)
        return b"", empty_arr, 0.0, "error"


"""
Unified Multi-Engine TTS Manager supporting:
1. VITS Hindi (Local Offline Neural Model: facebook/mms-tts-hin)
2. Edge-TTS (Free Cloud Neural TTS: hi-IN-SwaraNeural / en-US-JennyNeural)
3. Cartesia TTS (Cloud Sonic-3)
"""

import asyncio
import io
import os
import time
from typing import Optional, Tuple
import edge_tts
import httpx
import numpy as np
import soundfile as sf
import torch
from loguru import logger
from transformers import AutoTokenizer, VitsModel

from config import config

# Singleton VITS Hindi model holder
_vits_model: Optional[VitsModel] = None
_vits_tokenizer = None
_vits_loading_lock = asyncio.Lock()


def get_vits_hindi_engine():
    """Lazy load and warm up local VITS Hindi model."""
    global _vits_model, _vits_tokenizer
    if _vits_model is not None and _vits_tokenizer is not None:
        return _vits_model, _vits_tokenizer

    model_id = os.getenv("VITS_MODEL_ID", "facebook/mms-tts-hin")
    logger.info(f"Loading local VITS Hindi TTS model ({model_id})...")
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)
    model.eval()

    # Warmup
    try:
        inputs = tokenizer("नमस्ते", return_tensors="pt")
        with torch.no_grad():
            _ = model(**inputs)
    except Exception as e:
        logger.debug(f"VITS warmup note: {e}")

    _vits_model = model
    _vits_tokenizer = tokenizer
    logger.info(f"VITS Hindi model loaded and ready in {(time.perf_counter() - t0)*1000:.1f}ms")
    return _vits_model, _vits_tokenizer


def synthesize_vits_hindi(text: str, target_sr: int = 16000) -> Tuple[bytes, float]:
    """Synthesize Hindi audio locally using VITS model."""
    t0 = time.perf_counter()
    model, tokenizer = get_vits_hindi_engine()

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        waveform = outputs.waveform[0].cpu().numpy()

    native_sr = model.config.sampling_rate

    # Convert to 16-bit PCM WAV bytes
    buffer = io.BytesIO()
    sf.write(buffer, waveform, native_sr, format="WAV", subtype="PCM_16")
    audio_bytes = buffer.getvalue()
    lat_ms = (time.perf_counter() - t0) * 1000

    return audio_bytes, lat_ms


def clean_tts_text(text: str) -> str:
    """Sanitize text for TTS synthesis."""
    if not text:
        return ""
    import re
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<thought>.*?</thought>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"`.*?`", "", cleaned)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
    cleaned = re.sub(r"_([^_]+)_", r"\1", cleaned)
    cleaned = re.sub(r"^#+\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^[-*•]\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"[^\w\s.,!?;:।'\-—%₹$€]", " ", cleaned, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


_EDGE_VOICES = {
    "hindi": "hi-IN-SwaraNeural",
    "english": "en-IN-NeerjaNeural",
    "chinese": "zh-CN-XiaoxiaoNeural",
    "spanish": "es-ES-ElviraNeural",
    "french": "fr-FR-DeniseNeural",
    "german": "de-DE-KatjaNeural",
    "japanese": "ja-JP-NanamiNeural",
    "russian": "ru-RU-SvetlanaNeural",
}


async def synthesize_speech(
    text: str,
    language: Optional[str] = "Hindi",
    sample_rate: int = 16000,
    preferred_engine: Optional[str] = None,
) -> Tuple[bytes, float, str]:
    """
    Synthesize audio for input text across supported engines:
    - 'edge': Microsoft Edge Neural Female Voice (hi-IN-SwaraNeural / en-IN-NeerjaNeural)
    - 'vits': Local offline VITS neural model for Hindi
    - 'cartesia': Cartesia Sonic-3 cloud API
    - 'auto': Edge-TTS / Cartesia with automatic failover
    """
    clean_text = clean_tts_text(text)
    if not clean_text:
        return b"", 0.0, "none"

    engine_pref = (preferred_engine or os.getenv("TTS_ENGINE", "edge")).lower().strip()
    lang_key = (language or "Hindi").lower().strip()

    # 1. High Definition Neural Edge-TTS Female Voice
    if engine_pref in ("edge", "edge_tts", "auto", "default"):
        try:
            t_fb = time.perf_counter()
            voice_name = _EDGE_VOICES.get(lang_key, "hi-IN-SwaraNeural" if "hi" in lang_key else "en-IN-NeerjaNeural")
            communicate = edge_tts.Communicate(clean_text, voice=voice_name)
            audio_buffer = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.extend(chunk["data"])

            lat_ms = (time.perf_counter() - t_fb) * 1000
            logger.info(f"Edge-TTS synthesized ({voice_name}) in {lat_ms:.1f}ms")
            return bytes(audio_buffer), lat_ms, f"Edge-TTS Female ({voice_name})"
        except Exception as e:
            logger.warning(f"Edge-TTS failed: {e}. Falling back...")

    # 2. Local VITS Engine for Devanagari Hindi
    if engine_pref in ("vits", "local_vits"):
        try:
            loop = asyncio.get_running_loop()
            audio_bytes, lat_ms = await loop.run_in_executor(
                None, synthesize_vits_hindi, clean_text, sample_rate
            )
            if audio_bytes:
                logger.info(f"VITS Local Hindi TTS synthesized in {lat_ms:.1f}ms")
                return audio_bytes, lat_ms, "VITS Neural (Local Offline)"
        except Exception as e:
            logger.warning(f"VITS synthesis error: {e}. Falling back to Neural Edge-TTS...")

    # 3. Cartesia TTS (if preferred and configured)
    if engine_pref == "cartesia" and config.tts.api_key:
        try:
            t0 = time.perf_counter()
            cartesia_headers = {
                "X-API-Key": config.tts.api_key,
                "Cartesia-Version": "2026-03-01",
                "Content-Type": "application/json",
            }
            cartesia_payload = {
                "model_id": config.tts.model,
                "transcript": clean_text,
                "voice": {"mode": "id", "id": config.tts.voice_id},
                "output_format": {
                    "container": "wav",
                    "sample_rate": sample_rate,
                    "encoding": "pcm_s16le",
                },
            }
            if lang_key in ("hi", "hindi"):
                cartesia_payload["language"] = "hi"
            elif lang_key in ("en", "english"):
                cartesia_payload["language"] = "en"

            async with httpx.AsyncClient(timeout=6.0) as client:
                resp = await client.post(
                    "https://api.cartesia.ai/tts/bytes",
                    headers=cartesia_headers,
                    json=cartesia_payload,
                )
                if resp.status_code == 200 and resp.content:
                    lat_ms = (time.perf_counter() - t0) * 1000
                    return resp.content, lat_ms, "Cartesia (Sonic-3)"
                elif resp.status_code == 402:
                    logger.warning("Cartesia quota exceeded. Falling back to Edge-TTS...")
        except Exception as e:
            logger.warning(f"Cartesia error: {e}")

    # Fallback to Edge-TTS
    try:
        t_fb = time.perf_counter()
        voice_name = _EDGE_VOICES.get(lang_key, "hi-IN-SwaraNeural" if "hi" in lang_key else "en-IN-NeerjaNeural")
        communicate = edge_tts.Communicate(clean_text, voice=voice_name)
        audio_buffer = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.extend(chunk["data"])

        lat_ms = (time.perf_counter() - t_fb) * 1000
        logger.info(f"Edge-TTS fallback synthesized ({voice_name}) in {lat_ms:.1f}ms")
        return bytes(audio_buffer), lat_ms, f"Edge-TTS ({voice_name})"
    except Exception as e:
        logger.error(f"Fallback TTS failed: {e}")
        return b"", 0.0, "error"


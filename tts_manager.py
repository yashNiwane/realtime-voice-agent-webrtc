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


_EDGE_VOICES = {
    "hindi": "hi-IN-SwaraNeural",
    "english": "en-US-JennyNeural",
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
    - 'vits': Local offline VITS neural model for Hindi
    - 'cartesia': Cartesia Sonic-3 cloud API
    - 'edge': Microsoft Edge Neural TTS
    - 'auto': VITS for Hindi / local, or Cartesia/Edge with automatic failover
    """
    if not text.strip():
        return b"", 0.0, "none"

    engine_pref = (preferred_engine or os.getenv("TTS_ENGINE", "vits")).lower().strip()
    lang_key = (language or "Hindi").lower().strip()

    # 1. Local VITS Engine for Hindi
    if engine_pref in ("vits", "local_vits") or (engine_pref == "auto" and "hi" in lang_key):
        try:
            loop = asyncio.get_running_loop()
            audio_bytes, lat_ms = await loop.run_in_executor(
                None, synthesize_vits_hindi, text, sample_rate
            )
            if audio_bytes:
                logger.info(f"VITS Local Hindi TTS synthesized in {lat_ms:.1f}ms")
                return audio_bytes, lat_ms, "VITS Neural (Local Offline)"
        except Exception as e:
            logger.warning(f"VITS synthesis error: {e}. Falling back to Neural Edge-TTS...")

    # 2. Cartesia TTS (if preferred and configured)
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
                "transcript": text,
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
                    logger.warning("Cartesia quota exceeded. Falling back to VITS/Edge-TTS...")
        except Exception as e:
            logger.warning(f"Cartesia error: {e}")

    # 3. Neural Edge-TTS Fallback
    try:
        t_fb = time.perf_counter()
        voice_name = _EDGE_VOICES.get(lang_key, "hi-IN-SwaraNeural" if "hi" in lang_key else "en-US-JennyNeural")
        communicate = edge_tts.Communicate(text, voice=voice_name)
        audio_buffer = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.extend(chunk["data"])

        lat_ms = (time.perf_counter() - t_fb) * 1000
        logger.info(f"Edge-TTS synthesized ({voice_name}) in {lat_ms:.1f}ms")
        return bytes(audio_buffer), lat_ms, f"Edge-TTS ({voice_name})"
    except Exception as e:
        logger.error(f"Fallback TTS failed: {e}")
        return b"", 0.0, "error"

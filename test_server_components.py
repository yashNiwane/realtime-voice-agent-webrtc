"""
Unit and Integration Tests for Server Components.
"""

import asyncio
import numpy as np
from loguru import logger

from server.config import config
from server.tools import execute_tool_call, COLLECTED_USER_DATA, TOOLS_SCHEMA
from server.vad_analyzer import SileroVADAnalyzer
from server.asr_engine import is_valid_speech_text
from server.llm_engine import ThinkTagFilter
from server.tts_engine import resample_to_48k_pcm16
from server.webrtc_server import ServerAudioStreamTrack


async def run_tests():
    logger.info("1. Testing Config Module...")
    assert config.asr.sample_rate == 16000
    assert config.webrtc.port == 7860
    assert len(config.webrtc.stun_servers) >= 2
    logger.info("   -> Config tests passed.")

    logger.info("2. Testing Tools Module...")
    res1 = await execute_tool_call("save_user_info", {"name": "Aditya Sharma", "phone": "9876543210"})
    assert "Aditya Sharma" in res1
    assert len(COLLECTED_USER_DATA) >= 1
    res2 = await execute_tool_call("get_current_time", {})
    assert "Current" in res2 or "current" in res2
    res3 = await execute_tool_call("get_current_weather", {"location": "Delhi"})
    assert "Delhi" in res3
    logger.info("   -> Tools tests passed.")

    logger.info("3. Testing ASR Hallucination Filter...")
    assert is_valid_speech_text("नमस्ते, आप कैसे हैं?") is True
    assert is_valid_speech_text("Hello, how can I help you?") is True
    assert is_valid_speech_text("thank you") is False
    assert is_valid_speech_text("......") is False
    assert is_valid_speech_text("") is False
    logger.info("   -> Hallucination filter tests passed.")

    logger.info("4. Testing LLM Think Tag Filter...")
    ttf = ThinkTagFilter()
    out = ttf.process("Hello <think>internal thoughts</think>world!")
    out += ttf.flush()
    assert out == "Hello world!", f"Expected 'Hello world!', got '{out}'"
    logger.info("   -> Think tag filter tests passed.")

    logger.info("5. Testing Neural VAD Analyzer...")
    vad = SileroVADAnalyzer()
    silence = np.zeros(512, dtype=np.float32)
    vres = vad.process_chunk(silence)
    assert vres.probability < 0.2
    assert vres.is_speech is False
    logger.info(f"   -> VAD tests passed (silence prob={vres.probability:.4f}).")

    logger.info("6. Testing TTS 48kHz Resampler...")
    audio16 = np.zeros(1600, dtype=np.float32)
    bytes_48k, pcm_int16 = resample_to_48k_pcm16(audio16, 16000)
    assert len(pcm_int16) == 4800
    assert len(bytes_48k) == 9600
    logger.info("   -> TTS resampler tests passed.")

    logger.info("7. Testing WebRTC ServerAudioStreamTrack Pacing & Flush...")
    track = ServerAudioStreamTrack()
    track.add_pcm16_audio(np.zeros(1920, dtype=np.int16))
    assert track._queue.qsize() == 2
    track.flush()
    assert track._queue.qsize() == 0
    assert track.is_interrupted is True
    logger.info("   -> ServerAudioStreamTrack tests passed.")

    logger.info("🎉 ALL SERVER COMPONENT TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    asyncio.run(run_tests())

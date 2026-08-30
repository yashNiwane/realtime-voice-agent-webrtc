"""
Kaggle GPU WebRTC Voice Agent Server Package.

Provides high-performance, low-latency components for real-time voice interaction:
- ASR Engine (Qwen3-ASR with CUDA fp16 & CPU int8 quantization)
- Neural VAD Analyzer (Silero VAD with speech hysteresis)
- LLM Engine (Ollama streaming with tool execution & think-tag filtering)
- Multi-Engine TTS (VITS Hindi, Edge-TTS, Cartesia Sonic-3)
- WebRTC Server (FastAPI + aiortc with zero-jitter audio track pacing)
"""

from server.config import config, ServerConfig
from server.tools import TOOLS_SCHEMA, execute_tool_call, COLLECTED_USER_DATA
from server.asr_engine import Qwen3ASREngine, is_valid_speech_text
from server.vad_analyzer import SileroVADAnalyzer, VADResult
from server.llm_engine import OllamaLLMEngine
from server.tts_engine import MultiEngineTTSManager
from server.webrtc_server import app, ServerAudioStreamTrack

__all__ = [
    "config",
    "ServerConfig",
    "TOOLS_SCHEMA",
    "execute_tool_call",
    "COLLECTED_USER_DATA",
    "Qwen3ASREngine",
    "is_valid_speech_text",
    "SileroVADAnalyzer",
    "VADResult",
    "OllamaLLMEngine",
    "MultiEngineTTSManager",
    "app",
    "ServerAudioStreamTrack",
]

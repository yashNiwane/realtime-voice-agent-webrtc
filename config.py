"""
Configuration module for Realtime Voice Agent with Qwen3-ASR, Ollama LLM, and Cartesia TTS.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def load_env_file(dotenv_path: Optional[str] = None):
    """Load key-value pairs from .env file into os.environ if not already present."""
    if dotenv_path is None:
        dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if not os.path.exists(dotenv_path):
            dotenv_path = os.path.join(os.getcwd(), ".env")

    if os.path.exists(dotenv_path):
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("\"'")
                    if k and k not in os.environ:
                        os.environ[k] = v


# Load environment variables on import
load_env_file()


def get_default_qwen3_model_path() -> str:
    """Find local cached Qwen3-ASR-0.6B snapshot or fallback to huggingface repo id."""
    user_home = Path.home()
    hub_path = user_home / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-ASR-0.6B" / "snapshots"
    if hub_path.exists():
        snapshots = list(hub_path.glob("*"))
        if snapshots:
            return str(snapshots[0])
    return "Qwen/Qwen3-ASR-0.6B"


@dataclass
class ASRConfig:
    model_path: str = os.getenv("QWEN_ASR_MODEL_PATH", get_default_qwen3_model_path())
    sample_rate: int = 16000
    chunk_size_ms: int = int(os.getenv("ASR_CHUNK_SIZE_MS", "50"))  # 50ms audio chunk
    language: Optional[str] = os.getenv("ASR_LANGUAGE", "Hindi")
    use_quantization: bool = os.getenv("ASR_QUANTIZE", "true").lower() in ("true", "1", "yes")
    num_threads: int = int(os.getenv("ASR_NUM_THREADS", "8"))
    max_new_tokens: int = int(os.getenv("ASR_MAX_NEW_TOKENS", "48"))


@dataclass
class LLMConfig:
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model: str = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")
    enable_thinking: bool = os.getenv("OLLAMA_THINKING", "false").lower() in ("true", "1", "yes")
    system_prompt: str = os.getenv(
        "LLM_SYSTEM_PROMPT",
        "You are a friendly, direct, and concise AI voice assistant. "
        "Do NOT output internal thoughts, reasoning steps, or <think> tags. "
        "Respond immediately with your direct spoken answer in 1-2 brief sentences."
    )


@dataclass
class TTSConfig:
    api_key: str = os.getenv("CARTESIA_API_KEY", "")
    voice_id: str = os.getenv("CARTESIA_VOICE_ID", "95d51f79-c397-46f9-b49a-23763d3eaa2d")
    model: str = os.getenv("CARTESIA_MODEL", "sonic-3")
    sample_rate: int = 16000


@dataclass
class AppConfig:
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "7860"))


config = AppConfig()

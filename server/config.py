"""
Configuration module for Kaggle GPU WebRTC Voice Agent.

Provides robust Pydantic-based configuration management with environment variable
parsing for ASR (Qwen3-ASR), VAD (Silero), LLM (Ollama / Gemma 4), TTS (VITS / Edge / Cartesia),
and WebRTC streaming server.
"""

import os
from pathlib import Path
from typing import List, Optional
import torch
from pydantic import BaseModel, Field


def load_env_file(dotenv_path: Optional[str] = None) -> None:
    """Load key-value pairs from .env file into os.environ if not already present."""
    search_paths = [
        dotenv_path,
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]
    for path in search_paths:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("\"'")
                        if k and k not in os.environ:
                            os.environ[k] = v
            break


# Load environment variables upon module import
load_env_file()


def get_default_qwen3_model_path() -> str:
    """Locate local cached Qwen3-ASR-0.6B snapshot or fallback to huggingface repo id."""
    user_home = Path.home()
    hub_path = user_home / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-ASR-0.6B" / "snapshots"
    if hub_path.exists():
        snapshots = list(hub_path.glob("*"))
        if snapshots:
            return str(snapshots[0])
    return "Qwen/Qwen3-ASR-0.6B"


def get_default_device() -> str:
    """Determine the optimal compute device (CUDA GPU if available, else CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_default_torch_dtype() -> str:
    """Return default torch data type based on hardware availability."""
    return "float16" if torch.cuda.is_available() else "int8"


class ASRConfig(BaseModel):
    """Configuration for Qwen3-ASR Speech-to-Text engine."""
    model_path: str = Field(
        default_factory=lambda: os.getenv("QWEN_ASR_MODEL_PATH", os.getenv("ASR_MODEL_PATH", get_default_qwen3_model_path())),
        description="Path or HuggingFace ID for Qwen3-ASR-0.6B model",
    )
    device: str = Field(
        default_factory=lambda: os.getenv("ASR_DEVICE", get_default_device()),
        description="Compute device: 'cuda' for GPU or 'cpu'",
    )
    torch_dtype: str = Field(
        default_factory=lambda: os.getenv("ASR_DTYPE", get_default_torch_dtype()),
        description="Torch precision: 'float16' / 'bfloat16' for CUDA, 'int8' dynamic for CPU",
    )
    sample_rate: int = Field(default=16000, description="ASR input sample rate in Hz")
    chunk_size_ms: int = Field(
        default_factory=lambda: int(os.getenv("ASR_CHUNK_SIZE_MS", "50")),
        description="Audio chunk processing window in milliseconds",
    )
    language: str = Field(
        default_factory=lambda: os.getenv("ASR_LANGUAGE", "Hindi"),
        description="Primary target language for speech recognition",
    )
    use_quantization: bool = Field(
        default_factory=lambda: os.getenv("ASR_QUANTIZE", "true").lower() in ("true", "1", "yes"),
        description="Enable dynamic INT8 quantization on CPU",
    )
    num_threads: int = Field(
        default_factory=lambda: int(os.getenv("ASR_NUM_THREADS", "8")),
        description="Number of CPU inference threads for PyTorch",
    )
    max_new_tokens: int = Field(
        default_factory=lambda: int(os.getenv("ASR_MAX_NEW_TOKENS", "48")),
        description="Maximum tokens generated per speech segment",
    )


class VADConfig(BaseModel):
    """Configuration for Silero Neural Voice Activity Detector."""
    confidence: float = Field(
        default_factory=lambda: float(os.getenv("VAD_CONFIDENCE", "0.45")),
        description="Speech probability threshold (0.0 - 1.0) to qualify as speech",
    )
    start_secs: float = Field(
        default_factory=lambda: float(os.getenv("VAD_START_SECS", "0.15")),
        description="Minimum duration of continuous speech to trigger speech start",
    )
    stop_secs: float = Field(
        default_factory=lambda: float(os.getenv("VAD_STOP_SECS", "0.45")),
        description="Duration of silence required to trigger speech end / turn completion",
    )
    sample_rate: int = Field(default=16000, description="VAD audio sample rate in Hz")
    chunk_size_samples: int = Field(default=512, description="VAD frame chunk size (512 samples = 32ms @ 16kHz)")
    pre_speech_pad_ms: int = Field(
        default_factory=lambda: int(os.getenv("VAD_PRE_PAD_MS", "250")),
        description="Duration of pre-speech audio buffer in ms to prevent clipping initial consonants",
    )
    min_speech_duration_ms: int = Field(
        default_factory=lambda: int(os.getenv("VAD_MIN_SPEECH_MS", "250")),
        description="Minimum valid utterance length to filter out transient pops/clicks",
    )


class LLMConfig(BaseModel):
    """Configuration for Local llama-cpp GPU LLM (Gemma 4 E2B Instruct) and Ollama fallback."""
    engine_type: str = Field(
        default_factory=lambda: os.getenv("LLM_ENGINE_TYPE", "llama_cpp").lower(),
        description="LLM engine: 'llama_cpp' (local fast GPU GGUF) or 'ollama'",
    )
    repo_id: str = Field(
        default_factory=lambda: os.getenv("LLM_REPO_ID", "unsloth/gemma-4-E2B-it-GGUF"),
        description="HuggingFace GGUF repository for local LLM",
    )
    filename: str = Field(
        default_factory=lambda: os.getenv("LLM_GGUF_FILENAME", "gemma-4-E2B-it-Q4_K_M.gguf"),
        description="GGUF model filename to download and load",
    )
    model_path: Optional[str] = Field(
        default_factory=lambda: os.getenv("LLM_MODEL_PATH", None),
        description="Direct local path to GGUF model file (if already downloaded)",
    )
    n_gpu_layers: int = Field(
        default_factory=lambda: int(os.getenv("LLM_N_GPU_LAYERS", "-1")),
        description="Number of layers to offload to GPU (-1 = all layers on CUDA)",
    )
    n_ctx: int = Field(
        default_factory=lambda: int(os.getenv("LLM_N_CTX", "2048")),
        description="Context window length in tokens",
    )
    base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        description="Ollama API base URL endpoint (used if engine_type='ollama')",
    )
    model: str = Field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gemma-4-e2b-it"),
        description="Model name identifier",
    )
    enable_thinking: bool = Field(
        default_factory=lambda: os.getenv("LLM_THINKING", "false").lower() in ("true", "1", "yes"),
        description="Whether thinking tokens / reasoning mode are enabled",
    )
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.6")),
        description="Sampling temperature for LLM text generation",
    )
    max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "256")),
        description="Maximum token budget for generated response",
    )
    system_prompt: str = Field(
        default_factory=lambda: os.getenv(
            "LLM_SYSTEM_PROMPT",
            "You are an ultra-fast, friendly, and helpful AI voice assistant for real-time conversation.\n"
            "Rules:\n"
            "1. Keep responses brief, natural, and conversational (1-2 sentences unless details are asked).\n"
            "2. Speak naturally in Hindi or English (or Hinglish) matching the user's language.\n"
            "3. When asked about user information, weather, or time, invoke the appropriate tools accurately.\n"
            "4. NEVER output internal thoughts, reasoning steps, or <think> tags."
        ),
        description="System prompt defining the voice persona and behavioral constraints",
    )


class TTSConfig(BaseModel):
    """Configuration for Multi-Engine Text-to-Speech synthesizer."""
    default_engine: str = Field(
        default_factory=lambda: os.getenv("TTS_ENGINE", "vits").lower(),
        description="Default TTS engine: 'vits' (local offline), 'edge' (free neural cloud), 'cartesia' (Sonic-3)",
    )
    sample_rate: int = Field(
        default_factory=lambda: int(os.getenv("TTS_SAMPLE_RATE", "48000")),
        description="Output audio sample rate for WebRTC stream (48000 Hz standard)",
    )
    vits_model_id: str = Field(
        default_factory=lambda: os.getenv("VITS_MODEL_ID", "facebook/mms-tts-hin"),
        description="HuggingFace model ID for local offline VITS Hindi synthesis",
    )
    vits_device: str = Field(
        default_factory=lambda: os.getenv("VITS_DEVICE", get_default_device()),
        description="Compute device for VITS neural inference ('cuda' or 'cpu')",
    )
    cartesia_api_key: str = Field(
        default_factory=lambda: os.getenv("CARTESIA_API_KEY", ""),
        description="API Key for Cartesia Sonic-3 cloud TTS",
    )
    cartesia_model: str = Field(
        default_factory=lambda: os.getenv("CARTESIA_MODEL", "sonic-3"),
        description="Cartesia model ID",
    )
    cartesia_voice_id: str = Field(
        default_factory=lambda: os.getenv("CARTESIA_VOICE_ID", "95d51f79-c397-46f9-b49a-23763d3eaa2d"),
        description="Cartesia voice identifier",
    )
    edge_voice_hi: str = Field(
        default_factory=lambda: os.getenv("EDGE_VOICE_HI", "hi-IN-SwaraNeural"),
        description="Microsoft Edge Neural Voice for Hindi",
    )
    edge_voice_en: str = Field(
        default_factory=lambda: os.getenv("EDGE_VOICE_EN", "en-US-JennyNeural"),
        description="Microsoft Edge Neural Voice for English",
    )


class WebRTCConfig(BaseModel):
    """Configuration for WebRTC media transport and signaling server."""
    stun_servers: List[str] = Field(
        default_factory=lambda: [
            "stun:stun.l.google.com:19302",
            "stun:stun1.l.google.com:19302",
            "stun:stun2.l.google.com:19302",
            "stun:stun.cloudflare.com:3478",
        ],
        description="STUN ICE servers for NAT traversal",
    )
    host: str = Field(
        default_factory=lambda: os.getenv("HOST", "0.0.0.0"),
        description="HTTP/WebRTC binding host address",
    )
    port: int = Field(
        default_factory=lambda: int(os.getenv("PORT", "7860")),
        description="HTTP/WebRTC binding port",
    )
    audio_sample_rate: int = Field(
        default=48000,
        description="WebRTC audio transport sample rate in Hz",
    )
    audio_channels: int = Field(
        default=1,
        description="Number of audio channels (1 = Mono, 2 = Stereo)",
    )
    frame_duration_ms: int = Field(
        default=20,
        description="WebRTC audio packet frame duration (20ms = 960 samples @ 48kHz)",
    )


class ServerConfig(BaseModel):
    """Root configuration aggregating all sub-systems."""
    asr: ASRConfig = Field(default_factory=ASRConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    webrtc: WebRTCConfig = Field(default_factory=WebRTCConfig)

    @classmethod
    def load_from_env(cls) -> "ServerConfig":
        """Load fresh configuration instance with latest environment overrides."""
        load_env_file()
        return cls(
            asr=ASRConfig(),
            vad=VADConfig(),
            llm=LLMConfig(),
            tts=TTSConfig(),
            webrtc=WebRTCConfig(),
        )


# Global singleton configuration instance
config = ServerConfig.load_from_env()

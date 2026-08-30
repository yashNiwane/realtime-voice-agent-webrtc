"""
Qwen3-ASR CUDA/CPU Optimized Speech-to-Text Engine.

Features:
- Automatic CUDA GPU acceleration with FP16/BF16 inference.
- Dynamic INT8 quantization on CPU for low-resource low-latency execution.
- Configurable language prompting (Hindi, English, Multilingual).
- Speech hallucination & noise artifact filtering.
- Thread-pool backed non-blocking async transcription helper.
"""

import asyncio
import concurrent.futures
import time
from typing import Optional, Tuple
import numpy as np
import torch
from loguru import logger
from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import parse_asr_output

from server.config import config, ASRConfig


def is_valid_speech_text(text: str) -> bool:
    """
    Filter out background noise artifacts, repetitive symbols, and silence hallucinations.

    Args:
        text: Raw transcription string from ASR output.

    Returns:
        True if the text contains valid spoken content, False otherwise.
    """
    if not text:
        return False
    cleaned = text.strip()
    if not cleaned:
        return False

    # Extract alphanumeric and Indic/Devanagari characters
    valid_chars = [c for c in cleaned if c.isalnum() or ("\u0900" <= c <= "\u097F")]
    if len(valid_chars) < 2:
        return False

    # Common ASR noise hallucinations on silence or background room rumble
    hallucination_phrases = {
        "thank you",
        "thanks for watching",
        "subtitles by",
        "amara.org",
        "[blank_audio]",
        "(music)",
        "(applause)",
        "(bell)",
        "(chime)",
        "you",
        "yeah",
        "bye",
        "goodbye",
        "...",
        "---",
        "***",
        "???",
        "okay",
    }
    if cleaned.lower() in hallucination_phrases:
        return False

    # Repetitive single character (e.g. "........", "aaaaaaa")
    if len(set(valid_chars)) == 1 and len(valid_chars) > 3:
        return False

    return True


class Qwen3ASREngine:
    """
    Production-grade inference engine for Qwen3-ASR-0.6B with CUDA and CPU optimizations.
    """

    def __init__(
        self,
        asr_config: Optional[ASRConfig] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        language: Optional[str] = None,
        use_quantization: Optional[bool] = None,
        num_threads: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ):
        cfg = asr_config or config.asr
        self.model_path = model_path or cfg.model_path
        self.language = language or cfg.language
        self.num_threads = num_threads or cfg.num_threads
        self.max_new_tokens = max_new_tokens or cfg.max_new_tokens

        # Resolve device
        requested_dev = (device or cfg.device).lower()
        if requested_dev.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(requested_dev)
            self.is_cuda = True
        else:
            self.device = torch.device("cpu")
            self.is_cuda = False
            torch.set_num_threads(self.num_threads)

        # Resolve precision / quantization
        self.dtype_str = (torch_dtype or cfg.torch_dtype).lower()
        self.use_quantization = (
            cfg.use_quantization if use_quantization is None else use_quantization
        )

        logger.info(
            f"🚀 Initializing Qwen3-ASR Engine (device={self.device}, dtype={self.dtype_str}, "
            f"quantize={self.use_quantization and not self.is_cuda}, threads={self.num_threads})..."
        )
        t0 = time.perf_counter()

        # Load base wrapper & processor
        self.wrapper = Qwen3ASRModel.from_pretrained(
            self.model_path,
            max_new_tokens=self.max_new_tokens,
        )
        self.model = self.wrapper.model
        self.processor = self.wrapper.processor

        # Hardware optimization
        if self.is_cuda:
            logger.info("Moving Qwen3-ASR model to CUDA GPU...")
            self.model = self.model.to(self.device)
            if self.dtype_str in ("float16", "fp16", "half"):
                self.model = self.model.half()
                logger.info("CUDA FP16 precision enabled for ultra-fast GPU decoding.")
            elif self.dtype_str in ("bfloat16", "bf16"):
                self.model = self.model.to(torch.bfloat16)
                logger.info("CUDA BF16 precision enabled.")
        else:
            if self.use_quantization:
                logger.info("Applying dynamic INT8 quantization on CPU Linear layers...")
                try:
                    self.model = torch.ao.quantization.quantize_dynamic(
                        self.model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    logger.info("CPU INT8 dynamic quantization applied successfully.")
                except Exception as e:
                    logger.warning(f"Failed to apply dynamic INT8 quantization: {e}. Using FP32.")

        self.model.eval()
        self.prompt = self.wrapper._build_text_prompt(context="", force_language=self.language)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="qwen_asr_worker"
        )

        load_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"✨ Qwen3-ASR Engine ready in {load_ms:.1f}ms on {self.device}")

        # Warm up engine
        self._warmup()

    def _warmup(self) -> None:
        """Warm up PyTorch JIT, CUDA kernels, and caches with synthetic audio."""
        try:
            dummy_pcm = np.zeros(8000, dtype=np.float32)  # 0.5s silence
            inputs = self.processor(
                text=[self.prompt], audio=[dummy_pcm], return_tensors="pt", padding=True
            )
            inputs = inputs.to(self.device)
            if self.is_cuda and self.dtype_str in ("float16", "fp16", "half"):
                if "input_features" in inputs and inputs["input_features"].dtype == torch.float32:
                    inputs["input_features"] = inputs["input_features"].half()

            with torch.inference_mode():
                _ = self.model.generate(**inputs, max_new_tokens=4)
            logger.debug("Qwen3-ASR warmup complete.")
        except Exception as e:
            logger.warning(f"Qwen3-ASR warmup note: {e}")

    @torch.inference_mode()
    def transcribe_pcm(self, pcm_data: np.ndarray, sr: int = 16000) -> Tuple[str, str, float]:
        """
        Transcribe 1D float32 or int16 PCM numpy array.

        Args:
            pcm_data: Audio waveform as 1D numpy array.
            sr: Input sample rate in Hz (default 16000).

        Returns:
            Tuple of (transcribed_text, detected_language, latency_ms).
        """
        t0 = time.perf_counter()
        if pcm_data is None or len(pcm_data) == 0:
            return "", self.language or "", 0.0

        if pcm_data.ndim > 1:
            pcm_data = pcm_data.flatten()

        # Convert int16 to float32 in range [-1.0, 1.0]
        if pcm_data.dtype == np.int16:
            pcm_float = pcm_data.astype(np.float32) / 32768.0
        elif pcm_data.dtype != np.float32:
            pcm_float = pcm_data.astype(np.float32)
        else:
            pcm_float = pcm_data

        # Resample if not 16kHz
        if sr != 16000 and len(pcm_float) > 0:
            target_samples = int(round(len(pcm_float) * 16000 / sr))
            pcm_float = np.interp(
                np.linspace(0, len(pcm_float), target_samples, endpoint=False),
                np.arange(len(pcm_float)),
                pcm_float,
            ).astype(np.float32)

        # Ignore tiny audio bursts (< 25ms / 400 samples)
        if len(pcm_float) < 400:
            return "", self.language or "", 0.0

        inputs = self.processor(
            text=[self.prompt], audio=[pcm_float], return_tensors="pt", padding=True
        )
        inputs = inputs.to(self.device)
        if self.is_cuda and self.dtype_str in ("float16", "fp16", "half"):
            if "input_features" in inputs and inputs["input_features"].dtype == torch.float32:
                inputs["input_features"] = inputs["input_features"].half()

        text_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        decoded = self.processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        raw_out = decoded[0] if decoded else ""
        lang, text = parse_asr_output(raw_out, user_language=self.language)
        latency_ms = (time.perf_counter() - t0) * 1000

        final_text = text.strip() if text else ""
        if not is_valid_speech_text(final_text):
            final_text = ""

        return final_text, lang or self.language or "", latency_ms

    async def async_transcribe_pcm(
        self, pcm_data: np.ndarray, sr: int = 16000
    ) -> Tuple[str, str, float]:
        """
        Non-blocking async wrapper to transcribe PCM audio on the background worker pool.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self.transcribe_pcm, pcm_data, sr
        )

    def close(self) -> None:
        """Release worker threads and resources."""
        self._executor.shutdown(wait=False)

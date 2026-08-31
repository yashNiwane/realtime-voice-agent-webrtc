"""
Silero Neural Voice Activity Detection (VAD) Analyzer.

Features:
- Real-time chunk-by-chunk speech probability inference (512 samples = 32ms @ 16kHz).
- Onset/offset speech hysteresis avoiding false triggers and premature speech cuts.
- Pre-speech circular buffer ensuring leading syllables/consonants are preserved.
- Automatic utterance extraction and silence trimming.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional
import numpy as np
import torch
from loguru import logger
from silero_vad import load_silero_vad

from server.config import config, VADConfig


@dataclass
class VADResult:
    """
    Structured outcome of processing a single 512-sample audio chunk.
    """
    probability: float
    is_speech: bool
    event: Optional[str] = None  # None, "SPEECH_START", or "SPEECH_END"
    audio_utterance: Optional[np.ndarray] = None  # Full collected float32 PCM utterance on SPEECH_END
    rms: float = 0.0


class SileroVADAnalyzer:
    """
    Stateful Voice Activity Detection analyzer powered by Silero Neural VAD.
    """

    def __init__(
        self,
        vad_config: Optional[VADConfig] = None,
        confidence: Optional[float] = None,
        start_secs: Optional[float] = None,
        stop_secs: Optional[float] = None,
        sample_rate: Optional[int] = None,
    ):
        cfg = vad_config or config.vad
        self.confidence = confidence if confidence is not None else cfg.confidence
        self.start_secs = start_secs if start_secs is not None else cfg.start_secs
        self.stop_secs = stop_secs if stop_secs is not None else cfg.stop_secs
        self.sample_rate = sample_rate or cfg.sample_rate
        self.chunk_size = cfg.chunk_size_samples  # 512 samples

        # Calculate chunk thresholds based on 32ms (512 samples @ 16kHz)
        chunk_duration_sec = self.chunk_size / self.sample_rate  # 0.032s
        self.start_chunks_required = max(1, int(round(self.start_secs / chunk_duration_sec)))
        self.stop_chunks_required = max(1, int(round(self.stop_secs / chunk_duration_sec)))

        # Pre-speech ring buffer (e.g. 250ms = ~8 chunks)
        pre_chunks_count = max(2, int(round((cfg.pre_speech_pad_ms / 1000.0) / chunk_duration_sec)))
        self.pre_buffer: Deque[np.ndarray] = deque(maxlen=pre_chunks_count)

        # Minimum speech duration in samples
        self.min_speech_samples = int(self.sample_rate * (cfg.min_speech_duration_ms / 1000.0))

        logger.info(
            f"🎙️ Silero VAD Initialized (thresh={self.confidence:.2f}, "
            f"start={self.start_secs}s ({self.start_chunks_required} chunks), "
            f"stop={self.stop_secs}s ({self.stop_chunks_required} chunks), "
            f"pre_pad={cfg.pre_speech_pad_ms}ms)..."
        )

        # Load Silero model
        self.model = load_silero_vad()
        self.model.eval()

        # State tracking
        self.is_speaking: bool = False
        self.consecutive_speech_chunks: int = 0
        self.consecutive_silence_chunks: int = 0
        self.speech_buffer: List[np.ndarray] = []

    def reset(self) -> None:
        """Reset internal speech state, buffers, and VAD model recurrent states."""
        self.is_speaking = False
        self.consecutive_speech_chunks = 0
        self.consecutive_silence_chunks = 0
        self.speech_buffer.clear()
        self.pre_buffer.clear()
        try:
            if hasattr(self.model, "reset_states"):
                self.model.reset_states()
        except Exception as e:
            logger.debug(f"VAD state reset note: {e}")

    @torch.inference_mode()
    def process_chunk(self, chunk_512: np.ndarray) -> VADResult:
        """
        Process a single 512-sample float32 audio chunk (32ms @ 16kHz).

        Args:
            chunk_512: 1D numpy array of 512 float32 samples in range [-1.0, 1.0].

        Returns:
            VADResult with speech probability, current speaking state, and speech events.
        """
        if chunk_512.ndim > 1:
            chunk_512 = chunk_512.flatten()

        # Ensure exactly 512 samples
        if len(chunk_512) != self.chunk_size:
            if len(chunk_512) < self.chunk_size:
                chunk_512 = np.pad(chunk_512, (0, self.chunk_size - len(chunk_512)), mode="constant")
            else:
                chunk_512 = chunk_512[: self.chunk_size]

        if chunk_512.dtype != np.float32:
            if chunk_512.dtype == np.int16:
                chunk_512 = chunk_512.astype(np.float32) / 32768.0
            else:
                chunk_512 = chunk_512.astype(np.float32)

        # Compute RMS and peak amplitude
        peak = float(np.max(np.abs(chunk_512)))
        rms = float(np.sqrt(np.mean(chunk_512**2)))

        tensor_chunk = torch.from_numpy(chunk_512).float()
        raw_prob = self.model(tensor_chunk, self.sample_rate)
        speech_prob = float(raw_prob.item())

        # Gate only on absolute digital silence / zero signal (RMS < 0.0008)
        if rms < 0.0008 and peak < 0.0015:
            speech_prob = min(speech_prob, 0.05)

        is_current_speech = (speech_prob >= self.confidence)
        event: Optional[str] = None
        utterance_pcm: Optional[np.ndarray] = None

        if is_current_speech:
            self.consecutive_speech_chunks += 1
            self.consecutive_silence_chunks = 0

            if not self.is_speaking:
                if self.consecutive_speech_chunks >= self.start_chunks_required:
                    # Trigger Speech Start
                    self.is_speaking = True
                    event = "SPEECH_START"
                    self.speech_buffer.clear()
                    # Add pre-speech buffer so leading consonants are never truncated
                    self.speech_buffer.extend(list(self.pre_buffer))
                    self.speech_buffer.append(chunk_512.copy())
                else:
                    # Maintain pre-speech buffer while confirming onset
                    self.pre_buffer.append(chunk_512.copy())
            else:
                self.speech_buffer.append(chunk_512.copy())

        else:
            # Current frame is non-speech / silence
            self.consecutive_speech_chunks = 0

            if not self.is_speaking:
                self.pre_buffer.append(chunk_512.copy())
            else:
                self.speech_buffer.append(chunk_512.copy())
                self.consecutive_silence_chunks += 1

                if self.consecutive_silence_chunks >= self.stop_chunks_required:
                    # Trigger Speech End
                    self.is_speaking = False
                    event = "SPEECH_END"

                    if self.speech_buffer:
                        full_audio = np.concatenate(self.speech_buffer)
                        # Trim trailing silence chunks
                        trim_samples = self.consecutive_silence_chunks * self.chunk_size
                        if len(full_audio) > trim_samples:
                            trimmed_audio = full_audio[:-trim_samples]
                        else:
                            trimmed_audio = full_audio

                        # Verify minimum utterance duration
                        if len(trimmed_audio) >= self.min_speech_samples:
                            utterance_pcm = trimmed_audio
                        else:
                            logger.debug("Discarded short speech burst below minimum duration threshold.")

                    self.speech_buffer.clear()
                    self.pre_buffer.clear()
                    self.reset()

        return VADResult(
            probability=speech_prob,
            is_speech=self.is_speaking,
            event=event,
            audio_utterance=utterance_pcm,
            rms=rms,
        )

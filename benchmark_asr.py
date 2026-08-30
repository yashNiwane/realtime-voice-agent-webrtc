"""
Benchmark script for Qwen3-ASR latency, chunk processing time, and Real-Time Factor (RTF).
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import time
import numpy as np
import soundfile as sf
import torch
from loguru import logger

from config import config
from qwen_stt_service import Qwen3ASREngine


def run_benchmark():
    print("=" * 70)
    print(" [BENCHMARK] Qwen3-ASR Latency & Real-Time Performance")
    print("=" * 70)
    print(f" * Model Path:      {config.asr.model_path}")
    print(f" * Quantization:    {config.asr.use_quantization} (INT8 Dynamic)")
    print(f" * CPU Threads:     {config.asr.num_threads}")
    print("=" * 70)

    print("\nLoading Qwen3-ASR Engine...")
    t0 = time.perf_counter()
    engine = Qwen3ASREngine(
        model_path=config.asr.model_path,
        language=config.asr.language,
        use_quantization=config.asr.use_quantization,
        num_threads=config.asr.num_threads,
        max_new_tokens=32,
    )
    load_time = (time.perf_counter() - t0) * 1000
    print(f"Engine loaded and warmed up in {load_time:.1f}ms\n")

    # 1. Chunk Latency Test (Simulating continuous real-time audio streams)
    chunk_durations_ms = [25, 50, 100, 200, 500]
    print("--- 1. Chunk Processing Latency (Target: <50ms-100ms) ---")
    print(f"{'Chunk Size (ms)':<18} | {'Samples':<10} | {'Processing Time (ms)':<22} | {'Status'}")
    print("-" * 65)

    for chunk_ms in chunk_durations_ms:
        num_samples = int(16000 * (chunk_ms / 1000.0))
        dummy_chunk = np.random.randn(num_samples).astype(np.float32) * 0.05

        latencies = []
        for _ in range(5):
            t_start = time.perf_counter()
            _ = engine.transcribe_pcm(dummy_chunk, sr=16000)
            latencies.append((time.perf_counter() - t_start) * 1000)

        avg_lat = np.mean(latencies[1:])  # Ignore first iteration
        status = "[OK] Sub-50ms" if avg_lat <= 50 else ("[OK] Sub-100ms" if avg_lat <= 100 else "[*] >100ms")
        print(f"{chunk_ms:<18} | {num_samples:<10} | {avg_lat:>18.2f} ms | {status}")

    # 2. Utterance Transcription Test with Real Speech Audio
    print("\n--- 2. Real Speech Utterance Transcription Test ---")
    try:
        data, sr = sf.read("test_cartesia.wav")
        audio_dur_sec = len(data) / sr
        print(f"Input Audio File: test_cartesia.wav ({audio_dur_sec:.2f} seconds)")

        t_start = time.perf_counter()
        text, lang, lat_ms = engine.transcribe_pcm(data, sr=sr)
        total_time_sec = (time.perf_counter() - t_start)
        rtf = total_time_sec / audio_dur_sec

        print(f"* Transcribed Text:   '{text}'")
        print(f"* Detected Language:  {lang}")
        print(f"* Latency:            {lat_ms:.1f} ms")
        print(f"* Real-Time Factor:   {rtf:.2f}x (Lower is faster, <1.0 means faster than real-time)")
    except Exception as e:
        print(f"Utterance test skipped: {e}")

    print("\n" + "=" * 70)
    print(" Benchmark completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()

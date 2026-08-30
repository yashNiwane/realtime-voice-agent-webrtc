"""
CLI Interactive Voice Assistant Runner.
Captures microphone input and plays back Cartesia TTS audio through local speakers using SoundDevice.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import asyncio
import signal
from loguru import logger

from pipecat.pipeline.runner import PipelineRunner

from config import config
from pipeline import create_voice_pipeline
from sounddevice_transport import SoundDeviceTransport, SoundDeviceTransportParams


async def main():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    print("=" * 70)
    print(" [AGENT] Realtime Voice Assistant: Qwen3-ASR + Ollama Gemma4 + Cartesia TTS")
    print("=" * 70)
    print(f" * ASR Model:   {config.asr.model_path}")
    print(f" * ASR Quant:   {config.asr.use_quantization} (INT8 Dynamic)")
    print(f" * ASR Chunk:   {config.asr.chunk_size_ms}ms")
    print(f" * LLM Model:   {config.llm.model} @ {config.llm.base_url}")
    print(f" * TTS Voice:   {config.tts.voice_id} (Cartesia sonic-3)")
    print("=" * 70)
    print("Initializing devices and loading models... (Please wait)")

    transport = SoundDeviceTransport(
        params=SoundDeviceTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=config.asr.sample_rate,
            audio_out_sample_rate=config.tts.sample_rate,
            chunk_size_ms=config.asr.chunk_size_ms,
        )
    )

    task, context = create_voice_pipeline(transport, config)
    runner = PipelineRunner()

    print("\n[READY] Agent is LIVE! Speak into your microphone (Press Ctrl+C to stop)...")

    # Handle graceful exit
    stop_event = asyncio.Event()

    def signal_handler():
        print("\nStopping voice assistant...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows signal handling fallback
            pass

    runner_task = asyncio.create_task(runner.run(task))

    try:
        await runner_task
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        await runner.cancel()
        print("Assistant stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")

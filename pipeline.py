"""
Pipeline orchestration module for Realtime Voice Agent with Qwen3-ASR, Ollama, and Cartesia TTS.
"""

import re
from typing import Optional

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TextFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.tts_service import TTSService
from pipecat.transports.base_transport import BaseTransport

from config import AppConfig, config
from qwen_stt_service import Qwen3ASRSTTService


class ThinkingFilterProcessor(FrameProcessor):
    """Filters out any <think>...</think> or <thought>...</thought> reasoning blocks from LLM stream before TTS."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._in_think_block = False
        self._buffer = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            self._in_think_block = False
            self._buffer = ""
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (LLMTextFrame, TextFrame)):
            text = frame.text
            self._buffer += text

            while True:
                if self._in_think_block:
                    if "</think>" in self._buffer:
                        self._buffer = self._buffer.split("</think>", 1)[1]
                        self._in_think_block = False
                    elif "</thought>" in self._buffer:
                        self._buffer = self._buffer.split("</thought>", 1)[1]
                        self._in_think_block = False
                    else:
                        self._buffer = ""
                        break
                else:
                    if "<think>" in self._buffer:
                        before, after = self._buffer.split("<think>", 1)
                        if before:
                            await self.push_frame(type(frame)(text=before), direction)
                        self._buffer = after
                        self._in_think_block = True
                    elif "<thought>" in self._buffer:
                        before, after = self._buffer.split("<thought>", 1)
                        if before:
                            await self.push_frame(type(frame)(text=before), direction)
                        self._buffer = after
                        self._in_think_block = True
                    else:
                        if len(self._buffer) > 0 and not self._buffer.endswith(("<", "<t", "<th", "<thi", "<thin")):
                            await self.push_frame(type(frame)(text=self._buffer), direction)
                            self._buffer = ""
                        break
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            if not self._in_think_block and self._buffer:
                clean_text = re.sub(r"<think>.*?</think>", "", self._buffer, flags=re.DOTALL)
                clean_text = re.sub(r"<thought>.*?</thought>", "", clean_text, flags=re.DOTALL)
                if clean_text:
                    await self.push_frame(LLMTextFrame(text=clean_text), direction)
            self._in_think_block = False
            self._buffer = ""

        await self.push_frame(frame, direction)


def create_voice_pipeline(
    transport: BaseTransport,
    app_config: Optional[AppConfig] = None,
) -> tuple[PipelineTask, LLMContext]:
    """
    Build and configure the complete Pipecat voice pipeline:
    Transport Input -> Silero VAD -> Qwen3-ASR STT -> User Context Aggregator ->
    Ollama LLM (Gemma4:31b-cloud) -> Thinking Filter -> Cartesia TTS -> Transport Output -> Assistant Context Aggregator
    """
    cfg = app_config or config

    logger.info("Building Voice Pipeline...")
    logger.info(f"ASR: Qwen3-ASR ({cfg.asr.model_path}, quantize={cfg.asr.use_quantization})")
    logger.info(f"LLM: Ollama ({cfg.llm.base_url}, model={cfg.llm.model}, thinking={cfg.llm.enable_thinking})")
    logger.info(f"TTS: Cartesia ({cfg.tts.model}, voice_id={cfg.tts.voice_id})")

    # 1. Speech-to-Text: Qwen3-ASR
    stt_service = Qwen3ASRSTTService(
        model_path=cfg.asr.model_path,
        language=cfg.asr.language,
        use_quantization=cfg.asr.use_quantization,
        num_threads=cfg.asr.num_threads,
        chunk_size_ms=cfg.asr.chunk_size_ms,
        sample_rate=cfg.asr.sample_rate,
        stream_interim_results=True,
    )

    # 2. LLM: Ollama Gemma 4 (gemma4:31b-cloud)
    llm_service = OLLamaLLMService(
        base_url=cfg.llm.base_url,
        settings=OLLamaLLMService.Settings(
            model=cfg.llm.model,
            extra={
                "think": cfg.llm.enable_thinking,
                "options": {"temperature": 0.3},
            }
        ),
    )
    # Register tool handlers
    from tools import save_user_info, get_current_weather, get_current_time, TOOLS_SCHEMA
    llm_service.register_function("save_user_info", save_user_info)
    llm_service.register_function("get_current_weather", get_current_weather)
    llm_service.register_function("get_current_time", get_current_time)

    # 3. Thinking Filter Processor (Strips any <think> reasoning tags)
    thinking_filter = ThinkingFilterProcessor()

    # 4. TTS: Dual-Engine TTS (Cartesia with Neural Fallback)
    class DualEngineTTSService(TTSService):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        async def run_tts(self, text: str):
            from tts_manager import synthesize_speech
            audio_bytes, lat, engine = await synthesize_speech(
                text=text,
                language=cfg.asr.language or "Hindi",
                sample_rate=self.sample_rate or cfg.tts.sample_rate,
            )
            if audio_bytes:
                yield TTSAudioRawFrame(
                    audio=audio_bytes,
                    sample_rate=self.sample_rate or cfg.tts.sample_rate,
                    num_channels=1,
                )

    tts_service = DualEngineTTSService(
        sample_rate=cfg.tts.sample_rate,
    )

    # 5. Context & Conversation Management
    context = LLMContext(
        messages=messages,
        tools=[save_user_info, get_current_weather, get_current_time],
    )

    # 6. VAD: Silero Neural VAD Analyzer (Accurate Voice vs Noise Discrimination)
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            confidence=0.75,   # High confidence threshold for real speech
            start_secs=0.20,   # Requires 200ms speech to reject clicks/bumps
            stop_secs=0.40,    # Natural end-of-speech cutoff
            min_volume=0.50,   # Rejects ambient background murmur/hum
        )
    )

    context_pair = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad_analyzer,
            user_turn_stop_timeout=3.0,
        ),
    )
    user_aggregator = context_pair.user()
    assistant_aggregator = context_pair.assistant()

    # 7. Pipeline assembly
    pipeline = Pipeline(
        [
            transport.input(),
            stt_service,
            user_aggregator,
            llm_service,
            thinking_filter,
            tts_service,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            send_initial_empty_metrics=False,
        ),
    )

    return task, context

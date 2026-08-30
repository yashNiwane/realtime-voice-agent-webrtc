"""
SoundDevice Audio Transport for Pipecat AI framework.
Provides low-latency microphone capture and speaker output using sounddevice on Windows/Linux/macOS.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import sounddevice as sd
from loguru import logger
from pydantic import Field

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams


class SoundDeviceTransportParams(TransportParams):
    """Configuration parameters for SoundDevice transport."""

    input_device_index: Optional[int] = None
    output_device_index: Optional[int] = None
    chunk_size_ms: int = 50


class SoundDeviceInputTransport(BaseInputTransport):
    """Captures microphone audio via sounddevice and streams InputAudioRawFrames into Pipecat pipeline."""

    _params: SoundDeviceTransportParams

    def __init__(self, params: SoundDeviceTransportParams, **kwargs):
        super().__init__(params, **kwargs)
        self._in_stream: Optional[sd.RawInputStream] = None
        self._sample_rate = 0

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        # Block size for streaming audio chunk
        blocksize = int(self.sample_rate * (self._params.chunk_size_ms / 1000.0))

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"SoundDevice input status: {status}")
            frame = InputAudioRawFrame(
                audio=bytes(indata),
                sample_rate=self.sample_rate,
                num_channels=self._params.audio_in_channels,
            )
            asyncio.run_coroutine_threadsafe(
                self.push_audio_frame(frame), self.get_event_loop()
            )

        try:
            self._in_stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=blocksize,
                device=self._params.input_device_index,
                channels=self._params.audio_in_channels,
                dtype="int16",
                callback=callback,
            )
            logger.info(
                f"SoundDevice input initialized (device={self._params.input_device_index or 'default'}, "
                f"sr={self.sample_rate}Hz, chunk={self._params.chunk_size_ms}ms)"
            )
        except Exception as e:
            logger.error(f"Failed to initialize SoundDevice input: {e}")

    async def cleanup(self):
        await super().cleanup()
        if self._in_stream:
            self._in_stream.stop()
            self._in_stream.close()
            self._in_stream = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if self._in_stream:
            self._in_stream.start()
            await self.set_transport_ready(frame)
            logger.info("SoundDevice microphone recording started.")


class SoundDeviceOutputTransport(BaseOutputTransport):
    """Plays audio frames through system speakers via sounddevice."""

    _params: SoundDeviceTransportParams

    def __init__(self, params: SoundDeviceTransportParams, **kwargs):
        super().__init__(params, **kwargs)
        self._out_stream: Optional[sd.RawOutputStream] = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sd_output")

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        try:
            self._out_stream = sd.RawOutputStream(
                samplerate=self.sample_rate,
                device=self._params.output_device_index,
                channels=self._params.audio_out_channels,
                dtype="int16",
            )
            logger.info(
                f"SoundDevice output initialized (device={self._params.output_device_index or 'default'}, "
                f"sr={self.sample_rate}Hz)"
            )
        except Exception as e:
            logger.error(f"Failed to initialize SoundDevice output: {e}")

    async def cleanup(self):
        try:
            await super().cleanup()
            if self._out_stream:
                self._out_stream.stop()
                self._out_stream.close()
                self._out_stream = None
        finally:
            self._executor.shutdown(wait=False)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if self._out_stream:
            self._out_stream.start()
            await self.set_transport_ready(frame)
            logger.info("SoundDevice speaker playback started.")

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        if self._out_stream and frame.audio:
            await self.get_event_loop().run_in_executor(
                self._executor, self._out_stream.write, frame.audio
            )
            return True
        return False


class SoundDeviceTransport(BaseTransport):
    """Unified full-duplex SoundDevice Transport for Pipecat."""

    def __init__(
        self,
        params: Optional[SoundDeviceTransportParams] = None,
        input_device_index: Optional[int] = None,
        output_device_index: Optional[int] = None,
        chunk_size_ms: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._params = params or SoundDeviceTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            input_device_index=input_device_index,
            output_device_index=output_device_index,
            chunk_size_ms=chunk_size_ms,
        )
        self._input: Optional[SoundDeviceInputTransport] = None
        self._output: Optional[SoundDeviceOutputTransport] = None

    def input(self) -> FrameProcessor:
        if not self._input:
            self._input = SoundDeviceInputTransport(self._params)
        return self._input

    def output(self) -> FrameProcessor:
        if not self._output:
            self._output = SoundDeviceOutputTransport(self._params)
        return self._output

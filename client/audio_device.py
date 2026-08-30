"""
Audio Device I/O and Streaming Utilities for Realtime WebRTC Client.
Provides:
- Device discovery and selection for Microphones and Speakers
- Low-latency microphone capture as an aiortc MediaStreamTrack
- Realtime speaker audio playback from incoming WebRTC MediaStreamTrack
- Software gain, mute control, RMS level metering, and buffer management
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import fractions
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import av
import numpy as np
import sounddevice as sd
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError

logger = logging.getLogger("audio_device")


class AudioDeviceManager:
    """Manages audio device discovery, querying, and selection."""

    @staticmethod
    def list_devices() -> List[Dict[str, Any]]:
        """Return a list of all audio devices available on the system."""
        devices = []
        try:
            device_list = sd.query_devices()
            default_in, default_out = sd.default.device
            host_apis = sd.query_hostapis()

            for idx, dev in enumerate(device_list):
                host_api_name = (
                    host_apis[dev["hostapi"]]["name"]
                    if 0 <= dev["hostapi"] < len(host_apis)
                    else "Unknown"
                )
                devices.append({
                    "index": idx,
                    "name": dev["name"],
                    "hostapi": host_api_name,
                    "max_input_channels": dev["max_input_channels"],
                    "max_output_channels": dev["max_output_channels"],
                    "default_samplerate": int(dev["default_samplerate"]),
                    "is_default_input": (idx == default_in),
                    "is_default_output": (idx == default_out),
                })
        except Exception as e:
            logger.error(f"Error querying audio devices: {e}")
        return devices

    @classmethod
    def get_input_devices(cls) -> List[Dict[str, Any]]:
        """Get all devices capable of audio input (microphones)."""
        return [d for d in cls.list_devices() if d["max_input_channels"] > 0]

    @classmethod
    def get_output_devices(cls) -> List[Dict[str, Any]]:
        """Get all devices capable of audio output (speakers/headphones)."""
        return [d for d in cls.list_devices() if d["max_output_channels"] > 0]

    @classmethod
    def get_default_input(cls) -> Optional[Dict[str, Any]]:
        """Get default microphone device information."""
        for d in cls.get_input_devices():
            if d["is_default_input"]:
                return d
        inputs = cls.get_input_devices()
        return inputs[0] if inputs else None

    @classmethod
    def get_default_output(cls) -> Optional[Dict[str, Any]]:
        """Get default speaker/headphone device information."""
        for d in cls.get_output_devices():
            if d["is_default_output"]:
                return d
        outputs = cls.get_output_devices()
        return outputs[0] if outputs else None

    @classmethod
    def find_device(
        cls, query: Optional[Any], is_input: bool = True
    ) -> Optional[int]:
        """
        Find device index by integer ID or case-insensitive substring search.
        Returns device index integer or None.
        """
        if query is None:
            default_dev = cls.get_default_input() if is_input else cls.get_default_output()
            return default_dev["index"] if default_dev else None

        # If already an integer index
        if isinstance(query, int) or (isinstance(query, str) and query.isdigit()):
            idx = int(query)
            devices = cls.get_input_devices() if is_input else cls.get_output_devices()
            for d in devices:
                if d["index"] == idx:
                    return idx
            logger.warning(f"Device index {idx} not found in valid {'input' if is_input else 'output'} devices.")
            return idx

        # Search by substring
        query_str = str(query).lower().strip()
        devices = cls.get_input_devices() if is_input else cls.get_output_devices()
        for d in devices:
            if query_str in d["name"].lower():
                return d["index"]

        logger.warning(f"No audio device matching '{query}' found. Using default.")
        default_dev = cls.get_default_input() if is_input else cls.get_default_output()
        return default_dev["index"] if default_dev else None

    @classmethod
    def print_devices_table(cls):
        """Print a structured overview table of audio devices."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title="Available Audio Devices", show_header=True, header_style="bold cyan")
            table.add_column("Idx", style="dim", width=4)
            table.add_column("Device Name", style="white", min_width=30)
            table.add_column("Type", style="magenta", width=10)
            table.add_column("Host API", style="green", width=14)
            table.add_column("Sample Rate", style="yellow", width=12)
            table.add_column("Default", style="bold green", width=10)

            for d in cls.list_devices():
                dev_type = []
                if d["max_input_channels"] > 0:
                    dev_type.append(f"In ({d['max_input_channels']}ch)")
                if d["max_output_channels"] > 0:
                    dev_type.append(f"Out ({d['max_output_channels']}ch)")
                type_str = ", ".join(dev_type)

                default_str = ""
                if d["is_default_input"] and d["is_default_output"]:
                    default_str = "IN/OUT *"
                elif d["is_default_input"]:
                    default_str = "IN *"
                elif d["is_default_output"]:
                    default_str = "OUT *"

                table.add_row(
                    str(d["index"]),
                    d["name"],
                    type_str,
                    d["hostapi"],
                    f"{d['default_samplerate']} Hz",
                    default_str,
                )
            console.print(table)
        except ImportError:
            print("\n=== Available Audio Devices ===")
            for d in cls.list_devices():
                in_out = f"In:{d['max_input_channels']} Out:{d['max_output_channels']}"
                def_mark = " [DEFAULT]" if (d["is_default_input"] or d["is_default_output"]) else ""
                print(f"[{d['index']:2d}] {d['name']} ({d['hostapi']}) - {in_out} - {d['default_samplerate']}Hz{def_mark}")
            print("===============================\n")


class MicrophoneAudioTrack(MediaStreamTrack):
    """
    aiortc MediaStreamTrack implementation that captures local microphone input
    using sounddevice and emits av.AudioFrame packets at a fixed sample rate.
    """

    kind = "audio"

    def __init__(
        self,
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_ms: int = 20,
        gain: float = 1.0,
    ):
        super().__init__()
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_ms = chunk_duration_ms
        self.samples_per_chunk = int(sample_rate * (chunk_duration_ms / 1000.0))
        self.gain = gain
        self.muted = False

        self._queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=100)
        self._pts = 0
        self._time_base = fractions.Fraction(1, self.sample_rate)
        self._stream: Optional[sd.RawInputStream] = None
        self._running = False
        self._last_rms: float = 0.0
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """Start microphone capture stream."""
        if self._running:
            return

        self._loop = loop or asyncio.get_event_loop()
        self._running = True

        def _sd_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Microphone overflow/status: {status}")

            if not self._running:
                return

            raw_bytes = bytes(indata)

            # Compute RMS level
            try:
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
                if len(samples) > 0:
                    self._last_rms = float(np.sqrt(np.mean(samples ** 2)))
            except Exception:
                pass

            if self.muted:
                # Send silence if muted
                raw_bytes = b"\x00" * len(raw_bytes)
            elif self.gain != 1.0:
                try:
                    samples = (samples * self.gain).clip(-32768, 32767).astype(np.int16)
                    raw_bytes = samples.tobytes()
                except Exception:
                    pass

            if self._loop and self._loop.is_running():
                try:
                    self._loop.call_soon_threadsafe(self._put_chunk, raw_bytes)
                except Exception:
                    pass

        try:
            self._stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=self.samples_per_chunk,
                device=self.device_index,
                channels=self.channels,
                dtype="int16",
                callback=_sd_callback,
            )
            self._stream.start()
            logger.info(
                f"Microphone capture started [Dev: {self.device_index or 'Default'}, "
                f"SR: {self.sample_rate}Hz, Ch: {self.channels}, Chunk: {self.chunk_duration_ms}ms]"
            )
        except Exception as e:
            logger.error(f"Failed to start sounddevice microphone stream: {e}")
            raise

    def _put_chunk(self, chunk: bytes):
        """Thread-safe queue insertion with buffer overflow handling."""
        try:
            if self._queue.full():
                # Drop oldest chunk to maintain low latency
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            self._queue.put_nowait(chunk)
        except Exception:
            pass

    async def recv(self) -> av.AudioFrame:
        """
        Produce the next av.AudioFrame for the WebRTC peer connection.
        Called continuously by aiortc.
        """
        if not self._running:
            raise MediaStreamError

        try:
            # Wait for next audio chunk from sounddevice callback
            chunk_data = await asyncio.wait_for(self._queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            # Generate silence if no chunk arrived to prevent WebRTC pipeline stall
            chunk_data = b"\x00" * (self.samples_per_chunk * self.channels * 2)

        if chunk_data is None or not self._running:
            raise MediaStreamError

        # Convert raw PCM16 bytes to numpy array
        pcm_array = np.frombuffer(chunk_data, dtype=np.int16)
        if self.channels == 1:
            pcm_array = pcm_array.reshape(1, -1)
        else:
            pcm_array = pcm_array.reshape(self.channels, -1)

        layout = "mono" if self.channels == 1 else "stereo"
        frame = av.AudioFrame.from_ndarray(pcm_array, format="s16", layout=layout)
        frame.sample_rate = self.sample_rate
        frame.pts = self._pts
        frame.time_base = self._time_base

        self._pts += self.samples_per_chunk
        return frame

    def get_rms(self) -> float:
        """Return current RMS energy value of microphone."""
        return self._last_rms

    def set_muted(self, muted: bool):
        """Mute/unmute microphone capture."""
        self.muted = muted
        logger.info(f"Microphone {'muted' if muted else 'unmuted'}.")

    def set_gain(self, gain: float):
        """Set software gain multiplier (1.0 = normal, 2.0 = +6dB)."""
        self.gain = max(0.0, min(gain, 5.0))

    def stop(self):
        """Stop microphone capture and release hardware resources."""
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.debug(f"Error stopping mic stream: {e}")
            self._stream = None
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        logger.info("Microphone capture stopped.")


class SpeakerAudioPlayer:
    """
    Consumes remote WebRTC MediaStreamTrack audio frames and plays them in realtime
    through local system speakers using sounddevice with zero glitching and interruption support.
    """

    def __init__(
        self,
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        volume: float = 1.0,
    ):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self.volume = volume
        self.muted = False

        self._stream: Optional[sd.RawOutputStream] = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sd_speaker")
        self._play_task: Optional[asyncio.Task] = None
        self._running = False
        self._track: Optional[MediaStreamTrack] = None

    def start(self, track: MediaStreamTrack, sample_rate: Optional[int] = None):
        """Attach incoming WebRTC track and begin playback loop."""
        if self._running:
            self.stop()

        self._track = track
        if sample_rate:
            self.sample_rate = sample_rate

        self._running = True

        try:
            self._stream = sd.RawOutputStream(
                samplerate=self.sample_rate,
                device=self.device_index,
                channels=self.channels,
                dtype="int16",
                blocksize=0,  # Let system choose optimal low-latency buffer
            )
            self._stream.start()
            logger.info(
                f"Speaker playback stream started [Dev: {self.device_index or 'Default'}, "
                f"SR: {self.sample_rate}Hz, Ch: {self.channels}]"
            )
        except Exception as e:
            logger.error(f"Failed to open speaker output stream: {e}")
            raise

        self._play_task = asyncio.create_task(self._playback_loop())

    async def _playback_loop(self):
        """Asynchronously pulls frames from WebRTC audio track and writes to sounddevice."""
        loop = asyncio.get_running_loop()
        logger.info("WebRTC Speaker playback loop active.")

        while self._running and self._track:
            try:
                frame: av.AudioFrame = await self._track.recv()
                if not self._running:
                    break

                # Ensure sample rate match
                if frame.sample_rate != self.sample_rate:
                    # Dynamically adjust output stream if track format changes
                    logger.debug(f"Audio frame sample rate change: {self.sample_rate} -> {frame.sample_rate}")
                    self.sample_rate = frame.sample_rate
                    if self._stream:
                        self._stream.stop()
                        self._stream.close()
                    self._stream = sd.RawOutputStream(
                        samplerate=self.sample_rate,
                        device=self.device_index,
                        channels=self.channels,
                        dtype="int16",
                    )
                    self._stream.start()

                # Extract raw PCM16 bytes
                pcm_ndarray = frame.to_ndarray()
                if pcm_ndarray.dtype != np.int16:
                    pcm_ndarray = (pcm_ndarray * 32767).clip(-32768, 32767).astype(np.int16)

                if self.muted:
                    continue

                if self.volume != 1.0:
                    pcm_ndarray = (pcm_ndarray.astype(np.float32) * self.volume).clip(-32768, 32767).astype(np.int16)

                raw_bytes = pcm_ndarray.tobytes()

                if self._stream and not self._stream.closed:
                    await loop.run_in_executor(self._executor, self._stream.write, raw_bytes)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    logger.debug(f"Speaker playback loop ended: {e}")
                break

    def flush(self):
        """Immediately abort queued playback (e.g. on user speech interruption)."""
        if self._stream and not self._stream.closed:
            try:
                self._stream.abort()
                self._stream.start()
                logger.debug("Speaker playback buffer flushed.")
            except Exception as e:
                logger.debug(f"Error flushing speaker stream: {e}")

    def set_volume(self, volume: float):
        """Set output speaker volume (0.0 to 2.0)."""
        self.volume = max(0.0, min(volume, 2.0))

    def set_muted(self, muted: bool):
        """Mute/unmute speaker output."""
        self.muted = muted

    def stop(self):
        """Stop playback loop and close sounddevice output stream."""
        self._running = False
        if self._play_task and not self._play_task.done():
            self._play_task.cancel()
            self._play_task = None

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.debug(f"Error closing speaker stream: {e}")
            self._stream = None

        logger.info("Speaker playback stopped.")

    def __del__(self):
        self._executor.shutdown(wait=False)

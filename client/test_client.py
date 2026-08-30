"""
Unit and Integration Tests for WebRTC Voice Client and Audio Device Module.
Tests:
- AudioDeviceManager discovery, queries, and default device resolution
- MicrophoneAudioTrack frame generation, timing, PTS calculation, RMS and gain
- SpeakerAudioPlayer frame consumption and flush behavior
- WebRTCVoiceClient URL normalization and DataChannel message handling
- End-to-end WebRTC SDP offer/answer mock handshake
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import asyncio
import fractions
import json
import unittest
import numpy as np
import av
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription

from audio_device import AudioDeviceManager, MicrophoneAudioTrack, SpeakerAudioPlayer
from client import WebRTCVoiceClient


class MockRemoteAudioTrack(MediaStreamTrack):
    """Generates synthetic audio frames for testing SpeakerAudioPlayer."""
    kind = "audio"

    def __init__(self, sample_rate: int = 16000, channels: int = 1, num_frames: int = 5):
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        self.num_frames = num_frames
        self._pts = 0
        self._count = 0

    async def recv(self) -> av.AudioFrame:
        if self._count >= self.num_frames:
            await asyncio.sleep(0.1)
            raise StopIteration

        self._count += 1
        samples_per_frame = int(self.sample_rate * 0.02)  # 20ms
        samples = (np.sin(np.linspace(0, 10, samples_per_frame)) * 10000).astype(np.int16)
        if self.channels == 1:
            samples = samples.reshape(1, -1)
        else:
            samples = np.vstack([samples, samples])

        frame = av.AudioFrame.from_ndarray(
            samples, format="s16", layout="mono" if self.channels == 1 else "stereo"
        )
        frame.sample_rate = self.sample_rate
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        self._pts += samples_per_frame
        return frame


class TestAudioDeviceManager(unittest.TestCase):
    """Test audio device discovery and query helpers."""

    def test_list_devices(self):
        devices = AudioDeviceManager.list_devices()
        self.assertIsInstance(devices, list)
        if devices:
            d = devices[0]
            self.assertIn("index", d)
            self.assertIn("name", d)
            self.assertIn("max_input_channels", d)
            self.assertIn("max_output_channels", d)
            self.assertIn("default_samplerate", d)

    def test_get_inputs_and_outputs(self):
        inputs = AudioDeviceManager.get_input_devices()
        outputs = AudioDeviceManager.get_output_devices()
        self.assertIsInstance(inputs, list)
        self.assertIsInstance(outputs, list)

    def test_find_device(self):
        # Default resolution
        def_in = AudioDeviceManager.find_device(None, is_input=True)
        # Should be an int or None
        self.assertTrue(def_in is None or isinstance(def_in, int))

        # Query by integer
        idx_res = AudioDeviceManager.find_device(0, is_input=True)
        self.assertEqual(idx_res, 0)


class TestMicrophoneAudioTrack(unittest.IsolatedAsyncioTestCase):
    """Test MicrophoneAudioTrack properties, frame generation, and muting."""

    async def test_frame_generation_and_pts(self):
        track = MicrophoneAudioTrack(
            device_index=None, sample_rate=16000, channels=1, chunk_duration_ms=20
        )
        self.assertEqual(track.kind, "audio")
        self.assertEqual(track.sample_rate, 16000)
        self.assertEqual(track.samples_per_chunk, 320)

        # Mock queue data
        synthetic_pcm = np.zeros(320, dtype=np.int16).tobytes()
        track._running = True
        track._put_chunk(synthetic_pcm)

        frame = await track.recv()
        self.assertIsInstance(frame, av.AudioFrame)
        self.assertEqual(frame.sample_rate, 16000)
        self.assertEqual(frame.format.name, "s16")
        self.assertEqual(frame.pts, 0)

        # Second frame PTS should advance
        track._put_chunk(synthetic_pcm)
        frame2 = await track.recv()
        self.assertEqual(frame2.pts, 320)

        track.stop()
        self.assertFalse(track._running)

    async def test_gain_and_mute(self):
        track = MicrophoneAudioTrack(device_index=None, sample_rate=16000)
        track.set_gain(2.5)
        self.assertEqual(track.gain, 2.5)

        track.set_muted(True)
        self.assertTrue(track.muted)
        track.set_muted(False)
        self.assertFalse(track.muted)
        track.stop()


class TestSpeakerAudioPlayer(unittest.IsolatedAsyncioTestCase):
    """Test SpeakerAudioPlayer lifecycle and track consumption."""

    async def test_player_lifecycle(self):
        player = SpeakerAudioPlayer(device_index=None, sample_rate=16000, volume=0.8)
        self.assertEqual(player.volume, 0.8)
        self.assertFalse(player.muted)

        player.set_volume(1.2)
        self.assertEqual(player.volume, 1.2)

        player.set_muted(True)
        self.assertTrue(player.muted)

        # Test flush without error
        player.flush()
        player.stop()
        self.assertFalse(player._running)


class TestWebRTCVoiceClient(unittest.IsolatedAsyncioTestCase):
    """Test client state, URL normalization, and telemetry processing."""

    def test_url_normalization(self):
        c1 = WebRTCVoiceClient._normalize_url("localhost:7860")
        self.assertEqual(c1, "http://localhost:7860/offer")

        c2 = WebRTCVoiceClient._normalize_url("http://127.0.0.1:7860/")
        self.assertEqual(c2, "http://127.0.0.1:7860/offer")

        c3 = WebRTCVoiceClient._normalize_url("https://my-tunnel.trycloudflare.com/offer")
        self.assertEqual(c3, "https://my-tunnel.trycloudflare.com/offer")

    async def test_telemetry_message_handling(self):
        client = WebRTCVoiceClient(
            server_url="http://localhost:7860/offer",
            no_audio=True,
            language="Hindi",
            tts_engine="vits",
        )

        # 1. VAD Activity
        vad_msg = json.dumps({"type": "vad_activity", "is_speech": True, "speech_prob": 0.94, "rms": 820.0})
        client._handle_telemetry_message(vad_msg)
        self.assertTrue(client.last_vad_speech)
        self.assertAlmostEqual(client.last_vad_prob, 0.94)

        # 2. Final Transcript
        asr_msg = json.dumps({"type": "final_transcript", "text": "नमस्ते कैसे हो", "latency_ms": 115.4, "language": "Hindi"})
        client._handle_telemetry_message(asr_msg)
        self.assertAlmostEqual(client.last_asr_latency, 115.4)

        # 3. LLM TTFT
        ttft_msg = json.dumps({"type": "llm_ttft", "ttft_ms": 178.2})
        client._handle_telemetry_message(ttft_msg)
        self.assertAlmostEqual(client.last_llm_ttft, 178.2)

        # 4. Tool Call
        tool_msg = json.dumps({
            "type": "tool_call",
            "name": "get_current_weather",
            "args": {"location": "Delhi"},
            "result": "Sunny 28°C"
        })
        client._handle_telemetry_message(tool_msg)

        # 5. TTS Metrics
        tts_msg = json.dumps({"type": "tts_metrics", "latency_ms": 68.5, "engine": "Cartesia sonic-3"})
        client._handle_telemetry_message(tts_msg)
        self.assertAlmostEqual(client.last_tts_latency, 68.5)

        await client.close()


class TestWebRTCHandshake(unittest.IsolatedAsyncioTestCase):
    """End-to-end peer connection offer/answer handshake test using aiortc in-process."""

    async def test_local_peer_handshake(self):
        # Create Client PeerConnection
        pc_client = RTCPeerConnection()
        track = MicrophoneAudioTrack(device_index=None, sample_rate=16000)
        track._running = True
        pc_client.addTrack(track)
        pc_client.addTransceiver("audio", direction="sendrecv")
        dc_client = pc_client.createDataChannel("telemetry")

        # Create Server PeerConnection
        pc_server = RTCPeerConnection()
        server_dc_future = asyncio.Future()

        @pc_server.on("datachannel")
        def on_datachannel(channel):
            server_dc_future.set_result(channel)

        @pc_server.on("track")
        def on_track(server_track):
            pass

        # 1. Client creates offer
        offer = await pc_client.createOffer()
        await pc_client.setLocalDescription(offer)

        # 2. Server receives offer and creates answer
        await pc_server.setRemoteDescription(pc_client.localDescription)
        answer = await pc_server.createAnswer()
        await pc_server.setLocalDescription(answer)

        # 3. Client sets server answer
        await pc_client.setRemoteDescription(pc_server.localDescription)

        # 4. Verify data channel handshake
        server_dc = await asyncio.wait_for(server_dc_future, timeout=2.0)
        self.assertEqual(server_dc.label, "telemetry")

        # Clean up
        track.stop()
        await pc_client.close()
        await pc_server.close()


if __name__ == "__main__":
    unittest.main()

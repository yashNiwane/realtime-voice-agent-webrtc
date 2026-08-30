"""
FastAPI + aiortc Production WebRTC Server for Voice Agent.

Features:
- Zero-jitter `ServerAudioStreamTrack` with monotonic `pts` pacing and instant buffer flush on interruption.
- Real-time client audio ingestion (48kHz downsampled to 16kHz mono PCM).
- Integrated Silero Neural VAD state machine and Qwen3-ASR inference pipeline.
- Streaming Ollama LLM with tool execution and think-tag filtering.
- Multi-Engine TTS (VITS Hindi, Edge-TTS, Cartesia Sonic-3) resampled to 48kHz.
- Real-time 'telemetry' DataChannel broadcasting VAD probability, ASR transcripts, LLM tokens, and tool badges.
- REST SDP offer/answer signaling endpoints (`POST /offer`) and system health endpoint (`GET /health`).
- Interactive, responsive Web UI dashboard (`GET /`).
"""

import asyncio
from contextlib import asynccontextmanager
import fractions
import json
import os
import time
from typing import Any, Dict, List, Optional, Set
import torch
import numpy as np
import scipy.signal
import av
import uvicorn
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    RTCDataChannel,
)
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel

from server.asr_engine import Qwen3ASREngine
from server.config import config
from server.llm_engine import OllamaLLMEngine
from server.tools import TOOLS_SCHEMA, COLLECTED_USER_DATA
from server.tts_engine import MultiEngineTTSManager
from server.vad_analyzer import SileroVADAnalyzer, VADResult

# Global Engine Singletons
asr_engine: Optional[Qwen3ASREngine] = None
llm_engine: Optional[OllamaLLMEngine] = None
tts_manager: Optional[MultiEngineTTSManager] = None
active_peer_connections: Set[RTCPeerConnection] = set()


class ServerAudioStreamTrack(MediaStreamTrack):
    """
    Zero-jitter WebRTC audio output stream track.
    Features:
    - 48,000 Hz 16-bit mono PCM output (960 samples per 20ms frame).
    - Monotonic PTS timestamps and wall-clock frame pacing.
    - Instant buffer flush on user speech barge-in / interruption.
    """

    kind = "audio"

    def __init__(self, sample_rate: int = 48000, frame_duration_ms: int = 20):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_samples = int(sample_rate * (frame_duration_ms / 1000.0))  # 960 samples
        self.time_base = fractions.Fraction(1, self.sample_rate)

        self._queue: asyncio.Queue[av.AudioFrame] = asyncio.Queue()
        self._pts: int = 0
        self._last_frame_time: float = 0.0
        self._is_speaking: bool = False
        self._interrupted: bool = False

        # Pre-generate 20ms silence frame for zero-allocation idle pacing
        silence_arr = np.zeros((1, self.frame_samples), dtype=np.int16)
        self._silence_frame = av.AudioFrame.from_ndarray(silence_arr, format="s16", layout="mono")
        self._silence_frame.sample_rate = self.sample_rate

    def flush(self) -> None:
        """Instantly flush all queued audio frames to halt assistant speech playback."""
        cleared_count = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                cleared_count += 1
            except (asyncio.QueueEmpty, ValueError):
                break
        self._interrupted = True
        self._is_speaking = False
        if cleared_count > 0:
            logger.info(f"⚡ [TRACK FLUSH] Flushed {cleared_count} frames on user interruption.")

    def reset_interrupt(self) -> None:
        """Reset interrupt flag before queueing new speech synthesis."""
        self._interrupted = False

    @property
    def is_interrupted(self) -> bool:
        """Check if current synthesis turn was cancelled by user barge-in."""
        return self._interrupted

    def add_pcm16_audio(self, pcm_data: np.ndarray) -> int:
        """
        Slice and queue 1D int16 48kHz audio array into 20ms AudioFrames.

        Args:
            pcm_data: 1D int16 numpy array at 48,000 Hz.

        Returns:
            Number of audio frames queued.
        """
        if pcm_data is None or len(pcm_data) == 0:
            return 0

        if self._interrupted:
            return 0

        self._is_speaking = True
        total_samples = len(pcm_data)
        frames_queued = 0

        for start_idx in range(0, total_samples, self.frame_samples):
            chunk = pcm_data[start_idx : start_idx + self.frame_samples]
            if len(chunk) < self.frame_samples:
                # Pad remainder with zeros
                chunk = np.pad(chunk, (0, self.frame_samples - len(chunk)), mode="constant")

            chunk_2d = chunk.reshape(1, self.frame_samples)
            frame = av.AudioFrame.from_ndarray(chunk_2d, format="s16", layout="mono")
            frame.sample_rate = self.sample_rate
            self._queue.put_nowait(frame)
            frames_queued += 1

        return frames_queued

    async def recv(self) -> av.AudioFrame:
        """
        Generate or fetch next 20ms audio frame with precise monotonic pacing.
        """
        # Precise 20ms pacing relative to wall clock
        now = time.monotonic()
        if self._last_frame_time > 0.0:
            elapsed = now - self._last_frame_time
            target_interval = self.frame_samples / self.sample_rate  # 0.020s
            if elapsed < target_interval:
                await asyncio.sleep(target_interval - elapsed)
        self._last_frame_time = time.monotonic()

        # Fetch frame or generate silence
        if not self._queue.empty() and not self._interrupted:
            try:
                frame = self._queue.get_nowait()
                self._queue.task_done()
                self._is_speaking = True
            except asyncio.QueueEmpty:
                frame = self._silence_frame
                self._is_speaking = False
        else:
            frame = self._silence_frame
            self._is_speaking = False

        # Assign monotonic PTS
        frame.pts = self._pts
        frame.time_base = self.time_base
        self._pts += self.frame_samples

        return frame


class PeerSession:
    """Encapsulates state, tasks, and data channels for a single connected WebRTC peer."""

    def __init__(self, pc: RTCPeerConnection, server_track: ServerAudioStreamTrack):
        self.pc = pc
        self.server_track = server_track
        self.telemetry_channel: Optional[RTCDataChannel] = None
        self.vad = SileroVADAnalyzer()
        self.messages: List[Dict[str, Any]] = llm_engine.create_session_messages() if llm_engine else []
        self.current_pipeline_task: Optional[asyncio.Task] = None
        self.preferred_language: str = config.asr.language
        self.preferred_tts_engine: str = config.tts.default_engine
        self.incoming_audio_buffer: bytearray = bytearray()

    def send_telemetry(self, payload: Dict[str, Any]) -> None:
        """Send JSON telemetry payload over WebRTC DataChannel if open."""
        if self.telemetry_channel and self.telemetry_channel.readyState == "open":
            try:
                self.telemetry_channel.send(json.dumps(payload))
            except Exception as e:
                logger.debug(f"DataChannel send error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and shutdown manager."""
    global asr_engine, llm_engine, tts_manager

    print("=" * 72)
    print(" 🚀 KAGGLE GPU WEBRTC VOICE AGENT SERVER STARTING")
    print(f" * Web Dashboard: http://localhost:{config.webrtc.port}")
    print(f" * ASR Engine:    {config.asr.model_path} (Device: {config.asr.device}, DType: {config.asr.torch_dtype})")
    print(f" * VAD Engine:    Silero Neural VAD (Confidence: {config.vad.confidence:.2f})")
    print(f" * LLM Engine:    {config.llm.model} @ {config.llm.base_url}")
    print(f" * TTS Engine:    {config.tts.default_engine} (Sample Rate: {config.tts.sample_rate}Hz)")
    print(f" * WebRTC STUN:   {config.webrtc.stun_servers}")
    print("=" * 72)

    logger.info("Initializing neural models and inference pipelines...")
    asr_engine = Qwen3ASREngine(asr_config=config.asr)
    llm_engine = OllamaLLMEngine(llm_config=config.llm)
    tts_manager = MultiEngineTTSManager(tts_config=config.tts)
    logger.info("✨ All server components initialized and warm.")

    yield

    logger.info("Closing active WebRTC peer connections...")
    for pc in list(active_peer_connections):
        try:
            await pc.close()
        except Exception:
            pass
    active_peer_connections.clear()
    if asr_engine:
        asr_engine.close()
    logger.info("Server shutdown complete.")


app = FastAPI(
    title="Kaggle GPU WebRTC Voice Agent",
    description="Real-time voice agent server powered by Qwen3-ASR, Silero VAD, Ollama LLM, and Multi-Engine TTS",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OfferModel(BaseModel):
    sdp: str
    type: str
    language: Optional[str] = None
    tts_engine: Optional[str] = None


async def run_voice_pipeline(
    audio_pcm16: np.ndarray, session: PeerSession
) -> None:
    """
    Execute full ASR -> LLM Streaming -> TTS Audio playback pipeline on completed speech utterance.
    """
    if asr_engine is None or llm_engine is None or tts_manager is None:
        return

    session.server_track.reset_interrupt()

    # 1. Automatic Speech Recognition (ASR)
    t0_asr = time.perf_counter()
    transcript, detected_lang, asr_lat = await asr_engine.async_transcribe_pcm(audio_pcm16, 16000)

    if not transcript:
        logger.debug("ASR returned empty or hallucinated transcript; ignoring.")
        return

    active_lang = session.preferred_language or detected_lang or "Hindi"
    logger.info(f"🎙️ [USER TRANSCRIBED] '{transcript}' ({active_lang}, latency={asr_lat:.1f}ms)")

    session.send_telemetry({
        "type": "final_transcript",
        "text": transcript,
        "language": active_lang,
        "latency_ms": round(asr_lat, 1),
    })

    # 2. LLM Streaming & Real-time Sentence Synthesis
    t0_llm = time.perf_counter()
    sentence_delimiters = (".", "!", "?", "।", "\n")
    current_sentence_buf = ""
    first_tts_started = False

    try:
        async for event in llm_engine.stream_response(transcript, session.messages):
            if session.server_track.is_interrupted:
                logger.info("🛑 Pipeline halted mid-generation due to barge-in interruption.")
                break

            if event.type == "token":
                token = event.content
                session.send_telemetry({"type": "llm_chunk", "token": token})
                current_sentence_buf += token

                # If sentence boundary reached and buffer has substantial content
                has_delim = any(delim in token for delim in sentence_delimiters)
                if has_delim and len(current_sentence_buf.strip()) > 15:
                    phrase_to_speak = current_sentence_buf.strip()
                    current_sentence_buf = ""

                    # Synthesize phrase
                    _, pcm16_arr, tts_lat, engine_name = await tts_manager.synthesize(
                        phrase_to_speak,
                        language=active_lang,
                        preferred_engine=session.preferred_tts_engine,
                    )

                    if not session.server_track.is_interrupted and len(pcm16_arr) > 0:
                        session.server_track.add_pcm16_audio(pcm16_arr)
                        if not first_tts_started:
                            first_tts_started = True
                            e2e_lat = (time.perf_counter() - t0_asr) * 1000
                            session.send_telemetry({
                                "type": "tts_start",
                                "engine": engine_name,
                                "tts_latency_ms": round(tts_lat, 1),
                                "e2e_latency_ms": round(e2e_lat, 1),
                            })

            elif event.type == "tool_call":
                logger.info(f"🔧 [TOOL EVENT] {event.tool_data['name']} executed.")
                session.send_telemetry({
                    "type": "tool_call",
                    "name": event.tool_data["name"],
                    "arguments": event.tool_data["arguments"],
                    "result": event.tool_data["result"],
                })

            elif event.type == "done":
                # Synthesize any trailing text in sentence buffer
                trailing_phrase = current_sentence_buf.strip()
                if trailing_phrase and not session.server_track.is_interrupted:
                    _, pcm16_arr, tts_lat, engine_name = await tts_manager.synthesize(
                        trailing_phrase,
                        language=active_lang,
                        preferred_engine=session.preferred_tts_engine,
                    )
                    if not session.server_track.is_interrupted and len(pcm16_arr) > 0:
                        session.server_track.add_pcm16_audio(pcm16_arr)

                llm_total_lat = (time.perf_counter() - t0_llm) * 1000
                session.send_telemetry({
                    "type": "llm_done",
                    "full_text": event.content,
                    "latency_ms": round(llm_total_lat, 1),
                })

    except asyncio.CancelledError:
        logger.debug("Voice pipeline task cancelled cleanly.")
    except Exception as e:
        logger.exception(f"Error in voice agent pipeline: {e}")
        session.send_telemetry({"type": "error", "message": str(e)})


async def handle_incoming_audio(
    track: MediaStreamTrack, session: PeerSession
) -> None:
    """
    Ingest 48kHz audio from client WebRTC track, downsample to 16kHz mono float32,
    and drive Silero Neural VAD state machine.
    """
    sample_buffer = np.zeros(0, dtype=np.float32)

    try:
        while True:
            frame = await track.recv()
            if frame is None:
                break

            # Convert frame to numpy
            raw_ndarray = frame.to_ndarray()
            if raw_ndarray.ndim > 1:
                # Average multi-channel / stereo down to mono
                mono_data = np.mean(raw_ndarray, axis=0)
            else:
                mono_data = raw_ndarray

            # Convert int16 to float32
            if mono_data.dtype == np.int16:
                float_data = mono_data.astype(np.float32) / 32768.0
            elif mono_data.dtype != np.float32:
                float_data = mono_data.astype(np.float32)
            else:
                float_data = mono_data

            # Downsample from incoming sample rate (usually 48kHz) to 16kHz
            incoming_sr = getattr(frame, "sample_rate", 48000) or 48000
            if incoming_sr != 16000 and len(float_data) > 0:
                if incoming_sr == 48000:
                    downsampled = scipy.signal.resample_poly(float_data, up=1, down=3)
                else:
                    target_len = int(round(len(float_data) * 16000 / incoming_sr))
                    downsampled = np.interp(
                        np.linspace(0, len(float_data), target_len, endpoint=False),
                        np.arange(len(float_data)),
                        float_data,
                    )
            else:
                downsampled = float_data

            # Append to sample buffer
            sample_buffer = np.concatenate((sample_buffer, downsampled))

            # Process in 512-sample (32ms @ 16kHz) chunks for Silero VAD
            while len(sample_buffer) >= 512:
                chunk_512 = sample_buffer[:512]
                sample_buffer = sample_buffer[512:]

                vad_res: VADResult = session.vad.process_chunk(chunk_512)

                # Send real-time VAD probability to UI
                session.send_telemetry({
                    "type": "vad_activity",
                    "prob": round(vad_res.probability, 3),
                    "speech_prob": round(vad_res.probability, 3),
                    "is_speech": vad_res.is_speech,
                })

                # Handle Barge-In / Interruption on SPEECH_START
                if vad_res.event == "SPEECH_START":
                    session.server_track.flush()
                    if session.current_pipeline_task and not session.current_pipeline_task.done():
                        session.current_pipeline_task.cancel()
                    session.send_telemetry({"type": "interrupt", "timestamp": time.time()})
                    logger.info("🛑 [BARGE-IN] User started speaking; flushed assistant playback.")

                # Handle Turn Completion on SPEECH_END
                elif vad_res.event == "SPEECH_END" and vad_res.audio_utterance is not None:
                    utterance_pcm = vad_res.audio_utterance
                    # Cancel any prior running task
                    if session.current_pipeline_task and not session.current_pipeline_task.done():
                        session.current_pipeline_task.cancel()

                    session.current_pipeline_task = asyncio.create_task(
                        run_voice_pipeline(utterance_pcm, session)
                    )

    except asyncio.CancelledError:
        logger.debug("Incoming audio track loop cancelled.")
    except Exception as e:
        logger.exception(f"Exception in incoming audio handler: {e}")


@app.post("/offer")
async def webrtc_offer(offer_data: OfferModel):
    """
    WebRTC SDP Offer signaling endpoint.
    Creates RTCPeerConnection, adds ServerAudioStreamTrack, and returns SDP Answer.
    """
    rtc_config = RTCConfiguration(
        iceServers=[RTCIceServer(urls=url) for url in config.webrtc.stun_servers]
    )
    pc = RTCPeerConnection(configuration=rtc_config)
    active_peer_connections.add(pc)

    # Create server audio track
    server_track = ServerAudioStreamTrack(
        sample_rate=config.webrtc.audio_sample_rate,
        frame_duration_ms=config.webrtc.frame_duration_ms,
    )
    pc.addTrack(server_track)

    session = PeerSession(pc, server_track)
    if offer_data.language:
        session.preferred_language = offer_data.language
    if offer_data.tts_engine:
        session.preferred_tts_engine = offer_data.tts_engine

    @pc.on("datachannel")
    def on_datachannel(channel: RTCDataChannel):
        logger.info(f"📡 WebRTC DataChannel received: '{channel.label}'")
        if channel.label == "telemetry":
            session.telemetry_channel = channel

            @channel.on("message")
            def on_message(message: str):
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    if msg_type == "set_language":
                        session.preferred_language = data.get("language", "Hindi")
                        logger.info(f"Updated session language to '{session.preferred_language}'")
                    elif msg_type == "set_engine":
                        session.preferred_tts_engine = data.get("engine", "vits")
                        logger.info(f"Updated session TTS engine to '{session.preferred_tts_engine}'")
                    elif msg_type == "interrupt":
                        session.server_track.flush()
                        if session.current_pipeline_task and not session.current_pipeline_task.done():
                            session.current_pipeline_task.cancel()
                    elif msg_type == "clear_history":
                        session.messages = llm_engine.create_session_messages() if llm_engine else []
                        session.send_telemetry({"type": "history_cleared"})
                except Exception as e:
                    logger.debug(f"DataChannel message parse error: {e}")

    @pc.on("track")
    def on_track(track: MediaStreamTrack):
        logger.info(f"📥 Received remote WebRTC track: kind='{track.kind}'")
        if track.kind == "audio":
            asyncio.create_task(handle_incoming_audio(track, session))

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"WebRTC PeerConnection state changed: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            active_peer_connections.discard(pc)
            if session.current_pipeline_task and not session.current_pipeline_task.done():
                session.current_pipeline_task.cancel()
            await pc.close()

    # Apply remote SDP offer
    offer = RTCSessionDescription(sdp=offer_data.sdp, type=offer_data.type)
    await pc.setRemoteDescription(offer)

    # Generate and set local SDP answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse(
        content={
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }
    )


@app.websocket("/ws")
async def websocket_audio_endpoint(websocket: WebSocket):
    """
    Dual-Mode WebSocket Voice Stream:
    Guarantees 100% audio transmission over Cloudflare Tunnel (HTTPS/WSS port 443)
    when Kaggle container symmetric NAT blocks inbound UDP WebRTC packets.
    """
    await websocket.accept()
    logger.info("📡 [WEBSOCKET CONNECTED] Client connected for dual-mode voice stream.")

    vad = SileroVADAnalyzer()
    session_messages = llm_engine.create_session_messages() if llm_engine else []
    preferred_lang = config.asr.language
    preferred_tts = config.tts.default_engine
    active_llm_task = None
    sample_buffer = np.zeros(0, dtype=np.float32)

    try:
        while True:
            message = await websocket.receive()
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "config":
                        if "language" in data:
                            preferred_lang = data["language"]
                        if "tts_engine" in data:
                            preferred_tts = data["tts_engine"]
                except Exception:
                    pass

            elif "bytes" in message:
                raw_bytes = message["bytes"]
                if not raw_bytes:
                    continue

                pcm16 = np.frombuffer(raw_bytes, dtype=np.int16)
                float_data = pcm16.astype(np.float32) / 32768.0
                sample_buffer = np.concatenate((sample_buffer, float_data))

                while len(sample_buffer) >= 512:
                    chunk_512 = sample_buffer[:512]
                    sample_buffer = sample_buffer[512:]
                    vad_res = vad.process_chunk(chunk_512)

                    await websocket.send_json({
                        "type": "vad_activity",
                        "prob": round(vad_res.probability, 3),
                        "speech_prob": round(vad_res.probability, 3),
                        "is_speech": vad_res.is_speech,
                    })

                    if vad_res.event == "SPEECH_START":
                        if active_llm_task and not active_llm_task.done():
                            active_llm_task.cancel()
                        await websocket.send_json({"type": "interrupt", "timestamp": time.time()})

                    elif vad_res.event == "SPEECH_END" and vad_res.audio_utterance is not None:
                        utterance = vad_res.audio_utterance
                        if active_llm_task and not active_llm_task.done():
                            active_llm_task.cancel()

                        async def run_ws_pipeline(audio_data):
                            t0_asr = time.perf_counter()
                            transcript, d_lang, asr_lat = await asr_engine.async_transcribe_pcm(audio_data, 16000)
                            if not transcript:
                                return

                            active_l = preferred_lang or d_lang or "Hindi"
                            logger.info(f"🎙️ [WS TRANSCRIBED] '{transcript}' ({active_l}, {asr_lat:.1f}ms)")
                            await websocket.send_json({
                                "type": "final_transcript",
                                "text": transcript,
                                "language": active_l,
                                "latency_ms": round(asr_lat, 1),
                            })

                            t0_llm = time.perf_counter()
                            current_sentence_buf = ""
                            sentence_delims = (".", "!", "?", "।", "\n")

                            async for event in llm_engine.stream_response(transcript, session_messages):
                                if event.type == "token":
                                    token = event.content
                                    await websocket.send_json({"type": "llm_token", "token": token})
                                    current_sentence_buf += token

                                    if any(d in token for d in sentence_delims) and len(current_sentence_buf.strip()) > 15:
                                        phrase = current_sentence_buf.strip()
                                        current_sentence_buf = ""
                                        audio_bytes, _, tts_lat, eng_name = await tts_manager.synthesize(
                                            phrase, language=active_l, preferred_engine=preferred_tts
                                        )
                                        if audio_bytes:
                                            import base64
                                            await websocket.send_json({
                                                "type": "tts_audio",
                                                "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
                                                "latency_ms": round(tts_lat, 1),
                                                "engine": eng_name,
                                            })

                                elif event.type == "tool_call":
                                    await websocket.send_json({
                                        "type": "tool_call",
                                        "name": event.tool_data["name"],
                                        "arguments": event.tool_data["arguments"],
                                        "result": event.tool_data["result"],
                                    })

                                elif event.type == "done":
                                    trailing = current_sentence_buf.strip()
                                    if trailing:
                                        audio_bytes, _, tts_lat, eng_name = await tts_manager.synthesize(
                                            trailing, language=active_l, preferred_engine=preferred_tts
                                        )
                                        if audio_bytes:
                                            import base64
                                            await websocket.send_json({
                                                "type": "tts_audio",
                                                "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
                                                "latency_ms": round(tts_lat, 1),
                                                "engine": eng_name,
                                            })
                                    llm_tot = (time.perf_counter() - t0_llm) * 1000
                                    await websocket.send_json({
                                        "type": "llm_done",
                                        "full_text": event.content,
                                        "latency_ms": round(llm_tot, 1)
                                    })

                        active_llm_task = asyncio.create_task(run_ws_pipeline(utterance))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.debug(f"WebSocket session ended: {e}")


@app.get("/health")
async def health_check():
    """System health and hardware status endpoint."""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "CPU (No GPU detected)"
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "gpu_available": gpu_available,
        "gpu_device": gpu_name,
        "active_webrtc_sessions": len(active_peer_connections),
        "models": {
            "asr": config.asr.model_path,
            "llm": config.llm.model,
            "tts_default": config.tts.default_engine,
        },
        "collected_user_records": len(COLLECTED_USER_DATA),
    }


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the interactive, real-time Web Voice Agent Web Dashboard."""
    import pathlib
    index_path = pathlib.Path(__file__).parent.parent / "client" / "index.html"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content=WEB_UI_HTML)


WEB_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Kaggle GPU WebRTC Voice Agent | Qwen3 + Gemma4 + VITS</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-gradient: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0f172a 50%, #020617 100%);
      --card-bg: rgba(15, 23, 42, 0.75);
      --card-border: rgba(99, 102, 241, 0.2);
      --card-glow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
      --primary: #6366f1;
      --primary-hover: #4f46e5;
      --primary-glow: rgba(99, 102, 241, 0.4);
      --accent: #06b6d4;
      --accent-green: #10b981;
      --accent-rose: #f43f5e;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --font-sans: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: var(--bg-gradient);
      color: var(--text-main);
      font-family: var(--font-sans);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 24px;
      overflow-x: hidden;
    }

    .container {
      width: 100%;
      max-width: 1100px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 24px;
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--card-border);
      border-radius: 20px;
      box-shadow: var(--card-glow);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .logo-icon {
      width: 44px;
      height: 44px;
      border-radius: 12px;
      background: linear-gradient(135deg, #6366f1, #06b6d4);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 22px;
      box-shadow: 0 0 20px var(--primary-glow);
    }

    .brand h1 {
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.5px;
      background: linear-gradient(90deg, #ffffff, #94a3b8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .brand p {
      font-size: 12px;
      color: var(--text-muted);
    }

    .status-badge {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 16px;
      border-radius: 9999px;
      font-size: 13px;
      font-weight: 600;
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid rgba(255, 255, 255, 0.1);
      transition: all 0.3s ease;
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--text-muted);
    }

    .status-badge.connected .status-dot {
      background: var(--accent-green);
      box-shadow: 0 0 10px var(--accent-green);
    }
    .status-badge.listening .status-dot {
      background: var(--accent);
      box-shadow: 0 0 12px var(--accent);
      animation: pulse 1.5s infinite;
    }
    .status-badge.speaking .status-dot {
      background: var(--primary);
      box-shadow: 0 0 12px var(--primary);
      animation: pulse 1s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(1.3); }
    }

    .telemetry-grid {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 14px;
    }

    .metric-card {
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
      transform: translateY(-2px);
      border-color: rgba(99, 102, 241, 0.4);
    }

    .metric-title {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--text-muted);
      font-weight: 600;
    }

    .metric-value {
      font-family: var(--font-mono);
      font-size: 22px;
      font-weight: 700;
      color: #fff;
    }

    .metric-sub {
      font-size: 11px;
      color: var(--text-muted);
    }

    .interaction-area {
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 20px;
    }

    .chat-panel {
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--card-border);
      border-radius: 20px;
      display: flex;
      flex-direction: column;
      height: 490px;
      overflow: hidden;
      box-shadow: var(--card-glow);
    }

    .panel-header {
      padding: 14px 20px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .panel-header h2 { font-size: 14px; font-weight: 600; }

    .chat-messages {
      flex: 1;
      padding: 20px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
      scroll-behavior: smooth;
    }

    .message {
      display: flex;
      flex-direction: column;
      gap: 6px;
      max-width: 85%;
      animation: fadeIn 0.25s ease-out;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .message.user { align-self: flex-end; }
    .message.assistant { align-self: flex-start; }

    .message-bubble {
      padding: 12px 18px;
      border-radius: 16px;
      font-size: 14px;
      line-height: 1.5;
    }

    .message.user .message-bubble {
      background: linear-gradient(135deg, #4f46e5, #6366f1);
      color: white;
      border-bottom-right-radius: 4px;
      box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }

    .message.assistant .message-bubble {
      background: rgba(30, 41, 59, 0.85);
      border: 1px solid rgba(255, 255, 255, 0.08);
      color: var(--text-main);
      border-bottom-left-radius: 4px;
    }

    .message-meta {
      font-size: 11px;
      color: var(--text-muted);
      display: flex;
      gap: 8px;
      padding: 0 4px;
    }
    .message.user .message-meta { justify-content: flex-end; }

    .tool-badge {
      align-self: flex-start;
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #34d399;
      font-family: var(--font-mono);
      font-size: 12px;
      padding: 8px 14px;
      border-radius: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
      max-width: 90%;
      animation: fadeIn 0.25s ease-out;
    }

    .controls-panel {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .control-card {
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--card-border);
      border-radius: 20px;
      padding: 20px;
      box-shadow: var(--card-glow);
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .visualizer-container {
      height: 70px;
      width: 100%;
      background: rgba(2, 6, 23, 0.6);
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }

    canvas#visualizer {
      width: 100%;
      height: 100%;
    }

    .mic-button {
      width: 100%;
      padding: 16px;
      border-radius: 14px;
      border: none;
      font-family: var(--font-sans);
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      background: linear-gradient(135deg, #6366f1, #4f46e5);
      color: white;
      box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
    }
    .mic-button:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 25px rgba(99, 102, 241, 0.6);
    }
    .mic-button.active {
      background: linear-gradient(135deg, #f43f5e, #e11d48);
      box-shadow: 0 0 30px rgba(244, 63, 94, 0.6);
      animation: micPulse 1.5s infinite;
    }

    @keyframes micPulse {
      0%, 100% { transform: scale(1); }
      50% { transform: scale(1.02); }
    }

    .setting-item {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .setting-label {
      font-size: 12px;
      color: var(--text-muted);
      font-weight: 600;
    }
    .setting-select {
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 10px;
      padding: 8px 12px;
      color: #fff;
      font-family: var(--font-sans);
      font-size: 13px;
      outline: none;
      transition: border-color 0.2s;
    }
    .setting-select:focus {
      border-color: var(--primary);
    }

    .btn-secondary {
      padding: 8px 12px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 10px;
      color: var(--text-muted);
      font-size: 12px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-secondary:hover {
      background: rgba(255, 255, 255, 0.1);
      color: #fff;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand">
        <div class="logo-icon">⚡</div>
        <div>
          <h1>Kaggle GPU WebRTC Voice Agent</h1>
          <p>Qwen3-ASR + Silero VAD + Ollama Gemma 4 + VITS / Edge / Cartesia</p>
        </div>
      </div>
      <div id="statusBadge" class="status-badge">
        <div class="status-dot"></div>
        <span id="statusText">Ready to Connect</span>
      </div>
    </header>

    <div class="telemetry-grid">
      <div class="metric-card">
        <div class="metric-title">VAD Probability</div>
        <div id="metricVAD" class="metric-value">0%</div>
        <div id="metricVADSub" class="metric-sub">Silence</div>
      </div>
      <div class="metric-card">
        <div class="metric-title">ASR Latency</div>
        <div id="metricASR" class="metric-value">0 ms</div>
        <div id="metricASRSub" class="metric-sub">Qwen3-ASR (0.6B)</div>
      </div>
      <div class="metric-card">
        <div class="metric-title">LLM Latency</div>
        <div id="metricLLM" class="metric-value">0 ms</div>
        <div id="metricLLMSub" class="metric-sub">Gemma 4 (Streaming)</div>
      </div>
      <div class="metric-card">
        <div class="metric-title">TTS Engine</div>
        <div id="metricTTS" class="metric-value">VITS</div>
        <div id="metricTTSSub" class="metric-sub">0 ms latency</div>
      </div>
      <div class="metric-card">
        <div class="metric-title">End-to-End Latency</div>
        <div id="metricE2E" class="metric-value">0 ms</div>
        <div class="metric-sub">Spoken to Spoken</div>
      </div>
    </div>

    <div class="interaction-area">
      <div class="chat-panel">
        <div class="panel-header">
          <h2>Live Conversation Timeline</h2>
          <button id="btnClear" class="btn-secondary">Clear Chat</button>
        </div>
        <div id="chatMessages" class="chat-messages">
          <div class="message assistant">
            <div class="message-bubble">
              नमस्ते! How can I assist you today? Click "Start WebRTC Call" and speak naturally into your microphone.
            </div>
            <div class="message-meta">
              <span>Agent</span> • <span>Ready</span>
            </div>
          </div>
        </div>
      </div>

      <div class="controls-panel">
        <div class="control-card">
          <div class="visualizer-container">
            <canvas id="visualizer"></canvas>
          </div>
          <button id="btnCall" class="mic-button">
            <span>🎙️ Start WebRTC Call</span>
          </button>
        </div>

        <div class="control-card">
          <div class="setting-item">
            <label class="setting-label" for="selectEngine">TTS Synthesizer Engine</label>
            <select id="selectEngine" class="setting-select">
              <option value="vits">VITS Hindi Neural (Local Offline)</option>
              <option value="edge">Edge-TTS (Cloud Neural Free)</option>
              <option value="cartesia">Cartesia Sonic-3 (Cloud)</option>
            </select>
          </div>

          <div class="setting-item">
            <label class="setting-label" for="selectLanguage">Language Model Bias</label>
            <select id="selectLanguage" class="setting-select">
              <option value="Hindi">Hindi (हिन्दी)</option>
              <option value="English">English</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  </div>

  <audio id="remoteAudio" autoplay playsinline style="display:none;"></audio>

  <script>
    let peerConnection = null;
    let dataChannel = null;
    let localStream = null;
    let isConnected = false;
    let audioCtx = null;
    let analyser = null;
    let currentAssistantMsgBubble = null;

    const btnCall = document.getElementById("btnCall");
    const btnClear = document.getElementById("btnClear");
    const statusBadge = document.getElementById("statusBadge");
    const statusText = document.getElementById("statusText");
    const chatMessages = document.getElementById("chatMessages");
    const selectEngine = document.getElementById("selectEngine");
    const selectLanguage = document.getElementById("selectLanguage");
    const remoteAudio = document.getElementById("remoteAudio");

    const metricVAD = document.getElementById("metricVAD");
    const metricVADSub = document.getElementById("metricVADSub");
    const metricASR = document.getElementById("metricASR");
    const metricLLM = document.getElementById("metricLLM");
    const metricTTS = document.getElementById("metricTTS");
    const metricTTSSub = document.getElementById("metricTTSSub");
    const metricE2E = document.getElementById("metricE2E");

    // Audio Visualizer
    const canvas = document.getElementById("visualizer");
    const ctx = canvas.getContext("2d");

    function renderVisualizer() {
      requestAnimationFrame(renderVisualizer);
      const width = canvas.width = canvas.offsetWidth;
      const height = canvas.height = canvas.offsetHeight;
      ctx.clearRect(0, 0, width, height);

      if (!analyser || !isConnected) {
        ctx.strokeStyle = "rgba(99, 102, 241, 0.2)";
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();
        return;
      }

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      analyser.getByteFrequencyData(dataArray);

      const barWidth = (width / bufferLength) * 2.5;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * height;
        const grad = ctx.createLinearGradient(0, height, 0, height - barHeight);
        grad.addColorStop(0, "#6366f1");
        grad.addColorStop(1, "#06b6d4");
        ctx.fillStyle = grad;
        ctx.fillRect(x, height - barHeight, barWidth - 1, barHeight);
        x += barWidth;
      }
    }
    renderVisualizer();

    async function startWebRTC() {
      try {
        statusText.textContent = "Requesting Microphone...";
        localStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            sampleRate: 48000
          }
        });

        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 64;
        const source = audioCtx.createMediaStreamSource(localStream);
        source.connect(analyser);

        statusText.textContent = "Negotiating WebRTC...";
        peerConnection = new RTCPeerConnection({
          iceServers: [
            { urls: "stun:stun.l.google.com:19302" },
            { urls: "stun:stun1.l.google.com:19302" }
          ]
        });

        dataChannel = peerConnection.createDataChannel("telemetry");
        setupDataChannel(dataChannel);

        peerConnection.ontrack = (event) => {
          remoteAudio.srcObject = event.streams[0];
        };

        localStream.getTracks().forEach((track) => peerConnection.addTrack(track, localStream));

        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);

        const response = await fetch("/offer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sdp: peerConnection.localDescription.sdp,
            type: peerConnection.localDescription.type,
            language: selectLanguage.value,
            tts_engine: selectEngine.value
          })
        });

        if (!response.ok) throw new Error("Signaling error HTTP " + response.status);
        const answer = await response.json();
        await peerConnection.setRemoteDescription(answer);

        isConnected = true;
        btnCall.classList.add("active");
        btnCall.innerHTML = "<span>🛑 End Call</span>";
        statusBadge.className = "status-badge connected";
        statusText.textContent = "Connected & Active";

      } catch (err) {
        console.error(err);
        statusBadge.className = "status-badge";
        statusText.textContent = "Connection Failed";
        alert("WebRTC Connection Error: " + err.message);
        stopWebRTC();
      }
    }

    function stopWebRTC() {
      if (dataChannel) { dataChannel.close(); dataChannel = null; }
      if (peerConnection) { peerConnection.close(); peerConnection = null; }
      if (localStream) { localStream.getTracks().forEach(t => t.stop()); localStream = null; }
      if (audioCtx) { audioCtx.close(); audioCtx = null; }
      isConnected = false;
      btnCall.classList.remove("active");
      btnCall.innerHTML = "<span>🎙️ Start WebRTC Call</span>";
      statusBadge.className = "status-badge";
      statusText.textContent = "Disconnected";
    }

    function setupDataChannel(channel) {
      channel.onopen = () => {
        channel.send(JSON.stringify({
          type: "set_language",
          language: selectLanguage.value
        }));
        channel.send(JSON.stringify({
          type: "set_engine",
          engine: selectEngine.value
        }));
      };

      channel.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleTelemetry(data);
        } catch (e) {
          console.error("Telemetry error:", e);
        }
      };
    }

    function handleTelemetry(data) {
      if (data.type === "vad") {
        const pct = Math.round(data.prob * 100);
        metricVAD.textContent = pct + "%";
        metricVADSub.textContent = data.is_speech ? "User Speaking 🎙️" : "Listening 👂";
        if (data.is_speech) {
          statusBadge.className = "status-badge listening";
          statusText.textContent = "User Speaking...";
        } else if (statusBadge.classList.contains("listening")) {
          statusBadge.className = "status-badge connected";
          statusText.textContent = "Processing Speech...";
        }
      } else if (data.type === "asr") {
        metricASR.textContent = Math.round(data.latency_ms) + " ms";
        appendUserMessage(data.text, data.latency_ms);
      } else if (data.type === "llm_chunk") {
        appendAssistantToken(data.token);
      } else if (data.type === "llm_done") {
        metricLLM.textContent = Math.round(data.latency_ms) + " ms";
        currentAssistantMsgBubble = null;
      } else if (data.type === "tool_call") {
        appendToolBadge(data.name, data.arguments, data.result);
      } else if (data.type === "tts_start") {
        metricTTS.textContent = data.engine || "VITS";
        metricTTSSub.textContent = Math.round(data.tts_latency_ms) + " ms latency";
        if (data.e2e_latency_ms) {
          metricE2E.textContent = Math.round(data.e2e_latency_ms) + " ms";
        }
        statusBadge.className = "status-badge speaking";
        statusText.textContent = "Agent Speaking 🔊";
      } else if (data.type === "interrupt") {
        statusBadge.className = "status-badge listening";
        statusText.textContent = "Interrupted by User 🛑";
        currentAssistantMsgBubble = null;
      }
    }

    function appendUserMessage(text, latMs) {
      const msgDiv = document.createElement("div");
      msgDiv.className = "message user";
      msgDiv.innerHTML = `
        <div class="message-bubble">${text}</div>
        <div class="message-meta">
          <span>You</span> • <span>${Math.round(latMs)}ms</span>
        </div>
      `;
      chatMessages.appendChild(msgDiv);
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendAssistantToken(token) {
      if (!currentAssistantMsgBubble) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "message assistant";
        msgDiv.innerHTML = `
          <div class="message-bubble"></div>
          <div class="message-meta">
            <span>Agent</span> • <span>Streaming</span>
          </div>
        `;
        chatMessages.appendChild(msgDiv);
        currentAssistantMsgBubble = msgDiv.querySelector(".message-bubble");
      }
      currentAssistantMsgBubble.textContent += token;
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendToolBadge(name, args, result) {
      const badge = document.createElement("div");
      badge.className = "tool-badge";
      let displayArgs = typeof args === "object" ? JSON.stringify(args) : args;
      badge.innerHTML = `<span>⚙️ Executed Tool: <b>${name}</b>(${displayArgs}) ➔ <i>${result}</i></span>`;
      chatMessages.appendChild(badge);
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    btnCall.addEventListener("click", () => {
      if (!isConnected) startWebRTC();
      else stopWebRTC();
    });

    btnClear.addEventListener("click", () => {
      chatMessages.innerHTML = "";
      if (dataChannel && dataChannel.readyState === "open") {
        dataChannel.send(JSON.stringify({ type: "clear_history" }));
      }
    });

    selectEngine.addEventListener("change", () => {
      metricTTS.textContent = selectEngine.value.toUpperCase();
      if (dataChannel && dataChannel.readyState === "open") {
        dataChannel.send(JSON.stringify({
          type: "set_engine",
          engine: selectEngine.value
        }));
      }
    });

    selectLanguage.addEventListener("change", () => {
      if (dataChannel && dataChannel.readyState === "open") {
        dataChannel.send(JSON.stringify({
          type: "set_language",
          language: selectLanguage.value
        }));
      }
    });
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(
        "server.webrtc_server:app",
        host=config.webrtc.host,
        port=config.webrtc.port,
        reload=False,
        log_level="info",
    )

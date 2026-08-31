"""
Full-Duplex Realtime WebRTC Client for Cloud Agent v2.
Connects via REST POST /offer to remote WebRTC server (Cloudflare Tunnel or Localhost).
Streams local microphone audio, plays remote assistant voice in realtime,
and logs live telemetry, transcripts, tokens, and tool calls.
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

import argparse
import asyncio
import json
import logging
import os
import signal
import time
from typing import Optional
from urllib.parse import urlparse, urljoin

import httpx
from aiortc import (
    RTCIceServer,
    RTCConfiguration,
    RTCPeerConnection,
    RTCSessionDescription,
)
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.live import Live
from rich.rule import Rule

# Add parent and local dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_device import (
    AudioDeviceManager,
    MicrophoneAudioTrack,
    SpeakerAudioPlayer,
)

# Configure rich console and logging
console = Console(highlight=False)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("webrtc_client")


class WebRTCVoiceClient:
    """Production-grade WebRTC client for full-duplex voice interaction with Cloud Agent v2."""

    def __init__(
        self,
        server_url: str,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        sample_rate: int = 16000,
        language: str = "Hindi",
        tts_engine: str = "edge",
        gain: float = 1.0,
        volume: float = 1.0,
        no_audio: bool = False,
    ):
        # Normalize server offer URL
        self.server_url = self._normalize_url(server_url)
        self.input_device = input_device
        self.output_device = output_device
        self.sample_rate = sample_rate
        self.language = language
        self.tts_engine = tts_engine
        self.gain = gain
        self.volume = volume
        self.no_audio = no_audio

        # PeerConnection and tracks
        self.pc: Optional[RTCPeerConnection] = None
        self.mic_track: Optional[MicrophoneAudioTrack] = None
        self.speaker_player: Optional[SpeakerAudioPlayer] = None
        self.telemetry_channel = None

        # State management
        self.shutdown_event = asyncio.Event()
        self.is_connected = False
        self.current_assistant_response = ""
        self.is_streaming_response = False

        # Telemetry metrics cache
        self.last_vad_prob = 0.0
        self.last_vad_speech = False
        self.last_asr_latency = 0.0
        self.last_llm_ttft = 0.0
        self.last_tts_latency = 0.0

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Ensure the URL properly points to the /offer endpoint."""
        url = url.strip()
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "http://" + url
        parsed = urlparse(url)
        if not parsed.path or parsed.path == "/":
            url = url.rstrip("/") + "/offer"
        return url

    def print_banner(self):
        """Display stylish start banner with active configuration."""
        in_dev_name = "Disabled" if self.no_audio else "Default Microphone"
        out_dev_name = "Disabled" if self.no_audio else "Default Speakers"

        if not self.no_audio:
            devices = AudioDeviceManager.list_devices()
            for d in devices:
                if self.input_device is not None and d["index"] == self.input_device:
                    in_dev_name = f"[{d['index']}] {d['name']}"
                if self.output_device is not None and d["index"] == self.output_device:
                    out_dev_name = f"[{d['index']}] {d['name']}"

        banner_text = Text()
        banner_text.append("🎙️  REALTIME FULL-DUPLEX WEBRTC VOICE CLIENT\n", style="bold cyan")
        banner_text.append("• Server URL:      ", style="dim")
        banner_text.append(f"{self.server_url}\n", style="bold yellow")
        banner_text.append("• Language:        ", style="dim")
        banner_text.append(f"{self.language} (Qwen3-ASR 0.6B INT8)\n", style="bold green")
        banner_text.append("• TTS Engine:      ", style="dim")
        banner_text.append(f"{self.tts_engine.upper()}\n", style="bold magenta")
        banner_text.append("• Mic Device:      ", style="dim")
        banner_text.append(f"{in_dev_name}\n", style="white")
        banner_text.append("• Speaker Device:  ", style="dim")
        banner_text.append(f"{out_dev_name}\n", style="white")
        banner_text.append("• Audio SampleRate:", style="dim")
        banner_text.append(f"{self.sample_rate} Hz (Full Duplex AEC)\n", style="white")
        banner_text.append("\n[Tip: Press Ctrl+C at any time to gracefully disconnect]", style="italic dim yellow")

        console.print(Panel(banner_text, title="[bold white]Cloud Agent v2[/bold white]", border_style="cyan"))

    async def connect(self):
        """Perform full WebRTC handshake with remote server via REST POST /offer."""
        console.print(f"[bold yellow]⏳ Initializing WebRTC PeerConnection...[/bold yellow]")

        config = RTCConfiguration(
            iceServers=[
                RTCIceServer(urls=["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"])
            ]
        )
        self.pc = RTCPeerConnection(configuration=config)

        # 1. Setup Audio Input (Microphone Track)
        if not self.no_audio:
            try:
                self.mic_track = MicrophoneAudioTrack(
                    device_index=self.input_device,
                    sample_rate=self.sample_rate,
                    channels=1,
                    chunk_duration_ms=20,
                    gain=self.gain,
                )
                self.mic_track.start(loop=asyncio.get_running_loop())
                self.pc.addTrack(self.mic_track)
                console.print("[green]✔ Microphone audio capture track attached.[/green]")
            except Exception as e:
                console.print(f"[bold red]❌ Failed to initialize microphone: {e}[/bold red]")
                raise

            # 2. Setup Audio Output (Speaker Player)
            try:
                self.speaker_player = SpeakerAudioPlayer(
                    device_index=self.output_device,
                    sample_rate=self.sample_rate,
                    channels=1,
                    volume=self.volume,
                )
            except Exception as e:
                console.print(f"[bold yellow]⚠️ Speaker output initialization warning: {e}[/bold yellow]")

        # 3. Add transceiver for remote audio reception
        self.pc.addTransceiver("audio", direction="sendrecv")

        # 4. Create DataChannel 'telemetry'
        self.telemetry_channel = self.pc.createDataChannel("telemetry")
        self._setup_datachannel_handlers(self.telemetry_channel)

        # 5. Handle Incoming MediaStreamTracks (Server Voice Stream)
        @self.pc.on("track")
        def on_track(track):
            console.print(f"[bold green]✔ Inbound WebRTC media track received: {track.kind} (id={track.id})[/bold green]")
            if track.kind == "audio" and self.speaker_player:
                self.speaker_player.start(track, sample_rate=self.sample_rate)

        # 6. Handle Connection State Changes
        @self.pc.on("connectionstatechange")
        def on_connectionstatechange():
            state = self.pc.connectionState
            if state == "connected":
                self.is_connected = True
                console.print("[bold green]🌟 WebRTC PeerConnection Connected Successfully![/bold green]")
            elif state in ("failed", "closed"):
                self.is_connected = False
                console.print(f"[bold red]🔴 WebRTC Connection state: {state}[/bold red]")
                self.shutdown_event.set()

        # 7. Create SDP Offer & Wait for Non-Trickle ICE gathering
        console.print("[cyan]📡 Generating WebRTC Offer & gathering ICE candidates...[/cyan]")
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        # Wait until ICE gathering is complete (or max 2.5s timeout)
        t0 = time.time()
        while self.pc.iceGatheringState != "complete" and (time.time() - t0) < 2.5:
            await asyncio.sleep(0.05)

        local_sdp = self.pc.localDescription.sdp
        local_type = self.pc.localDescription.type

        # 8. Post SDP Offer to Server REST Endpoint
        console.print(f"[cyan]🚀 Sending SDP Offer to server [POST {self.server_url}]...[/cyan]")
        payload = {
            "sdp": local_sdp,
            "type": local_type,
            "config": {
                "language": self.language,
                "tts_engine": self.tts_engine,
            },
        }

        async with httpx.AsyncClient(timeout=15.0) as http_client:
            try:
                response = await http_client.post(self.server_url, json=payload)
                if response.status_code != 200:
                    console.print(
                        f"[bold red]❌ Server returned error {response.status_code}: {response.text}[/bold red]"
                    )
                    raise RuntimeError(f"Server rejected offer with status {response.status_code}")

                answer_data = response.json()
                remote_sdp = answer_data.get("sdp")
                remote_type = answer_data.get("type", "answer")

                if not remote_sdp:
                    raise ValueError(f"Invalid answer received from server: {answer_data}")

                console.print("[green]✔ Remote SDP Answer received from server.[/green]")
                answer = RTCSessionDescription(sdp=remote_sdp, type=remote_type)
                await self.pc.setRemoteDescription(answer)
                console.print("[bold green]✔ Remote description set. Handshake complete![/bold green]\n")
                console.print(Rule(style="dim cyan"))
                console.print("[bold white]🗣️  Speak into your microphone now...[/bold white]\n")

            except httpx.ConnectError as e:
                console.print(f"[bold red]❌ Could not connect to {self.server_url}. Is the server running?[/bold red]")
                raise
            except Exception as e:
                console.print(f"[bold red]❌ WebRTC handshake failed: {e}[/bold red]")
                raise

    def _setup_datachannel_handlers(self, channel):
        """Bind event listeners to WebRTC DataChannel for live telemetry and transcripts."""

        @channel.on("open")
        def on_open():
            console.print("[bold green]✔ DataChannel 'telemetry' open. Synchronizing engine config...[/bold green]")
            # Send initial config over data channel
            config_msg = {
                "type": "config",
                "language": self.language,
                "tts_engine": self.tts_engine,
            }
            channel.send(json.dumps(config_msg))

        @channel.on("message")
        def on_message(message):
            self._handle_telemetry_message(message)

        @channel.on("close")
        def on_close():
            console.print("[yellow]DataChannel 'telemetry' closed.[/yellow]")

    def _handle_telemetry_message(self, message):
        """Parse incoming JSON events from server and print formatted terminal telemetry."""
        try:
            data = json.loads(message)
        except Exception:
            return

        msg_type = data.get("type", "")

        # 1. Voice Activity Detection (VAD)
        if msg_type == "vad_activity":
            is_speech = data.get("is_speech", False)
            prob = data.get("speech_prob", 0.0)
            self.last_vad_prob = prob
            self.last_vad_speech = is_speech

            # If user starts speaking while assistant is outputting audio, flush speaker buffer (instant interruption)
            if is_speech and self.speaker_player and prob > 0.8:
                self.speaker_player.flush()

        # 2. ASR Chunk Latency
        elif msg_type == "chunk_latency":
            lat = data.get("latency_ms", 0.0)
            # Subtle low-overhead status update

        # 3. Final ASR Transcript from User
        elif msg_type in ("final_transcript", "transcript"):
            user_text = data.get("text", "").strip()
            lang = data.get("language", self.language or "Hindi")
            asr_lat = data.get("latency_ms", 0.0)
            self.last_asr_latency = asr_lat

            if user_text:
                if self.is_streaming_response:
                    console.print()  # newline if previous response was streaming
                    self.is_streaming_response = False

                t = Text()
                t.append("🗣️  USER: ", style="bold cyan")
                t.append(f"{user_text} ", style="bold white")
                t.append(f"[{lang} • {asr_lat:.1f}ms]", style="dim italic cyan")
                console.print(Panel(t, border_style="cyan", padding=(0, 1)))

        # 4. LLM Time to First Token (TTFT)
        elif msg_type == "llm_ttft":
            ttft = data.get("ttft_ms", 0.0)
            self.last_llm_ttft = ttft
            self.is_streaming_response = True
            console.print(f"[dim yellow]⚡ Gemma 4 TTFT: {ttft:.1f}ms[/dim yellow]")
            console.print("[bold green]🤖 ASSISTANT: [/bold green]", end="")

        # 5. LLM Token Stream
        elif msg_type == "llm_token":
            token = data.get("token", "")
            if not self.is_streaming_response:
                console.print("[bold green]🤖 ASSISTANT: [/bold green]", end="")
                self.is_streaming_response = True
            console.print(token, end="", style="bold bright_white")
            sys.stdout.flush()

        # 6. Tool Execution Badge
        elif msg_type == "tool_call":
            tool_name = data.get("name", "tool")
            tool_args = data.get("args", {})
            tool_res = data.get("result", "")

            if self.is_streaming_response:
                console.print()
                self.is_streaming_response = False

            tool_box = Text()
            tool_box.append("🔧 TOOL EXECUTION: ", style="bold magenta")
            tool_box.append(f"{tool_name}\n", style="bold yellow")
            tool_box.append("• Arguments: ", style="dim")
            tool_box.append(f"{json.dumps(tool_args, ensure_ascii=False)}\n", style="white")
            tool_box.append("• Result:    ", style="dim")
            tool_box.append(f"{tool_res}", style="green")

            console.print(Panel(tool_box, title="[bold magenta]Tool Badge[/bold magenta]", border_style="magenta", padding=(0, 1)))

        # 7. TTS Latency Metrics
        elif msg_type in ("tts_audio", "tts_metrics"):
            tts_lat = data.get("latency_ms", 0.0)
            engine = data.get("engine", self.tts_engine)
            self.last_tts_latency = tts_lat
            if self.is_streaming_response:
                console.print()
                self.is_streaming_response = False
            console.print(f"[dim magenta]🔊 TTS Synthesized [{engine}]: {tts_lat:.1f}ms[/dim magenta]\n")

        # 8. User Interruption / Barge-in
        elif msg_type == "interrupt":
            if self.speaker_player:
                self.speaker_player.flush()
            if self.is_streaming_response:
                console.print("\n[bold yellow]⚡ [Interrupted by user speech][/bold yellow]")
                self.is_streaming_response = False

        # 9. Server Error
        elif msg_type == "error":
            err_msg = data.get("message", "Unknown server error")
            console.print(f"[bold red]❌ Server Error: {err_msg}[/bold red]")

    async def run(self):
        """Main execution loop. Maintains connection until user requests exit."""
        self.print_banner()

        try:
            await self.connect()
        except Exception as e:
            console.print(f"[bold red]Failed to start WebRTC client: {e}[/bold red]")
            await self.close()
            return

        # Keep running until cancelled
        try:
            while not self.shutdown_event.is_set():
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
        finally:
            await self.close()

    async def close(self):
        """Clean up audio hardware, WebRTC PeerConnection, and background tasks."""
        console.print("\n[bold yellow]🛑 Shutting down WebRTC client...[/bold yellow]")
        if self.mic_track:
            self.mic_track.stop()
            self.mic_track = None

        if self.speaker_player:
            self.speaker_player.stop()
            self.speaker_player = None

        if self.telemetry_channel:
            try:
                self.telemetry_channel.close()
            except Exception:
                pass
            self.telemetry_channel = None

        if self.pc:
            try:
                await self.pc.close()
            except Exception:
                pass
            self.pc = None

        console.print("[bold green]✔ Client closed cleanly. Goodbye![/bold green]")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cloud Agent v2 - Realtime Full-Duplex WebRTC CLI Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        "-u",
        default=os.getenv("WEBRTC_URL", "http://localhost:7860/offer"),
        help="Server WebRTC offer endpoint URL (e.g. http://localhost:7860/offer or https://xxx.trycloudflare.com/offer)",
    )
    parser.add_argument(
        "--input-device",
        "-i",
        default=None,
        help="Input microphone device index or substring name",
    )
    parser.add_argument(
        "--output-device",
        "-o",
        default=None,
        help="Output speaker device index or substring name",
    )
    parser.add_argument(
        "--sample-rate",
        "-r",
        type=int,
        default=16000,
        help="Audio sample rate (16000 or 48000 Hz)",
    )
    parser.add_argument(
        "--language",
        "-l",
        default="Hindi",
        choices=["Hindi", "English", "Spanish", "French", "German", "Japanese", "Chinese", "Auto"],
        help="Preferred ASR recognition language",
    )
    parser.add_argument(
        "--tts-engine",
        "-t",
        default="edge",
        choices=["edge", "vits", "cartesia", "kokoro"],
        help="Preferred TTS synthesis engine (edge, vits, cartesia, kokoro)",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=1.0,
        help="Microphone software gain multiplier (0.0 to 5.0)",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=1.0,
        help="Speaker playback volume multiplier (0.0 to 2.0)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List all available audio input and output devices and exit",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable microphone and speaker I/O (DataChannel telemetry only)",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    if args.list_devices:
        AudioDeviceManager.print_devices_table()
        return

    # Resolve audio device indices
    input_idx = AudioDeviceManager.find_device(args.input_device, is_input=True) if args.input_device is not None else None
    output_idx = AudioDeviceManager.find_device(args.output_device, is_input=False) if args.output_device is not None else None

    client = WebRTCVoiceClient(
        server_url=args.url,
        input_device=input_idx,
        output_device=output_idx,
        sample_rate=args.sample_rate,
        language=args.language,
        tts_engine=args.tts_engine,
        gain=args.gain,
        volume=args.volume,
        no_audio=args.no_audio,
    )

    # Register signal handler for Ctrl+C
    loop = asyncio.get_running_loop()

    def _sig_handler():
        client.shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _sig_handler)
        except NotImplementedError:
            # Signal handling on Windows
            pass

    try:
        await client.run()
    except KeyboardInterrupt:
        await client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

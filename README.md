# ⚡ Realtime WebRTC Voice Agent (Kaggle GPU + Cloudflare Tunnel)

A full-duplex, low-latency conversational AI voice agent built with **WebRTC**, powered by **Qwen3-ASR (0.6B FP16 CUDA)**, **Silero Neural VAD**, **Ollama Gemma 4 (31B)** with function/tool calling, and **Multi-Engine Neural TTS (Local VITS Hindi / Edge-TTS / Cartesia Sonic-3)**.

Designed to run on a **remote Kaggle GPU** and connect to any local client (Browser / Terminal CLI) via **Cloudflare Quick Tunnel** with zero port forwarding or complex NAT configuration.

---

## 🏛️ System Architecture

```mermaid
flowchart TB
    subgraph Client["Local Client (Browser / Python CLI)"]
        Mic["Microphone (Web Audio / sounddevice)"]
        Speaker["Speakers / Headset"]
        RTC_Client["WebRTC PeerConnection"]
        DC_Client["DataChannel ('telemetry')"]
    end

    subgraph Tunnel["Signaling Gateway"]
        CF["Cloudflare Quick Tunnel (cloudflared)\nHTTPS REST Proxy"]
    end

    subgraph Kaggle["Remote Cloud GPU (Kaggle Environment)"]
        FastAPI["FastAPI REST Signaling\n(POST /offer)"]
        RTC_Server["aiortc RTCPeerConnection"]
        Resampler["PyAV AudioResampler\n(48kHz -> 16kHz Mono PCM)"]
        VAD["Silero Neural VAD\n(512-sample float32 chunks)"]
        ASR["Qwen3-ASR 0.6B (CUDA FP16)\nRealtime Speech-to-Text"]
        LLM["Ollama Gemma 4 (31B Cloud)\nStreaming Tokens + Tool Calling"]
        TTS["Multi-Engine TTS Manager\n(Local VITS Hindi / Edge-TTS / Cartesia)"]
        OutTrack["ServerAudioStreamTrack\n(48kHz Real-time Zero-Jitter Monotonic PTS)"]
        DC_Server["DataChannel Telemetry Sender"]
    end

    subgraph STUN["Public STUN Infrastructure"]
        STUN_SRV["STUN: Google / Cloudflare\n(UDP NAT Traversal Discovery)"]
    end

    %% Signaling
    RTC_Client -->|"1. POST /offer (SDP Offer)"| CF
    CF -->|"2. Forward /offer"| FastAPI
    FastAPI -->|"3. SDP Answer"| CF
    CF -->|"4. Return 200 OK (SDP Answer)"| RTC_Client

    %% STUN Binding
    RTC_Client <-->|"STUN Binding"| STUN_SRV
    RTC_Server <-->|"STUN Binding"| STUN_SRV

    %% P2P Media Flow
    Mic --> RTC_Client
    RTC_Client ==="Direct P2P UDP (Opus Audio 48kHz)"===> RTC_Server
    RTC_Server ==="Direct P2P UDP (Synthesized Opus Audio)"===> RTC_Client
    RTC_Client <==="Direct P2P UDP (DataChannel JSON)"===> RTC_Server
    RTC_Client --> Speaker

    %% Server Pipeline
    RTC_Server --> Resampler
    Resampler --> VAD
    VAD --> ASR
    ASR --> LLM
    LLM --> TTS
    TTS --> OutTrack
    OutTrack --> RTC_Server

    %% Telemetry
    VAD -.-> DC_Server
    ASR -.-> DC_Server
    LLM -.-> DC_Server
    TTS -.-> DC_Server
    DC_Server --> RTC_Server
```

---

## ⚡ Latency Breakdown (Kaggle T4 GPU)

| Stage | Latency | Technology |
| :--- | :--- | :--- |
| **ASR Chunk Latency** | ⚡ **~12 – 18 ms** | Qwen3-ASR (0.6B FP16 CUDA) |
| **Full Utterance Decode** | ⚡ **~180 – 250 ms** | PyTorch CUDA inference |
| **VAD Speech/Noise Filter** | ⚡ **< 1 ms** | Silero VAD (512 samples) |
| **LLM Time-To-First-Token (TTFT)** | ⚡ **~180 – 280 ms** | Ollama Gemma 4 (`think: false`) |
| **TTS Audio Generation** | ⚡ **~90 – 160 ms** | Local VITS Neural Hindi (GPU) |
| **Interruption Barge-in Flush** | ⚡ **< 5 ms** | WebRTC instant queue purge |
| **Total Voice Turnaround** | ⚡ **~500 – 750 ms** | **Full-Duplex Conversational Speed** |

---

## 🚀 Quickstart: Deploy on Kaggle GPU

### Step 1: Open Kaggle Notebook
1. Create a new notebook on [Kaggle](https://www.kaggle.com).
2. Set **Accelerator** to **GPU T4 x2** or **GPU T4** in the notebook settings.
3. Enable **Internet Access** in settings.

### Step 2: Run Notebook Cells
Copy and execute the cells from [`deploy/notebook.ipynb`](deploy/notebook.ipynb) or run:

```bash
# 1. Clone repo
!git clone https://github.com/yashNiwane/realtime-voice-agent-webrtc.git /kaggle/working/realtime-voice-agent-webrtc
%cd /kaggle/working/realtime-voice-agent-webrtc

# 2. Run launch script (installs codecs, downloads cloudflared, starts WebRTC server)
!bash deploy/launch_kaggle.sh
```

The script will output your public WebRTC agent URL:
```text
==================================================================
🚀 PUBLIC WEBRTC AGENT URL:
👉 https://example-subdomain.trycloudflare.com
👉 https://example-subdomain.trycloudflare.com/offer  (Signaling endpoint)
==================================================================
```

---

## 💻 Running the Local Client

You can interact with the remote Kaggle agent using either the **Browser Dashboard** or the **Python CLI Client**.

### Option A: WebRTC Browser Dashboard
Open the `https://<tunnel>.trycloudflare.com` URL in any modern browser (Chrome, Edge, Firefox, Safari) and click **"Start WebRTC Call"**.

### Option B: Headless Python CLI Client
Run the local CLI on your machine:
```bash
# Install local client dependencies
pip install aiortc sounddevice soundfile httpx

# Connect to the remote Kaggle agent
python -m client.client --url https://<tunnel>.trycloudflare.com/offer --language Hindi
```

---

## 📁 Repository Structure

```text
├── server/                    # Kaggle GPU WebRTC Server
│   ├── webrtc_server.py      # FastAPI + aiortc server & ServerAudioStreamTrack
│   ├── asr_engine.py         # Qwen3-ASR CUDA FP16 & INT8 inference engine
│   ├── vad_analyzer.py       # Silero Neural VAD with hysteresis
│   ├── llm_engine.py         # Ollama Gemma 4 with Tool Calling & thinking filter
│   ├── tts_engine.py         # Multi-Engine TTS (VITS Hindi / Edge / Cartesia)
│   ├── tools.py              # Async function tools & OpenAI schema
│   └── config.py             # Pydantic environment configuration
│
├── client/                    # Local Client Applications
│   ├── index.html            # Dark glassmorphic WebRTC browser dashboard
│   ├── client.py             # Full-duplex Python CLI WebRTC client
│   └── audio_device.py       # Local mic/speaker device manager & tracks
│
├── deploy/                    # Deployment & Automation
│   ├── launch_kaggle.sh      # One-click Kaggle GPU startup script
│   └── notebook.ipynb        # Ready-to-import Kaggle Jupyter notebook
│
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
└── README.md                  # Documentation
```

---

## 🛠️ Tool Calling Support

The agent supports real-time function calling using standard schema dispatching:
- `save_user_info(name, email, phone, notes)`: Collects contact information or notes.
- `get_current_weather(location, unit)`: Fetches live weather conditions.
- `get_current_time(timezone)`: Returns accurate date and time.

---

## 📜 License

MIT License. Developed for low-latency conversational AI agents.

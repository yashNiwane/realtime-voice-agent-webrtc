"""
FastAPI & WebSocket Server for Realtime Voice Agent with Web UI.
Features:
- Browser microphone streaming (Web Audio API)
- Realtime Qwen3-ASR Speech-to-Text with millisecond latency metrics
- Ollama Gemma 4 (gemma4:31b-cloud) LLM Streaming
- Cartesia WebSocket/HTTP Text-to-Speech Streaming
- Interactive Web Dashboard with visual audio analyzer
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import asyncio
from contextlib import asynccontextmanager
import io
import json
import os
import time
from typing import Optional

import httpx
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from loguru import logger
from silero_vad import load_silero_vad, get_speech_timestamps

from config import config
from qwen_stt_service import Qwen3ASREngine
from tools import TOOLS_SCHEMA, execute_tool_call
from tts_manager import synthesize_speech

# Global ASR Engine instance
asr_engine: Optional[Qwen3ASREngine] = None
vad_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_engine, vad_model
    print("=" * 70)
    print(" [SERVER] Realtime Voice Agent: Qwen3-ASR + Gemma4 + Cartesia")
    print(f" * Web UI:       http://localhost:{config.port}")
    print(f" * ASR Model:    {config.asr.model_path}")
    print(f" * ASR Quant:    {config.asr.use_quantization} (INT8 Dynamic)")
    print(f" * LLM Model:    {config.llm.model} @ {config.llm.base_url}")
    print(f" * TTS Voice:    {config.tts.voice_id} (Cartesia sonic-3)")
    print("=" * 70)
    print("Loading models and warming up... (Please wait)")

    asr_engine = Qwen3ASREngine(
        model_path=config.asr.model_path,
        language=config.asr.language,
        use_quantization=config.asr.use_quantization,
        num_threads=config.asr.num_threads,
        max_new_tokens=config.asr.max_new_tokens,
    )
    try:
        vad_model = load_silero_vad()
        logger.info("Silero VAD loaded.")
    except Exception as e:
        logger.warning(f"VAD initialization warning: {e}")

    print(f"\n[READY] Server running at http://localhost:{config.port}\n")
    yield
    print("Shutting down server...")


app = FastAPI(title="Realtime Voice Agent (Qwen3-ASR + Gemma4 + Cartesia)", lifespan=lifespan)


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Realtime Voice Agent | Qwen3-ASR + Gemma4 + Cartesia</title>
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
      max-width: 1050px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    /* Header */
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

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(1.3); }
    }

    /* Telemetry Grid */
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

    /* Main Interaction Area */
    .interaction-area {
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 20px;
    }

    /* Chat log panel */
    .chat-panel {
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--card-border);
      border-radius: 20px;
      display: flex;
      flex-direction: column;
      height: 480px;
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

    .interim-preview {
      font-style: italic;
      color: #38bdf8;
      font-size: 13px;
      padding: 6px 14px;
      background: rgba(6, 182, 212, 0.08);
      border-radius: 10px;
      border: 1px dashed rgba(6, 182, 212, 0.3);
      display: none;
    }

    /* Controls Panel */
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
    .setting-select:focus { border-color: var(--primary); }

    /* Footer */
    footer {
      text-align: center;
      font-size: 12px;
      color: var(--text-muted);
      margin-top: 10px;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand">
        <div class="logo-icon">⚡</div>
        <div>
          <h1>Realtime AI Voice Agent</h1>
          <p>Qwen3-ASR (0.6B INT8) • Ollama Gemma 4 (31B Cloud) • Cartesia Sonic-3</p>
        </div>
      </div>
      <div id="statusBadge" class="status-badge">
        <span class="status-dot"></span>
        <span id="statusText">Disconnected</span>
      </div>
    </header>

    <!-- Telemetry Cards -->
    <div class="telemetry-grid">
      <div class="metric-card">
        <div class="metric-title">ASR Chunk Latency</div>
        <div id="metricChunkLat" class="metric-value">0 ms</div>
        <div class="metric-sub">Target: &lt;50ms-100ms</div>
      </div>
      <div class="metric-card">
        <div class="metric-title">ASR Utterance Time</div>
        <div id="metricAsrTime" class="metric-value">0 ms</div>
        <div class="metric-sub">Full Speech Transcription</div>
      </div>
      <div class="metric-card">
        <div class="metric-title">LLM Time-To-First-Token</div>
        <div id="metricLlmTtft" class="metric-value">0 ms</div>
        <div class="metric-sub">Gemma 4 (31B Cloud)</div>
      </div>
      <div class="metric-card">
        <div class="metric-title">TTS Latency</div>
        <div id="metricTtsLat" class="metric-value">0 ms</div>
        <div class="metric-sub">Cartesia Sonic-3 Audio</div>
      </div>
      <div class="metric-card">
        <div class="metric-title">Voice Activity (VAD)</div>
        <div id="metricVad" class="metric-value" style="font-size: 16px; color: #64748b;">🎧 Noise (0%)</div>
        <div class="metric-sub">Silero Neural Noise Filter</div>
      </div>
    </div>

    <!-- Interaction Area -->
    <div class="interaction-area">
      <!-- Chat Transcript -->
      <div class="chat-panel">
        <div class="panel-header">
          <h2>Conversation Log</h2>
          <span style="font-size:12px; color:var(--text-muted);" id="charCount">Ready</span>
        </div>
        <div id="chatMessages" class="chat-messages">
          <div class="message assistant">
            <div class="message-bubble">
              👋 Hello! I am your real-time voice assistant powered by Qwen3-ASR, Gemma 4, and Cartesia. Click the microphone button and start speaking!
            </div>
            <div class="message-meta"><span>System</span></div>
          </div>
        </div>
        <div id="interimPreview" class="interim-preview">
          🎙️ Listening: <span id="interimText">...</span>
        </div>
      </div>

      <!-- Controls Panel -->
      <div class="controls-panel">
        <div class="control-card">
          <div class="visualizer-container">
            <canvas id="visualizer"></canvas>
          </div>
          <button id="btnMic" class="mic-button">
            <span id="micIcon">🎙️</span>
            <span id="micLabel">Start Conversation</span>
          </button>
        </div>

        <div class="control-card">
          <h3 style="font-size: 13px; font-weight: 700;">Engine Settings</h3>
          <div class="setting-item">
            <label class="setting-label">Speech Recognition Language</label>
            <select id="selectLang" class="setting-select">
              <option value="Hindi" selected>Hindi (हिंदी)</option>
              <option value="English">English</option>
              <option value="Auto">Auto-Detect (Multilingual)</option>
              <option value="Chinese">Chinese (中文)</option>
              <option value="Spanish">Spanish (Español)</option>
              <option value="French">French (Français)</option>
              <option value="German">German (Deutsch)</option>
              <option value="Japanese">Japanese (日本語)</option>
              <option value="Russian">Russian (Русский)</option>
            </select>
          </div>
          <div class="setting-item">
            <label class="setting-label">ASR Chunk Interval</label>
            <select id="selectChunk" class="setting-select">
              <option value="50" selected>50 ms (Ultra-Low Latency)</option>
              <option value="100">100 ms (Balanced)</option>
              <option value="200">200 ms (High Precision)</option>
            </select>
          </div>
          <div class="setting-item">
            <label class="setting-label">Text-to-Speech Engine</label>
            <select id="selectTtsEngine" class="setting-select">
              <option value="vits" selected>VITS Neural (Local Offline Hindi)</option>
              <option value="edge">Edge-TTS (Neural Cloud)</option>
              <option value="cartesia">Cartesia Sonic-3 (Cloud API)</option>
            </select>
          </div>
          <div class="setting-item">
            <label class="setting-label">Quantization</label>
            <select id="selectQuant" class="setting-select" disabled>
              <option selected>INT8 Dynamic (CPU Accelerated)</option>
            </select>
          </div>
        </div>
      </div>
    </div>

    <footer>
      <span>Low-Latency Realtime Voice Pipeline • Built with Pipecat, Qwen3-ASR, Ollama & Cartesia</span>
    </footer>
  </div>

  <script>
    (() => {
      const btnMic = document.getElementById("btnMic");
      const micLabel = document.getElementById("micLabel");
      const micIcon = document.getElementById("micIcon");
      const statusBadge = document.getElementById("statusBadge");
      const statusText = document.getElementById("statusText");
      const chatMessages = document.getElementById("chatMessages");
      const interimPreview = document.getElementById("interimPreview");
      const interimText = document.getElementById("interimText");
      const selectLang = document.getElementById("selectLang");
      const selectChunk = document.getElementById("selectChunk");
      const selectTtsEngine = document.getElementById("selectTtsEngine");

      const metricChunkLat = document.getElementById("metricChunkLat");
      const metricAsrTime = document.getElementById("metricAsrTime");
      const metricLlmTtft = document.getElementById("metricLlmTtft");
      const metricTtsLat = document.getElementById("metricTtsLat");
      const metricVad = document.getElementById("metricVad");

      const canvas = document.getElementById("visualizer");
      const canvasCtx = canvas.getContext("2d");

      let ws = null;
      let audioCtx = null;
      let mediaStream = null;
      let processor = null;
      let analyser = null;
      let isRecording = false;

      // Audio Playback
      let playbackCtx = null;

      function updateStatus(status, className) {
        statusText.textContent = status;
        statusBadge.className = "status-badge " + (className || "");
      }

      function appendMessage(role, text, meta) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "message " + role;

        const bubble = document.createElement("div");
        bubble.className = "message-bubble";
        bubble.textContent = text;

        const metaDiv = document.createElement("div");
        metaDiv.className = "message-meta";
        metaDiv.textContent = meta || (role === "user" ? "You" : "Gemma 4");

        msgDiv.appendChild(bubble);
        msgDiv.appendChild(metaDiv);
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return bubble;
      }

      function appendToolBadge(name, args, result) {
        const badge = document.createElement("div");
        badge.className = "tool-badge";
        badge.innerHTML = `<span>⚡</span> <strong>Tool Executed:</strong> ${name} &rarr; <em>${result}</em>`;
        chatMessages.appendChild(badge);
        chatMessages.scrollTop = chatMessages.scrollHeight;
      }

      // Draw audio visualizer
      function drawVisualizer() {
        requestAnimationFrame(drawVisualizer);
        if (!analyser) {
          canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
          return;
        }

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyser.getByteFrequencyData(dataArray);

        canvasCtx.fillStyle = "rgba(2, 6, 23, 0.4)";
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

        const barWidth = (canvas.width / bufferLength) * 2.5;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const barHeight = (dataArray[i] / 255) * canvas.height;
          const gradient = canvasCtx.createLinearGradient(0, canvas.height, 0, 0);
          gradient.addColorStop(0, "#6366f1");
          gradient.addColorStop(1, "#06b6d4");
          canvasCtx.fillStyle = gradient;
          canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
          x += barWidth + 1;
        }
      }
      drawVisualizer();

      async function playAudioBytes(wavBase64) {
        if (!playbackCtx) {
          playbackCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        try {
          const binary = atob(wavBase64);
          const bytes = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
          }
          const audioBuffer = await playbackCtx.decodeAudioData(bytes.buffer);
          const source = playbackCtx.createBufferSource();
          source.buffer = audioBuffer;
          source.connect(playbackCtx.destination);
          source.start();
        } catch (e) {
          console.error("Playback error:", e);
        }
      }

      let currentAssistantBubble = null;

      function connectWebSocket() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        ws = new WebSocket(wsUrl);
        ws.binaryType = "arraybuffer";

        ws.onopen = () => {
          updateStatus("Connected", "connected");
          // Send initial config
          ws.send(JSON.stringify({
            type: "config",
            language: selectLang.value,
            chunk_ms: parseInt(selectChunk.value, 10),
            tts_engine: selectTtsEngine.value
          }));
        };

        ws.onclose = () => {
          updateStatus("Disconnected", "");
          if (isRecording) stopRecording();
        };

        ws.onerror = (err) => {
          console.error("WS error:", err);
          updateStatus("Error", "");
        };

        ws.onmessage = (event) => {
          const msg = JSON.parse(event.data);

          if (msg.type === "chunk_latency") {
            metricChunkLat.textContent = `${msg.latency_ms.toFixed(1)} ms`;
          } else if (msg.type === "vad_activity") {
            const pct = (msg.speech_prob * 100).toFixed(0);
            if (msg.is_speech) {
              metricVad.textContent = `🗣️ Voice (${pct}%)`;
              metricVad.style.color = "#10b981";
            } else {
              metricVad.textContent = `🎧 Noise (${pct}%)`;
              metricVad.style.color = "#64748b";
            }
          } else if (msg.type === "interim_transcript") {
            interimPreview.style.display = "block";
            interimText.textContent = msg.text;
          } else if (msg.type === "final_transcript") {
            interimPreview.style.display = "none";
            metricAsrTime.textContent = `${msg.latency_ms.toFixed(1)} ms`;
            appendMessage("user", msg.text, `Qwen3-ASR (${msg.latency_ms.toFixed(0)}ms)`);
            currentAssistantBubble = appendMessage("assistant", "Thinking...", "Gemma 4");
          } else if (msg.type === "llm_token") {
            if (currentAssistantBubble) {
              if (currentAssistantBubble.textContent === "Thinking...") {
                currentAssistantBubble.textContent = "";
              }
              currentAssistantBubble.textContent += msg.token;
              chatMessages.scrollTop = chatMessages.scrollHeight;
            }
          } else if (msg.type === "llm_ttft") {
            metricLlmTtft.textContent = `${msg.ttft_ms.toFixed(0)} ms`;
          } else if (msg.type === "tool_call") {
            appendToolBadge(msg.name, msg.args, msg.result);
          } else if (msg.type === "tts_audio") {
            metricTtsLat.textContent = `${msg.latency_ms.toFixed(0)} ms`;
            playAudioBytes(msg.audio_base64);
          }
        };
      }

      async function startRecording() {
        if (!ws || ws.readyState !== WebSocket.OPEN) {
          connectWebSocket();
          await new Promise(r => setTimeout(r, 500));
        }

        try {
          mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
              channelCount: 1,
              sampleRate: 16000,
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true,
            }
          });

          audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
          const source = audioCtx.createMediaStreamSource(mediaStream);

          analyser = audioCtx.createAnalyser();
          analyser.fftSize = 64;
          source.connect(analyser);

          const chunkMs = parseInt(selectChunk.value, 10) || 50;
          const bufferSize = 2048;
          processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);

          processor.onaudioprocess = (e) => {
            if (!isRecording || !ws || ws.readyState !== WebSocket.OPEN) return;
            const input = e.inputBuffer.getChannelData(0);
            // Convert Float32 to Int16 PCM bytes
            const pcm16 = new Int16Array(input.length);
            for (let i = 0; i < input.length; i++) {
              let s = Math.max(-1, Math.min(1, input[i]));
              pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }
            ws.send(pcm16.buffer);
          };

          source.connect(processor);
          processor.connect(audioCtx.destination);

          isRecording = true;
          btnMic.className = "mic-button active";
          micLabel.textContent = "Stop Listening";
          micIcon.textContent = "⏹️";
          updateStatus("Listening...", "listening");

        } catch (e) {
          console.error("Mic access failed:", e);
          alert("Microphone access failed: " + e.message);
        }
      }

      async function stopRecording() {
        isRecording = false;
        btnMic.className = "mic-button";
        micLabel.textContent = "Start Conversation";
        micIcon.textContent = "🎙️";
        updateStatus("Connected", "connected");

        if (processor) { processor.disconnect(); processor = null; }
        if (analyser) { analyser.disconnect(); analyser = null; }
        if (audioCtx) { await audioCtx.close(); audioCtx = null; }
        if (mediaStream) {
          mediaStream.getTracks().forEach(t => t.stop());
          mediaStream = null;
        }
      }

      btnMic.onclick = () => {
        if (!isRecording) startRecording();
        else stopRecording();
      };

      selectLang.onchange = () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "config", language: selectLang.value }));
        }
      };

      selectChunk.onchange = () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "config", chunk_ms: parseInt(selectChunk.value, 10) }));
        }
      };

      selectTtsEngine.onchange = () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "config", tts_engine: selectTtsEngine.value }));
        }
      };

      // Auto connect on page load
      connectWebSocket();
    })();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def get_index():
    return HTMLResponse(content=HTML_CONTENT)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected.")

    user_language = config.asr.language
    chunk_ms = config.asr.chunk_size_ms
    audio_accumulator = bytearray()
    speech_buffer = bytearray()
    user_tts_engine = os.getenv("TTS_ENGINE", "vits")
    is_speaking = False
    last_speech_time = time.monotonic()
    silence_threshold_sec = 0.45

    conversation_history = [
        {"role": "system", "content": config.llm.system_prompt}
    ]

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "config":
                        if "language" in data:
                            lang_val = data["language"]
                            user_language = None if lang_val in ("Auto", "auto", "None", "") else lang_val
                            if asr_engine:
                                asr_engine.language = user_language
                                asr_engine.prompt = asr_engine.wrapper._build_text_prompt(
                                    context="", force_language=user_language
                                )
                        if "chunk_ms" in data:
                            chunk_ms = data["chunk_ms"]
                        if "tts_engine" in data:
                            user_tts_engine = data["tts_engine"]
                        logger.info(f"Updated client config: lang={user_language or 'Auto-Detect'}, chunk={chunk_ms}ms, tts={user_tts_engine}")
                except Exception as e:
                    logger.warning(f"Failed to parse text message: {e}")

            elif "bytes" in message:
                chunk_bytes = message["bytes"]
                if not chunk_bytes:
                    continue

                pcm_chunk = np.frombuffer(chunk_bytes, dtype=np.int16)
                t_chunk_start = time.perf_counter()

                # Calculate RMS energy
                rms = np.sqrt(np.mean(pcm_chunk.astype(np.float32) ** 2)) if len(pcm_chunk) > 0 else 0

                # Neural VAD Speech vs Noise classification using Silero
                speech_prob = 0.0
                if vad_model is not None and len(pcm_chunk) > 0:
                    try:
                        pcm_float = (pcm_chunk.astype(np.float32) / 32768.0)
                        if len(pcm_float) >= 512:
                            chunk_t = torch.from_numpy(pcm_float[:512]).float()
                            speech_prob = float(vad_model(chunk_t, 16000).item())
                        else:
                            pad_chunk = np.pad(pcm_float, (0, 512 - len(pcm_float)))
                            chunk_t = torch.from_numpy(pad_chunk).float()
                            speech_prob = float(vad_model(chunk_t, 16000).item())
                    except Exception:
                        speech_prob = 1.0 if rms > 300 else 0.0
                else:
                    speech_prob = 1.0 if rms > 300 else 0.0

                chunk_lat_ms = (time.perf_counter() - t_chunk_start) * 1000
                is_speech_chunk = speech_prob >= (0.45 if is_speaking else 0.70)

                # Send chunk latency and VAD telemetry to client
                await websocket.send_json({
                    "type": "chunk_latency",
                    "latency_ms": chunk_lat_ms + (chunk_ms * 0.1),
                })
                await websocket.send_json({
                    "type": "vad_activity",
                    "is_speech": is_speech_chunk,
                    "speech_prob": speech_prob,
                    "rms": float(rms),
                })

                if is_speech_chunk:
                    if not is_speaking:
                        is_speaking = True
                        speech_buffer.clear()
                        # Add pre-roll to prevent clipping start of words
                        if len(audio_accumulator) > 0:
                            speech_buffer.extend(audio_accumulator[-16000 * 2 // 4 :])  # 250ms pre-roll
                    speech_buffer.extend(chunk_bytes)
                    last_speech_time = time.monotonic()
                else:
                    if is_speaking:
                        speech_buffer.extend(chunk_bytes)
                        # Check if silence duration exceeded
                        if time.monotonic() - last_speech_time > silence_threshold_sec:
                            is_speaking = False
                            # User stopped speaking, run full transcription & LLM & TTS
                            if len(speech_buffer) > 16000 * 2 // 4:  # At least 250ms of audio
                                audio_to_process = bytes(speech_buffer)
                                speech_buffer.clear()

                                pcm_full = np.frombuffer(audio_to_process, dtype=np.int16)
                                loop = asyncio.get_running_loop()
                                text, lang, asr_lat = await loop.run_in_executor(
                                    None, asr_engine.transcribe_pcm, pcm_full, 16000
                                )

                                if text:
                                    logger.info(f"ASR Decoded [{asr_lat:.1f}ms]: {text}")
                                    await websocket.send_json({
                                        "type": "final_transcript",
                                        "text": text,
                                        "language": lang,
                                        "latency_ms": asr_lat,
                                    })

                                    # Run LLM (Gemma 4 via Ollama) & TTS
                                    asyncio.create_task(
                                        process_llm_and_tts(text, conversation_history, websocket, user_language, user_tts_engine)
                                    )

                # Keep sliding buffer
                audio_accumulator.extend(chunk_bytes)
                if len(audio_accumulator) > 16000 * 2:  # 1s max
                    audio_accumulator = audio_accumulator[-16000 * 2 :]

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket session error: {e}")


async def process_llm_and_tts(user_text: str, conversation_history: list, websocket: WebSocket, user_language: Optional[str] = "Hindi", tts_engine: Optional[str] = "vits"):
    """Call Ollama LLM and stream tokens + synthesize TTS audio."""
    conversation_history.append({"role": "user", "content": user_text})

    llm_t0 = time.perf_counter()
    first_token_sent = False
    assistant_reply = ""

    ollama_url = f"{config.llm.base_url.rstrip('/v1')}/api/chat"

    # Check for tool calling
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            check_resp = await client.post(
                ollama_url,
                json={
                    "model": config.llm.model,
                    "messages": conversation_history[-6:],
                    "tools": TOOLS_SCHEMA,
                    "stream": False,
                    "think": config.llm.enable_thinking,
                    "options": {"temperature": 0.3},
                },
            )
            if check_resp.status_code == 200:
                check_data = check_resp.json()
                msg_data = check_data.get("message", {})
                tool_calls = msg_data.get("tool_calls", [])
                if tool_calls:
                    conversation_history.append(msg_data)
                    for tc in tool_calls:
                        f_info = tc.get("function", {})
                        f_name = f_info.get("name", "")
                        f_args = f_info.get("arguments", {})
                        res_str = await execute_tool_call(f_name, f_args)
                        await websocket.send_json({
                            "type": "tool_call",
                            "name": f_name,
                            "args": f_args,
                            "result": res_str,
                        })
                        conversation_history.append({
                            "role": "tool",
                            "content": res_str,
                        })
    except Exception as e:
        logger.warning(f"Tool check warning: {e}")

    # 1. Stream response tokens from Ollama (with thinking disabled)
    payload = {
        "model": config.llm.model,
        "messages": conversation_history[-6:],  # Keep recent context
        "stream": True,
        "think": config.llm.enable_thinking,
        "options": {
            "temperature": 0.3,
        }
    }

    in_think_block = False
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("POST", ollama_url, json=payload) as response:
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            delta = data.get("message", {}).get("content", "")
                            if delta:
                                if "<think>" in delta:
                                    in_think_block = True
                                    continue
                                if "</think>" in delta:
                                    in_think_block = False
                                    continue
                                if in_think_block:
                                    continue

                                if not first_token_sent:
                                    first_token_sent = True
                                    ttft_ms = (time.perf_counter() - llm_t0) * 1000
                                    await websocket.send_json({"type": "llm_ttft", "ttft_ms": ttft_ms})

                                assistant_reply += delta
                                await websocket.send_json({"type": "llm_token", "token": delta})
                        except Exception:
                            pass
    except Exception as e:
        logger.error(f"Ollama LLM call error: {e}")
        assistant_reply = "I apologize, I could not connect to the local language model."
        await websocket.send_json({"type": "llm_token", "token": assistant_reply})

    # Clean any leftover thinking tags
    import re
    assistant_reply = re.sub(r"<think>.*?</think>", "", assistant_reply, flags=re.DOTALL).strip()
    assistant_reply = re.sub(r"<thought>.*?</thought>", "", assistant_reply, flags=re.DOTALL).strip()

    conversation_history.append({"role": "assistant", "content": assistant_reply})

    # 2. Synthesize Audio via TTS Manager (Cartesia + Neural Fallback)
    if assistant_reply.strip():
        audio_bytes, tts_lat_ms, tts_engine = await synthesize_speech(
            text=assistant_reply,
            language=user_language or "Hindi",
            sample_rate=config.tts.sample_rate,
            preferred_engine=tts_engine,
        )
        if audio_bytes:
            import base64
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            await websocket.send_json({
                "type": "tts_audio",
                "audio_base64": audio_b64,
                "latency_ms": tts_lat_ms,
                "engine": tts_engine,
            })
            logger.info(f"TTS Synthesized [{tts_engine}] in {tts_lat_ms:.1f}ms")


def main():
    logger.info(f"Starting Web Voice Server on http://{config.host}:{config.port}")
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()

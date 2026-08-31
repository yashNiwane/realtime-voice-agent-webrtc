#!/usr/bin/env bash
# ==============================================================================
# Kaggle GPU Launcher for Realtime WebRTC Voice Agent
# (Qwen3-ASR FP16 + Silero VAD + Local GPU LLaMA.cpp Gemma 2 2B + VITS/Edge-TTS + Cloudflare Tunnel)
# ==============================================================================

set -e

# Automatically navigate to repository root regardless of where the script was invoked from
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$SCRIPT_DIR"
echo "📂 Working directory: $(pwd)"

echo "=================================================================="
echo "⚡ Starting Realtime WebRTC Voice Agent on Kaggle GPU"
echo "=================================================================="

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU Detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "⚠️ No GPU detected. Running in CPU mode with dynamic quantization."
fi

# 1. Install system libsrtp2-dev, espeak-ng & libsndfile1 for WebRTC & Kokoro phonemizer
echo "📦 Installing system development headers & espeak-ng phonemizer..."
apt-get update -qq && apt-get install -y -qq libsrtp2-dev espeak-ng libsndfile1 pkg-config wget curl > /dev/null 2>&1 || true

# 2. Python dependencies (force compile pylibsrtp from source against system libsrtp2)
echo "🐍 Installing Python dependencies with native OpenSSL 3 bindings, llama-cpp CUDA & Kokoro-82M..."
pip install --quiet --no-binary pylibsrtp --no-cache-dir pylibsrtp
pip install --quiet llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122 || pip install --quiet llama-cpp-python
pip install --quiet kokoro soundfile misaki
pip install --quiet -r requirements.txt

# Verify and pre-download Kokoro models for zero-latency warm start
echo "⚡ Pre-downloading and warming Kokoro-82M neural pipelines (Hindi 'hf_alpha' & English 'af_heart')..."
python -c "
import pylibsrtp, aiortc, llama_cpp
from kokoro import KPipeline
print('📥 Downloading Kokoro Hindi pipeline...')
KPipeline(lang_code='h')
print('📥 Downloading Kokoro English pipeline...')
KPipeline(lang_code='a')
print('✅ Kokoro-82M models downloaded and verified on GPU/CPU!')
"

# 3. Download Cloudflare Tunnel if not present
if [ ! -f "cloudflared" ]; then
    echo "🌐 Downloading Cloudflare Tunnel (cloudflared)..."
    wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
    chmod +x cloudflared
fi

# 4. Start Cloudflare Tunnel in background
echo "🚇 Starting Cloudflare Quick Tunnel..."
rm -f /tmp/tunnel.log
./cloudflared tunnel --url http://localhost:7860 > /tmp/tunnel.log 2>&1 &

# Wait for tunnel URL
TUNNEL_URL=""
for i in {1..25}; do
    sleep 1
    if [ -f "/tmp/tunnel.log" ]; then
        TUNNEL_URL=$(grep -o 'https://[-0-9a-z]*\.trycloudflare\.com' /tmp/tunnel.log | tail -n 1 || true)
        if [ -n "$TUNNEL_URL" ]; then
            break
        fi
    fi
done

echo "=================================================================="
if [ -n "$TUNNEL_URL" ]; then
    echo "🚀 PUBLIC WEBRTC AGENT URL:"
    echo "👉 $TUNNEL_URL"
    echo "👉 $TUNNEL_URL/offer  (REST Signaling endpoint for CLI/Web clients)"
else
    echo "⚠️ Cloudflare tunnel URL is still initializing. Check /tmp/tunnel.log."
fi
echo "=================================================================="

# 5. Start WebRTC Server with CUDA acceleration, Local llama.cpp Gemma 4 E2B INT8 & Kokoro-82M GPU TTS
echo "🔥 Launching WebRTC Server on port 7860 with Kokoro-82M & Gemma 4 E2B INT8..."
export HOST="0.0.0.0"
export PORT="7860"
export DEVICE="cuda"
export TORCH_DTYPE="float16"
export LLM_ENGINE_TYPE="llama_cpp"
export LLM_N_GPU_LAYERS="-1"
export LLM_REPO_ID="unsloth/gemma-4-E2B-it-GGUF"
export LLM_GGUF_FILENAME="gemma-4-E2B-it-Q8_0.gguf"
export LLM_MODEL="gemma-4-e2b-it-int8"
export TTS_ENGINE="kokoro"
export KOKORO_VOICE_HI="hf_alpha"
export KOKORO_VOICE_EN="af_heart"
export KOKORO_SPEED="1.05"
export PYTHONFAULTHANDLER="1"

python -m server.webrtc_server

#!/usr/bin/env bash
# ==============================================================================
# Kaggle GPU Launcher for Realtime WebRTC Voice Agent
# (Qwen3-ASR FP16 + Silero VAD + Ollama Gemma 4 + VITS/Edge-TTS + Cloudflare Tunnel)
# ==============================================================================

set -e

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

# 1. Install system libsrtp2-dev for OpenSSL 3.0 compatibility (prevents pylibsrtp wheel segfault)
echo "📦 Installing system libsrtp2 development headers..."
apt-get update -qq && apt-get install -y -qq libsrtp2-dev pkg-config wget curl > /dev/null 2>&1 || true

# 2. Python dependencies (force compile pylibsrtp from source against system libsrtp2)
echo "🐍 Installing Python dependencies with native OpenSSL 3 bindings..."
pip install --quiet --no-binary pylibsrtp --no-cache-dir pylibsrtp
pip install --quiet -r requirements.txt

# Verify aiortc / pylibsrtp import
python -c "import pylibsrtp, aiortc; print('✅ aiortc & pylibsrtp verified successfully on Python runtime!')"

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

# 5. Start WebRTC Server with CUDA acceleration
echo "🔥 Launching WebRTC Server on port 7860..."
export HOST="0.0.0.0"
export PORT="7860"
export DEVICE="cuda"
export TORCH_DTYPE="float16"
export PYTHONFAULTHANDLER="1"

python -m server.webrtc_server

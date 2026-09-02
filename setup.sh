#!/bin/bash
# Jarvis Pi5 — one-shot setup script
# Run once after cloning the repo on your Raspberry Pi 5
set -e

echo ""
echo "╔══════════════════════════════════╗"
echo "║   J.A.R.V.I.S — Pi5 Setup       ║"
echo "╚══════════════════════════════════╝"
echo ""

# ── SYSTEM PACKAGES ───────────────────────────────────────────────────────────
echo "[1/7] Installing system packages..."
sudo apt update -qq
sudo apt install -y \
    python3-pip python3-venv \
    portaudio19-dev \
    alsa-utils espeak-ng \
    git cmake build-essential \
    libsdl2-dev ffmpeg

# ── DISABLE SWAP (mandatory for SD card life + LLM performance) ───────────────
echo "[2/7] Disabling swap..."
sudo swapoff -a
sudo systemctl disable dphys-swapfile 2>/dev/null || true
echo "Swap disabled."

# ── ENABLE PCIe GEN3 (if you have the M.2 HAT — doubles SSD read speed) ──────
echo "[3/7] Checking PCIe Gen3..."
if ! grep -q "pciex1_gen=3" /boot/firmware/config.txt 2>/dev/null; then
    echo "# Added by Jarvis setup — enables PCIe Gen3 for M.2 HAT" | sudo tee -a /boot/firmware/config.txt
    echo "dtparam=pciex1_gen=3" | sudo tee -a /boot/firmware/config.txt
    echo "PCIe Gen3 enabled (reboot required to take effect)."
else
    echo "PCIe Gen3 already enabled."
fi

# ── PYTHON VIRTUAL ENVIRONMENT ────────────────────────────────────────────────
echo "[4/7] Setting up Python environment..."
python3 -m venv ai_env
source ai_env/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "Python packages installed."

# ── OLLAMA ────────────────────────────────────────────────────────────────────
echo "[5/7] Installing Ollama..."
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.ai/install.sh | sh
else
    echo "Ollama already installed."
fi

# Configure Ollama for Pi 5
sudo mkdir -p /etc/systemd/system/ollama.service.d
cat <<EOF | sudo tee /etc/systemd/system/ollama.service.d/pi5.conf
[Service]
Environment="OLLAMA_FLASH_ATTN=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_NUM_PARALLEL=1"
EOF
sudo systemctl daemon-reload
sudo systemctl restart ollama
sleep 3

echo "[6/7] Pulling LLM models (this may take a while on first run)..."
ollama pull qwen3.5:4b
ollama pull nomic-embed-text

# ── PIPER TTS (British voice) ─────────────────────────────────────────────────
echo "[7/7] Installing Piper TTS..."
mkdir -p ~/piper
cd ~/piper

if [ ! -f "piper" ]; then
    echo "Downloading Piper binary for ARM64..."
    wget -q "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz"
    tar -xzf piper_linux_aarch64.tar.gz --strip-components=1
    rm piper_linux_aarch64.tar.gz
fi

if [ ! -f "en_GB-alan-medium.onnx" ]; then
    echo "Downloading British voice model (Alan, medium quality)..."
    wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx"
    wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json"
fi

cd -

# ── WHISPER.CPP ───────────────────────────────────────────────────────────────
if [ ! -f ~/whisper.cpp/main ]; then
    echo "Building Whisper.cpp for wake word detection..."
    cd ~
    git clone --depth=1 https://github.com/ggerganov/whisper.cpp
    cd whisper.cpp
    cmake -B build -DGGML_NATIVE=ON
    cmake --build build -j4
    cp build/bin/main .
    mkdir -p models
    bash ./models/download-ggml-model.sh tiny.en
    cd -
    echo "Whisper.cpp built."
else
    echo "Whisper.cpp already built."
fi

# ── DONE ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   Setup complete!                                ║"
echo "║                                                  ║"
echo "║   To run Jarvis:                                 ║"
echo "║     source ai_env/bin/activate                   ║"
echo "║     python assistant.py                          ║"
echo "║                                                  ║"
echo "║   Optional: set your NewsAPI key for headlines:  ║"
echo "║     export NEWS_API_KEY=your_key_here            ║"
echo "║                                                  ║"
echo "║   Reboot recommended (PCIe Gen3 change)          ║"
echo "╚══════════════════════════════════════════════════╝"

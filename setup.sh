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
echo "[1/8] Installing system packages..."
sudo apt update -qq
sudo apt install -y \
    python3-pip python3-venv python3-tk python3-dev \
    portaudio19-dev \
    alsa-utils espeak-ng \
    git cmake build-essential \
    libsdl2-dev ffmpeg

# ── DISABLE SWAP (mandatory for SD card life + LLM performance) ───────────────
echo "[2/8] Disabling swap..."
sudo swapoff -a
sudo systemctl disable dphys-swapfile 2>/dev/null || true
echo "Swap disabled."

# ── ENABLE PCIe GEN3 (if you have the M.2 HAT — doubles SSD read speed) ──────
echo "[3/8] Checking PCIe Gen3..."
if ! grep -q "pciex1_gen=3" /boot/firmware/config.txt 2>/dev/null; then
    echo "# Added by Jarvis setup — enables PCIe Gen3 for M.2 HAT" | sudo tee -a /boot/firmware/config.txt
    echo "dtparam=pciex1_gen=3" | sudo tee -a /boot/firmware/config.txt
    echo "PCIe Gen3 enabled (reboot required to take effect)."
else
    echo "PCIe Gen3 already enabled."
fi

# ── PYTHON VIRTUAL ENVIRONMENT ────────────────────────────────────────────────
echo "[4/8] Setting up Python environment..."
python3 -m venv ai_env
source ai_env/bin/activate
pip install --upgrade pip -q
if ! pip install -r requirements.txt -q; then
    # Install one by one so a single failure (e.g. no wheel for this OS) doesn't block the rest;
    # Jarvis falls back to the previous voice setup for anything missing.
    echo "Some packages failed — installing individually..."
    grep -v '^\s*#' requirements.txt | grep . | while read -r pkg; do
        pip install -q "$pkg" || echo "WARNING: could not install $pkg"
    done
fi
echo "Python packages installed."

# ── OLLAMA ────────────────────────────────────────────────────────────────────
echo "[5/8] Installing Ollama..."
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.ai/install.sh | sh
else
    echo "Ollama already installed."
fi

# Configure Ollama for Pi 5
sudo mkdir -p /etc/systemd/system/ollama.service.d
cat <<EOF | sudo tee /etc/systemd/system/ollama.service.d/pi5.conf
[Service]
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_NUM_PARALLEL=1"
EOF
sudo systemctl daemon-reload
sudo systemctl restart ollama
sleep 3

echo "[6/8] Pulling LLM models (this may take a while on first run)..."
ollama pull qwen3.5:4b

# ── VOICES ────────────────────────────────────────────────────────────────────
echo "[7/8] Downloading voices and the speech recogniser..."

# Kokoro — the default, most natural voice (int8 model: the fastest on a CPU)
mkdir -p ~/kokoro
KOKORO_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
for f in kokoro-v1.0.int8.onnx voices-v1.0.bin; do
    if [ ! -s ~/kokoro/$f ]; then
        echo "Downloading Kokoro $f..."
        curl -fL --retry 3 -o ~/kokoro/$f.part "$KOKORO_URL/$f" && mv ~/kokoro/$f.part ~/kokoro/$f \
            || echo "WARNING: Kokoro $f download failed — Jarvis will use the Piper voice."
    fi
done

# Piper voices (faster fallback; switch with JARVIS_TTS=piper JARVIS_VOICE=<name>). Existing files are skipped.
mkdir -p ~/piper
python -m piper.download_voices en_GB-alan-medium en_GB-cori-high en_US-ryan-high --data-dir ~/piper \
    || echo "WARNING: Piper voice download failed."

# Moonshine speech recogniser model (JARVIS_STT_MODEL, default medium) — cached in ~/.cache/moonshine_voice
python -c "import stt; stt.prefetch()" || echo "WARNING: Moonshine model download failed — it will retry when Jarvis starts."

# ── WHISPER.CPP (fallback speech recogniser) ──────────────────────────────────
echo "[8/8] Checking Whisper.cpp fallback..."
# Note: upstream renamed 'main' to 'whisper-cli'; build/bin/main is a stub that only prints a warning
[ -d ~/whisper.cpp/.git ] || git clone --depth=1 https://github.com/ggml-org/whisper.cpp ~/whisper.cpp
if [ ! -x ~/whisper.cpp/build/bin/whisper-cli ]; then
    echo "Building Whisper.cpp..."
    cd ~/whisper.cpp
    cmake -B build -DGGML_NATIVE=ON
    cmake --build build -j4 --config Release
    cd -
    echo "Whisper.cpp built."
else
    echo "Whisper.cpp already built."
fi
# Only used if Moonshine or the wake word model can't load
[ -f ~/whisper.cpp/models/ggml-tiny.en.bin ] || bash ~/whisper.cpp/models/download-ggml-model.sh tiny.en
[ -f ~/whisper.cpp/models/ggml-base.en.bin ] || bash ~/whisper.cpp/models/download-ggml-model.sh base.en

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

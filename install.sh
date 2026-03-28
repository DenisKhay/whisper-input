#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
CONFIG_DIR="$HOME/.config/whisper-input"
SERVICE_DIR="$HOME/.config/systemd/user"

echo "=== whisper-input installer ==="
echo ""

# 1. Check system dependencies
echo "Checking system dependencies..."
MISSING=()
for cmd in xdotool xclip; do
    if ! command -v "$cmd" &>/dev/null; then
        MISSING+=("$cmd")
    fi
done

# Check portaudio
if ! ldconfig -p 2>/dev/null | grep -q libportaudio; then
    MISSING+=("portaudio19-dev" "libportaudio2")
fi

# Check cuBLAS
CUBLAS_PATH=""
if [ -f /usr/local/lib/ollama/cuda_v12/libcublas.so.12 ]; then
    CUBLAS_PATH="/usr/local/lib/ollama/cuda_v12"
elif ldconfig -p 2>/dev/null | grep -q libcublas.so.12; then
    CUBLAS_PATH=""  # system-wide, no override needed
else
    echo "WARNING: libcublas.so.12 not found. GPU transcription may not work."
    echo "         Install CUDA toolkit or Ollama to get it."
fi

# Check pipewire-alsa
if ! dpkg -l pipewire-alsa &>/dev/null; then
    MISSING+=("pipewire-alsa")
fi

# Check appindicator for tray
if ! dpkg -l gir1.2-appindicator3-0.1 &>/dev/null; then
    MISSING+=("gir1.2-appindicator3-0.1")
fi

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "Missing packages: ${MISSING[*]}"
    echo "Install with:"
    echo "  sudo apt install -y ${MISSING[*]}"
    echo ""
    read -p "Install now? [Y/n] " -r
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        sudo apt install -y "${MISSING[@]}"
    else
        echo "Please install them manually and re-run this script."
        exit 1
    fi
fi
echo "System dependencies OK"
echo ""

# 2. Create/update venv
echo "Setting up Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install -q -r "$SCRIPT_DIR/requirements.txt"
pip install -q scipy
pip install -q -e "$SCRIPT_DIR"

# Link gi for tray icon support
GI_SYS="/usr/lib/python3/dist-packages/gi"
GI_VENV="$VENV_DIR/lib/python3.*/site-packages/gi"
if [ -d "$GI_SYS" ] && ! ls $GI_VENV &>/dev/null; then
    GI_TARGET=$(echo "$VENV_DIR"/lib/python3.*/site-packages/)
    ln -sf "$GI_SYS" "$GI_TARGET/gi"
    echo "Linked gi for tray icon support"
fi

echo "Python dependencies OK"
echo ""

# 3. Audio device selection
if [ ! -f "$CONFIG_DIR/config.yaml" ]; then
    echo "=== Audio Device Setup ==="
    echo "Let's find your microphone."
    echo ""
    python3 "$SCRIPT_DIR/test_mic.py"
    echo ""
    read -p "Which device number worked best? " AUDIO_DEV

    mkdir -p "$CONFIG_DIR"
    cat > "$CONFIG_DIR/config.yaml" <<EOF
audio_device: $AUDIO_DEV
EOF
    echo "Config saved to $CONFIG_DIR/config.yaml"
else
    echo "Config already exists at $CONFIG_DIR/config.yaml"
fi
echo ""

# 4. Create systemd user service
echo "Setting up systemd user service..."
mkdir -p "$SERVICE_DIR"

LD_LINE=""
if [ -n "$CUBLAS_PATH" ]; then
    LD_LINE="Environment=LD_LIBRARY_PATH=$CUBLAS_PATH"
fi

cat > "$SERVICE_DIR/whisper-input.service" <<EOF
[Unit]
Description=whisper-input voice-to-text daemon
After=graphical-session.target

[Service]
Type=simple
ExecStart=$VENV_DIR/bin/python3 -m whisper_input.main
WorkingDirectory=$SCRIPT_DIR
$LD_LINE
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable whisper-input.service
echo "Service installed and enabled"
echo ""

# 5. Start service
read -p "Start whisper-input now? [Y/n] " -r
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    systemctl --user start whisper-input.service
    sleep 2
    if systemctl --user is-active whisper-input.service &>/dev/null; then
        echo ""
        echo "=== whisper-input is running! ==="
        echo ""
        echo "  Hotkey:  Ctrl+\`  (hold to record, release to transcribe)"
        echo "  Tray:    Look for colored circle in system tray"
        echo "  Config:  $CONFIG_DIR/config.yaml"
        echo "  Logs:    journalctl --user -u whisper-input -f"
        echo ""
        echo "  To switch to toggle mode: right-click tray icon"
        echo "  To stop:   systemctl --user stop whisper-input"
        echo "  To start:  systemctl --user start whisper-input"
    else
        echo "Service failed to start. Check logs:"
        echo "  journalctl --user -u whisper-input -e"
    fi
else
    echo ""
    echo "=== Installation complete ==="
    echo ""
    echo "  Start with:  systemctl --user start whisper-input"
    echo "  Or manually:  $SCRIPT_DIR/run.sh -v"
fi

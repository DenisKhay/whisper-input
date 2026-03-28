#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$HOME/.local/share/whisper-input"
BIN_DIR="$HOME/.local/bin"
APP_DIR="$HOME/.local/share/applications"
CONFIG_DIR="$HOME/.config/whisper-input"
SERVICE_DIR="$HOME/.config/systemd/user"
AUTOSTART_DIR="$HOME/.config/autostart"
ICON_DIR="$HOME/.local/share/icons/hicolor/128x128/apps"

echo "╔══════════════════════════════════╗"
echo "║   whisper-input installer        ║"
echo "╚══════════════════════════════════╝"
echo ""

# ── 1. System dependencies ──────────────────────────────────────────
echo "[1/6] Checking system dependencies..."
MISSING=()
for cmd in xdotool xclip; do
    command -v "$cmd" &>/dev/null || MISSING+=("$cmd")
done
ldconfig -p 2>/dev/null | grep -q libportaudio || MISSING+=("portaudio19-dev" "libportaudio2")
dpkg -l pipewire-alsa &>/dev/null 2>&1 || MISSING+=("pipewire-alsa")
dpkg -l gir1.2-appindicator3-0.1 &>/dev/null 2>&1 || MISSING+=("gir1.2-appindicator3-0.1")

# Detect cuBLAS
CUBLAS_PATH=""
if [ -f /usr/local/lib/ollama/cuda_v12/libcublas.so.12 ]; then
    CUBLAS_PATH="/usr/local/lib/ollama/cuda_v12"
elif ! ldconfig -p 2>/dev/null | grep -q libcublas.so.12; then
    echo "  WARNING: libcublas.so.12 not found. GPU may not work."
    echo "           Install CUDA toolkit or Ollama."
fi

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "  Missing: ${MISSING[*]}"
    read -p "  Install now? [Y/n] " -r
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        sudo apt install -y "${MISSING[@]}"
    else
        echo "  Please install manually and re-run."; exit 1
    fi
fi
echo "  OK"
echo ""

# ── 2. Copy source to install dir ───────────────────────────────────
echo "[2/6] Installing to $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR"
# Copy source files (not .git, not venv)
rsync -a --delete \
    --exclude='.git' \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='*.egg-info' \
    "$REPO_DIR/" "$INSTALL_DIR/"

# Create/update venv
if [ ! -d "$INSTALL_DIR/venv" ]; then
    python3 -m venv "$INSTALL_DIR/venv"
fi
source "$INSTALL_DIR/venv/bin/activate"
pip install -q -r "$INSTALL_DIR/requirements.txt"
pip install -q scipy
pip install -q -e "$INSTALL_DIR"

# Link gi for tray icon
GI_SYS="/usr/lib/python3/dist-packages/gi"
GI_TARGET=$(echo "$INSTALL_DIR"/venv/lib/python3.*/site-packages/)
if [ -d "$GI_SYS" ] && [ ! -e "$GI_TARGET/gi" ]; then
    ln -sf "$GI_SYS" "$GI_TARGET/gi"
fi
echo "  OK"
echo ""

# ── 3. Generate app icon ────────────────────────────────────────────
echo "[3/6] Creating application icon..."
mkdir -p "$ICON_DIR"
python3 -c "
from PIL import Image, ImageDraw
size = 128
img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)
# Outer circle (dark)
draw.ellipse([4, 4, size-4, size-4], fill='#2D3748')
# Inner mic shape (simplified)
cx, cy = size//2, size//2
# Mic body
draw.rounded_rectangle([cx-14, cy-30, cx+14, cy+10], radius=14, fill='#E2E8F0')
# Mic stand
draw.rectangle([cx-2, cy+10, cx+2, cy+25], fill='#E2E8F0')
draw.rectangle([cx-16, cy+25, cx+16, cy+28], fill='#E2E8F0')
# Arc around mic
draw.arc([cx-22, cy-20, cx+22, cy+18], start=0, end=180, fill='#E2E8F0', width=3)
img.save('$ICON_DIR/whisper-input.png')
"
echo "  OK"
echo ""

# ── 4. Create CLI launcher + .desktop file ──────────────────────────
echo "[4/6] Creating launcher and desktop entry..."
mkdir -p "$BIN_DIR" "$APP_DIR"

# CLI launcher
cat > "$BIN_DIR/whisper-input" <<'LAUNCHER'
#!/bin/bash
INSTALL_DIR="$HOME/.local/share/whisper-input"
source "$INSTALL_DIR/venv/bin/activate"
LAUNCHER

# Append LD_LIBRARY_PATH if needed
if [ -n "$CUBLAS_PATH" ]; then
    echo "export LD_LIBRARY_PATH=$CUBLAS_PATH:\$LD_LIBRARY_PATH" >> "$BIN_DIR/whisper-input"
fi

cat >> "$BIN_DIR/whisper-input" <<'LAUNCHER'

case "${1:-start}" in
    start)
        exec python3 -m whisper_input.main "${@:2}"
        ;;
    stop)
        systemctl --user stop whisper-input.service 2>/dev/null
        pkill -f "whisper_input.main" 2>/dev/null
        echo "whisper-input stopped"
        ;;
    restart)
        systemctl --user restart whisper-input.service
        echo "whisper-input restarted"
        ;;
    status)
        systemctl --user status whisper-input.service
        ;;
    log|logs)
        journalctl --user -u whisper-input -f
        ;;
    update)
        echo "Updating whisper-input..."
        REPO="$HOME/Projects/whisper-input"
        if [ -d "$REPO/.git" ]; then
            cd "$REPO" && git pull
            exec "$REPO/install.sh"
        else
            echo "Source repo not found at $REPO"
            exit 1
        fi
        ;;
    mic)
        python3 "$INSTALL_DIR/test_mic.py"
        ;;
    *)
        echo "Usage: whisper-input {start|stop|restart|status|logs|update|mic}"
        echo ""
        echo "  start [-v]  Start the daemon (foreground)"
        echo "  stop        Stop the daemon"
        echo "  restart     Restart the systemd service"
        echo "  status      Show service status"
        echo "  logs        Follow service logs"
        echo "  update      Pull latest and reinstall"
        echo "  mic         Test microphone"
        ;;
esac
LAUNCHER
chmod +x "$BIN_DIR/whisper-input"

# Desktop entry
cat > "$APP_DIR/whisper-input.desktop" <<EOF
[Desktop Entry]
Name=Whisper Input
Comment=Voice-to-text input daemon (Ctrl+\` to record)
Exec=$BIN_DIR/whisper-input start
Icon=whisper-input
Terminal=false
Type=Application
Categories=Utility;Accessibility;
Keywords=voice;speech;whisper;dictation;
StartupNotify=false
EOF

echo "  OK"
echo ""

# ── 5. Audio device setup ───────────────────────────────────────────
echo "[5/6] Audio configuration..."
if [ ! -f "$CONFIG_DIR/config.yaml" ]; then
    echo "  Let's find your microphone."
    echo ""
    python3 "$INSTALL_DIR/test_mic.py"
    echo ""
    read -p "  Which device number worked best? " AUDIO_DEV
    mkdir -p "$CONFIG_DIR"
    cat > "$CONFIG_DIR/config.yaml" <<EOF
audio_device: $AUDIO_DEV
EOF
    echo "  Config saved"
else
    echo "  Config already exists at $CONFIG_DIR/config.yaml"
fi
echo ""

# ── 6. Systemd service ─────────────────────────────────────────────
echo "[6/6] Setting up auto-start service..."
mkdir -p "$SERVICE_DIR"

cat > "$SERVICE_DIR/whisper-input.service" <<EOF
[Unit]
Description=Whisper Input — voice-to-text daemon
After=graphical-session.target

[Service]
Type=simple
ExecStart=$BIN_DIR/whisper-input start
WorkingDirectory=$INSTALL_DIR
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable whisper-input.service
echo "  OK"
echo ""

# ── Done ────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════╗"
echo "║   Installation complete!         ║"
echo "╚══════════════════════════════════╝"
echo ""
echo "  Launch:     whisper-input start"
echo "  Stop:       whisper-input stop"
echo "  Status:     whisper-input status"
echo "  Logs:       whisper-input logs"
echo "  Update:     whisper-input update"
echo "  Test mic:   whisper-input mic"
echo "  Config:     $CONFIG_DIR/config.yaml"
echo ""
echo "  Auto-starts on login via systemd."
echo "  Also available in your app menu as 'Whisper Input'."
echo ""

read -p "Start whisper-input now? [Y/n] " -r
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    systemctl --user start whisper-input.service
    sleep 2
    if systemctl --user is-active whisper-input.service &>/dev/null; then
        echo ""
        echo "  Running! Hold Ctrl+\` to record, release to transcribe."
        echo "  Right-click tray icon to switch to toggle mode."
    else
        echo "  Failed to start. Run: whisper-input logs"
    fi
fi

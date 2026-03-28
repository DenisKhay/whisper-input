# whisper-input

Voice-to-text input daemon for Linux. Press a hotkey, speak, get text typed into any focused window.

Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with GPU acceleration for near-instant transcription. Supports English and Russian with automatic language detection.

## How it works

1. Press **Ctrl+`** — recording starts (tray icon turns red)
2. Speak naturally
3. Press **Ctrl+`** again — recording stops, text is transcribed and pasted (tray icon turns yellow, then grey)

Two modes available:
- **Toggle** (default): tap to start, tap to stop
- **Hold**: hold the hotkey while speaking, release to transcribe

## Requirements

- Linux with X11 (KDE Plasma, GNOME, etc.)
- Python 3.10+
- NVIDIA GPU with CUDA support
- Working microphone

## Installation

```bash
git clone https://github.com/denisk/whisper-input.git
cd whisper-input
./install.sh
```

The installer will:
- Install system dependencies (xdotool, xclip, portaudio, etc.)
- Set up a Python virtual environment with all packages
- Run an interactive microphone test to find the right audio device
- Create a CLI tool at `~/.local/bin/whisper-input`
- Add a desktop entry (shows up in your app menu)
- Set up a systemd user service that auto-starts on login

The first launch downloads the Whisper `medium` model (~1.5 GB).

## Usage

```bash
whisper-input start      # run in foreground
whisper-input stop       # stop the daemon
whisper-input restart    # restart the systemd service
whisper-input status     # check if running
whisper-input logs       # follow service logs
whisper-input update     # pull latest changes and reinstall
whisper-input mic        # test microphone devices
```

Or just launch "Whisper Input" from your app menu.

## Configuration

Config file: `~/.config/whisper-input/config.yaml`

```yaml
# Audio device index (run `whisper-input mic` to find yours)
audio_device: 5

# Recording mode: "toggle" (tap to start/stop) or "hold" (hold to record)
mode: toggle

# Hotkey combination (pynput format)
hotkey: "<ctrl>+`"

# Whisper model: tiny, base, small, medium, large-v3
model: medium

# GPU settings
device: cuda
compute_type: float16

# Language: "auto" (detects English/Russian), "en", or "ru"
language: auto

# Silence timeout in seconds (toggle mode safety net)
silence_timeout: 3
```

## System tray

The tray icon shows the current state:
- **Grey** — idle, ready to record
- **Red** — recording
- **Yellow** — transcribing

Right-click the tray icon to switch between hold/toggle mode or quit.

## Troubleshooting

**No audio captured / very quiet:**
Run `whisper-input mic` to test different audio devices. Pick the one with the highest amplitude when you speak. Update `audio_device` in the config.

**Wrong language detected:**
Language detection is restricted to English and Russian. If you primarily use one language, set `language: en` or `language: ru` in the config to skip auto-detection.

**cuBLAS not found:**
The CUDA cuBLAS library is needed for GPU inference. If you have [Ollama](https://ollama.com/) installed, it's already available. Otherwise install the CUDA toolkit.

**Tray icon not showing:**
Install AppIndicator support: `sudo apt install gir1.2-appindicator3-0.1`

**Text not pasting:**
The daemon uses `Shift+Insert` to paste. If your target application doesn't support this, the text is also copied to your clipboard — use `Ctrl+V` manually.

## Tech stack

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-based Whisper inference
- [sounddevice](https://python-sounddevice.readthedocs.io/) — audio capture via PortAudio
- [pynput](https://pynput.readthedocs.io/) — global hotkey detection
- [pystray](https://pystray.readthedocs.io/) — system tray icon
- [scipy](https://scipy.org/) — audio resampling

## License

MIT

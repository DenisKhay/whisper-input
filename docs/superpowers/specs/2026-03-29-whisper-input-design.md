# whisper-input — Voice-to-Text Input Daemon

## Overview

A Python background daemon that listens for a configurable hotkey, records voice via microphone, transcribes using faster-whisper (medium model, GPU), and outputs the text by simulating keyboard input into the focused window + copying to clipboard.

## Requirements

- Press-and-hold or toggle hotkey to record voice
- Near-instant transcription using GPU-accelerated faster-whisper
- Text typed into focused window via xdotool + copied to clipboard
- System tray icon showing state (idle/recording/transcribing)
- Configurable via YAML config file
- English primary, Russian secondary (auto-detect)

## Environment

- Linux (Ubuntu), KDE Plasma
- Python 3.12
- NVIDIA RTX 4060 Laptop GPU (8GB VRAM)
- CUDA 13.0
- Audio: sof-hda-dsp (built-in mic)
- ffmpeg available

## Components

### 1. Audio Recorder (`recorder.py`)

- `sounddevice` library for audio capture
- Records 16kHz mono float32 (Whisper's expected format)
- Accumulates audio chunks into a numpy buffer while recording is active
- Exposes `start()`, `stop() -> np.ndarray` interface
- In toggle mode: silence detection auto-stops after configurable timeout (default 3s)

### 2. Transcription Engine (`transcriber.py`)

- `faster-whisper` with `medium` model
- CTranslate2 backend, `device=cuda`, `compute_type=float16`
- Model loaded once at daemon startup, stays resident in VRAM (~3GB)
- Language: auto-detect (no hint) — handles English/Russian mixing
- Returns transcribed text as a string, stripped of leading/trailing whitespace
- Segments joined with spaces

### 3. Hotkey Listener (`hotkey.py`)

- `pynput` for global hotkey detection
- Two modes:
  - `hold` — starts recording on key press, stops on key release, triggers transcription
  - `toggle` — starts recording on first press, stops on second press, triggers transcription
- Default hotkey: `Super+V` (configurable)
- Callback-based: calls provided `on_start` and `on_stop` functions

### 4. Text Output (`output.py`)

- `xdotool type --clearmodifiers` to simulate keyboard input into the focused window
- `xclip -selection clipboard` to copy text to clipboard
- 100ms delay before typing to allow hotkey release to propagate
- Handles multi-line text (xdotool types it as-is)

### 5. System Tray (`tray.py`)

- `pystray` with Pillow-generated icons
- Three states with colored mic icons:
  - Idle: grey
  - Recording: red
  - Transcribing: yellow
- Right-click menu:
  - Toggle mode (hold/toggle) — shows current mode
  - Quit
- Optional — daemon runs fine without a display (headless mode)

### 6. Configuration (`config.py`)

YAML file at `~/.config/whisper-input/config.yaml`:

```yaml
model: medium
device: cuda
compute_type: float16
hotkey: <super>+v
mode: hold
silence_timeout: 3
language: auto
```

- Loaded at startup with defaults for any missing keys
- Config file created from example on first run if missing

### 7. Main Entry Point (`main.py`)

- Parses CLI args (optional config path override)
- Loads config
- Initializes transcriber (loads model — may take a few seconds)
- Initializes recorder, hotkey listener, tray
- Wires callbacks: hotkey -> recorder -> transcriber -> output
- Runs event loop
- Clean shutdown on SIGINT/SIGTERM or tray quit

## Project Structure

```
whisper-input/
  whisper_input/
    __init__.py
    main.py
    recorder.py
    transcriber.py
    hotkey.py
    output.py
    tray.py
    config.py
  config.example.yaml
  requirements.txt
  setup.py
```

## Dependencies

### Python packages
- `faster-whisper` — CTranslate2-based Whisper inference
- `sounddevice` — audio capture via PortAudio
- `numpy` — audio buffer handling
- `pynput` — global hotkey detection
- `pystray` — system tray icon
- `Pillow` — icon generation for tray
- `PyYAML` — config parsing

### System packages (apt)
- `xdotool` — simulate keyboard input
- `xclip` — clipboard access
- `portaudio19-dev` — required by sounddevice

## Data Flow

```
Hotkey press
  -> recorder.start()
  -> tray: red (recording)

Hotkey release / toggle stop
  -> audio_buffer = recorder.stop()
  -> tray: yellow (transcribing)
  -> text = transcriber.transcribe(audio_buffer)
  -> output.type_text(text)
  -> output.copy_to_clipboard(text)
  -> tray: grey (idle)
```

## Error Handling

- No audio device: log error, exit with message
- Transcription returns empty: skip output, log warning
- xdotool/xclip missing: check at startup, exit with message listing missing deps
- Model download: faster-whisper auto-downloads on first run, log progress

## Future Considerations (not in scope)

- Systemd user service for auto-start
- Per-application language hints
- Noise gate / VAD before sending to Whisper
- Sound feedback (beep on start/stop recording)

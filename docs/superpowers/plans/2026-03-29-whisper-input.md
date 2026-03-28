# whisper-input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python daemon that records voice via hotkey, transcribes with faster-whisper on GPU, and types the result into the focused window.

**Architecture:** Single-process daemon. Components: config loader, audio recorder (sounddevice), transcriber (faster-whisper), hotkey listener (pynput), text output (xdotool+xclip), system tray (pystray). Main wires them together with callbacks.

**Tech Stack:** Python 3.12, faster-whisper, sounddevice, pynput, pystray, PyYAML, xdotool, xclip

---

### Task 1: Project Scaffolding & Dependencies

**Files:**
- Create: `/home/denisk/Projects/whisper-input/whisper_input/__init__.py`
- Create: `/home/denisk/Projects/whisper-input/requirements.txt`
- Create: `/home/denisk/Projects/whisper-input/setup.py`
- Create: `/home/denisk/Projects/whisper-input/config.example.yaml`

- [ ] **Step 1: Create requirements.txt**

```
faster-whisper
sounddevice
numpy
pynput
pystray
Pillow
PyYAML
```

- [ ] **Step 2: Create setup.py**

```python
from setuptools import setup, find_packages

setup(
    name="whisper-input",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "faster-whisper",
        "sounddevice",
        "numpy",
        "pynput",
        "pystray",
        "Pillow",
        "PyYAML",
    ],
    entry_points={
        "console_scripts": [
            "whisper-input=whisper_input.main:main",
        ],
    },
)
```

- [ ] **Step 3: Create config.example.yaml**

```yaml
model: medium
device: cuda
compute_type: float16
hotkey: "<cmd>+v"
mode: hold
silence_timeout: 3
language: auto
```

- [ ] **Step 4: Create whisper_input/__init__.py**

Empty file.

- [ ] **Step 5: Install system dependencies**

Run: `sudo apt install -y xdotool xclip portaudio19-dev`

- [ ] **Step 6: Create venv and install Python dependencies**

Run:
```bash
cd /home/denisk/Projects/whisper-input
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

- [ ] **Step 7: Verify faster-whisper can load on GPU**

Run:
```bash
cd /home/denisk/Projects/whisper-input
source venv/bin/activate
python3 -c "from faster_whisper import WhisperModel; print('faster-whisper OK')"
```
Expected: `faster-whisper OK`

- [ ] **Step 8: Commit**

```bash
cd /home/denisk/Projects/whisper-input
git add whisper_input/__init__.py requirements.txt setup.py config.example.yaml
git commit -m "Project scaffolding and dependencies"
```

---

### Task 2: Configuration Module

**Files:**
- Create: `/home/denisk/Projects/whisper-input/whisper_input/config.py`
- Create: `/home/denisk/Projects/whisper-input/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
import os
import tempfile
import pytest
import yaml

from whisper_input.config import load_config, DEFAULT_CONFIG


def test_load_defaults_when_no_file():
    config = load_config("/nonexistent/path/config.yaml")
    assert config["model"] == "medium"
    assert config["device"] == "cuda"
    assert config["compute_type"] == "float16"
    assert config["hotkey"] == "<cmd>+v"
    assert config["mode"] == "hold"
    assert config["silence_timeout"] == 3
    assert config["language"] == "auto"


def test_load_partial_override():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"model": "large-v3", "mode": "toggle"}, f)
        path = f.name
    try:
        config = load_config(path)
        assert config["model"] == "large-v3"
        assert config["mode"] == "toggle"
        assert config["device"] == "cuda"  # default preserved
    finally:
        os.unlink(path)


def test_load_full_override():
    custom = {
        "model": "small",
        "device": "cpu",
        "compute_type": "int8",
        "hotkey": "<ctrl>+<shift>+r",
        "mode": "toggle",
        "silence_timeout": 5,
        "language": "en",
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(custom, f)
        path = f.name
    try:
        config = load_config(path)
        for key, value in custom.items():
            assert config[key] == value
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /home/denisk/Projects/whisper-input
source venv/bin/activate
python3 -m pytest tests/test_config.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'whisper_input.config'`

- [ ] **Step 3: Write implementation**

```python
"""Configuration loading with defaults."""

import os
from pathlib import Path

import yaml

DEFAULT_CONFIG = {
    "model": "medium",
    "device": "cuda",
    "compute_type": "float16",
    "hotkey": "<cmd>+v",
    "mode": "hold",
    "silence_timeout": 3,
    "language": "auto",
}

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "whisper-input" / "config.yaml"


def load_config(path: str | None = None) -> dict:
    """Load config from YAML file, falling back to defaults for missing keys."""
    config = dict(DEFAULT_CONFIG)
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH

    if config_path.exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
        if isinstance(user_config, dict):
            config.update(user_config)

    return config
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /home/denisk/Projects/whisper-input
source venv/bin/activate
python3 -m pytest tests/test_config.py -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
cd /home/denisk/Projects/whisper-input
git add whisper_input/config.py tests/test_config.py
git commit -m "Add configuration module with YAML loading and defaults"
```

---

### Task 3: Audio Recorder

**Files:**
- Create: `/home/denisk/Projects/whisper-input/whisper_input/recorder.py`
- Create: `/home/denisk/Projects/whisper-input/tests/test_recorder.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from whisper_input.recorder import Recorder


def test_start_stop_returns_audio():
    """Recorder accumulates chunks and returns concatenated float32 array."""
    recorder = Recorder(samplerate=16000, channels=1)

    # Simulate the sounddevice callback with fake audio chunks
    chunk1 = np.random.rand(1600, 1).astype(np.float32)
    chunk2 = np.random.rand(1600, 1).astype(np.float32)

    with patch("whisper_input.recorder.sd") as mock_sd:
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        recorder.start()

        # Simulate callbacks
        recorder._audio_callback(chunk1, 1600, None, None)
        recorder._audio_callback(chunk2, 1600, None, None)

        audio = recorder.stop()

    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert len(audio) == 3200
    np.testing.assert_array_equal(audio[:1600], chunk1.flatten())
    np.testing.assert_array_equal(audio[1600:], chunk2.flatten())


def test_stop_without_start_returns_empty():
    recorder = Recorder(samplerate=16000, channels=1)
    audio = recorder.stop()
    assert audio.dtype == np.float32
    assert len(audio) == 0


def test_start_clears_previous_chunks():
    recorder = Recorder(samplerate=16000, channels=1)
    chunk = np.ones((800, 1), dtype=np.float32)

    with patch("whisper_input.recorder.sd") as mock_sd:
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        recorder.start()
        recorder._audio_callback(chunk, 800, None, None)
        recorder.stop()

        # Start again — previous chunks should be gone
        recorder.start()
        audio = recorder.stop()

    assert len(audio) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /home/denisk/Projects/whisper-input
source venv/bin/activate
python3 -m pytest tests/test_recorder.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
"""Audio recording via sounddevice."""

import logging

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class Recorder:
    def __init__(self, samplerate: int = 16000, channels: int = 1):
        self.samplerate = samplerate
        self.channels = channels
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            logger.warning("sounddevice status: %s", status)
        self._chunks.append(indata.copy())

    def start(self) -> None:
        """Start recording from the default input device."""
        self._chunks.clear()
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("Recording started")

    def stop(self) -> np.ndarray:
        """Stop recording and return the captured audio as a 1D float32 array."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._chunks:
            audio = np.concatenate(self._chunks, axis=0).flatten()
            duration = len(audio) / self.samplerate
            logger.info("Captured %.2fs of audio (%d samples)", duration, len(audio))
            return audio

        return np.array([], dtype=np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /home/denisk/Projects/whisper-input
source venv/bin/activate
python3 -m pytest tests/test_recorder.py -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
cd /home/denisk/Projects/whisper-input
git add whisper_input/recorder.py tests/test_recorder.py
git commit -m "Add audio recorder with start/stop streaming capture"
```

---

### Task 4: Transcription Engine

**Files:**
- Create: `/home/denisk/Projects/whisper-input/whisper_input/transcriber.py`
- Create: `/home/denisk/Projects/whisper-input/tests/test_transcriber.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from whisper_input.transcriber import Transcriber


def test_transcribe_joins_segments():
    """Transcriber joins segment texts with spaces and strips whitespace."""
    mock_model = MagicMock()
    seg1 = MagicMock()
    seg1.text = " Hello world"
    seg2 = MagicMock()
    seg2.text = " how are you"
    mock_info = MagicMock()

    mock_model.transcribe.return_value = (iter([seg1, seg2]), mock_info)

    with patch("whisper_input.transcriber.WhisperModel", return_value=mock_model):
        transcriber = Transcriber(model="tiny", device="cpu", compute_type="int8")
        result = transcriber.transcribe(np.zeros(16000, dtype=np.float32))

    assert result == "Hello world how are you"


def test_transcribe_empty_audio_returns_empty():
    mock_model = MagicMock()
    mock_info = MagicMock()
    mock_model.transcribe.return_value = (iter([]), mock_info)

    with patch("whisper_input.transcriber.WhisperModel", return_value=mock_model):
        transcriber = Transcriber(model="tiny", device="cpu", compute_type="int8")
        result = transcriber.transcribe(np.array([], dtype=np.float32))

    assert result == ""


def test_transcribe_passes_language_none_for_auto():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter([]), MagicMock())

    with patch("whisper_input.transcriber.WhisperModel", return_value=mock_model):
        transcriber = Transcriber(model="tiny", device="cpu", compute_type="int8", language="auto")
        transcriber.transcribe(np.zeros(16000, dtype=np.float32))

    call_kwargs = mock_model.transcribe.call_args[1]
    assert call_kwargs["language"] is None


def test_transcribe_passes_explicit_language():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter([]), MagicMock())

    with patch("whisper_input.transcriber.WhisperModel", return_value=mock_model):
        transcriber = Transcriber(model="tiny", device="cpu", compute_type="int8", language="en")
        transcriber.transcribe(np.zeros(16000, dtype=np.float32))

    call_kwargs = mock_model.transcribe.call_args[1]
    assert call_kwargs["language"] == "en"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /home/denisk/Projects/whisper-input
source venv/bin/activate
python3 -m pytest tests/test_transcriber.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
"""Speech-to-text transcription via faster-whisper."""

import logging

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class Transcriber:
    def __init__(
        self,
        model: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "auto",
    ):
        self.language = None if language == "auto" else language
        logger.info("Loading Whisper model '%s' on %s (%s)...", model, device, compute_type)
        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        logger.info("Model loaded")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a float32 16kHz audio array to text."""
        if len(audio) == 0:
            return ""

        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            vad_filter=True,
        )

        text = " ".join(seg.text for seg in segments).strip()
        if text:
            logger.info("Transcribed (%s, %.0f%%): %s", info.language, info.language_probability * 100, text)
        else:
            logger.warning("Transcription returned empty text")

        return text
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /home/denisk/Projects/whisper-input
source venv/bin/activate
python3 -m pytest tests/test_transcriber.py -v
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
cd /home/denisk/Projects/whisper-input
git add whisper_input/transcriber.py tests/test_transcriber.py
git commit -m "Add transcription engine wrapping faster-whisper"
```

---

### Task 5: Text Output

**Files:**
- Create: `/home/denisk/Projects/whisper-input/whisper_input/output.py`
- Create: `/home/denisk/Projects/whisper-input/tests/test_output.py`

- [ ] **Step 1: Write the failing test**

```python
import subprocess
from unittest.mock import patch, call

from whisper_input.output import type_text, copy_to_clipboard, output_text, check_dependencies


def test_type_text_calls_xdotool():
    with patch("whisper_input.output.subprocess.run") as mock_run:
        type_text("hello world")
    mock_run.assert_called_once_with(
        ["xdotool", "type", "--clearmodifiers", "--delay", "12", "hello world"],
        check=True,
    )


def test_copy_to_clipboard_calls_xclip():
    with patch("whisper_input.output.subprocess.run") as mock_run:
        copy_to_clipboard("hello world")
    mock_run.assert_called_once_with(
        ["xclip", "-selection", "clipboard"],
        input="hello world",
        text=True,
        check=True,
    )


def test_output_text_does_both(monkeypatch):
    calls = []
    monkeypatch.setattr("whisper_input.output.type_text", lambda t: calls.append(("type", t)))
    monkeypatch.setattr("whisper_input.output.copy_to_clipboard", lambda t: calls.append(("clip", t)))
    monkeypatch.setattr("whisper_input.output.time.sleep", lambda _: None)
    output_text("hi")
    assert ("type", "hi") in calls
    assert ("clip", "hi") in calls


def test_output_text_skips_empty():
    with patch("whisper_input.output.type_text") as mock_type:
        output_text("")
    mock_type.assert_not_called()


def test_check_dependencies_ok():
    with patch("whisper_input.output.shutil.which", return_value="/usr/bin/xdotool"):
        missing = check_dependencies()
    assert missing == []


def test_check_dependencies_missing():
    def fake_which(cmd):
        return None if cmd == "xclip" else "/usr/bin/" + cmd

    with patch("whisper_input.output.shutil.which", side_effect=fake_which):
        missing = check_dependencies()
    assert "xclip" in missing
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /home/denisk/Projects/whisper-input
source venv/bin/activate
python3 -m pytest tests/test_output.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
"""Text output via xdotool (keyboard simulation) and xclip (clipboard)."""

import logging
import shutil
import subprocess
import time

logger = logging.getLogger(__name__)

REQUIRED_COMMANDS = ["xdotool", "xclip"]


def check_dependencies() -> list[str]:
    """Return list of missing system commands."""
    return [cmd for cmd in REQUIRED_COMMANDS if shutil.which(cmd) is None]


def type_text(text: str) -> None:
    """Type text into the focused window using xdotool."""
    subprocess.run(
        ["xdotool", "type", "--clearmodifiers", "--delay", "12", text],
        check=True,
    )
    logger.debug("Typed %d characters via xdotool", len(text))


def copy_to_clipboard(text: str) -> None:
    """Copy text to clipboard using xclip."""
    subprocess.run(
        ["xclip", "-selection", "clipboard"],
        input=text,
        text=True,
        check=True,
    )
    logger.debug("Copied to clipboard")


def output_text(text: str) -> None:
    """Type text into focused window and copy to clipboard."""
    if not text:
        return
    time.sleep(0.15)  # let hotkey release propagate
    copy_to_clipboard(text)
    type_text(text)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /home/denisk/Projects/whisper-input
source venv/bin/activate
python3 -m pytest tests/test_output.py -v
```
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
cd /home/denisk/Projects/whisper-input
git add whisper_input/output.py tests/test_output.py
git commit -m "Add text output via xdotool and xclip"
```

---

### Task 6: Hotkey Listener

**Files:**
- Create: `/home/denisk/Projects/whisper-input/whisper_input/hotkey.py`
- Create: `/home/denisk/Projects/whisper-input/tests/test_hotkey.py`

- [ ] **Step 1: Write the failing test**

```python
from unittest.mock import MagicMock, patch
from pynput import keyboard

from whisper_input.hotkey import HotkeyListener


def test_hold_mode_press_triggers_start():
    on_start = MagicMock()
    on_stop = MagicMock()

    with patch("whisper_input.hotkey.keyboard.Listener") as MockListener:
        mock_listener_instance = MagicMock()
        # canonical just returns the key as-is for testing
        mock_listener_instance.canonical = lambda k: k
        MockListener.return_value = mock_listener_instance

        listener = HotkeyListener(
            hotkey_str="<cmd>+v",
            mode="hold",
            on_start=on_start,
            on_stop=on_stop,
        )
        listener.start()

        # Get the on_press/on_release callbacks passed to pynput Listener
        call_kwargs = MockListener.call_args[1]
        on_press = call_kwargs["on_press"]
        on_release = call_kwargs["on_release"]

    # Simulate pressing cmd then v
    on_press(keyboard.Key.cmd)
    on_press(keyboard.KeyCode.from_char("v"))
    on_start.assert_called_once()

    # Simulate releasing v
    on_release(keyboard.KeyCode.from_char("v"))
    on_stop.assert_called_once()


def test_toggle_mode_two_presses():
    on_start = MagicMock()
    on_stop = MagicMock()

    with patch("whisper_input.hotkey.keyboard.Listener") as MockListener:
        mock_listener_instance = MagicMock()
        mock_listener_instance.canonical = lambda k: k
        MockListener.return_value = mock_listener_instance

        listener = HotkeyListener(
            hotkey_str="<cmd>+v",
            mode="toggle",
            on_start=on_start,
            on_stop=on_stop,
        )
        listener.start()

        call_kwargs = MockListener.call_args[1]
        on_press = call_kwargs["on_press"]
        on_release = call_kwargs["on_release"]

    # First combo press — starts recording
    on_press(keyboard.Key.cmd)
    on_press(keyboard.KeyCode.from_char("v"))
    on_start.assert_called_once()

    # Release keys
    on_release(keyboard.KeyCode.from_char("v"))
    on_release(keyboard.Key.cmd)

    # Second combo press — stops recording
    on_press(keyboard.Key.cmd)
    on_press(keyboard.KeyCode.from_char("v"))
    on_stop.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /home/denisk/Projects/whisper-input
source venv/bin/activate
python3 -m pytest tests/test_hotkey.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
"""Global hotkey listener using pynput."""

import logging

from pynput import keyboard

logger = logging.getLogger(__name__)


class HotkeyListener:
    def __init__(
        self,
        hotkey_str: str,
        mode: str,
        on_start: callable,
        on_stop: callable,
    ):
        self.mode = mode
        self._on_start = on_start
        self._on_stop = on_stop
        self._target_keys = set(keyboard.HotKey.parse(hotkey_str))
        self._pressed_keys: set = set()
        self._active = False
        self._listener: keyboard.Listener | None = None

    def _canonical(self, key):
        if self._listener is not None:
            return self._listener.canonical(key)
        return key

    def _on_press(self, key):
        canon = self._canonical(key)
        self._pressed_keys.add(canon)

        if self._target_keys.issubset(self._pressed_keys):
            if self.mode == "hold":
                if not self._active:
                    self._active = True
                    logger.info("Hold hotkey pressed — starting")
                    self._on_start()
            elif self.mode == "toggle":
                if not self._active:
                    self._active = True
                    logger.info("Toggle hotkey pressed — starting")
                    self._on_start()
                else:
                    self._active = False
                    logger.info("Toggle hotkey pressed — stopping")
                    self._on_stop()

    def _on_release(self, key):
        canon = self._canonical(key)

        if self.mode == "hold" and self._active:
            if canon in self._target_keys:
                self._active = False
                logger.info("Hold hotkey released — stopping")
                self._on_stop()

        self._pressed_keys.discard(canon)

    def start(self) -> None:
        """Start listening for the hotkey."""
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        logger.info("Hotkey listener started (mode=%s, keys=%s)", self.mode, self._target_keys)

    def stop(self) -> None:
        """Stop the hotkey listener."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /home/denisk/Projects/whisper-input
source venv/bin/activate
python3 -m pytest tests/test_hotkey.py -v
```
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
cd /home/denisk/Projects/whisper-input
git add whisper_input/hotkey.py tests/test_hotkey.py
git commit -m "Add hotkey listener with hold and toggle modes"
```

---

### Task 7: System Tray Icon

**Files:**
- Create: `/home/denisk/Projects/whisper-input/whisper_input/tray.py`

- [ ] **Step 1: Write implementation**

The tray is UI-only and difficult to unit test meaningfully. We test it manually in Task 8.

```python
"""System tray icon showing daemon state."""

import logging
import threading

from PIL import Image, ImageDraw
import pystray

logger = logging.getLogger(__name__)

# Colors for each state
COLORS = {
    "idle": "#808080",
    "recording": "#FF0000",
    "transcribing": "#FFD700",
}


def _create_icon_image(color: str) -> Image.Image:
    """Create a simple colored circle icon."""
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = 4
    draw.ellipse([margin, margin, size - margin, size - margin], fill=color)
    return img


class TrayIcon:
    def __init__(self, on_quit: callable, on_mode_toggle: callable | None = None):
        self._on_quit = on_quit
        self._on_mode_toggle = on_mode_toggle
        self._current_mode = "hold"
        self._icon: pystray.Icon | None = None
        self._thread: threading.Thread | None = None

    def _build_menu(self):
        return pystray.Menu(
            pystray.MenuItem(
                lambda _: f"Mode: {self._current_mode}",
                self._toggle_mode,
            ),
            pystray.MenuItem("Quit", self._quit),
        )

    def _toggle_mode(self):
        if self._on_mode_toggle:
            self._current_mode = "toggle" if self._current_mode == "hold" else "hold"
            self._on_mode_toggle(self._current_mode)
            self._icon.update_menu()
            logger.info("Mode toggled to %s", self._current_mode)

    def _quit(self):
        logger.info("Quit requested from tray")
        self._on_quit()
        if self._icon:
            self._icon.stop()

    def start(self, mode: str = "hold") -> None:
        """Start the tray icon in a background thread."""
        self._current_mode = mode
        self._icon = pystray.Icon(
            "whisper-input",
            icon=_create_icon_image(COLORS["idle"]),
            title="whisper-input (idle)",
            menu=self._build_menu(),
        )
        self._thread = threading.Thread(target=self._icon.run, daemon=True)
        self._thread.start()
        logger.info("Tray icon started")

    def set_state(self, state: str) -> None:
        """Update the tray icon to reflect current state: idle, recording, transcribing."""
        if self._icon is None:
            return
        self._icon.icon = _create_icon_image(COLORS.get(state, COLORS["idle"]))
        self._icon.title = f"whisper-input ({state})"

    def stop(self) -> None:
        if self._icon:
            self._icon.stop()
            self._icon = None
```

- [ ] **Step 2: Commit**

```bash
cd /home/denisk/Projects/whisper-input
git add whisper_input/tray.py
git commit -m "Add system tray icon with state colors and mode toggle"
```

---

### Task 8: Main Entry Point — Wire Everything Together

**Files:**
- Create: `/home/denisk/Projects/whisper-input/whisper_input/main.py`

- [ ] **Step 1: Write implementation**

```python
"""Main entry point — wires all components together."""

import argparse
import logging
import signal
import sys
import threading

from whisper_input.config import load_config
from whisper_input.output import check_dependencies, output_text
from whisper_input.recorder import Recorder
from whisper_input.transcriber import Transcriber
from whisper_input.hotkey import HotkeyListener
from whisper_input.tray import TrayIcon

logger = logging.getLogger("whisper_input")


class App:
    def __init__(self, config: dict):
        self.config = config
        self._shutdown_event = threading.Event()
        self._recorder = Recorder()
        self._transcriber: Transcriber | None = None
        self._hotkey: HotkeyListener | None = None
        self._tray: TrayIcon | None = None

    def _on_recording_start(self) -> None:
        logger.info("Recording started")
        if self._tray:
            self._tray.set_state("recording")
        self._recorder.start()

    def _on_recording_stop(self) -> None:
        logger.info("Recording stopped, transcribing...")
        audio = self._recorder.stop()
        if len(audio) == 0:
            logger.warning("No audio captured")
            if self._tray:
                self._tray.set_state("idle")
            return

        if self._tray:
            self._tray.set_state("transcribing")

        text = self._transcriber.transcribe(audio)
        output_text(text)

        if self._tray:
            self._tray.set_state("idle")

    def _on_mode_toggle(self, new_mode: str) -> None:
        if self._hotkey:
            self._hotkey.mode = new_mode
            logger.info("Switched to %s mode", new_mode)

    def _shutdown(self) -> None:
        logger.info("Shutting down...")
        self._shutdown_event.set()
        if self._hotkey:
            self._hotkey.stop()
        if self._tray:
            self._tray.stop()

    def run(self) -> None:
        # Check system deps
        missing = check_dependencies()
        if missing:
            print(f"Missing system dependencies: {', '.join(missing)}")
            print("Install with: sudo apt install " + " ".join(missing))
            sys.exit(1)

        # Load model
        self._transcriber = Transcriber(
            model=self.config["model"],
            device=self.config["device"],
            compute_type=self.config["compute_type"],
            language=self.config["language"],
        )

        # Start tray
        self._tray = TrayIcon(
            on_quit=self._shutdown,
            on_mode_toggle=self._on_mode_toggle,
        )
        self._tray.start(mode=self.config["mode"])

        # Start hotkey listener
        self._hotkey = HotkeyListener(
            hotkey_str=self.config["hotkey"],
            mode=self.config["mode"],
            on_start=self._on_recording_start,
            on_stop=self._on_recording_stop,
        )
        self._hotkey.start()

        # Handle signals
        signal.signal(signal.SIGINT, lambda *_: self._shutdown())
        signal.signal(signal.SIGTERM, lambda *_: self._shutdown())

        print(f"whisper-input running (mode={self.config['mode']}, hotkey={self.config['hotkey']})")
        print("Press Ctrl+C or use tray icon to quit.")

        self._shutdown_event.wait()


def main():
    parser = argparse.ArgumentParser(description="Voice-to-text input daemon")
    parser.add_argument("-c", "--config", help="Path to config YAML file")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    app = App(config)
    app.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it starts up**

Run:
```bash
cd /home/denisk/Projects/whisper-input
source venv/bin/activate
timeout 10 python3 -m whisper_input.main -v 2>&1 || true
```
Expected: Should print model loading logs, then "whisper-input running", then timeout kills it. If there are import errors, fix them.

- [ ] **Step 3: Commit**

```bash
cd /home/denisk/Projects/whisper-input
git add whisper_input/main.py
git commit -m "Add main entry point wiring all components together"
```

---

### Task 9: End-to-End Manual Test

- [ ] **Step 1: Start the daemon**

Run in a terminal:
```bash
cd /home/denisk/Projects/whisper-input
source venv/bin/activate
python3 -m whisper_input.main -v
```

Wait for "whisper-input running" message.

- [ ] **Step 2: Test hold mode**

1. Open a text editor or terminal
2. Hold Super+V and say something in English
3. Release Super+V
4. Verify: text appears in the focused window and is in clipboard (Ctrl+V to check)

- [ ] **Step 3: Test toggle mode**

1. Right-click tray icon → click "Mode: hold" to switch to toggle
2. Press Super+V once, speak, press Super+V again
3. Verify text appears

- [ ] **Step 4: Test Russian**

1. Say something in Russian while recording
2. Verify it transcribes correctly

- [ ] **Step 5: Fix any issues found during testing**

If issues are found, fix them and commit each fix separately.

- [ ] **Step 6: Final commit if any fixes were made**

---

### Task 10: Run Full Test Suite

- [ ] **Step 1: Run all tests**

Run:
```bash
cd /home/denisk/Projects/whisper-input
source venv/bin/activate
python3 -m pytest tests/ -v
```
Expected: All tests pass (15 tests across 4 test files).

- [ ] **Step 2: Fix any failures and commit**

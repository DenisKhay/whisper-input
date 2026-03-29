"""Configuration loading with defaults."""

import os
from pathlib import Path

import yaml

DEFAULT_CONFIG = {
    "model": "medium",
    "device": "cuda",
    "compute_type": "float16",
    "hotkey": "<ctrl>+`",
    "mode": "hold",
    "silence_timeout": 3,
    "language": "auto",
    "audio_device": None,
    "wakeword": {
        "enabled": False,
        "word": "claude",
        "stop_word": "over",
        "cancel_phrase": "claude cancel",
        "timeout": 5,
        "beep": True,
    },
}

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "whisper-input" / "config.yaml"


def load_config(path: str | None = None) -> dict:
    """Load config from YAML file, falling back to defaults for missing keys."""
    config = dict(DEFAULT_CONFIG)
    config["wakeword"] = dict(DEFAULT_CONFIG["wakeword"])
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH

    if config_path.exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
        if isinstance(user_config, dict):
            wakeword_override = user_config.pop("wakeword", None)
            config.update(user_config)
            if isinstance(wakeword_override, dict):
                config["wakeword"].update(wakeword_override)

    return config

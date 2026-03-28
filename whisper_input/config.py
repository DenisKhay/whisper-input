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

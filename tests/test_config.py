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

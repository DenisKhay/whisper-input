import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

from whisper_input.wakeword import WakeWordListener, match_wake, match_stop, match_cancel


def test_match_wake_exact():
    assert match_wake("claude", "claude") is True


def test_match_wake_in_sentence():
    assert match_wake("hey claude how are you", "claude") is True


def test_match_wake_fuzzy():
    assert match_wake("cloud", "claude") is True
    assert match_wake("claud", "claude") is True


def test_match_wake_negative():
    assert match_wake("hello world", "claude") is False


def test_match_stop():
    assert match_stop("over", "over") is True
    assert match_stop("yeah over", "over") is True
    assert match_stop("hello world", "over") is False


def test_match_cancel():
    assert match_cancel("claude cancel", "claude cancel") is True
    assert match_cancel("claude, cancel", "claude cancel") is True
    assert match_cancel("claud cancel", "claude cancel") is True
    assert match_cancel("just cancel", "claude cancel") is False


def test_state_transitions():
    """Test the full state machine: idle -> wake -> recording -> stop."""
    on_start = MagicMock()
    on_stop = MagicMock()
    on_cancel = MagicMock()

    listener = WakeWordListener.__new__(WakeWordListener)
    listener._config = {
        "word": "claude",
        "stop_word": "over",
        "cancel_phrase": "claude cancel",
        "timeout": 5,
        "beep": False,
    }
    listener._on_start = on_start
    listener._on_stop = on_stop
    listener._on_cancel = on_cancel
    listener._state = "idle"

    # Wake word detected
    listener._handle_keyword("hey claude")
    assert listener._state == "recording"
    on_start.assert_called_once()

    # Stop word detected
    listener._handle_command("over")
    assert listener._state == "idle"
    on_stop.assert_called_once()


def test_cancel_during_recording():
    on_start = MagicMock()
    on_stop = MagicMock()
    on_cancel = MagicMock()

    listener = WakeWordListener.__new__(WakeWordListener)
    listener._config = {
        "word": "claude",
        "stop_word": "over",
        "cancel_phrase": "claude cancel",
        "timeout": 5,
        "beep": False,
    }
    listener._on_start = on_start
    listener._on_stop = on_stop
    listener._on_cancel = on_cancel
    listener._state = "idle"

    listener._handle_keyword("claude")
    assert listener._state == "recording"

    listener._handle_command("claude cancel")
    assert listener._state == "idle"
    on_cancel.assert_called_once()
    on_stop.assert_not_called()

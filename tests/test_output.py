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

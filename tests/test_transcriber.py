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

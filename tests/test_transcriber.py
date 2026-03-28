import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from whisper_input.transcriber import Transcriber


def _mock_model_with_detect(detected_lang="en"):
    """Create a mock model that supports detect_language."""
    mock_model = MagicMock()
    mock_model.detect_language.return_value = (
        detected_lang,
        [("en", 0.9), ("ru", 0.1)],
    )
    return mock_model


def test_transcribe_joins_segments():
    """Transcriber joins segment texts with spaces and strips whitespace."""
    mock_model = _mock_model_with_detect()
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
    mock_model = _mock_model_with_detect()
    mock_model.transcribe.return_value = (iter([]), MagicMock())

    with patch("whisper_input.transcriber.WhisperModel", return_value=mock_model):
        transcriber = Transcriber(model="tiny", device="cpu", compute_type="int8")
        result = transcriber.transcribe(np.array([], dtype=np.float32))

    assert result == ""


def test_auto_detects_language_from_allowed_set():
    mock_model = _mock_model_with_detect()
    mock_model.transcribe.return_value = (iter([]), MagicMock())

    with patch("whisper_input.transcriber.WhisperModel", return_value=mock_model):
        transcriber = Transcriber(model="tiny", device="cpu", compute_type="int8", language="auto")
        transcriber.transcribe(np.zeros(16000, dtype=np.float32))

    # Should use detected language, not None
    call_kwargs = mock_model.transcribe.call_args[1]
    assert call_kwargs["language"] == "en"


def test_transcribe_passes_explicit_language():
    mock_model = _mock_model_with_detect()
    mock_model.transcribe.return_value = (iter([]), MagicMock())

    with patch("whisper_input.transcriber.WhisperModel", return_value=mock_model):
        transcriber = Transcriber(model="tiny", device="cpu", compute_type="int8", language="en")
        transcriber.transcribe(np.zeros(16000, dtype=np.float32))

    call_kwargs = mock_model.transcribe.call_args[1]
    assert call_kwargs["language"] == "en"
    # Should NOT call detect_language when language is explicit
    mock_model.detect_language.assert_not_called()

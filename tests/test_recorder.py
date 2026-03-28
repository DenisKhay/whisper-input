import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from whisper_input.recorder import Recorder


def test_start_stop_returns_audio():
    """Recorder accumulates chunks and returns concatenated float32 array."""
    recorder = Recorder(samplerate=16000, channels=1)

    chunk1 = np.random.rand(1600, 1).astype(np.float32)
    chunk2 = np.random.rand(1600, 1).astype(np.float32)

    with patch("whisper_input.recorder.sd") as mock_sd:
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        recorder.start()

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

        recorder.start()
        audio = recorder.stop()

    assert len(audio) == 0

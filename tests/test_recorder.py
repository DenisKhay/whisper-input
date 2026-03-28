import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from whisper_input.recorder import Recorder, _resample


def _make_recorder(**kwargs):
    """Create Recorder with mocked sounddevice to avoid device queries."""
    with patch("whisper_input.recorder.sd") as mock_sd:
        mock_sd.query_devices.return_value = {"default_samplerate": 16000.0}
        mock_sd.default.device = (0, 0)
        recorder = Recorder(**kwargs)
    return recorder


def test_start_stop_returns_audio():
    """Recorder accumulates chunks and returns concatenated float32 array."""
    recorder = _make_recorder(target_samplerate=16000, channels=1)

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
    recorder = _make_recorder(target_samplerate=16000, channels=1)
    audio = recorder.stop()
    assert audio.dtype == np.float32
    assert len(audio) == 0


def test_start_clears_previous_chunks():
    recorder = _make_recorder(target_samplerate=16000, channels=1)
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


def test_resample_48k_to_16k():
    """Resampling 48kHz to 16kHz produces correct length."""
    audio_48k = np.random.rand(48000).astype(np.float32)  # 1 second at 48kHz
    audio_16k = _resample(audio_48k, 48000, 16000)
    assert audio_16k.dtype == np.float32
    assert len(audio_16k) == 16000


def test_resample_same_rate_noop():
    audio = np.random.rand(16000).astype(np.float32)
    result = _resample(audio, 16000, 16000)
    np.testing.assert_array_equal(result, audio)

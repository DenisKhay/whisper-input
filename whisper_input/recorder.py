"""Audio recording via sounddevice."""

import logging

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000


def _resample(audio: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio using linear interpolation."""
    if orig_rate == target_rate:
        return audio
    duration = len(audio) / orig_rate
    target_len = int(duration * target_rate)
    indices = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


class Recorder:
    def __init__(self, target_samplerate: int = WHISPER_SAMPLE_RATE, channels: int = 1):
        self.target_samplerate = target_samplerate
        self.channels = channels
        self._device_samplerate = int(sd.query_devices(sd.default.device[0])["default_samplerate"])
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        logger.info("Device sample rate: %d Hz, target: %d Hz", self._device_samplerate, self.target_samplerate)

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            logger.warning("sounddevice status: %s", status)
        self._chunks.append(indata.copy())

    def start(self) -> None:
        """Start recording from the default input device."""
        self._chunks.clear()
        self._stream = sd.InputStream(
            samplerate=self._device_samplerate,
            channels=self.channels,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("Recording started at %d Hz", self._device_samplerate)

    def stop(self) -> np.ndarray:
        """Stop recording and return audio resampled to target rate as a 1D float32 array."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._chunks:
            audio = np.concatenate(self._chunks, axis=0).flatten()
            duration = len(audio) / self._device_samplerate
            logger.info("Captured %.2fs of audio (%d samples at %d Hz)", duration, len(audio), self._device_samplerate)
            audio = _resample(audio, self._device_samplerate, self.target_samplerate)
            logger.info("Resampled to %d Hz (%d samples)", self.target_samplerate, len(audio))
            return audio

        return np.array([], dtype=np.float32)

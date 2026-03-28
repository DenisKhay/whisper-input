"""Audio recording via sounddevice."""

import logging

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from math import gcd

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000


def _resample(audio: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio using polyphase filtering (anti-aliased)."""
    if orig_rate == target_rate:
        return audio
    divisor = gcd(orig_rate, target_rate)
    up = target_rate // divisor
    down = orig_rate // divisor
    resampled = resample_poly(audio, up, down)
    return resampled.astype(np.float32)


class Recorder:
    def __init__(self, target_samplerate: int = WHISPER_SAMPLE_RATE, channels: int = 1, device: int | None = None):
        self.target_samplerate = target_samplerate
        self.channels = channels
        self._device = device
        dev_info = sd.query_devices(device if device is not None else sd.default.device[0])
        self._device_samplerate = int(dev_info["default_samplerate"])
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        logger.info("Audio device: %s (%d Hz), target: %d Hz", dev_info["name"], self._device_samplerate, self.target_samplerate)

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
            device=self._device,
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

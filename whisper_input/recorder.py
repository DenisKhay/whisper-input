"""Audio recording via sounddevice."""

import logging

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class Recorder:
    def __init__(self, samplerate: int = 16000, channels: int = 1):
        self.samplerate = samplerate
        self.channels = channels
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            logger.warning("sounddevice status: %s", status)
        self._chunks.append(indata.copy())

    def start(self) -> None:
        """Start recording from the default input device."""
        self._chunks.clear()
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("Recording started")

    def stop(self) -> np.ndarray:
        """Stop recording and return the captured audio as a 1D float32 array."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._chunks:
            audio = np.concatenate(self._chunks, axis=0).flatten()
            duration = len(audio) / self.samplerate
            logger.info("Captured %.2fs of audio (%d samples)", duration, len(audio))
            return audio

        return np.array([], dtype=np.float32)

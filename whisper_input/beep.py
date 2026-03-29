"""Audio feedback beeps."""

import logging
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


def generate_beep(freq: float = 800, duration: float = 0.15, samplerate: int = 48000, volume: float = 0.3) -> np.ndarray:
    """Generate a sine wave beep as float32 array."""
    t = np.linspace(0, duration, int(duration * samplerate), endpoint=False, dtype=np.float32)
    return (np.sin(2 * np.pi * freq * t) * volume).astype(np.float32)


def play_beep(freq: float = 800, duration: float = 0.15, volume: float = 0.3) -> None:
    """Play a short beep through the default output device."""
    try:
        samplerate = int(sd.query_devices(sd.default.device[1])["default_samplerate"])
        audio = generate_beep(freq=freq, duration=duration, samplerate=samplerate, volume=volume)
        sd.play(audio, samplerate=samplerate)
    except Exception as e:
        logger.warning("Failed to play beep: %s", e)


def beep_wake() -> None:
    """Play wake word confirmation beep (high)."""
    play_beep(freq=800, duration=0.15, volume=0.3)


def beep_cancel() -> None:
    """Play cancel confirmation beep (low)."""
    play_beep(freq=400, duration=0.15, volume=0.3)

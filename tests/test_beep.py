import numpy as np
from whisper_input.beep import generate_beep


def test_generate_beep_shape_and_dtype():
    audio = generate_beep(freq=800, duration=0.15, samplerate=48000)
    assert audio.dtype == np.float32
    expected_len = int(0.15 * 48000)
    assert len(audio) == expected_len


def test_generate_beep_amplitude():
    audio = generate_beep(freq=800, duration=0.15, samplerate=48000, volume=0.3)
    assert np.max(np.abs(audio)) <= 0.3 + 0.01


def test_generate_beep_different_frequencies():
    beep_high = generate_beep(freq=800, duration=0.1, samplerate=16000)
    beep_low = generate_beep(freq=400, duration=0.1, samplerate=16000)
    assert not np.array_equal(beep_high, beep_low)

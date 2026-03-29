"""Wake word listener using silero-vad + Whisper tiny."""

import logging
import threading
import time

import numpy as np
import torch
import sounddevice as sd
from silero_vad import load_silero_vad, VADIterator

logger = logging.getLogger(__name__)

# Fuzzy variants of "claude" that Whisper might produce (English + Russian)
CLAUDE_VARIANTS = {"claude", "claud", "cloud", "klaud", "clod", "клод", "клот", "клоуд"}

VAD_SAMPLE_RATE = 16000
VAD_CHUNK_SIZE = 512  # 32ms at 16kHz, required by silero-vad


def match_wake(text: str, word: str) -> bool:
    """Check if text contains the wake word (fuzzy matching)."""
    text_lower = text.lower()
    for variant in CLAUDE_VARIANTS:
        if variant in text_lower:
            return True
    return word.lower() in text_lower


def match_stop(text: str, stop_word: str) -> bool:
    """Check if text contains the stop word."""
    return stop_word.lower() in text.lower()


def match_cancel(text: str, cancel_phrase: str) -> bool:
    """Check if text contains the cancel phrase (fuzzy on 'claude' part)."""
    text_lower = text.lower()
    for variant in CLAUDE_VARIANTS:
        if variant in text_lower and "cancel" in text_lower:
            return True
    return cancel_phrase.lower().replace(",", "").replace(" ", "") in text_lower.replace(",", "").replace(" ", "")


class WakeWordListener:
    def __init__(
        self,
        config: dict,
        transcriber,
        recorder,
        on_start: callable,
        on_stop: callable,
        on_cancel: callable,
        audio_device: int | None = None,
    ):
        self._config = config
        self._transcriber = transcriber
        self._recorder = recorder
        self._on_start = on_start
        self._on_stop = on_stop
        self._on_cancel = on_cancel
        self._audio_device = audio_device
        self._state = "idle"  # idle | recording
        self._vad_stream = None

        # Detect device sample rate
        dev_info = sd.query_devices(audio_device if audio_device is not None else sd.default.device[0])
        self._device_rate = int(dev_info["default_samplerate"])
        # Blocksize at device rate that gives ~32ms (same as VAD_CHUNK_SIZE at 16kHz)
        self._device_blocksize = int(VAD_CHUNK_SIZE * self._device_rate / VAD_SAMPLE_RATE)
        logger.info("VAD stream: device '%s' at %d Hz, blocksize %d", dev_info["name"], self._device_rate, self._device_blocksize)

        # VAD setup
        torch.set_num_threads(1)
        self._vad_model = load_silero_vad(onnx=False)
        self._vad_iterator = VADIterator(
            self._vad_model,
            threshold=0.3,
            sampling_rate=VAD_SAMPLE_RATE,
            min_silence_duration_ms=800,
            speech_pad_ms=0,
        )
        logger.info("Wake word listener initialized")

        # Rolling pre-buffer: keeps last ~500ms of audio so we don't clip the start
        self._pre_buffer_chunks = int(0.5 * VAD_SAMPLE_RATE / VAD_CHUNK_SIZE)  # ~16 chunks
        self._pre_buffer: list[np.ndarray] = []

        # Buffer for collecting speech segments
        self._speech_buffer: list[np.ndarray] = []
        self._collecting_speech = False
        self._last_speech_time = 0.0
        self._recording_start_time = 0.0

    def _handle_keyword(self, text: str) -> None:
        """Handle text from keyword detection (idle state)."""
        if self._state == "idle" and match_wake(text, self._config["word"]):
            self._state = "recording"
            self._recording_start_time = time.time()
            self._last_speech_time = time.time()
            logger.info("Wake word detected: '%s'", text)
            self._on_start()

    def _handle_command(self, text: str) -> None:
        """Handle text from command detection (recording state, after a pause)."""
        if self._state != "recording":
            return

        if match_cancel(text, self._config["cancel_phrase"]):
            self._state = "idle"
            logger.info("Cancel detected: '%s'", text)
            self._on_cancel()
        elif match_stop(text, self._config["stop_word"]):
            self._state = "idle"
            logger.info("Stop word detected: '%s'", text)
            self._on_stop()

    def _resample_to_vad(self, audio: np.ndarray) -> np.ndarray:
        """Resample audio from device rate to VAD rate (16kHz)."""
        if self._device_rate == VAD_SAMPLE_RATE:
            return audio
        # Simple decimation — device_rate is always a multiple of 16kHz for common rates
        ratio = self._device_rate / VAD_SAMPLE_RATE
        indices = np.round(np.arange(VAD_CHUNK_SIZE) * ratio).astype(int)
        indices = np.clip(indices, 0, len(audio) - 1)
        return audio[indices].astype(np.float32)

    def _vad_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Called by sounddevice for each audio chunk."""
        if status:
            logger.warning("VAD stream status: %s", status)

        raw_chunk = indata[:, 0].copy()
        # Resample to 16kHz for VAD (silero requires exactly 512 samples at 16kHz)
        vad_chunk = self._resample_to_vad(raw_chunk)
        audio_tensor = torch.from_numpy(vad_chunk)

        # Always maintain a rolling pre-buffer
        self._pre_buffer.append(vad_chunk.copy())
        if len(self._pre_buffer) > self._pre_buffer_chunks:
            self._pre_buffer.pop(0)

        speech_dict = self._vad_iterator(audio_tensor, return_seconds=False)

        if speech_dict is not None:
            if "start" in speech_dict:
                self._collecting_speech = True
                # Start with the pre-buffer so we capture the beginning of speech
                self._speech_buffer = list(self._pre_buffer)
            elif "end" in speech_dict:
                self._collecting_speech = False
                if self._speech_buffer:
                    speech_audio = np.concatenate(self._speech_buffer)
                    self._speech_buffer.clear()
                    threading.Thread(
                        target=self._process_speech_segment,
                        args=(speech_audio,),
                        daemon=True,
                    ).start()

        if self._collecting_speech:
            self._speech_buffer.append(vad_chunk)

    def _process_speech_segment(self, audio: np.ndarray) -> None:
        """Process a detected speech segment through Whisper."""
        try:
            text = self._transcriber.transcribe(audio)
            if not text:
                return

            logger.debug("VAD segment transcribed: '%s'", text)

            if self._state == "idle":
                self._handle_keyword(text)
            # Note: stop/cancel commands not checked here because the VAD stream
            # is paused during recording. Use hotkey (Ctrl+`) to stop recording
            # after voice activation. Voice stop commands require audio device
            # sharing which is a future enhancement.
        except Exception as e:
            logger.error("Error processing speech segment: %s", e)

    def start(self) -> None:
        """Start the VAD listening stream."""
        self._vad_stream = sd.InputStream(
            samplerate=self._device_rate,
            channels=1,
            dtype="float32",
            blocksize=self._device_blocksize,
            device=self._audio_device,
            callback=self._vad_callback,
        )
        self._vad_stream.start()
        self._last_speech_time = time.time()
        logger.info("Wake word listener started (word='%s', stop='%s')",
                     self._config["word"], self._config["stop_word"])

    def pause(self) -> None:
        """Pause the VAD stream to free the audio device for recording."""
        if self._vad_stream is not None:
            self._vad_stream.stop()
            self._vad_stream.close()
            self._vad_stream = None
            self._collecting_speech = False
            self._speech_buffer.clear()
            logger.debug("VAD stream paused")

    def resume(self) -> None:
        """Resume the VAD stream after recording is done."""
        if self._vad_stream is None:
            self._vad_model.reset_states()
            self._vad_iterator.reset_states()
            self._pre_buffer.clear()
            self._vad_stream = sd.InputStream(
                samplerate=self._device_rate,
                channels=1,
                dtype="float32",
                blocksize=self._device_blocksize,
                device=self._audio_device,
                callback=self._vad_callback,
            )
            self._vad_stream.start()
            self._last_speech_time = time.time()
            logger.debug("VAD stream resumed")

    def stop(self) -> None:
        """Stop the VAD listening stream."""
        self.pause()
        self._vad_model.reset_states()
        self._vad_iterator.reset_states()
        logger.info("Wake word listener stopped")

    @property
    def is_active(self) -> bool:
        return self._state == "recording"

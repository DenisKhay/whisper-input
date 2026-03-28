"""Speech-to-text transcription via faster-whisper."""

import logging

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class Transcriber:
    def __init__(
        self,
        model: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "auto",
    ):
        self.language = None if language == "auto" else language
        logger.info("Loading Whisper model '%s' on %s (%s)...", model, device, compute_type)
        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        logger.info("Model loaded")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a float32 16kHz audio array to text."""
        if len(audio) == 0:
            return ""

        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            vad_filter=False,
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()
        if text:
            logger.info("Transcribed (%s, %.0f%%): %s", info.language, info.language_probability * 100, text)
        else:
            logger.warning("Transcription returned empty text")

        return text

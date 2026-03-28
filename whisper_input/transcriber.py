"""Speech-to-text transcription via faster-whisper."""

import logging

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Languages to consider during auto-detection
ALLOWED_LANGUAGES = {"en", "ru"}


class Transcriber:
    def __init__(
        self,
        model: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "auto",
    ):
        self._auto = language == "auto"
        self._language = None if self._auto else language
        logger.info("Loading Whisper model '%s' on %s (%s)...", model, device, compute_type)
        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        logger.info("Model loaded")

    def _detect_language(self, audio: np.ndarray) -> str:
        """Detect language, restricted to allowed set."""
        _lang, _prob, all_probs = self._model.detect_language(audio)
        # Filter to allowed languages and pick the best
        filtered = {lang: prob for lang, prob in all_probs if lang in ALLOWED_LANGUAGES}
        if not filtered:
            return "en"
        best = max(filtered, key=filtered.get)
        logger.info("Language detection: %s (%.0f%%)", best, filtered[best] * 100)
        return best

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a float32 16kHz audio array to text."""
        if len(audio) == 0:
            return ""

        language = self._language
        if self._auto:
            language = self._detect_language(audio)

        segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=False,
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()
        if text:
            logger.info("Transcribed (%s): %s", language, text)
        else:
            logger.warning("Transcription returned empty text")

        return text

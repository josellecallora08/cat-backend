"""Speech-to-Text service using Faster-Whisper with CTranslate2.

Provides a Protocol interface for testability and a concrete implementation
that lazily loads the Whisper model. If faster-whisper is not installed,
a MockSTTService is available for testing.

Requirements: 3.1, 3.6
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Protocol

logger = logging.getLogger(__name__)

# Supported languages for detection
SUPPORTED_LANGUAGES = {"en": "english", "tl": "tagalog"}

# Confidence threshold below which results are considered unreliable
LOW_CONFIDENCE_THRESHOLD = 0.4


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription.

    Attributes:
        text: The transcribed text content.
        language: Detected or specified language code (e.g. 'en', 'tl').
        confidence: Average confidence score from the model (0.0 to 1.0).
        duration_ms: Duration of the audio input in milliseconds.
    """

    text: str
    language: str
    confidence: float
    duration_ms: int


class STTServiceProtocol(Protocol):
    """Protocol for STT service implementations, enabling easy mocking."""

    def transcribe(
        self, audio: bytes, *, language: Optional[str] = None
    ) -> TranscriptionResult:
        """Transcribe a PCM 16kHz audio buffer to text.

        Args:
            audio: Raw PCM 16-bit signed little-endian audio at 16kHz sample rate.
            language: Optional language hint ('en' or 'tl'). If None, auto-detect.

        Returns:
            TranscriptionResult with transcribed text, language, confidence, duration.
        """
        ...


class STTService:
    """Speech-to-Text service wrapping Faster-Whisper with CTranslate2.

    The model is loaded lazily on first transcribe() call so that importing
    this module does not require the model to be available.
    """

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cpu",
        compute_type: str = "float32",
    ):
        """Initialize STT service configuration.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v2).
            device: Compute device ('cpu' or 'cuda').
            compute_type: CTranslate2 compute type ('float16', 'float32', 'int8').
        """
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model = None

    def _load_model(self) -> None:
        """Lazily load the Faster-Whisper model."""
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "faster-whisper is not installed. "
                "Install it with: pip install faster-whisper"
            ) from e

        logger.info(
            "Loading Faster-Whisper model: size=%s, device=%s, compute_type=%s",
            self._model_size,
            self._device,
            self._compute_type,
        )
        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        logger.info("Faster-Whisper model loaded successfully.")

    def _calculate_duration_ms(self, audio: bytes) -> int:
        """Calculate audio duration in milliseconds from PCM 16kHz 16-bit mono data.

        PCM 16-bit mono at 16kHz = 32000 bytes per second.
        """
        bytes_per_second = 16000 * 2  # 16kHz * 2 bytes per sample (16-bit)
        duration_seconds = len(audio) / bytes_per_second
        return int(duration_seconds * 1000)

    def transcribe(
        self, audio: bytes, *, language: Optional[str] = None
    ) -> TranscriptionResult:
        """Transcribe a PCM 16kHz audio buffer to text.

        Args:
            audio: Raw PCM 16-bit signed little-endian audio at 16kHz sample rate.
            language: Optional language hint ('en' or 'tl'). If None, auto-detect.

        Returns:
            TranscriptionResult with transcribed text, language, confidence, duration.

        Raises:
            RuntimeError: If faster-whisper is not installed.
            ValueError: If audio buffer is empty.
        """
        if not audio:
            raise ValueError("Audio buffer is empty.")

        if self._model is None:
            self._load_model()

        import numpy as np

        duration_ms = self._calculate_duration_ms(audio)

        # Convert PCM bytes to float32 numpy array normalized to [-1.0, 1.0]
        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        # Build transcription kwargs
        transcribe_kwargs: dict = {}
        if language and language in SUPPORTED_LANGUAGES:
            transcribe_kwargs["language"] = language

        start_time = time.time()
        segments, info = self._model.transcribe(
            audio_array,
            beam_size=5,
            **transcribe_kwargs,
        )

        # Collect all segments
        text_parts: list[str] = []
        confidences: list[float] = []

        for segment in segments:
            text_parts.append(segment.text.strip())
            # avg_logprob is log probability; convert to approximate confidence
            # Using exp(avg_logprob) as a rough confidence proxy
            import math

            confidence = math.exp(segment.avg_logprob) if segment.avg_logprob else 0.0
            confidences.append(confidence)

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.debug("Transcription completed in %d ms", elapsed_ms)

        # Combine text from all segments
        full_text = " ".join(text_parts).strip()

        # Calculate average confidence
        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        # Determine detected language
        detected_language = info.language if info.language else "en"

        # Handle low-confidence or empty results (Requirement 3.6)
        if not full_text or avg_confidence < LOW_CONFIDENCE_THRESHOLD:
            logger.warning(
                "Low-confidence or empty transcription: confidence=%.3f, text='%s'",
                avg_confidence,
                full_text[:50],
            )
            return TranscriptionResult(
                text=full_text,
                language=detected_language,
                confidence=avg_confidence,
                duration_ms=duration_ms,
            )

        return TranscriptionResult(
            text=full_text,
            language=detected_language,
            confidence=avg_confidence,
            duration_ms=duration_ms,
        )


class MockSTTService:
    """Mock STT service for testing without Faster-Whisper installed.

    Returns configurable transcription results for unit testing.
    """

    def __init__(
        self,
        default_text: str = "Hello, this is a test transcription.",
        default_language: str = "en",
        default_confidence: float = 0.92,
    ):
        """Initialize mock with default return values.

        Args:
            default_text: Default transcribed text to return.
            default_language: Default language code to return.
            default_confidence: Default confidence score to return.
        """
        self._default_text = default_text
        self._default_language = default_language
        self._default_confidence = default_confidence
        self._call_count = 0
        self._last_audio: Optional[bytes] = None
        self._last_language: Optional[str] = None

    @property
    def call_count(self) -> int:
        """Number of times transcribe has been called."""
        return self._call_count

    @property
    def last_audio(self) -> Optional[bytes]:
        """The last audio buffer passed to transcribe."""
        return self._last_audio

    @property
    def last_language(self) -> Optional[str]:
        """The last language hint passed to transcribe."""
        return self._last_language

    def _calculate_duration_ms(self, audio: bytes) -> int:
        """Calculate audio duration from PCM 16kHz 16-bit mono data."""
        bytes_per_second = 16000 * 2
        duration_seconds = len(audio) / bytes_per_second
        return int(duration_seconds * 1000)

    def transcribe(
        self, audio: bytes, *, language: Optional[str] = None
    ) -> TranscriptionResult:
        """Return a mock transcription result.

        Args:
            audio: Raw PCM audio buffer.
            language: Optional language hint.

        Returns:
            TranscriptionResult with configured default values.

        Raises:
            ValueError: If audio buffer is empty.
        """
        if not audio:
            raise ValueError("Audio buffer is empty.")

        self._call_count += 1
        self._last_audio = audio
        self._last_language = language

        duration_ms = self._calculate_duration_ms(audio)

        # Use language hint if provided, otherwise default
        detected_language = language if language else self._default_language

        return TranscriptionResult(
            text=self._default_text,
            language=detected_language,
            confidence=self._default_confidence,
            duration_ms=duration_ms,
        )

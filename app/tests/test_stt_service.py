"""Unit tests for STT service (Faster-Whisper integration).

Tests use MockSTTService and mocked Faster-Whisper to verify:
- TranscriptionResult structure and fields
- Language detection for English and Tagalog
- Low-confidence / empty result handling
- Empty audio buffer rejection
- Duration calculation from PCM data
- Lazy model loading behavior

Requirements: 3.1, 3.6
"""

import math
import sys
from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_numpy():
    """Create a mock numpy module sufficient for STTService.transcribe."""
    mock_np = MagicMock()
    mock_np.int16 = "int16"
    mock_np.float32 = "float32"
    # pytest.approx checks np.bool_ for type checking
    mock_np.bool_ = bool

    # np.frombuffer(...).astype(...) / 32768.0 → just return a MagicMock array
    mock_array = MagicMock()
    mock_array.astype.return_value = mock_array
    mock_array.__truediv__ = lambda self, other: self
    mock_np.frombuffer.return_value = mock_array
    return mock_np

from app.services.voice.stt_service import (
    LOW_CONFIDENCE_THRESHOLD,
    SUPPORTED_LANGUAGES,
    MockSTTService,
    STTService,
    TranscriptionResult,
)


# --- TranscriptionResult tests ---


class TestTranscriptionResult:
    """Tests for the TranscriptionResult dataclass."""

    def test_dataclass_fields(self):
        """TranscriptionResult has expected fields."""
        field_names = {f.name for f in fields(TranscriptionResult)}
        assert field_names == {"text", "language", "confidence", "duration_ms"}

    def test_creation(self):
        """TranscriptionResult can be created with valid values."""
        result = TranscriptionResult(
            text="Hello world",
            language="en",
            confidence=0.95,
            duration_ms=3200,
        )
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.confidence == 0.95
        assert result.duration_ms == 3200

    def test_language_codes(self):
        """Supported language codes are correct."""
        assert "en" in SUPPORTED_LANGUAGES
        assert "tl" in SUPPORTED_LANGUAGES
        assert SUPPORTED_LANGUAGES["en"] == "english"
        assert SUPPORTED_LANGUAGES["tl"] == "tagalog"


# --- MockSTTService tests ---


class TestMockSTTService:
    """Tests for MockSTTService behavior."""

    def test_default_transcription(self):
        """Mock returns configured default values."""
        service = MockSTTService()
        # 1 second of 16kHz 16-bit mono PCM = 32000 bytes
        audio = b"\x00" * 32000
        result = service.transcribe(audio)

        assert result.text == "Hello, this is a test transcription."
        assert result.language == "en"
        assert result.confidence == 0.92
        assert result.duration_ms == 1000

    def test_custom_defaults(self):
        """Mock respects custom default configuration."""
        service = MockSTTService(
            default_text="Kamusta po",
            default_language="tl",
            default_confidence=0.88,
        )
        audio = b"\x00" * 32000
        result = service.transcribe(audio)

        assert result.text == "Kamusta po"
        assert result.language == "tl"
        assert result.confidence == 0.88

    def test_language_hint_override(self):
        """Mock uses language hint when provided."""
        service = MockSTTService(default_language="en")
        audio = b"\x00" * 32000
        result = service.transcribe(audio, language="tl")

        assert result.language == "tl"

    def test_empty_audio_raises(self):
        """Mock rejects empty audio buffer."""
        service = MockSTTService()
        with pytest.raises(ValueError, match="Audio buffer is empty"):
            service.transcribe(b"")

    def test_call_tracking(self):
        """Mock tracks call count and last arguments."""
        service = MockSTTService()
        audio1 = b"\x00" * 16000
        audio2 = b"\x01" * 32000

        assert service.call_count == 0
        assert service.last_audio is None
        assert service.last_language is None

        service.transcribe(audio1, language="en")
        assert service.call_count == 1
        assert service.last_audio == audio1
        assert service.last_language == "en"

        service.transcribe(audio2, language="tl")
        assert service.call_count == 2
        assert service.last_audio == audio2
        assert service.last_language == "tl"

    def test_duration_calculation(self):
        """Mock correctly calculates duration from PCM data.

        PCM 16kHz 16-bit mono = 32000 bytes/sec
        """
        service = MockSTTService()

        # 500ms of audio = 16000 bytes
        audio_500ms = b"\x00" * 16000
        result = service.transcribe(audio_500ms)
        assert result.duration_ms == 500

        # 2 seconds of audio = 64000 bytes
        audio_2s = b"\x00" * 64000
        result = service.transcribe(audio_2s)
        assert result.duration_ms == 2000


# --- STTService tests (with mocked Faster-Whisper) ---


class TestSTTService:
    """Tests for STTService with mocked Faster-Whisper model."""

    @pytest.fixture(autouse=True)
    def _mock_numpy(self):
        """Mock numpy for all tests since it's not installed in dev."""
        mock_np = _make_mock_numpy()
        with patch.dict(sys.modules, {"numpy": mock_np}):
            yield

    def _make_mock_segment(self, text: str, avg_logprob: float):
        """Create a mock transcription segment."""
        segment = MagicMock()
        segment.text = text
        segment.avg_logprob = avg_logprob
        return segment

    def _make_mock_info(self, language: str = "en"):
        """Create a mock transcription info object."""
        info = MagicMock()
        info.language = language
        return info

    def test_lazy_model_loading(self):
        """Model is not loaded until first transcribe call."""
        service = STTService(model_size="tiny", device="cpu")
        assert service._model is None

    def test_empty_audio_raises(self):
        """STTService rejects empty audio buffer."""
        service = STTService(model_size="tiny", device="cpu")
        with pytest.raises(ValueError, match="Audio buffer is empty"):
            service.transcribe(b"")

    @patch("app.services.voice.stt_service.STTService._load_model")
    def test_successful_transcription(self, mock_load):
        """STTService returns correct TranscriptionResult for valid audio."""
        service = STTService(model_size="tiny", device="cpu")

        # Set up mock model
        mock_model = MagicMock()
        segments = [
            self._make_mock_segment("Hello, I'm calling about your account.", -0.2)
        ]
        info = self._make_mock_info("en")
        mock_model.transcribe.return_value = (iter(segments), info)
        service._model = mock_model

        # 1 second of PCM audio
        audio = b"\x00" * 32000
        result = service.transcribe(audio)

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello, I'm calling about your account."
        assert result.language == "en"
        assert result.confidence == pytest.approx(math.exp(-0.2), rel=1e-3)
        assert result.duration_ms == 1000

    @patch("app.services.voice.stt_service.STTService._load_model")
    def test_tagalog_language_detection(self, mock_load):
        """STTService detects Tagalog language."""
        service = STTService(model_size="tiny", device="cpu")

        mock_model = MagicMock()
        segments = [self._make_mock_segment("Magandang araw po", -0.15)]
        info = self._make_mock_info("tl")
        mock_model.transcribe.return_value = (iter(segments), info)
        service._model = mock_model

        audio = b"\x00" * 32000
        result = service.transcribe(audio)

        assert result.language == "tl"
        assert result.text == "Magandang araw po"

    @patch("app.services.voice.stt_service.STTService._load_model")
    def test_language_hint_passed_to_model(self, mock_load):
        """STTService passes language hint to Whisper model."""
        service = STTService(model_size="tiny", device="cpu")

        mock_model = MagicMock()
        segments = [self._make_mock_segment("Kamusta po", -0.1)]
        info = self._make_mock_info("tl")
        mock_model.transcribe.return_value = (iter(segments), info)
        service._model = mock_model

        audio = b"\x00" * 32000
        service.transcribe(audio, language="tl")

        # Verify language was passed to model.transcribe
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs.get("language") == "tl"

    @patch("app.services.voice.stt_service.STTService._load_model")
    def test_low_confidence_result(self, mock_load):
        """STTService handles low-confidence transcription results."""
        service = STTService(model_size="tiny", device="cpu")

        mock_model = MagicMock()
        # Very low log probability = low confidence
        low_logprob = math.log(LOW_CONFIDENCE_THRESHOLD - 0.1)
        segments = [self._make_mock_segment("um", low_logprob)]
        info = self._make_mock_info("en")
        mock_model.transcribe.return_value = (iter(segments), info)
        service._model = mock_model

        audio = b"\x00" * 32000
        result = service.transcribe(audio)

        assert result.confidence < LOW_CONFIDENCE_THRESHOLD
        assert result.text == "um"

    @patch("app.services.voice.stt_service.STTService._load_model")
    def test_empty_transcription_result(self, mock_load):
        """STTService handles empty transcription (no speech detected)."""
        service = STTService(model_size="tiny", device="cpu")

        mock_model = MagicMock()
        # No segments returned
        segments: list = []
        info = self._make_mock_info("en")
        mock_model.transcribe.return_value = (iter(segments), info)
        service._model = mock_model

        audio = b"\x00" * 32000
        result = service.transcribe(audio)

        assert result.text == ""
        assert result.confidence == 0.0
        assert result.duration_ms == 1000

    @patch("app.services.voice.stt_service.STTService._load_model")
    def test_multiple_segments_combined(self, mock_load):
        """STTService combines multiple transcription segments."""
        service = STTService(model_size="tiny", device="cpu")

        mock_model = MagicMock()
        segments = [
            self._make_mock_segment("Hello.", -0.1),
            self._make_mock_segment("I'm calling about your account.", -0.2),
            self._make_mock_segment("Can we discuss a payment plan?", -0.15),
        ]
        info = self._make_mock_info("en")
        mock_model.transcribe.return_value = (iter(segments), info)
        service._model = mock_model

        audio = b"\x00" * 96000  # 3 seconds
        result = service.transcribe(audio)

        assert "Hello." in result.text
        assert "I'm calling about your account." in result.text
        assert "Can we discuss a payment plan?" in result.text
        assert result.duration_ms == 3000

        # Confidence should be average of all segments
        expected_confidence = (
            math.exp(-0.1) + math.exp(-0.2) + math.exp(-0.15)
        ) / 3
        assert result.confidence == pytest.approx(expected_confidence, rel=1e-3)

    @patch("app.services.voice.stt_service.STTService._load_model")
    def test_duration_calculation_various_lengths(self, mock_load):
        """STTService correctly calculates duration for various audio lengths."""
        service = STTService(model_size="tiny", device="cpu")

        mock_model = MagicMock()
        segments = [self._make_mock_segment("test", -0.1)]
        info = self._make_mock_info("en")
        mock_model.transcribe.return_value = (iter(segments), info)
        service._model = mock_model

        # 250ms = 8000 bytes
        audio = b"\x00" * 8000
        result = service.transcribe(audio)
        assert result.duration_ms == 250

    def test_missing_faster_whisper_raises_runtime_error(self):
        """STTService raises RuntimeError when faster-whisper is not installed."""
        service = STTService(model_size="tiny", device="cpu")

        with patch.dict("sys.modules", {"faster_whisper": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'faster_whisper'"),
            ):
                # Force model load attempt
                with pytest.raises(RuntimeError, match="faster-whisper is not installed"):
                    service._load_model()

    @patch("app.services.voice.stt_service.STTService._load_model")
    def test_unsupported_language_hint_ignored(self, mock_load):
        """STTService ignores unsupported language hints gracefully."""
        service = STTService(model_size="tiny", device="cpu")

        mock_model = MagicMock()
        segments = [self._make_mock_segment("test", -0.1)]
        info = self._make_mock_info("en")
        mock_model.transcribe.return_value = (iter(segments), info)
        service._model = mock_model

        audio = b"\x00" * 32000
        # Pass unsupported language - should not crash, just not pass to model
        result = service.transcribe(audio, language="fr")

        # Should not pass unsupported language to model
        call_kwargs = mock_model.transcribe.call_args[1]
        assert "language" not in call_kwargs
        assert result.language == "en"  # Uses detected language from info

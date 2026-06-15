"""Unit tests for TTS service.

Tests use MockTTSService to verify interface behavior, sentence splitting,
streaming logic, and error handling without requiring piper-tts installed.

Requirements: 3.2
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest

from app.services.voice.tts_service import (
    AudioStream,
    MockTTSService,
    TTSSynthesisError,
    TTS_CHANNELS,
    TTS_SAMPLE_RATE,
    _split_sentences,
)


# --- Helpers ---


async def _async_iter(items: list[str]) -> AsyncIterator[str]:
    """Create an async iterator from a list of strings."""
    for item in items:
        yield item


# --- Tests for _split_sentences ---


class TestSplitSentences:
    """Tests for the sentence boundary splitting helper."""

    def test_single_sentence_no_split(self):
        result = _split_sentences("Hello world")
        assert result == ["Hello world"]

    def test_two_sentences_period(self):
        result = _split_sentences("Hello world. How are you?")
        assert result == ["Hello world.", "How are you?"]

    def test_multiple_sentences_mixed_punctuation(self):
        result = _split_sentences("Stop! What happened? I see. Okay then.")
        assert result == ["Stop!", "What happened?", "I see.", "Okay then."]

    def test_empty_string(self):
        result = _split_sentences("")
        assert result == []

    def test_whitespace_only(self):
        result = _split_sentences("   ")
        assert result == []

    def test_no_terminal_punctuation(self):
        result = _split_sentences("Hello world, how are you")
        assert result == ["Hello world, how are you"]

    def test_abbreviations_not_split(self):
        # No space after period means no split
        result = _split_sentences("Dr.Smith went home")
        assert result == ["Dr.Smith went home"]

    def test_exclamation_boundary(self):
        result = _split_sentences("Wait! I need to tell you something.")
        assert result == ["Wait!", "I need to tell you something."]

    def test_multiple_spaces_between_sentences(self):
        result = _split_sentences("First sentence.   Second sentence.")
        assert result == ["First sentence.", "Second sentence."]


# --- Tests for AudioStream ---


class TestAudioStream:
    """Tests for the AudioStream dataclass."""

    def test_default_values(self):
        stream = AudioStream(data=b"\x00\x01")
        assert stream.data == b"\x00\x01"
        assert stream.sample_rate == TTS_SAMPLE_RATE
        assert stream.channels == TTS_CHANNELS

    def test_custom_values(self):
        stream = AudioStream(data=b"\x00", sample_rate=16000, channels=2)
        assert stream.sample_rate == 16000
        assert stream.channels == 2


# --- Tests for MockTTSService.synthesize ---


class TestMockTTSSynthesize:
    """Tests for MockTTSService.synthesize method."""

    @pytest.fixture
    def tts(self) -> MockTTSService:
        return MockTTSService()

    async def test_synthesize_returns_audio_stream(self, tts: MockTTSService):
        result = await tts.synthesize("Hello world")
        assert isinstance(result, AudioStream)
        assert result.sample_rate == TTS_SAMPLE_RATE
        assert result.channels == TTS_CHANNELS

    async def test_synthesize_produces_nonzero_data(self, tts: MockTTSService):
        result = await tts.synthesize("Hello")
        assert len(result.data) > 0

    async def test_synthesize_data_proportional_to_text(self, tts: MockTTSService):
        short = await tts.synthesize("Hi")
        long = await tts.synthesize("This is a much longer sentence for testing")
        assert len(long.data) > len(short.data)

    async def test_synthesize_records_calls(self, tts: MockTTSService):
        await tts.synthesize("Test text", "en")
        await tts.synthesize("Otro texto", "es")
        assert tts.synthesize_calls == [("Test text", "en"), ("Otro texto", "es")]

    async def test_synthesize_empty_text_raises(self, tts: MockTTSService):
        with pytest.raises(TTSSynthesisError) as exc_info:
            await tts.synthesize("")
        assert exc_info.value.original_text == ""

    async def test_synthesize_whitespace_only_raises(self, tts: MockTTSService):
        with pytest.raises(TTSSynthesisError):
            await tts.synthesize("   ")

    async def test_synthesize_failure_mode(self, tts: MockTTSService):
        tts.set_should_fail(True)
        with pytest.raises(TTSSynthesisError) as exc_info:
            await tts.synthesize("Some text")
        assert exc_info.value.original_text == "Some text"

    async def test_synthesize_default_language(self, tts: MockTTSService):
        await tts.synthesize("Hello")
        assert tts.synthesize_calls[0] == ("Hello", "en")

    async def test_synthesize_even_byte_count(self, tts: MockTTSService):
        """PCM 16-bit requires even number of bytes."""
        result = await tts.synthesize("Odd")
        assert len(result.data) % 2 == 0


# --- Tests for MockTTSService.synthesize_streaming ---


class TestMockTTSSynthesizeStreaming:
    """Tests for streaming synthesis with sentence boundary chunking."""

    @pytest.fixture
    def tts(self) -> MockTTSService:
        return MockTTSService()

    async def test_single_sentence_streamed_at_end(self, tts: MockTTSService):
        chunks = ["Hello ", "world"]
        results = []
        async for audio in tts.synthesize_streaming(_async_iter(chunks)):
            results.append(audio)
        # Single sentence, flushed at end
        assert len(results) == 1
        assert len(results[0]) > 0

    async def test_two_sentences_produce_two_audio_chunks(self, tts: MockTTSService):
        # "Hello world. " triggers sentence boundary, then "How are you?" at flush
        chunks = ["Hello world. ", "How are you?"]
        results = []
        async for audio in tts.synthesize_streaming(_async_iter(chunks)):
            results.append(audio)
        assert len(results) == 2

    async def test_streaming_incremental_tokens(self, tts: MockTTSService):
        # Simulates LLM token-by-token streaming
        chunks = ["I ", "am ", "fine. ", "Thank ", "you. ", "Goodbye!"]
        results = []
        async for audio in tts.synthesize_streaming(_async_iter(chunks)):
            results.append(audio)
        # "I am fine." -> audio, "Thank you." -> audio, "Goodbye!" -> flush
        assert len(results) == 3

    async def test_empty_stream_produces_nothing(self, tts: MockTTSService):
        chunks: list[str] = []
        results = []
        async for audio in tts.synthesize_streaming(_async_iter(chunks)):
            results.append(audio)
        assert len(results) == 0

    async def test_whitespace_only_stream_produces_nothing(self, tts: MockTTSService):
        chunks = ["  ", "  "]
        results = []
        async for audio in tts.synthesize_streaming(_async_iter(chunks)):
            results.append(audio)
        assert len(results) == 0

    async def test_streaming_with_failure_raises(self, tts: MockTTSService):
        tts.set_should_fail(True)
        chunks = ["Hello world. ", "More text."]
        with pytest.raises(TTSSynthesisError):
            async for _ in tts.synthesize_streaming(_async_iter(chunks)):
                pass

    async def test_streaming_preserves_sentence_text(self, tts: MockTTSService):
        chunks = ["First sentence. ", "Second sentence."]
        results = []
        async for audio in tts.synthesize_streaming(_async_iter(chunks)):
            results.append(audio)
        # Verify synthesize was called with the right sentences
        assert ("First sentence.", "en") in tts.synthesize_calls
        assert ("Second sentence.", "en") in tts.synthesize_calls


# --- Tests for TTSSynthesisError ---


class TestTTSSynthesisError:
    """Tests for the TTSSynthesisError exception."""

    def test_error_message(self):
        err = TTSSynthesisError("Something went wrong")
        assert str(err) == "Something went wrong"

    def test_original_text_preserved(self):
        err = TTSSynthesisError("Failed", original_text="Hello world")
        assert err.original_text == "Hello world"

    def test_default_original_text_empty(self):
        err = TTSSynthesisError("Failed")
        assert err.original_text == ""


# --- Tests for TTSService import guard ---


class TestTTSServiceImportGuard:
    """Tests that TTSService raises ImportError when piper is unavailable."""

    def test_import_error_when_piper_missing(self):
        """TTSService should raise ImportError if piper-tts is not installed."""
        # piper-tts is not installed in test environment
        with pytest.raises(ImportError, match="piper-tts is not installed"):
            from app.services.voice.tts_service import TTSService

            TTSService()

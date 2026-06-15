"""Text-to-Speech service using Piper for fast local neural synthesis.

Provides TTS synthesis with streaming support for low-latency voice output.
Uses a Protocol/interface pattern with conditional Piper import to allow
testing without the piper-tts dependency installed.

Requirements: 3.2
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import AsyncIterator, Protocol


# Audio format constants
TTS_SAMPLE_RATE = 22050  # Piper default output rate
TTS_CHANNELS = 1  # Mono audio
TTS_SAMPLE_WIDTH = 2  # 16-bit PCM

# Sentence boundary pattern for streaming chunking
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


class TTSSynthesisError(Exception):
    """Raised when TTS synthesis fails."""

    def __init__(self, message: str, original_text: str = ""):
        super().__init__(message)
        self.original_text = original_text


@dataclass
class AudioStream:
    """Container for synthesized PCM audio data."""

    data: bytes
    sample_rate: int = TTS_SAMPLE_RATE
    channels: int = TTS_CHANNELS


class TTSServiceProtocol(Protocol):
    """Protocol for TTS service implementations, enabling easy mocking."""

    async def synthesize(self, text: str, language: str = "en") -> AudioStream: ...

    async def synthesize_streaming(
        self, text_chunks: AsyncIterator[str]
    ) -> AsyncIterator[bytes]: ...


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences at boundary punctuation.

    Splits on sentence-ending punctuation (.!?) followed by whitespace.
    Returns non-empty segments only.

    Args:
        text: Input text to split into sentences.

    Returns:
        List of sentence strings (non-empty, stripped).
    """
    parts = _SENTENCE_BOUNDARY_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


class TTSService:
    """Text-to-Speech using Piper for fast local synthesis.

    Attempts to load the piper-tts library at initialization. If unavailable,
    raises ImportError. Use MockTTSService for testing environments.

    Target: sub-200ms first-byte latency for sentence-level synthesis.
    """

    def __init__(self, voice_model: str = "en_US-lessac-medium"):
        """Initialize TTS service with specified Piper voice model.

        Args:
            voice_model: Piper voice model identifier to load.

        Raises:
            ImportError: If piper-tts is not installed.
            TTSSynthesisError: If the voice model cannot be loaded.
        """
        self.voice_model = voice_model
        self._voice = None

        try:
            from piper import PiperVoice  # type: ignore[import-untyped]

            self._piper_voice_cls = PiperVoice
        except ImportError as e:
            raise ImportError(
                "piper-tts is not installed. Install with: pip install piper-tts"
            ) from e

        try:
            self._voice = self._piper_voice_cls.load(voice_model)
        except Exception as e:
            raise TTSSynthesisError(
                f"Failed to load voice model '{voice_model}': {e}"
            )

    async def synthesize(self, text: str, language: str = "en") -> AudioStream:
        """Synthesize text to PCM audio.

        Runs Piper synthesis in a thread pool to avoid blocking the event loop.

        Args:
            text: Text to synthesize.
            language: Language code (currently used for logging/future routing).

        Returns:
            AudioStream containing PCM audio data.

        Raises:
            TTSSynthesisError: If synthesis fails.
        """
        if not text or not text.strip():
            raise TTSSynthesisError("Cannot synthesize empty text", original_text=text)

        try:
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(None, self._synthesize_sync, text)
            return AudioStream(
                data=audio_data,
                sample_rate=TTS_SAMPLE_RATE,
                channels=TTS_CHANNELS,
            )
        except TTSSynthesisError:
            raise
        except Exception as e:
            raise TTSSynthesisError(
                f"Synthesis failed: {e}", original_text=text
            ) from e

    def _synthesize_sync(self, text: str) -> bytes:
        """Synchronous synthesis using Piper voice.

        Args:
            text: Text to synthesize.

        Returns:
            Raw PCM bytes (16-bit, mono, 22050Hz).
        """
        import io
        import wave

        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wav_file:
            wav_file.setnchannels(TTS_CHANNELS)
            wav_file.setsampwidth(TTS_SAMPLE_WIDTH)
            wav_file.setframerate(TTS_SAMPLE_RATE)
            self._voice.synthesize(text, wav_file)

        # Extract raw PCM data (skip WAV header)
        audio_buffer.seek(0)
        with wave.open(audio_buffer, "rb") as wav_file:
            pcm_data = wav_file.readframes(wav_file.getnframes())

        return pcm_data

    async def synthesize_streaming(
        self, text_chunks: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        """Stream synthesis as LLM tokens arrive, chunked by sentence boundaries.

        Accumulates incoming text chunks until a sentence boundary is detected,
        then synthesizes each complete sentence for low-latency first-byte delivery.

        Args:
            text_chunks: Async iterator of text chunks (e.g., from LLM streaming).

        Yields:
            PCM audio bytes for each synthesized sentence.

        Raises:
            TTSSynthesisError: If synthesis fails for a sentence (logged, continues
                with remaining sentences where possible).
        """
        buffer = ""

        async for chunk in text_chunks:
            buffer += chunk

            # Check for sentence boundaries in accumulated buffer
            sentences = _split_sentences(buffer)

            if len(sentences) > 1:
                # Synthesize all complete sentences (all except the last partial)
                for sentence in sentences[:-1]:
                    try:
                        audio_stream = await self.synthesize(sentence)
                        yield audio_stream.data
                    except TTSSynthesisError:
                        # On failure, skip this sentence's audio
                        # The text can be sent as fallback via data channel
                        raise

                # Keep the last (potentially incomplete) segment in buffer
                buffer = sentences[-1]

        # Synthesize any remaining text in the buffer
        if buffer.strip():
            try:
                audio_stream = await self.synthesize(buffer)
                yield audio_stream.data
            except TTSSynthesisError:
                raise


class MockTTSService:
    """Mock TTS service for testing without piper-tts installed.

    Generates deterministic synthetic PCM data based on input text length.
    Implements the same interface as TTSService.
    """

    def __init__(self, voice_model: str = "en_US-lessac-medium"):
        """Initialize mock TTS service.

        Args:
            voice_model: Voice model name (stored but not loaded).
        """
        self.voice_model = voice_model
        self.synthesize_calls: list[tuple[str, str]] = []
        self._should_fail = False

    def set_should_fail(self, fail: bool) -> None:
        """Configure whether synthesis should raise errors (for testing)."""
        self._should_fail = fail

    async def synthesize(self, text: str, language: str = "en") -> AudioStream:
        """Generate mock audio data.

        Produces deterministic PCM bytes proportional to text length.
        Approximately 100 bytes of audio per character of input.

        Args:
            text: Text to "synthesize".
            language: Language code.

        Returns:
            AudioStream with mock PCM data.

        Raises:
            TTSSynthesisError: If set_should_fail(True) was called, or text is empty.
        """
        if not text or not text.strip():
            raise TTSSynthesisError("Cannot synthesize empty text", original_text=text)

        if self._should_fail:
            raise TTSSynthesisError(
                "Mock synthesis failure", original_text=text
            )

        self.synthesize_calls.append((text, language))

        # Generate deterministic mock PCM data
        # ~100 bytes per character simulates realistic audio length
        num_bytes = len(text.strip()) * 100
        # Ensure even number of bytes for 16-bit samples
        num_bytes = num_bytes + (num_bytes % 2)
        mock_pcm = b"\x00\x01" * (num_bytes // 2)

        return AudioStream(
            data=mock_pcm,
            sample_rate=TTS_SAMPLE_RATE,
            channels=TTS_CHANNELS,
        )

    async def synthesize_streaming(
        self, text_chunks: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        """Stream mock synthesis, chunked by sentence boundaries.

        Uses the same sentence-splitting logic as the real service.

        Args:
            text_chunks: Async iterator of text chunks.

        Yields:
            Mock PCM audio bytes for each sentence.
        """
        buffer = ""

        async for chunk in text_chunks:
            buffer += chunk

            sentences = _split_sentences(buffer)

            if len(sentences) > 1:
                for sentence in sentences[:-1]:
                    audio_stream = await self.synthesize(sentence)
                    yield audio_stream.data

                buffer = sentences[-1]

        if buffer.strip():
            audio_stream = await self.synthesize(buffer)
            yield audio_stream.data

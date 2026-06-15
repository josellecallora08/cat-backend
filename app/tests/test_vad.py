"""Unit tests for VAD processor and audio buffering.

Validates: Requirements 3.3, 3.6
"""

import struct

import pytest

from app.services.voice.audio_buffer import AudioBuffer, MAX_BUFFER_DURATION_MS, MAX_FRAMES
from app.services.voice.vad import (
    DEFAULT_SILENCE_THRESHOLD_MS,
    FRAME_DURATION_MS,
    FRAME_SIZE_BYTES,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    EnergyVADBackend,
    VADProcessor,
    VADResult,
)


# --- Helpers ---

def make_silent_frame(num_bytes: int = FRAME_SIZE_BYTES) -> bytes:
    """Create a silent PCM frame (all zeros)."""
    return b"\x00" * num_bytes


def make_speech_frame(amplitude: int = 5000, num_bytes: int = FRAME_SIZE_BYTES) -> bytes:
    """Create a PCM frame with a constant tone (speech-like energy)."""
    num_samples = num_bytes // SAMPLE_WIDTH
    return struct.pack(f"<{num_samples}h", *([amplitude] * num_samples))


def make_low_energy_frame(amplitude: int = 50, num_bytes: int = FRAME_SIZE_BYTES) -> bytes:
    """Create a PCM frame with very low energy (below threshold)."""
    num_samples = num_bytes // SAMPLE_WIDTH
    return struct.pack(f"<{num_samples}h", *([amplitude] * num_samples))


# --- EnergyVADBackend Tests ---

class TestEnergyVADBackend:
    """Tests for the energy-based VAD fallback."""

    def test_silent_frame_detected_as_non_speech(self):
        backend = EnergyVADBackend(energy_threshold=300.0)
        frame = make_silent_frame()
        assert backend.is_speech(frame) is False

    def test_loud_frame_detected_as_speech(self):
        backend = EnergyVADBackend(energy_threshold=300.0)
        frame = make_speech_frame(amplitude=5000)
        assert backend.is_speech(frame) is True

    def test_low_energy_frame_below_threshold(self):
        backend = EnergyVADBackend(energy_threshold=300.0)
        frame = make_low_energy_frame(amplitude=50)
        assert backend.is_speech(frame) is False

    def test_frame_at_threshold_boundary(self):
        """Frame with energy exactly at threshold is considered speech."""
        backend = EnergyVADBackend(energy_threshold=300.0)
        # Amplitude of 300 gives RMS of exactly 300
        frame = make_speech_frame(amplitude=300)
        assert backend.is_speech(frame) is True

    def test_empty_frame_is_not_speech(self):
        backend = EnergyVADBackend()
        assert backend.is_speech(b"") is False
        assert backend.is_speech(b"\x00") is False

    def test_custom_threshold(self):
        # Very low threshold: even quiet audio is "speech"
        backend = EnergyVADBackend(energy_threshold=10.0)
        frame = make_low_energy_frame(amplitude=50)
        assert backend.is_speech(frame) is True


# --- VADProcessor Tests ---

class TestVADProcessor:
    """Tests for the VADProcessor class."""

    def test_frame_size_constants(self):
        """Verify frame size is 640 bytes for 20ms at 16kHz/16-bit."""
        assert FRAME_SIZE_BYTES == 640
        assert FRAME_DURATION_MS == 20
        assert SAMPLE_RATE == 16000
        assert SAMPLE_WIDTH == 2

    def test_process_silent_frame_returns_non_speech(self):
        vad = VADProcessor(backend=EnergyVADBackend())
        result = vad.process_frame(make_silent_frame())
        assert result.is_speech is False
        assert result.duration_silent_ms == FRAME_DURATION_MS

    def test_process_speech_frame_returns_speech(self):
        vad = VADProcessor(backend=EnergyVADBackend())
        result = vad.process_frame(make_speech_frame())
        assert result.is_speech is True
        assert result.duration_silent_ms == 0

    def test_silence_accumulates(self):
        vad = VADProcessor(backend=EnergyVADBackend())
        # Process 5 silent frames = 100ms
        for i in range(5):
            result = vad.process_frame(make_silent_frame())
        assert result.duration_silent_ms == 100

    def test_speech_resets_silence_counter(self):
        vad = VADProcessor(backend=EnergyVADBackend())
        # Accumulate silence
        vad.process_frame(make_silent_frame())
        vad.process_frame(make_silent_frame())
        # Speech resets
        result = vad.process_frame(make_speech_frame())
        assert result.duration_silent_ms == 0
        assert result.is_speech is True

    def test_is_speech_ended_false_without_prior_speech(self):
        """End-of-utterance requires prior speech detection."""
        vad = VADProcessor(backend=EnergyVADBackend(), silence_threshold_ms=500)
        # Feed lots of silence but no speech was seen
        for _ in range(30):
            vad.process_frame(make_silent_frame())
        assert vad.is_speech_ended() is False

    def test_is_speech_ended_true_after_speech_and_silence(self):
        """End-of-utterance detected after speech followed by 500ms silence."""
        vad = VADProcessor(backend=EnergyVADBackend(), silence_threshold_ms=500)
        # Feed speech
        vad.process_frame(make_speech_frame())
        # Feed 500ms silence = 25 frames at 20ms each
        for _ in range(25):
            vad.process_frame(make_silent_frame())
        assert vad.is_speech_ended() is True

    def test_is_speech_ended_false_below_threshold(self):
        """Not ended if silence hasn't reached threshold."""
        vad = VADProcessor(backend=EnergyVADBackend(), silence_threshold_ms=500)
        vad.process_frame(make_speech_frame())
        # Feed 480ms silence = 24 frames
        for _ in range(24):
            vad.process_frame(make_silent_frame())
        assert vad.is_speech_ended() is False

    def test_custom_silence_threshold(self):
        """Custom threshold of 200ms."""
        vad = VADProcessor(backend=EnergyVADBackend(), silence_threshold_ms=200)
        vad.process_frame(make_speech_frame())
        # Feed 200ms silence = 10 frames
        for _ in range(10):
            vad.process_frame(make_silent_frame())
        assert vad.is_speech_ended() is True

    def test_reset_clears_state(self):
        vad = VADProcessor(backend=EnergyVADBackend())
        vad.process_frame(make_speech_frame())
        for _ in range(25):
            vad.process_frame(make_silent_frame())
        assert vad.is_speech_ended() is True

        vad.reset()
        assert vad.is_speech_ended() is False
        # After reset, no speech seen, so silence won't trigger end
        vad.process_frame(make_silent_frame())
        assert vad.is_speech_ended() is False

    def test_default_silence_threshold_is_500ms(self):
        vad = VADProcessor(backend=EnergyVADBackend())
        assert vad.silence_threshold_ms == DEFAULT_SILENCE_THRESHOLD_MS
        assert DEFAULT_SILENCE_THRESHOLD_MS == 500


# --- AudioBuffer Tests ---

class TestAudioBuffer:
    """Tests for the AudioBuffer class."""

    def test_empty_buffer_properties(self):
        buf = AudioBuffer()
        assert buf.duration_ms == 0
        assert buf.frame_count == 0
        assert buf.is_empty() is True
        assert buf.is_overflowing() is False

    def test_append_single_frame(self):
        buf = AudioBuffer()
        frame = make_speech_frame()
        buf.append(frame)
        assert buf.frame_count == 1
        assert buf.duration_ms == FRAME_DURATION_MS

    def test_append_multiple_frames(self):
        buf = AudioBuffer()
        for _ in range(10):
            buf.append(make_speech_frame())
        assert buf.frame_count == 10
        assert buf.duration_ms == 200  # 10 * 20ms

    def test_flush_returns_all_data(self):
        buf = AudioBuffer()
        frame1 = make_speech_frame(amplitude=1000)
        frame2 = make_speech_frame(amplitude=2000)
        buf.append(frame1)
        buf.append(frame2)

        data = buf.flush()
        assert data == frame1 + frame2
        assert buf.is_empty() is True

    def test_flush_empty_buffer_returns_empty_bytes(self):
        buf = AudioBuffer()
        assert buf.flush() == b""

    def test_reset_clears_buffer(self):
        buf = AudioBuffer()
        buf.append(make_speech_frame())
        buf.append(make_speech_frame())
        buf.reset()
        assert buf.is_empty() is True
        assert buf.duration_ms == 0

    def test_overflow_at_30_seconds(self):
        """Buffer overflows at 30 seconds (1500 frames at 20ms each)."""
        buf = AudioBuffer()
        assert MAX_FRAMES == 1500

        # Fill to exactly max
        for _ in range(MAX_FRAMES):
            buf.append(make_silent_frame())

        assert buf.is_overflowing() is True
        assert buf.duration_ms == MAX_BUFFER_DURATION_MS

    def test_overflow_discards_oldest_frames(self):
        """When buffer is full, adding a new frame drops the oldest."""
        buf = AudioBuffer(max_duration_ms=60)  # 3 frames max (60ms / 20ms)

        frame_a = b"\x01" * FRAME_SIZE_BYTES
        frame_b = b"\x02" * FRAME_SIZE_BYTES
        frame_c = b"\x03" * FRAME_SIZE_BYTES
        frame_d = b"\x04" * FRAME_SIZE_BYTES

        buf.append(frame_a)
        buf.append(frame_b)
        buf.append(frame_c)
        assert buf.frame_count == 3

        # Adding a 4th frame should discard frame_a
        buf.append(frame_d)
        assert buf.frame_count == 3

        data = buf.flush()
        assert data == frame_b + frame_c + frame_d
        assert frame_a not in data

    def test_overflow_keeps_buffer_at_max_size(self):
        """Overflow never allows buffer to exceed max frames."""
        buf = AudioBuffer(max_duration_ms=100)  # 5 frames max
        for _ in range(20):
            buf.append(make_speech_frame())
        assert buf.frame_count == 5
        assert buf.duration_ms == 100

    def test_duration_ms_is_correct(self):
        buf = AudioBuffer()
        for _ in range(50):
            buf.append(make_silent_frame())
        assert buf.duration_ms == 1000  # 50 * 20ms = 1000ms

    def test_is_overflowing_just_below_max(self):
        """Buffer at max-1 frames is not overflowing."""
        buf = AudioBuffer(max_duration_ms=60)  # 3 frames max
        buf.append(make_silent_frame())
        buf.append(make_silent_frame())
        assert buf.is_overflowing() is False

    def test_custom_max_duration(self):
        buf = AudioBuffer(max_duration_ms=1000)  # 1 second = 50 frames
        for _ in range(50):
            buf.append(make_silent_frame())
        assert buf.is_overflowing() is True
        assert buf.frame_count == 50


# --- Integration: VAD + AudioBuffer ---

class TestVADAndBufferIntegration:
    """Tests for VAD and AudioBuffer working together."""

    def test_typical_utterance_flow(self):
        """Simulate: silence → speech → silence → end-of-utterance."""
        vad = VADProcessor(backend=EnergyVADBackend(), silence_threshold_ms=500)
        buf = AudioBuffer()

        # Pre-speech silence (should not trigger end)
        for _ in range(10):
            frame = make_silent_frame()
            result = vad.process_frame(frame)
            if result.is_speech:
                buf.append(frame)

        assert buf.is_empty()
        assert vad.is_speech_ended() is False

        # Speech begins
        speech_frames = []
        for _ in range(50):  # 1 second of speech
            frame = make_speech_frame()
            result = vad.process_frame(frame)
            buf.append(frame)
            speech_frames.append(frame)

        assert not buf.is_empty()
        assert vad.is_speech_ended() is False

        # Post-speech silence until end-of-utterance
        for _ in range(25):  # 500ms
            frame = make_silent_frame()
            result = vad.process_frame(frame)
            buf.append(frame)

        assert vad.is_speech_ended() is True

        # Flush buffer for STT
        audio_data = buf.flush()
        assert len(audio_data) > 0
        assert buf.is_empty()

        # Reset for next utterance
        vad.reset()
        assert vad.is_speech_ended() is False

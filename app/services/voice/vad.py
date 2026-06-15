"""Voice Activity Detection (VAD) processor.

Detects end-of-utterance by monitoring silence duration. Uses a pluggable
backend architecture: attempts webrtcvad first, falls back to a simple
energy-based detector.

Requirements: 3.3, 3.6
"""

from __future__ import annotations

import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Audio format constants
SAMPLE_RATE = 16000  # 16 kHz
SAMPLE_WIDTH = 2  # 16-bit PCM = 2 bytes per sample
FRAME_DURATION_MS = 20  # 20ms frames
FRAME_SIZE_BYTES = SAMPLE_RATE * SAMPLE_WIDTH * FRAME_DURATION_MS // 1000  # 640 bytes

# VAD configuration defaults
DEFAULT_SILENCE_THRESHOLD_MS = 500  # End-of-utterance after 500ms silence


@dataclass
class VADResult:
    """Result of processing a single audio frame through VAD."""

    is_speech: bool
    duration_silent_ms: int


class VADBackend(ABC):
    """Abstract base class for VAD detection backends."""

    @abstractmethod
    def is_speech(self, frame: bytes) -> bool:
        """Determine if a frame contains speech.

        Args:
            frame: Raw PCM audio frame (16kHz, 16-bit mono, 640 bytes for 20ms).

        Returns:
            True if the frame contains speech.
        """
        ...


class EnergyVADBackend(VADBackend):
    """Simple energy-based VAD fallback.

    Computes RMS energy of the frame and compares against a threshold.
    Works without external dependencies.
    """

    def __init__(self, energy_threshold: float = 300.0):
        """Initialize energy-based VAD.

        Args:
            energy_threshold: RMS energy threshold for speech detection.
                Values below this are considered silence.
        """
        self.energy_threshold = energy_threshold

    def is_speech(self, frame: bytes) -> bool:
        """Detect speech using RMS energy of the PCM samples."""
        if len(frame) < 2:
            return False

        num_samples = len(frame) // SAMPLE_WIDTH
        samples = struct.unpack(f"<{num_samples}h", frame[:num_samples * SAMPLE_WIDTH])

        if not samples:
            return False

        # Compute RMS energy
        sum_squares = sum(s * s for s in samples)
        rms = (sum_squares / num_samples) ** 0.5

        return rms >= self.energy_threshold


class WebRTCVADBackend(VADBackend):
    """WebRTC-based VAD backend using the webrtcvad library.

    Only available if webrtcvad is installed.
    """

    def __init__(self, aggressiveness: int = 2):
        """Initialize WebRTC VAD.

        Args:
            aggressiveness: VAD aggressiveness mode (0-3). Higher values
                are more aggressive about filtering non-speech.
        """
        import webrtcvad

        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame: bytes) -> bool:
        """Detect speech using WebRTC VAD."""
        return self._vad.is_speech(frame, SAMPLE_RATE)


def _create_default_backend() -> VADBackend:
    """Create the best available VAD backend.

    Tries webrtcvad first, falls back to energy-based.
    """
    try:
        return WebRTCVADBackend()
    except (ImportError, Exception):
        return EnergyVADBackend()


@dataclass
class VADProcessor:
    """Voice Activity Detection processor.

    Processes audio frames and detects end-of-utterance based on
    accumulated silence duration exceeding a configurable threshold.

    Audio format: 16kHz, 16-bit PCM mono, 20ms frames (640 bytes).
    """

    silence_threshold_ms: int = DEFAULT_SILENCE_THRESHOLD_MS
    backend: VADBackend = field(default_factory=_create_default_backend)
    _silent_frames: int = field(default=0, init=False, repr=False)
    _has_seen_speech: bool = field(default=False, init=False, repr=False)

    def process_frame(self, frame: bytes) -> VADResult:
        """Process an audio frame and return VAD result.

        Args:
            frame: Raw PCM audio data. Should be 640 bytes for a 20ms
                frame at 16kHz/16-bit mono.

        Returns:
            VADResult with speech detection and current silence duration.
        """
        speech_detected = self.backend.is_speech(frame)

        if speech_detected:
            self._has_seen_speech = True
            self._silent_frames = 0
        else:
            self._silent_frames += 1

        duration_silent_ms = self._silent_frames * FRAME_DURATION_MS

        return VADResult(
            is_speech=speech_detected,
            duration_silent_ms=duration_silent_ms,
        )

    def is_speech_ended(self) -> bool:
        """Check if speech has ended (silence exceeds threshold).

        Returns True only if we have previously seen speech and then
        accumulated silence exceeding the configured threshold.

        Returns:
            True if end-of-utterance detected.
        """
        if not self._has_seen_speech:
            return False

        duration_silent_ms = self._silent_frames * FRAME_DURATION_MS
        return duration_silent_ms >= self.silence_threshold_ms

    def reset(self) -> None:
        """Reset VAD state for a new utterance."""
        self._silent_frames = 0
        self._has_seen_speech = False

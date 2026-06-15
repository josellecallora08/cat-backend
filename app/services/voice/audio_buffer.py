"""Audio buffer for accumulating PCM frames until VAD signals speech end.

Provides a ring-buffer-like overflow mechanism that discards oldest frames
when the buffer exceeds the maximum duration (30 seconds).

Requirements: 3.3, 3.6
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from app.services.voice.vad import FRAME_DURATION_MS, FRAME_SIZE_BYTES, SAMPLE_RATE, SAMPLE_WIDTH

# Buffer limits
MAX_BUFFER_DURATION_MS = 30_000  # 30 seconds max
MAX_FRAMES = MAX_BUFFER_DURATION_MS // FRAME_DURATION_MS  # 1500 frames at 20ms each


@dataclass
class AudioBuffer:
    """Accumulates PCM audio frames for STT processing.

    Frames are appended until the caller flushes the buffer (typically
    triggered by VAD end-of-utterance detection). If the buffer exceeds
    30 seconds of audio, oldest frames are automatically discarded.

    Audio format: 16kHz, 16-bit PCM mono, 20ms frames (640 bytes each).
    """

    max_duration_ms: int = MAX_BUFFER_DURATION_MS
    _frames: deque = field(default_factory=deque, init=False, repr=False)

    @property
    def _max_frames(self) -> int:
        """Maximum number of frames before overflow."""
        return self.max_duration_ms // FRAME_DURATION_MS

    @property
    def duration_ms(self) -> int:
        """Current buffer duration in milliseconds."""
        return len(self._frames) * FRAME_DURATION_MS

    @property
    def frame_count(self) -> int:
        """Number of frames currently in the buffer."""
        return len(self._frames)

    def append(self, frame: bytes) -> None:
        """Append a PCM audio frame to the buffer.

        If the buffer would exceed max duration, the oldest frame is
        discarded to make room.

        Args:
            frame: Raw PCM audio frame data.
        """
        if len(self._frames) >= self._max_frames:
            # Overflow: discard oldest frame
            self._frames.popleft()

        self._frames.append(frame)

    def flush(self) -> bytes:
        """Return all accumulated PCM data and clear the buffer.

        Returns:
            Concatenated PCM audio bytes from all buffered frames.
        """
        data = b"".join(self._frames)
        self._frames.clear()
        return data

    def reset(self) -> None:
        """Clear the buffer without returning data."""
        self._frames.clear()

    def is_overflowing(self) -> bool:
        """Check if the buffer has reached its maximum capacity.

        Returns:
            True if the buffer contains 30+ seconds of audio.
        """
        return len(self._frames) >= self._max_frames

    def is_empty(self) -> bool:
        """Check if the buffer is empty.

        Returns:
            True if no frames are buffered.
        """
        return len(self._frames) == 0

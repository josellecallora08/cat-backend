"""In-memory event store with ring buffer and TTL-based eviction."""

import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone

from app.schemas.event import EventPayload


class EventStore:
    """In-memory ring buffer storing recent events for gap-fill replay.

    Events are stored with timestamps and evicted when they exceed the
    configured TTL. A monotonically increasing sequence counter is used
    to assign unique sequence numbers to each event.
    """

    def __init__(self, ttl_seconds: int = 300) -> None:
        """Initialize the event store.

        Args:
            ttl_seconds: Time-to-live in seconds for stored events.
                Events older than this are evicted on the next append.
        """
        self._ttl = timedelta(seconds=ttl_seconds)
        self._buffer: deque[tuple[datetime, EventPayload]] = deque()
        self._lock = asyncio.Lock()
        self._seq: int = 0

    async def next_seq(self) -> int:
        """Return and increment the global sequence counter.

        Acquires the asyncio lock to guarantee monotonically increasing
        sequence numbers across concurrent coroutines.

        Returns:
            The next sequence number (monotonically increasing).
        """
        async with self._lock:
            self._seq += 1
            return self._seq

    def append(self, event: EventPayload) -> None:
        """Store an event and evict entries older than TTL.

        Args:
            event: The event payload to store.
        """
        now = datetime.now(timezone.utc)
        cutoff = now - self._ttl

        # Evict expired entries from the front of the deque
        while self._buffer and self._buffer[0][0] <= cutoff:
            self._buffer.popleft()

        self._buffer.append((now, event))

    def get_after(self, last_seq: int) -> list[EventPayload] | None:
        """Return events with seq > last_seq.

        Args:
            last_seq: The last sequence number the client received.

        Returns:
            A list of events newer than last_seq, ordered by sequence number.
            Returns None if last_seq is older than the oldest stored event,
            indicating the client needs a full resync.
        """
        if not self._buffer:
            return []

        oldest_seq = self._buffer[0][1].seq

        if last_seq < oldest_seq:
            return None

        return [event for _, event in self._buffer if event.seq > last_seq]

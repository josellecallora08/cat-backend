"""TranscriptManager service for real-time transcript recording and persistence.

Manages buffering, ordering, and persistence of transcript entries during
active sessions. Uses the db_retry wrapper for resilient database writes.

Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import List
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Transcript
from app.services.db_retry import retry_db_operation

logger = logging.getLogger(__name__)


class TranscriptValidationError(Exception):
    """Raised when a transcript entry fails validation."""

    pass


class TranscriptManager:
    """Records utterances in real-time and persists completed transcripts.

    Maintains an in-memory buffer per session for real-time buffering,
    then flushes to the database via the retry wrapper on persist().
    """

    VALID_SPEAKERS = ("agent", "debtor")

    def __init__(self, db: AsyncSession) -> None:
        self._db = db
        self._buffer: dict[UUID, list[Transcript]] = defaultdict(list)
        self._sequence_counters: dict[UUID, int] = defaultdict(int)

    async def append_entry(
        self,
        session_id: UUID,
        speaker: str,
        text: str,
        timestamp: datetime,
    ) -> Transcript:
        """Validate and buffer a new transcript entry.

        Args:
            session_id: The session this entry belongs to.
            speaker: Must be "agent" or "debtor".
            text: The utterance text (must be non-empty).
            timestamp: Timestamp with millisecond precision.

        Returns:
            The created Transcript model instance.

        Raises:
            TranscriptValidationError: If speaker or text is invalid.
        """
        # Validate speaker
        if speaker not in self.VALID_SPEAKERS:
            raise TranscriptValidationError(
                f"Invalid speaker '{speaker}'. Must be one of {self.VALID_SPEAKERS}"
            )

        # Validate text is non-empty
        if not text or not text.strip():
            raise TranscriptValidationError("Utterance text must be non-empty")

        # Determine next sequence number for this session
        sequence_number = await self._get_next_sequence_number(session_id)

        # Create the Transcript model
        entry = Transcript(
            session_id=session_id,
            speaker=speaker,
            utterance_text=text,
            timestamp_ms=timestamp,
            sequence_number=sequence_number,
        )

        # Add to buffer
        self._buffer[session_id].append(entry)

        logger.debug(
            "Buffered transcript entry for session %s: speaker=%s, seq=%d",
            session_id,
            speaker,
            sequence_number,
        )

        return entry

    async def get_transcript(self, session_id: UUID) -> List[Transcript]:
        """Return all transcript entries for a session ordered by sequence_number.

        Queries the database for persisted entries ordered ascending by
        sequence_number.

        Args:
            session_id: The session to retrieve the transcript for.

        Returns:
            List of Transcript entries ordered by sequence_number ascending.
        """
        stmt = (
            select(Transcript)
            .where(Transcript.session_id == session_id)
            .order_by(Transcript.sequence_number.asc())
        )
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def get_agent_utterance_count(self, session_id: UUID) -> int:
        """Count transcript entries where speaker is 'agent'.

        Args:
            session_id: The session to count agent utterances for.

        Returns:
            Number of agent utterances in the session.
        """
        stmt = (
            select(func.count())
            .select_from(Transcript)
            .where(
                Transcript.session_id == session_id,
                Transcript.speaker == "agent",
            )
        )
        result = await self._db.execute(stmt)
        return result.scalar_one()

    async def persist(self, session_id: UUID) -> None:
        """Flush buffered entries to the database using the db_retry wrapper.

        Moves all buffered entries for the given session into the database
        with retry logic for resilience.

        Args:
            session_id: The session whose buffered entries to persist.
        """
        entries = self._buffer.get(session_id, [])
        if not entries:
            logger.debug("No buffered entries to persist for session %s", session_id)
            return

        async def _flush_entries() -> None:
            for entry in entries:
                self._db.add(entry)
            await self._db.commit()

        await retry_db_operation(
            _flush_entries,
            session_id=str(session_id),
            data=entries,
        )

        # Clear the buffer after successful persistence
        self._buffer.pop(session_id, None)
        logger.info(
            "Persisted %d transcript entries for session %s",
            len(entries),
            session_id,
        )

    async def _get_next_sequence_number(self, session_id: UUID) -> int:
        """Determine the next sequence number for a session.

        Considers both buffered entries and any already-persisted entries
        in the database.

        Args:
            session_id: The session to get the next sequence number for.

        Returns:
            The next sequence number to assign.
        """
        # Check if we already have a counter in memory
        if session_id in self._sequence_counters and self._sequence_counters[session_id] > 0:
            next_seq = self._sequence_counters[session_id]
            self._sequence_counters[session_id] += 1
            return next_seq

        # Query the database for the current max sequence number
        stmt = (
            select(func.max(Transcript.sequence_number))
            .where(Transcript.session_id == session_id)
        )
        result = await self._db.execute(stmt)
        db_max = result.scalar_one_or_none()

        # Consider buffered entries as well
        buffer_max = -1
        if session_id in self._buffer and self._buffer[session_id]:
            buffer_max = max(e.sequence_number for e in self._buffer[session_id])

        # Next sequence is max of db and buffer + 1
        current_max = max(db_max if db_max is not None else -1, buffer_max)
        next_seq = current_max + 1

        # Store the counter for future calls
        self._sequence_counters[session_id] = next_seq + 1

        return next_seq

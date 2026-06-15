"""Unit tests for TranscriptManager service.

Tests use an async SQLite in-memory database to validate transcript
buffering, ordering, persistence, and validation behavior.
"""

import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base
from app.models import Scenario, Session, Transcript
from app.services.transcript_manager import TranscriptManager, TranscriptValidationError


@pytest_asyncio.fixture
async def async_engine():
    """Create an async SQLite engine for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(async_engine):
    """Provide a fresh async database session for each test."""
    session_factory = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def session_id(db_session: AsyncSession):
    """Create a scenario and session in the DB and return the session ID."""
    scenario = Scenario(
        id=uuid.uuid4(),
        name="Test Scenario",
        scenario_type="FINANCIAL_HARDSHIP",
        description="A test scenario",
        debtor_profile={"name": "John", "outstanding_balance": 1000, "days_past_due": 30,
                        "personality_profile": "cooperative", "conversation_goal": "pay"},
    )
    db_session.add(scenario)
    await db_session.flush()

    sess = Session(
        id=uuid.uuid4(),
        scenario_id=scenario.id,
        agent_id=uuid.uuid4(),
        status="active",
    )
    db_session.add(sess)
    await db_session.commit()
    return sess.id


@pytest_asyncio.fixture
async def transcript_manager(db_session: AsyncSession):
    """Create a TranscriptManager instance."""
    return TranscriptManager(db=db_session)


class TestAppendEntry:
    """Tests for TranscriptManager.append_entry."""

    @pytest.mark.asyncio
    async def test_append_valid_agent_entry(self, transcript_manager, session_id):
        """Valid agent entry is buffered with correct sequence number."""
        ts = datetime.now(timezone.utc)
        entry = await transcript_manager.append_entry(session_id, "agent", "Hello, this is collections.", ts)

        assert entry.speaker == "agent"
        assert entry.utterance_text == "Hello, this is collections."
        assert entry.timestamp_ms == ts
        assert entry.sequence_number == 0
        assert entry.session_id == session_id

    @pytest.mark.asyncio
    async def test_append_valid_debtor_entry(self, transcript_manager, session_id):
        """Valid debtor entry is buffered with correct sequence number."""
        ts = datetime.now(timezone.utc)
        entry = await transcript_manager.append_entry(session_id, "debtor", "I can't pay right now.", ts)

        assert entry.speaker == "debtor"
        assert entry.utterance_text == "I can't pay right now."
        assert entry.sequence_number == 0

    @pytest.mark.asyncio
    async def test_append_multiple_entries_increments_sequence(self, transcript_manager, session_id):
        """Multiple entries get incrementing sequence numbers."""
        ts = datetime.now(timezone.utc)

        e1 = await transcript_manager.append_entry(session_id, "agent", "Hello", ts)
        e2 = await transcript_manager.append_entry(session_id, "debtor", "Hi", ts)
        e3 = await transcript_manager.append_entry(session_id, "agent", "How can I help?", ts)

        assert e1.sequence_number == 0
        assert e2.sequence_number == 1
        assert e3.sequence_number == 2

    @pytest.mark.asyncio
    async def test_append_invalid_speaker_raises(self, transcript_manager, session_id):
        """Invalid speaker value raises TranscriptValidationError."""
        ts = datetime.now(timezone.utc)
        with pytest.raises(TranscriptValidationError, match="Invalid speaker"):
            await transcript_manager.append_entry(session_id, "customer", "Hello", ts)

    @pytest.mark.asyncio
    async def test_append_empty_text_raises(self, transcript_manager, session_id):
        """Empty text raises TranscriptValidationError."""
        ts = datetime.now(timezone.utc)
        with pytest.raises(TranscriptValidationError, match="non-empty"):
            await transcript_manager.append_entry(session_id, "agent", "", ts)

    @pytest.mark.asyncio
    async def test_append_whitespace_only_text_raises(self, transcript_manager, session_id):
        """Whitespace-only text raises TranscriptValidationError."""
        ts = datetime.now(timezone.utc)
        with pytest.raises(TranscriptValidationError, match="non-empty"):
            await transcript_manager.append_entry(session_id, "agent", "   ", ts)


class TestGetTranscript:
    """Tests for TranscriptManager.get_transcript."""

    @pytest.mark.asyncio
    async def test_get_empty_transcript(self, transcript_manager, session_id):
        """Empty session returns empty list."""
        result = await transcript_manager.get_transcript(session_id)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_transcript_returns_ordered_entries(self, transcript_manager, session_id, db_session):
        """Persisted entries are returned ordered by sequence_number."""
        ts = datetime.now(timezone.utc)

        # Manually insert entries out of order
        entries = [
            Transcript(session_id=session_id, speaker="debtor", utterance_text="Response", timestamp_ms=ts, sequence_number=2),
            Transcript(session_id=session_id, speaker="agent", utterance_text="Hello", timestamp_ms=ts, sequence_number=0),
            Transcript(session_id=session_id, speaker="agent", utterance_text="How are you?", timestamp_ms=ts, sequence_number=1),
        ]
        for e in entries:
            db_session.add(e)
        await db_session.commit()

        result = await transcript_manager.get_transcript(session_id)

        assert len(result) == 3
        assert result[0].sequence_number == 0
        assert result[1].sequence_number == 1
        assert result[2].sequence_number == 2
        assert result[0].utterance_text == "Hello"
        assert result[1].utterance_text == "How are you?"
        assert result[2].utterance_text == "Response"


class TestGetAgentUtteranceCount:
    """Tests for TranscriptManager.get_agent_utterance_count."""

    @pytest.mark.asyncio
    async def test_count_zero_when_empty(self, transcript_manager, session_id):
        """No entries returns count of 0."""
        count = await transcript_manager.get_agent_utterance_count(session_id)
        assert count == 0

    @pytest.mark.asyncio
    async def test_count_only_agent_entries(self, transcript_manager, session_id, db_session):
        """Only counts entries where speaker='agent'."""
        ts = datetime.now(timezone.utc)

        entries = [
            Transcript(session_id=session_id, speaker="agent", utterance_text="Hello", timestamp_ms=ts, sequence_number=0),
            Transcript(session_id=session_id, speaker="debtor", utterance_text="Hi", timestamp_ms=ts, sequence_number=1),
            Transcript(session_id=session_id, speaker="agent", utterance_text="Can we talk?", timestamp_ms=ts, sequence_number=2),
            Transcript(session_id=session_id, speaker="debtor", utterance_text="Sure", timestamp_ms=ts, sequence_number=3),
            Transcript(session_id=session_id, speaker="agent", utterance_text="Great", timestamp_ms=ts, sequence_number=4),
        ]
        for e in entries:
            db_session.add(e)
        await db_session.commit()

        count = await transcript_manager.get_agent_utterance_count(session_id)
        assert count == 3


class TestPersist:
    """Tests for TranscriptManager.persist."""

    @pytest.mark.asyncio
    async def test_persist_flushes_buffer_to_db(self, transcript_manager, session_id, db_session):
        """Persisted entries are retrievable from the database."""
        ts = datetime.now(timezone.utc)

        await transcript_manager.append_entry(session_id, "agent", "Hello", ts)
        await transcript_manager.append_entry(session_id, "debtor", "Hi there", ts)

        # Before persist, DB should be empty for this session
        result = await transcript_manager.get_transcript(session_id)
        assert len(result) == 0

        # Persist
        await transcript_manager.persist(session_id)

        # After persist, entries should be in DB
        result = await transcript_manager.get_transcript(session_id)
        assert len(result) == 2
        assert result[0].speaker == "agent"
        assert result[1].speaker == "debtor"

    @pytest.mark.asyncio
    async def test_persist_clears_buffer(self, transcript_manager, session_id):
        """After persist, the internal buffer is cleared."""
        ts = datetime.now(timezone.utc)

        await transcript_manager.append_entry(session_id, "agent", "Hello", ts)
        await transcript_manager.persist(session_id)

        # Buffer should be empty
        assert session_id not in transcript_manager._buffer or len(transcript_manager._buffer[session_id]) == 0

    @pytest.mark.asyncio
    async def test_persist_empty_buffer_is_noop(self, transcript_manager, session_id):
        """Persisting with no buffered entries does nothing."""
        # Should not raise
        await transcript_manager.persist(session_id)

    @pytest.mark.asyncio
    async def test_sequence_continues_after_persist(self, transcript_manager, session_id):
        """Sequence numbers continue correctly after a persist."""
        ts = datetime.now(timezone.utc)

        await transcript_manager.append_entry(session_id, "agent", "Hello", ts)
        await transcript_manager.append_entry(session_id, "debtor", "Hi", ts)
        await transcript_manager.persist(session_id)

        # New entry should get sequence_number = 2
        entry = await transcript_manager.append_entry(session_id, "agent", "Continuing...", ts)
        assert entry.sequence_number == 2

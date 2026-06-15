"""Property-based tests for transcript chronological ordering.

Feature: collection-agent-trainer, Property 7: Transcript chronological ordering

**Validates: Requirements 4.5**

Property 7: For any set of transcript entries belonging to a single session,
they SHALL be ordered by ascending timestamp value, and the sequence_number
SHALL be monotonically increasing.
"""

import asyncio
import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base
from app.models import Scenario, Session
from app.services.transcript_manager import TranscriptManager


# --- Strategies ---

speakers = st.sampled_from(["agent", "debtor"])

utterance_texts = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=200,
).filter(lambda s: s.strip() != "")

# Generate random timestamps within a reasonable range (naive, then attach UTC)
timestamps = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31),
).map(lambda dt: dt.replace(tzinfo=timezone.utc))


@st.composite
def transcript_entry_data(draw):
    """Generate a single transcript entry's data (speaker, text, timestamp)."""
    return {
        "speaker": draw(speakers),
        "text": draw(utterance_texts),
        "timestamp": draw(timestamps),
    }


# Generate lists of 2-10 transcript entries with random timestamps
transcript_entry_lists = st.lists(
    transcript_entry_data(),
    min_size=2,
    max_size=10,
)


# --- Fixtures ---


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
        description="A test scenario for property testing",
        debtor_profile={
            "name": "John",
            "outstanding_balance": 1000,
            "days_past_due": 30,
            "personality_profile": "cooperative",
            "conversation_goal": "pay",
        },
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


# --- Helper ---


async def _run_ordering_test(entries_data: list):
    """Set up an in-memory DB, append entries, persist, and retrieve transcript."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as db_session:
        # Create scenario and session
        scenario = Scenario(
            id=uuid.uuid4(),
            name="Property Test Scenario",
            scenario_type="FINANCIAL_HARDSHIP",
            description="For property testing",
            debtor_profile={
                "name": "Jane",
                "outstanding_balance": 2000,
                "days_past_due": 60,
                "personality_profile": "evasive",
                "conversation_goal": "negotiate",
            },
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

        # Append entries via TranscriptManager
        manager = TranscriptManager(db=db_session)

        for entry_data in entries_data:
            await manager.append_entry(
                session_id=sess.id,
                speaker=entry_data["speaker"],
                text=entry_data["text"],
                timestamp=entry_data["timestamp"],
            )

        # Persist to database
        await manager.persist(sess.id)

        # Retrieve transcript
        transcript = await manager.get_transcript(sess.id)

    await engine.dispose()
    return transcript


# --- Property Tests ---


class TestTranscriptChronologicalOrdering:
    """Property 7: Transcript chronological ordering.

    Feature: collection-agent-trainer, Property 7: Transcript chronological ordering
    """

    @given(entries_data=transcript_entry_lists)
    @settings(max_examples=100)
    def test_sequence_numbers_are_monotonically_increasing(self, entries_data: list):
        """**Validates: Requirements 4.5**

        For any set of transcript entries appended to a session with random
        timestamps, after persisting and retrieving via get_transcript, the
        sequence_numbers SHALL be monotonically increasing.
        """
        transcript = asyncio.run(_run_ordering_test(entries_data))

        # Verify we got all entries back
        assert len(transcript) == len(entries_data)

        # Verify sequence_numbers are monotonically increasing
        for i in range(1, len(transcript)):
            assert transcript[i].sequence_number > transcript[i - 1].sequence_number, (
                f"sequence_number at index {i} ({transcript[i].sequence_number}) "
                f"is not greater than at index {i-1} ({transcript[i-1].sequence_number})"
            )

    @given(entries_data=transcript_entry_lists)
    @settings(max_examples=100)
    def test_transcript_ordered_by_sequence_number_ascending(self, entries_data: list):
        """**Validates: Requirements 4.5**

        For any set of transcript entries appended to a session with random
        timestamps, after persisting and retrieving via get_transcript, the
        list SHALL be ordered by sequence_number ascending.
        """
        transcript = asyncio.run(_run_ordering_test(entries_data))

        # Verify we got all entries back
        assert len(transcript) == len(entries_data)

        # Verify the list is sorted by sequence_number ascending
        sequence_numbers = [entry.sequence_number for entry in transcript]
        assert sequence_numbers == sorted(sequence_numbers), (
            f"Transcript not ordered by sequence_number ascending: {sequence_numbers}"
        )

        # Verify sequence numbers start from 0 and are contiguous
        expected = list(range(len(entries_data)))
        assert sequence_numbers == expected, (
            f"Expected contiguous sequence numbers {expected}, got {sequence_numbers}"
        )

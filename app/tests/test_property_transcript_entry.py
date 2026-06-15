"""Property-based tests for transcript entry structural completeness.

Feature: collection-agent-trainer, Property 6: Transcript entry structural completeness

**Validates: Requirements 4.3**

Property 6: For any transcript entry produced by the system, it SHALL have a non-empty
speaker identifier (either "agent" or "debtor"), a timestamp with millisecond precision,
and non-empty utterance_text.
"""

import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base
from app.models import Scenario, Session
from app.services.transcript_manager import TranscriptManager, TranscriptValidationError


# --- Strategies ---

# Valid speakers
valid_speakers = st.sampled_from(["agent", "debtor"])

# Non-empty text (at least 1 printable character, stripping won't produce empty)
valid_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z")),
    min_size=1,
    max_size=500,
).filter(lambda t: t.strip() != "")

# Datetime timestamps with millisecond precision
valid_timestamps = st.datetimes(
    min_value=datetime(2000, 1, 1),
    max_value=datetime(2030, 12, 31),
    timezones=st.just(timezone.utc),
)

# Invalid speakers (anything not "agent" or "debtor")
invalid_speakers = st.text(min_size=1, max_size=50).filter(
    lambda s: s not in ("agent", "debtor")
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
        description="A test scenario",
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


@pytest_asyncio.fixture
async def transcript_manager(db_session: AsyncSession):
    """Create a TranscriptManager instance."""
    return TranscriptManager(db=db_session)


# --- Property Tests ---


class TestTranscriptEntryStructuralCompleteness:
    """Property 6: Transcript entry structural completeness.

    Feature: collection-agent-trainer, Property 6: Transcript entry structural completeness
    """

    @given(speaker=valid_speakers, text=valid_text, timestamp=valid_timestamps)
    @settings(max_examples=100)
    def test_valid_entries_have_complete_structure(
        self, speaker: str, text: str, timestamp: datetime
    ):
        """Valid entries produced by TranscriptManager have valid speaker, non-empty text, and datetime timestamp.

        **Validates: Requirements 4.3**
        """
        import asyncio

        async def _run():
            engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            session_factory = async_sessionmaker(
                engine, class_=AsyncSession, expire_on_commit=False
            )
            async with session_factory() as db_session:
                # Create prerequisite records
                scenario = Scenario(
                    id=uuid.uuid4(),
                    name="Test",
                    scenario_type="FINANCIAL_HARDSHIP",
                    description="test",
                    debtor_profile={"name": "John", "outstanding_balance": 1000,
                                    "days_past_due": 30, "personality_profile": "cooperative",
                                    "conversation_goal": "pay"},
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

                manager = TranscriptManager(db=db_session)
                entry = await manager.append_entry(sess.id, speaker, text, timestamp)

                # Verify structural completeness
                assert entry.speaker in ("agent", "debtor"), (
                    f"Speaker must be 'agent' or 'debtor', got '{entry.speaker}'"
                )
                assert entry.utterance_text and entry.utterance_text.strip() != "", (
                    f"Utterance text must be non-empty, got '{entry.utterance_text}'"
                )
                assert isinstance(entry.timestamp_ms, datetime), (
                    f"Timestamp must be a datetime, got {type(entry.timestamp_ms)}"
                )

            await engine.dispose()

        asyncio.run(_run())

    @given(speaker=invalid_speakers)
    @settings(max_examples=100)
    def test_invalid_speakers_are_rejected(self, speaker: str):
        """Invalid speaker values are rejected by TranscriptManager.

        **Validates: Requirements 4.3**
        """
        import asyncio

        async def _run():
            engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            session_factory = async_sessionmaker(
                engine, class_=AsyncSession, expire_on_commit=False
            )
            async with session_factory() as db_session:
                scenario = Scenario(
                    id=uuid.uuid4(),
                    name="Test",
                    scenario_type="FINANCIAL_HARDSHIP",
                    description="test",
                    debtor_profile={"name": "John", "outstanding_balance": 1000,
                                    "days_past_due": 30, "personality_profile": "cooperative",
                                    "conversation_goal": "pay"},
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

                manager = TranscriptManager(db=db_session)
                timestamp = datetime.now(timezone.utc)

                with pytest.raises(TranscriptValidationError):
                    await manager.append_entry(sess.id, speaker, "Valid text", timestamp)

            await engine.dispose()

        asyncio.run(_run())

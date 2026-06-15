"""Property-based tests for session artifact association.

Feature: collection-agent-trainer, Property 15: Session artifact association

For any training artifact (evaluation, coaching report, learning plan, transcript entry)
created by the system, it SHALL reference a valid, existing session_id.

Validates: Requirements 8.2
"""

import uuid
from datetime import datetime, timezone

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base
from app.models import (
    CoachingReport,
    Evaluation,
    LearningPlan,
    Scenario,
    Session,
    Transcript,
)


# --- Fixtures ---


@pytest.fixture
async def async_db():
    """Create an in-memory SQLite database for testing with foreign keys enabled."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)

    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


# --- Strategies ---

# Random text for artifact content
artifact_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=1,
    max_size=100,
).filter(lambda s: s.strip() != "")

# Random scores
scores = st.integers(min_value=0, max_value=100)

# Random speaker values
speakers = st.sampled_from(["agent", "debtor"])

# Random sequence numbers
sequence_numbers = st.integers(min_value=0, max_value=1000)

# Evaluation category scores
category_scores_strategy = st.lists(
    st.fixed_dictionaries({
        "category": st.sampled_from([
            "call_opening", "compliance", "empathy_communication", "negotiation_resolution"
        ]),
        "score": scores,
    }),
    min_size=1,
    max_size=4,
)

# Strengths and weaknesses
strength_weakness_items = st.lists(
    st.fixed_dictionaries({
        "description": artifact_text,
        "category": st.sampled_from([
            "call_opening", "compliance", "empathy_communication", "negotiation_resolution"
        ]),
        "transcript_excerpt": artifact_text,
    }),
    min_size=1,
    max_size=5,
)

# Mistakes by category
mistakes_strategy = st.fixed_dictionaries({
    "call_opening": st.lists(
        st.fixed_dictionaries({
            "transcript_position": st.integers(min_value=0, max_value=50),
            "transcript_excerpt": artifact_text,
            "explanation": artifact_text,
            "recommended_alternative": artifact_text,
        }),
        min_size=0,
        max_size=3,
    ),
    "compliance": st.lists(
        st.fixed_dictionaries({
            "transcript_position": st.integers(min_value=0, max_value=50),
            "transcript_excerpt": artifact_text,
            "explanation": artifact_text,
            "recommended_alternative": artifact_text,
        }),
        min_size=0,
        max_size=3,
    ),
})

# Weak competencies for learning plans
weak_competencies_strategy = st.lists(
    st.fixed_dictionaries({
        "category": st.sampled_from([
            "call_opening", "compliance", "empathy_communication", "negotiation_resolution"
        ]),
        "score": st.integers(min_value=0, max_value=69),
        "recommended_scenario": st.sampled_from([
            "Financial Hardship", "Payment Arrangement",
            "Compliance Fundamentals", "Call Opening Basics",
        ]),
    }),
    min_size=0,
    max_size=4,
)


# --- Helpers ---


def _make_scenario() -> Scenario:
    """Create a valid Scenario instance for testing."""
    return Scenario(
        id=uuid.uuid4(),
        name="Test Scenario",
        scenario_type="FINANCIAL_HARDSHIP",
        description="A test scenario",
        debtor_profile={
            "name": "Test Debtor",
            "outstanding_balance": "5000.00",
            "days_past_due": 30,
            "personality_profile": "cooperative",
            "conversation_goal": "negotiate payment",
        },
        is_active=True,
    )


def _make_session(scenario_id: uuid.UUID) -> Session:
    """Create a valid Session instance for testing."""
    return Session(
        id=uuid.uuid4(),
        scenario_id=scenario_id,
        agent_id=uuid.uuid4(),
        status="completed",
        persona_context={"name": "Test Persona", "emotional_state": 3},
    )


# --- Property Tests ---


class TestSessionArtifactAssociation:
    """Property 15: Session artifact association.

    Feature: collection-agent-trainer, Property 15: Session artifact association
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        overall_score=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
        cat_scores=category_scores_strategy,
        strengths=strength_weakness_items,
        weaknesses=strength_weakness_items,
        is_too_short=st.booleans(),
    )
    @pytest.mark.asyncio
    async def test_evaluation_references_valid_session(
        self,
        async_db: AsyncSession,
        overall_score: float,
        cat_scores: list,
        strengths: list,
        weaknesses: list,
        is_too_short: bool,
    ):
        """**Validates: Requirements 8.2**

        For any evaluation artifact created, its session_id SHALL reference
        a valid, existing session in the sessions table.
        """
        # Create scenario and session
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.flush()

        session = _make_session(scenario.id)
        async_db.add(session)
        await async_db.flush()

        # Create evaluation artifact
        evaluation = Evaluation(
            id=uuid.uuid4(),
            session_id=session.id,
            overall_score=overall_score,
            category_scores=cat_scores,
            strengths=strengths,
            weaknesses=weaknesses,
            is_too_short=is_too_short,
        )
        async_db.add(evaluation)
        await async_db.flush()

        # Verify the evaluation's session_id references an existing session
        stmt = select(Session).where(Session.id == evaluation.session_id)
        result = await async_db.execute(stmt)
        referenced_session = result.scalar_one_or_none()

        assert referenced_session is not None, (
            f"Evaluation {evaluation.id} references session_id {evaluation.session_id} "
            "which does not exist in the sessions table"
        )
        assert referenced_session.id == session.id

        # Cleanup for next iteration
        await async_db.rollback()

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        mistakes=mistakes_strategy,
        total_mistakes=st.integers(min_value=0, max_value=20),
        no_mistakes=st.booleans(),
    )
    @pytest.mark.asyncio
    async def test_coaching_report_references_valid_session(
        self,
        async_db: AsyncSession,
        mistakes: dict,
        total_mistakes: int,
        no_mistakes: bool,
    ):
        """**Validates: Requirements 8.2**

        For any coaching report artifact created, its session_id SHALL reference
        a valid, existing session in the sessions table.
        """
        # Create scenario and session
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.flush()

        session = _make_session(scenario.id)
        async_db.add(session)
        await async_db.flush()

        # Create coaching report artifact
        coaching_report = CoachingReport(
            id=uuid.uuid4(),
            session_id=session.id,
            mistakes_by_category=mistakes,
            total_mistakes=total_mistakes,
            no_mistakes=no_mistakes,
        )
        async_db.add(coaching_report)
        await async_db.flush()

        # Verify the coaching report's session_id references an existing session
        stmt = select(Session).where(Session.id == coaching_report.session_id)
        result = await async_db.execute(stmt)
        referenced_session = result.scalar_one_or_none()

        assert referenced_session is not None, (
            f"CoachingReport {coaching_report.id} references session_id "
            f"{coaching_report.session_id} which does not exist in the sessions table"
        )
        assert referenced_session.id == session.id

        # Cleanup for next iteration
        await async_db.rollback()

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        weak_competencies=weak_competencies_strategy,
        all_passing=st.booleans(),
    )
    @pytest.mark.asyncio
    async def test_learning_plan_references_valid_session(
        self,
        async_db: AsyncSession,
        weak_competencies: list,
        all_passing: bool,
    ):
        """**Validates: Requirements 8.2**

        For any learning plan artifact created, its session_id SHALL reference
        a valid, existing session in the sessions table.
        """
        # Create scenario and session
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.flush()

        session = _make_session(scenario.id)
        async_db.add(session)
        await async_db.flush()

        # Create learning plan artifact
        learning_plan = LearningPlan(
            id=uuid.uuid4(),
            session_id=session.id,
            agent_id=uuid.uuid4(),
            weak_competencies=weak_competencies,
            all_passing=all_passing,
        )
        async_db.add(learning_plan)
        await async_db.flush()

        # Verify the learning plan's session_id references an existing session
        stmt = select(Session).where(Session.id == learning_plan.session_id)
        result = await async_db.execute(stmt)
        referenced_session = result.scalar_one_or_none()

        assert referenced_session is not None, (
            f"LearningPlan {learning_plan.id} references session_id "
            f"{learning_plan.session_id} which does not exist in the sessions table"
        )
        assert referenced_session.id == session.id

        # Cleanup for next iteration
        await async_db.rollback()

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        speaker=speakers,
        utterance_text=artifact_text,
        seq_num=sequence_numbers,
    )
    @pytest.mark.asyncio
    async def test_transcript_entry_references_valid_session(
        self,
        async_db: AsyncSession,
        speaker: str,
        utterance_text: str,
        seq_num: int,
    ):
        """**Validates: Requirements 8.2**

        For any transcript entry artifact created, its session_id SHALL reference
        a valid, existing session in the sessions table.
        """
        # Create scenario and session
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.flush()

        session = _make_session(scenario.id)
        async_db.add(session)
        await async_db.flush()

        # Create transcript entry artifact
        transcript_entry = Transcript(
            id=uuid.uuid4(),
            session_id=session.id,
            speaker=speaker,
            utterance_text=utterance_text,
            timestamp_ms=datetime.now(timezone.utc),
            sequence_number=seq_num,
        )
        async_db.add(transcript_entry)
        await async_db.flush()

        # Verify the transcript entry's session_id references an existing session
        stmt = select(Session).where(Session.id == transcript_entry.session_id)
        result = await async_db.execute(stmt)
        referenced_session = result.scalar_one_or_none()

        assert referenced_session is not None, (
            f"Transcript {transcript_entry.id} references session_id "
            f"{transcript_entry.session_id} which does not exist in the sessions table"
        )
        assert referenced_session.id == session.id

        # Cleanup for next iteration
        await async_db.rollback()

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        num_transcripts=st.integers(min_value=1, max_value=5),
        speaker=speakers,
    )
    @pytest.mark.asyncio
    async def test_all_artifacts_for_session_reference_same_valid_session(
        self,
        async_db: AsyncSession,
        num_transcripts: int,
        speaker: str,
    ):
        """**Validates: Requirements 8.2**

        When multiple artifacts (evaluation, coaching report, learning plan,
        and transcript entries) are created for a session, ALL of them SHALL
        reference the same valid, existing session_id.
        """
        # Create scenario and session
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.flush()

        session = _make_session(scenario.id)
        async_db.add(session)
        await async_db.flush()

        # Create all artifact types for this session
        evaluation = Evaluation(
            id=uuid.uuid4(),
            session_id=session.id,
            overall_score=75.0,
            category_scores=[{"category": "compliance", "score": 80}],
            strengths=[{"description": "Good opening", "category": "call_opening", "transcript_excerpt": "Hello"}],
            weaknesses=[{"description": "Missed compliance", "category": "compliance", "transcript_excerpt": "..."}],
            is_too_short=False,
        )
        async_db.add(evaluation)

        coaching_report = CoachingReport(
            id=uuid.uuid4(),
            session_id=session.id,
            mistakes_by_category={"compliance": []},
            total_mistakes=0,
            no_mistakes=True,
        )
        async_db.add(coaching_report)

        learning_plan = LearningPlan(
            id=uuid.uuid4(),
            session_id=session.id,
            agent_id=uuid.uuid4(),
            weak_competencies=[],
            all_passing=True,
        )
        async_db.add(learning_plan)

        # Create transcript entries
        for i in range(num_transcripts):
            transcript = Transcript(
                id=uuid.uuid4(),
                session_id=session.id,
                speaker=speaker,
                utterance_text=f"Utterance {i}",
                timestamp_ms=datetime.now(timezone.utc),
                sequence_number=i,
            )
            async_db.add(transcript)

        await async_db.flush()

        # Verify ALL artifacts reference the same valid session
        all_artifact_session_ids = [
            evaluation.session_id,
            coaching_report.session_id,
            learning_plan.session_id,
        ]

        # Query all transcripts for this session
        stmt = select(Transcript).where(Transcript.session_id == session.id)
        result = await async_db.execute(stmt)
        transcripts = result.scalars().all()
        all_artifact_session_ids.extend([t.session_id for t in transcripts])

        # All session_ids should be identical and reference the existing session
        for artifact_session_id in all_artifact_session_ids:
            assert artifact_session_id == session.id

        # Verify the session actually exists
        stmt = select(Session).where(Session.id == session.id)
        result = await async_db.execute(stmt)
        referenced_session = result.scalar_one_or_none()
        assert referenced_session is not None

        # Cleanup for next iteration
        await async_db.rollback()

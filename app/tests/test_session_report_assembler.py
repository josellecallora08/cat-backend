"""Unit tests for deterministic session report assembly.

Feature: session-report-generation, task 2.2.

Validates: Requirements 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.12, 2.5
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload
from sqlalchemy import select

from app.database import Base
from app.models import CoachingReport, Evaluation, LearningPlan, Scenario, Session, Transcript
from app.services.session_report_assembler import (
    assemble_report_payload,
    canonical_serialize,
    compute_content_hash,
)


@pytest.fixture
async def async_db():
    """In-memory SQLite database with foreign keys enabled."""
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


def _make_scenario() -> Scenario:
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


def _make_session(scenario_id: uuid.UUID, **overrides) -> Session:
    now = datetime.now(timezone.utc)
    defaults = dict(
        id=uuid.uuid4(),
        scenario_id=scenario_id,
        agent_id=uuid.uuid4(),
        status="completed",
        persona_context={"name": "Test Persona", "communication_style": "calm", "emotional_state": 3},
        created_at=now - timedelta(minutes=10),
        ended_at=now,
    )
    defaults.update(overrides)
    return Session(**defaults)


async def _reload_session(async_db: AsyncSession, session_id: uuid.UUID) -> Session:
    """Reload a session the same way session_service.get_session does."""
    stmt = (
        select(Session)
        .options(selectinload(Session.campaign))
        .where(Session.id == session_id)
    )
    result = await async_db.execute(stmt)
    return result.scalar_one()


@pytest.mark.asyncio
async def test_assembles_minimal_session_with_no_artifacts(async_db):
    """A completed session with no artifacts yields explicit absent sections."""
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()

    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.commit()

    reloaded = await _reload_session(async_db, session.id)
    payload = await assemble_report_payload(async_db, reloaded)

    assert payload.summary.session_id == session.id
    assert payload.summary.scenario_id == scenario.id
    assert payload.summary.duration_seconds == pytest.approx(600.0, abs=1.0)
    assert payload.transcript.available is True
    assert payload.transcript.entries == []
    assert payload.transcript.reason_code == "empty_transcript"
    assert payload.evaluation.available is False
    assert payload.evaluation.reason
    assert payload.evaluation.reason_code == "artifact_missing"
    assert payload.coaching.available is False
    assert payload.learning_plan.available is False


@pytest.mark.asyncio
async def test_transcript_entries_ordered_by_sequence(async_db):
    """Transcript entries are returned strictly in sequence order."""
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.flush()

    now = datetime.now(timezone.utc)
    for seq, speaker in [(2, "debtor"), (0, "agent"), (1, "debtor")]:
        async_db.add(Transcript(
            id=uuid.uuid4(), session_id=session.id, speaker=speaker,
            utterance_text=f"utterance {seq}", timestamp_ms=now, sequence_number=seq,
        ))
    await async_db.commit()

    reloaded = await _reload_session(async_db, session.id)
    payload = await assemble_report_payload(async_db, reloaded)

    sequences = [e.sequence_number for e in payload.transcript.entries]
    assert sequences == [0, 1, 2]


@pytest.mark.asyncio
async def test_legacy_evaluation_branch_used_without_canonical_result(async_db):
    """Evaluation without rubric_result.categories uses the legacy branch."""
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.flush()

    async_db.add(Evaluation(
        id=uuid.uuid4(), session_id=session.id, overall_score=72.5,
        category_scores=[{"category": "compliance", "score": 80, "strengths": [], "weaknesses": []}],
        strengths=[{"description": "Good opening", "category": "call_opening", "transcript_excerpt": "Hi"}],
        weaknesses=[{"description": "Missed disclosure", "category": "compliance", "transcript_excerpt": "..."}],
        is_too_short=False,
    ))
    await async_db.commit()

    reloaded = await _reload_session(async_db, session.id)
    payload = await assemble_report_payload(async_db, reloaded)

    assert payload.evaluation.available is True
    assert payload.evaluation.mode == "legacy"
    assert payload.evaluation.reason_code == "legacy_only"
    assert payload.evaluation.canonical is None
    assert payload.evaluation.legacy is not None
    assert payload.evaluation.legacy.overall_score == 72.5


@pytest.mark.asyncio
async def test_too_short_evaluation_is_terminal_without_score(async_db):
    """is_too_short evaluations are terminal and carry no fabricated score."""
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.flush()

    async_db.add(Evaluation(
        id=uuid.uuid4(), session_id=session.id, overall_score=0.0,
        category_scores=[], strengths=[], weaknesses=[], is_too_short=True,
    ))
    await async_db.commit()

    reloaded = await _reload_session(async_db, session.id)
    payload = await assemble_report_payload(async_db, reloaded)

    assert payload.evaluation.mode == "too_short"
    assert payload.evaluation.reason_code == "session_too_short"
    assert payload.evaluation.canonical is None
    assert payload.evaluation.legacy is None
    assert payload.evaluation.reason


@pytest.mark.asyncio
async def test_canonical_evaluation_branch_used_when_categories_present(async_db):
    """A rubric_result with non-empty categories uses the canonical branch."""
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.flush()

    canonical_result = {
        "status": "evaluated",
        "summary": "Agent performed well overall.",
        "categories": [{
            "rubric_block_id": "opening",
            "category": "Opening",
            "raw_score": 90,
            "penalty_total": 0,
            "penalized_score": 90,
            "weight": 50,
            "weighted_contribution": 45,
            "passing_score": 70,
            "passed": True,
            "evidence": [{
                "sequence_number": 0, "speaker": "agent",
                "excerpt": "Hello", "explanation": "Proper greeting",
            }],
            "strengths": [],
            "violations": [],
            "failed_criteria": [],
            "recommendation_inputs": [],
        }],
        "weighted_total": 45,
        "passing_score": 70,
        "passed": False,
        "applied_techniques": {"techniques_used": [], "reason_if_empty": "None observed"},
        "missed_opportunities": {"missed_techniques": [], "reason_if_empty": "None observed"},
        "recommendations": [],
    }
    async_db.add(Transcript(
        id=uuid.uuid4(), session_id=session.id, speaker="agent",
        utterance_text="Hello", timestamp_ms=datetime.now(timezone.utc), sequence_number=0,
    ))
    async_db.add(Evaluation(
        id=uuid.uuid4(), session_id=session.id, overall_score=45.0,
        category_scores=[], strengths=[], weaknesses=[], is_too_short=False,
        rubric_result=canonical_result, weighted_total=45.0, passing_score=70, passed=False,
    ))
    await async_db.commit()

    reloaded = await _reload_session(async_db, session.id)
    payload = await assemble_report_payload(async_db, reloaded)

    assert payload.evaluation.mode == "canonical"
    assert payload.evaluation.legacy is None
    assert payload.evaluation.canonical is not None
    assert payload.evaluation.canonical.categories[0].rubric_block_id == "opening"


@pytest.mark.asyncio
async def test_not_applicable_canonical_result_is_terminal(async_db):
    """A not_applicable canonical status is terminal with an explicit reason."""
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.flush()

    canonical_result = {
        "status": "not_applicable",
        "summary": "No rubric was active for this session.",
        "categories": [{
            "rubric_block_id": "opening",
            "category": "Opening",
            "raw_score": None,
            "penalty_total": 0,
            "penalized_score": None,
            "weight": 50,
            "weighted_contribution": 0,
            "passing_score": 70,
            "passed": False,
            "evidence": [],
            "strengths": [],
            "violations": [],
            "failed_criteria": [],
            "recommendation_inputs": [],
        }],
        "weighted_total": 0,
        "passing_score": 70,
        "passed": False,
        "applied_techniques": {"techniques_used": [], "reason_if_empty": "N/A"},
        "missed_opportunities": {"missed_techniques": [], "reason_if_empty": "N/A"},
        "recommendations": [],
    }
    async_db.add(Evaluation(
        id=uuid.uuid4(), session_id=session.id, overall_score=0.0,
        category_scores=[], strengths=[], weaknesses=[], is_too_short=False,
        rubric_result=canonical_result,
    ))
    await async_db.commit()

    reloaded = await _reload_session(async_db, session.id)
    payload = await assemble_report_payload(async_db, reloaded)

    assert payload.evaluation.mode == "not_applicable"
    assert payload.evaluation.reason == "No rubric was active for this session."


@pytest.mark.asyncio
async def test_coaching_canonical_suppresses_legacy_mistakes(async_db):
    """When canonical coaching is present, legacy mistake entries are suppressed."""
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.flush()

    rubric_coaching = {
        "standard_version_id": None,
        "standard_version_number": None,
        "blocks": [{
            "rubric_block_id": "opening",
            "block_name": "Call Opening",
            "display_order": 1,
            "recommendations": [{
                "rubric_block_id": "opening",
                "criterion_id": "greet-properly",
                "evidence_sequence_number": 0,
                "explanation": "Missed identity verification",
                "recommended_response": "Please confirm your full name and date of birth.",
                "coaching_advice": "Always verify identity before discussing account details.",
            }],
        }],
    }
    async_db.add(Transcript(
        id=uuid.uuid4(), session_id=session.id, speaker="agent",
        utterance_text="Please confirm your full name and date of birth.",
        timestamp_ms=datetime.now(timezone.utc), sequence_number=0,
    ))
    async_db.add(CoachingReport(
        id=uuid.uuid4(), session_id=session.id,
        mistakes_by_category={
            "compliance": [{
                "transcript_position": 0, "transcript_excerpt": "...",
                "category": "compliance", "explanation": "legacy mistake",
                "recommended_alternative": "should not appear",
            }],
            "_rubric_coaching": rubric_coaching,
        },
        total_mistakes=1, no_mistakes=False,
    ))
    await async_db.commit()

    reloaded = await _reload_session(async_db, session.id)
    payload = await assemble_report_payload(async_db, reloaded)

    assert payload.coaching.mode == "canonical"
    assert payload.coaching.legacy_mistakes_by_category == {}
    assert len(payload.coaching.blocks) == 1
    assert payload.coaching.blocks[0].rubric_block_id == "opening"


@pytest.mark.asyncio
async def test_coaching_legacy_branch_used_without_canonical_markers(async_db):
    """Without any canonical coaching markers, legacy mistakes are returned."""
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.flush()
    async_db.add(Transcript(
        id=uuid.uuid4(), session_id=session.id, speaker="agent",
        utterance_text="legacy transcript", timestamp_ms=datetime.now(timezone.utc), sequence_number=0,
    ))

    async_db.add(CoachingReport(
        id=uuid.uuid4(), session_id=session.id,
        mistakes_by_category={
            "compliance": [{
                "transcript_position": 0, "transcript_excerpt": "...",
                "category": "compliance", "explanation": "legacy mistake",
                "recommended_alternative": "verify identity first",
            }],
        },
        total_mistakes=1, no_mistakes=False,
    ))
    await async_db.commit()

    reloaded = await _reload_session(async_db, session.id)
    payload = await assemble_report_payload(async_db, reloaded)

    assert payload.coaching.mode == "legacy"
    assert payload.coaching.blocks == []
    assert "compliance" in payload.coaching.legacy_mistakes_by_category


@pytest.mark.asyncio
async def test_learning_plan_section_reflects_stored_items(async_db):
    """Learning plan items are surfaced without modification."""
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.flush()

    async_db.add(LearningPlan(
        id=uuid.uuid4(), session_id=session.id, agent_id=session.agent_id,
        weak_competencies=[{"category": "compliance", "score": 55, "recommended_scenario": "Compliance Fundamentals"}],
        all_passing=False,
    ))
    await async_db.commit()

    reloaded = await _reload_session(async_db, session.id)
    payload = await assemble_report_payload(async_db, reloaded)

    assert payload.learning_plan.available is True
    assert payload.learning_plan.all_passing is False
    assert payload.learning_plan.items[0].category == "compliance"


@pytest.mark.asyncio
async def test_assembly_is_deterministic_for_identical_inputs(async_db):
    """Two independent assemblies of the same stored data hash identically."""
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.flush()

    now = datetime.now(timezone.utc)
    async_db.add(Transcript(
        id=uuid.uuid4(), session_id=session.id, speaker="agent",
        utterance_text="Hello", timestamp_ms=now, sequence_number=0,
    ))
    await async_db.commit()

    reloaded_1 = await _reload_session(async_db, session.id)
    payload_1 = await assemble_report_payload(async_db, reloaded_1)
    hash_1 = compute_content_hash(payload_1)

    reloaded_2 = await _reload_session(async_db, session.id)
    payload_2 = await assemble_report_payload(async_db, reloaded_2)
    hash_2 = compute_content_hash(payload_2)

    assert hash_1 == hash_2
    assert canonical_serialize(payload_1) == canonical_serialize(payload_2)


@pytest.mark.asyncio
async def test_bounded_query_count_independent_of_transcript_length(async_db):
    """Assembly issues the same number of queries regardless of transcript size."""
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.flush()

    now = datetime.now(timezone.utc)
    for i in range(50):
        async_db.add(Transcript(
            id=uuid.uuid4(), session_id=session.id, speaker="agent" if i % 2 == 0 else "debtor",
            utterance_text=f"utterance {i}", timestamp_ms=now, sequence_number=i,
        ))
    await async_db.commit()

    reloaded = await _reload_session(async_db, session.id)

    query_count = 0

    @event.listens_for(async_db.sync_session.bind, "before_cursor_execute")
    def _count(conn, cursor, statement, parameters, context, executemany):
        nonlocal query_count
        if statement.strip().upper().startswith("SELECT"):
            query_count += 1

    try:
        await assemble_report_payload(async_db, reloaded)
    finally:
        event.remove(async_db.sync_session.bind, "before_cursor_execute", _count)

    # Exactly 4 artifact queries (transcript, evaluation, coaching, learning plan).
    assert query_count == 4


@pytest.mark.asyncio
async def test_missing_completed_timestamp_is_rejected(async_db):
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id, ended_at=None)
    async_db.add(session)
    await async_db.commit()

    reloaded = await _reload_session(async_db, session.id)
    with pytest.raises(ValueError, match="ended_at"):
        await assemble_report_payload(async_db, reloaded)


@pytest.mark.asyncio
async def test_duplicate_transcript_sequence_is_rejected(async_db):
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.flush()
    now = datetime.now(timezone.utc)
    for text in ("first", "second"):
        async_db.add(Transcript(
            id=uuid.uuid4(), session_id=session.id, speaker="agent",
            utterance_text=text, timestamp_ms=now, sequence_number=0,
        ))
    await async_db.commit()

    reloaded = await _reload_session(async_db, session.id)
    with pytest.raises(ValueError, match="sequence"):
        await assemble_report_payload(async_db, reloaded)


@pytest.mark.asyncio
async def test_canonical_cross_reference_is_rejected(async_db):
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.flush()
    result = {
        "status": "evaluated", "summary": "Stored result.",
        "categories": [{
            "rubric_block_id": "opening", "category": "Opening", "raw_score": 80,
            "penalty_total": 0, "penalized_score": 80, "weight": 100,
            "weighted_contribution": 80, "passing_score": 70, "passed": True,
            "evidence": [{
                "sequence_number": 99, "speaker": "agent", "excerpt": "missing",
                "explanation": "Missing transcript reference",
            }],
            "strengths": [], "violations": [], "failed_criteria": [],
            "recommendation_inputs": [],
        }],
        "weighted_total": 80, "passing_score": 70, "passed": True,
        "applied_techniques": {"techniques_used": [], "reason_if_empty": "None."},
        "missed_opportunities": {"missed_techniques": [], "reason_if_empty": "None."},
        "recommendations": [],
    }
    async_db.add(Evaluation(
        id=uuid.uuid4(), session_id=session.id, overall_score=80,
        category_scores=[], strengths=[], weaknesses=[], rubric_result=result,
        weighted_total=80, passing_score=70, passed=True, is_too_short=False,
    ))
    await async_db.commit()

    reloaded = await _reload_session(async_db, session.id)
    with pytest.raises(ValueError, match="cross-reference"):
        await assemble_report_payload(async_db, reloaded)


@pytest.mark.asyncio
async def test_learning_plan_mismatched_scenario_is_rejected(async_db):
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.flush()
    async_db.add(LearningPlan(
        id=uuid.uuid4(), session_id=session.id, agent_id=session.agent_id,
        weak_competencies=[{
            "category": "compliance", "score": 55, "scenario_id": str(uuid.uuid4()),
            "rubric_block_id": "block", "criterion_id": "criterion",
            "practice_focus": "Practice the criterion",
        }],
        all_passing=False,
    ))
    await async_db.commit()

    reloaded = await _reload_session(async_db, session.id)
    with pytest.raises(ValueError, match="scenario"):
        await assemble_report_payload(async_db, reloaded)


@pytest.mark.asyncio
async def test_query_count_is_independent_of_rubric_block_count(async_db):
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.flush()
    blocks = [{
        "rubric_block_id": f"block-{index}", "block_name": f"Block {index}",
        "display_order": index, "recommendations": [],
    } for index in range(40)]
    async_db.add(CoachingReport(
        id=uuid.uuid4(), session_id=session.id,
        mistakes_by_category={"_rubric_coaching": {
            "standard_version_id": None, "standard_version_number": None, "blocks": blocks,
        }},
        total_mistakes=0, no_mistakes=True,
    ))
    await async_db.commit()
    reloaded = await _reload_session(async_db, session.id)
    query_count = 0

    @event.listens_for(async_db.sync_session.bind, "before_cursor_execute")
    def _count(conn, cursor, statement, parameters, context, executemany):
        nonlocal query_count
        if statement.strip().upper().startswith("SELECT"):
            query_count += 1

    try:
        payload = await assemble_report_payload(async_db, reloaded)
    finally:
        event.remove(async_db.sync_session.bind, "before_cursor_execute", _count)

    assert len(payload.coaching.blocks) == 40
    assert query_count == 4


@pytest.mark.asyncio
async def test_assembly_does_not_invoke_business_result_services(async_db):
    from unittest.mock import patch
    from app.services.coaching_engine import CoachingEngine
    from app.services.evaluation_pipeline import EvaluationPipeline
    from app.services.learning_plan_generator import LearningPlanGenerator

    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.commit()
    reloaded = await _reload_session(async_db, session.id)

    with (
        patch.object(EvaluationPipeline, "run", side_effect=AssertionError("scoring recomputed")),
        patch.object(CoachingEngine, "generate_report", side_effect=AssertionError("coaching recomputed")),
        patch.object(LearningPlanGenerator, "generate", side_effect=AssertionError("learning plan recomputed")),
    ):
        payload = await assemble_report_payload(async_db, reloaded)

    assert payload.summary.session_id == session.id

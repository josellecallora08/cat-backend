"""Unit tests for the session report generation and retrieval service.

Feature: session-report-generation, task 3.

Validates: Requirements 2.2, 2.3, 2.4, 3.6, 6.1, 6.2
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from app.database import Base
from app.models import Scenario, Session, SessionReport, SessionReportStatus
from app.schemas import SessionStatus, TranscriptEntry
from app.schemas.rubric_evaluation import (
    CanonicalEvaluationResult,
    RubricAppliedTechniques,
    RubricCategoryScore,
    RubricEvidence,
    RubricMissedOpportunities,
)
from app.schemas.session_report import (
    CoachingSection,
    EvaluationSection,
    LearningPlanSection,
    LegacyEvaluationResult,
    ReportReasonCode,
    SessionReportPayload,
    SessionReportSummary,
    TranscriptSection,
)
from app.services.session_report_service import (
    SessionNotCompletedError,
    generate_report,
    get_current_report,
    get_report_status,
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


def _make_session(scenario_id: uuid.UUID, status: str = "completed") -> Session:
    created_at = datetime.now(timezone.utc)
    return Session(
        id=uuid.uuid4(),
        scenario_id=scenario_id,
        agent_id=uuid.uuid4(),
        status=status,
        created_at=created_at,
        ended_at=created_at if status == "completed" else None,
        persona_context={"name": "Test Persona", "communication_style": "calm", "emotional_state": 3},
    )


async def _reload_session(async_db: AsyncSession, session_id: uuid.UUID) -> Session:
    stmt = (
        select(Session)
        .options(selectinload(Session.campaign))
        .where(Session.id == session_id)
    )
    result = await async_db.execute(stmt)
    return result.scalar_one()


@pytest.mark.asyncio
async def test_generate_report_creates_ready_row_with_hash(async_db):
    """A successful generation stores status=ready, a payload, and a content_hash."""
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.commit()

    reloaded = await _reload_session(async_db, session.id)
    generator_id = uuid.uuid4()
    with patch(
        "app.services.session_report_service.log_session_report_generated"
    ) as audit_log:
        report = await generate_report(async_db, reloaded, generated_by=generator_id)

    assert report.status == SessionReportStatus.READY
    assert report.report_version == 1
    assert report.payload is not None
    assert report.content_hash and len(report.content_hash) == 64
    assert report.reason_code is None
    assert report.updated_at is not None
    assert report.failure_reason is None
    audit_log.assert_called_once_with(str(session.id), 1, str(generator_id))


@pytest.mark.asyncio
async def test_generate_report_refuses_non_completed_session(async_db):
    """Generation is refused for a pending or active session."""
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id, status="active")
    async_db.add(session)
    await async_db.commit()

    reloaded = await _reload_session(async_db, session.id)
    with pytest.raises(SessionNotCompletedError):
        await generate_report(async_db, reloaded)

    # No row should have been inserted.
    stmt = select(SessionReport).where(SessionReport.session_id == session.id)
    rows = (await async_db.execute(stmt)).scalars().all()
    assert rows == []


@pytest.mark.asyncio
async def test_regeneration_increments_version_and_preserves_prior_payload(async_db):
    """A second generation call creates version 2; version 1's payload is untouched."""
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.commit()

    reloaded_1 = await _reload_session(async_db, session.id)
    generator_id = uuid.uuid4()
    report_1 = await generate_report(async_db, reloaded_1, generated_by=generator_id)
    assert report_1.report_version == 1
    original_snapshot = {
        "payload": report_1.payload,
        "content_hash": report_1.content_hash,
        "status": report_1.status,
        "generated_by": report_1.generated_by,
        "created_at": report_1.created_at,
        "updated_at": report_1.updated_at,
    }

    reloaded_2 = await _reload_session(async_db, session.id)
    report_2 = await generate_report(async_db, reloaded_2)
    assert report_2.report_version == 2

    stmt = select(SessionReport).where(
        SessionReport.session_id == session.id,
        SessionReport.report_version == 1,
    )
    row_1_after = (await async_db.execute(stmt)).scalar_one()
    assert {
        "payload": row_1_after.payload,
        "content_hash": row_1_after.content_hash,
        "status": row_1_after.status,
        "generated_by": row_1_after.generated_by,
        "created_at": row_1_after.created_at,
        "updated_at": row_1_after.updated_at,
    } == original_snapshot


@pytest.mark.asyncio
async def test_get_current_report_returns_highest_ready_version(async_db):
    """get_current_report returns the highest report_version with status=ready."""
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.commit()

    reloaded = await _reload_session(async_db, session.id)
    await generate_report(async_db, reloaded)
    reloaded = await _reload_session(async_db, session.id)
    latest = await generate_report(async_db, reloaded)

    current = await get_current_report(async_db, session.id)
    assert current is not None
    assert current.id == latest.id
    assert current.report_version == 2


@pytest.mark.asyncio
async def test_get_current_report_returns_none_when_absent(async_db):
    """get_current_report returns None when no report has been generated."""
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.commit()

    current = await get_current_report(async_db, session.id)
    assert current is None


@pytest.mark.asyncio
async def test_failed_assembly_records_failure_without_payload(async_db):
    """An assembly failure is recorded as status=failed with no payload, and re-raised."""
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.commit()

    reloaded = await _reload_session(async_db, session.id)

    with patch(
        "app.services.session_report_service.assemble_report_payload",
        side_effect=RuntimeError("boom"),
    ), patch(
        "app.services.session_report_service.log_session_report_generation_failed"
    ) as audit_log:
        with pytest.raises(RuntimeError):
            await generate_report(async_db, reloaded)

    audit_log.assert_called_once_with(str(session.id), 1, "generation_failed")

    stmt = select(SessionReport).where(SessionReport.session_id == session.id)
    row = (await async_db.execute(stmt)).scalar_one()
    assert row.status == SessionReportStatus.FAILED
    assert row.payload is None
    assert row.content_hash is None
    assert row.reason_code == "generation_failed"
    assert row.failure_reason == "Report generation failed"
    assert "boom" not in row.failure_reason  # no raw exception text leaked
    assert row.report_version == 1


@pytest.mark.asyncio
async def test_failed_then_retry_produces_version_2(async_db):
    """After a failed version 1, a successful retry lands at version 2."""
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.commit()

    reloaded = await _reload_session(async_db, session.id)
    with patch(
        "app.services.session_report_service.assemble_report_payload",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RuntimeError):
            await generate_report(async_db, reloaded)

    reloaded = await _reload_session(async_db, session.id)
    report = await generate_report(async_db, reloaded)
    assert report.report_version == 2
    assert report.status == SessionReportStatus.READY


@pytest.mark.asyncio
async def test_pending_row_is_typed_and_payload_free_before_assembly(async_db):
    """An in-flight generation is visible as a pending null-payload marker."""
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.commit()
    reloaded = await _reload_session(async_db, session.id)

    async def inspect_pending(db, current_session):
        row = (
            await db.execute(
                select(SessionReport).where(SessionReport.session_id == current_session.id)
            )
        ).scalar_one()
        assert row.status == SessionReportStatus.PENDING
        assert row.reason_code == "generation_pending"
        assert row.payload is None
        assert row.content_hash is None
        raise RuntimeError("private assembly detail")

    with patch(
        "app.services.session_report_service.assemble_report_payload",
        side_effect=inspect_pending,
    ), pytest.raises(RuntimeError):
        await generate_report(async_db, reloaded)

    row = (
        await async_db.execute(
            select(SessionReport).where(SessionReport.session_id == session.id)
        )
    ).scalar_one()
    assert row.status == SessionReportStatus.FAILED
    assert row.reason_code == "generation_failed"
    assert row.payload is None
    assert row.content_hash is None


@pytest.mark.asyncio
async def test_payload_validation_failure_cannot_create_ready_row(async_db):
    """A structurally invalid alternate assembler result is recorded as failed."""
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.commit()
    reloaded = await _reload_session(async_db, session.id)

    with patch(
        "app.services.session_report_service.assemble_report_payload",
        return_value={"summary": "invalid"},
    ), pytest.raises(ValidationError):
        await generate_report(async_db, reloaded)

    row = (
        await async_db.execute(
            select(SessionReport).where(SessionReport.session_id == session.id)
        )
    ).scalar_one()
    assert row.status == SessionReportStatus.FAILED
    assert row.reason_code == "generation_failed"
    assert row.payload is None
    assert row.content_hash is None


@pytest.mark.asyncio
async def test_failed_latest_attempt_does_not_hide_older_ready_snapshot(async_db):
    """A failed regeneration leaves the previous ready snapshot current."""
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.commit()

    first = await generate_report(async_db, await _reload_session(async_db, session.id))
    with patch(
        "app.services.session_report_service.assemble_report_payload",
        side_effect=RuntimeError("regeneration detail"),
    ), pytest.raises(RuntimeError):
        await generate_report(async_db, await _reload_session(async_db, session.id))

    current = await get_current_report(async_db, session.id)
    assert current is not None
    assert current.id == first.id
    assert current.report_version == 1
    failed = (
        await async_db.execute(
            select(SessionReport).where(
                SessionReport.session_id == session.id,
                SessionReport.status == SessionReportStatus.FAILED,
            )
        )
    ).scalar_one()
    assert failed.report_version == 2
    assert failed.payload is None
    assert failed.content_hash is None



def _status_payload(session: Session, variant: str) -> SessionReportPayload:
    """Build minimal validated payloads for status-resolution examples."""
    now = session.created_at
    summary = SessionReportSummary(
        session_id=session.id,
        scenario_id=session.scenario_id,
        agent_id=session.agent_id,
        status=SessionStatus.COMPLETED,
        created_at=now,
        ended_at=session.ended_at,
        duration_seconds=0,
    )
    transcript = TranscriptSection(
        available=True,
        entries=[]
        if variant == "empty_transcript"
        else [
            TranscriptEntry(
                speaker="agent",
                text="Hello",
                timestamp=now,
                sequence_number=0,
            )
        ],
    )
    if variant == "not_applicable":
        evaluation = EvaluationSection(
            available=True,
            mode="not_applicable",
            reason_code=ReportReasonCode.NOT_APPLICABLE,
        )
    elif variant == "too_short":
        evaluation = EvaluationSection(
            available=True,
            mode="too_short",
            reason_code=ReportReasonCode.SESSION_TOO_SHORT,
        )
    elif variant == "empty_transcript":
        evaluation = EvaluationSection(
            available=False,
            reason_code=ReportReasonCode.ARTIFACT_MISSING,
        )
    elif variant == "legacy_only":
        evaluation = EvaluationSection(
            available=True,
            mode="legacy",
            reason_code=ReportReasonCode.LEGACY_ONLY,
            legacy=LegacyEvaluationResult(overall_score=80),
        )
    else:
        evidence = [] if variant == "no_evidence" else [
            RubricEvidence(
                sequence_number=0,
                speaker="agent",
                excerpt="Hello",
                explanation="Greeting evidence",
            )
        ]
        canonical = CanonicalEvaluationResult(
            status="evaluated",
            summary="Evaluated",
            categories=[
                RubricCategoryScore(
                    rubric_block_id="block-one",
                    category="Opening",
                    raw_score=80,
                    penalty_total=0,
                    penalized_score=80,
                    weight=100,
                    weighted_contribution=80,
                    passing_score=70,
                    passed=True,
                    evidence=evidence,
                    strengths=[],
                    violations=[],
                    failed_criteria=[],
                    recommendation_inputs=[],
                )
            ],
            weighted_total=80,
            passing_score=70,
            passed=True,
            applied_techniques=RubricAppliedTechniques(
                techniques_used=[], reason_if_empty="No techniques recorded"
            ),
            missed_opportunities=RubricMissedOpportunities(
                missed_techniques=[], reason_if_empty="No missed opportunities"
            ),
        )
        evaluation = EvaluationSection(
            available=True,
            mode="canonical",
            reason_code=(
                ReportReasonCode.NO_EVIDENCE if variant == "no_evidence" else None
            ),
            canonical=canonical,
            weighted_total=80,
            passing_score=70,
            passed=True,
        )
    return SessionReportPayload(
        summary=summary,
        transcript=transcript,
        evaluation=evaluation,
        coaching=CoachingSection(available=True, mode="legacy"),
        learning_plan=LearningPlanSection(available=True, all_passing=True),
    )


async def _insert_status_row(
    async_db: AsyncSession,
    session: Session,
    *,
    status: str,
    version: int,
    payload: SessionReportPayload | None = None,
    reason_code: str | None = None,
) -> SessionReport:
    now = session.created_at
    row = SessionReport(
        session_id=session.id,
        agent_id=session.agent_id,
        status=status,
        report_version=version,
        payload=payload.model_dump(mode="json") if payload is not None else None,
        content_hash="a" * 64 if payload is not None else None,
        reason_code=reason_code,
        created_at=now,
        updated_at=now,
    )
    async_db.add(row)
    await async_db.commit()
    return row


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "variant, expected_status",
    [
        ("ready", "ready"),
        ("not_applicable", "not_applicable"),
        ("too_short", "too_short"),
        ("legacy_only", "legacy_only"),
        ("empty_transcript", "empty_transcript"),
        ("no_evidence", "no_evidence"),
    ],
)
async def test_get_report_status_resolves_readable_variants(
    async_db, variant, expected_status
):
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.commit()

    payload = _status_payload(session, variant)
    await _insert_status_row(
        async_db,
        session,
        status=SessionReportStatus.READY,
        version=1,
        payload=payload,
    )

    result = await get_report_status(async_db, session.id)
    assert result.status == expected_status
    assert result.report.payload.summary.session_id == session.id
    assert result.report.payload.evaluation is not None


@pytest.mark.asyncio
async def test_get_report_status_returns_missing_for_no_attempt(async_db):
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.commit()

    with patch(
        "app.services.session_report_service._probe_report_artifacts",
        return_value=[],
    ):
        result = await get_report_status(async_db, session.id)

    assert result.status == "missing"
    assert result.report is None
    assert result.reason.code == ReportReasonCode.ARTIFACT_MISSING


@pytest.mark.asyncio
async def test_get_report_status_returns_incomplete_for_missing_artifacts(async_db):
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.commit()

    result = await get_report_status(async_db, session.id)

    assert result.status == "incomplete"
    assert result.report is None
    assert result.missing_sections == [ReportReasonCode.ARTIFACT_MISSING]


@pytest.mark.asyncio
async def test_get_report_status_returns_generating_without_payload(async_db):
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.commit()

    await _insert_status_row(
        async_db,
        session,
        status=SessionReportStatus.PENDING,
        version=1,
        reason_code=ReportReasonCode.GENERATION_PENDING,
    )
    result = await get_report_status(async_db, session.id)

    assert result.status == "generating"
    assert result.report is None
    assert result.latest_attempt.reason.code == ReportReasonCode.GENERATION_PENDING


@pytest.mark.asyncio
async def test_get_report_status_returns_failed_only_without_payload(async_db):
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.commit()

    await _insert_status_row(
        async_db,
        session,
        status=SessionReportStatus.FAILED,
        version=1,
        reason_code=ReportReasonCode.GENERATION_FAILED,
    )
    result = await get_report_status(async_db, session.id)

    assert result.status == "failed"
    assert result.report is None
    assert result.latest_attempt.reason.code == ReportReasonCode.GENERATION_FAILED


@pytest.mark.asyncio
async def test_get_report_status_keeps_older_ready_when_latest_attempt_failed(async_db):
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id)
    async_db.add(session)
    await async_db.commit()

    payload = _status_payload(session, "ready")
    first = await _insert_status_row(
        async_db,
        session,
        status=SessionReportStatus.READY,
        version=1,
        payload=payload,
    )
    await _insert_status_row(
        async_db,
        session,
        status=SessionReportStatus.FAILED,
        version=2,
        reason_code=ReportReasonCode.GENERATION_FAILED,
    )

    result = await get_report_status(async_db, session.id)

    assert result.status == "ready"
    assert result.report.report_version == first.report_version
    assert result.latest_attempt.status == "failed"
    assert result.latest_attempt.report_version == 2
    assert result.report.payload is not None

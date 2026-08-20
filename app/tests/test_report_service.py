"""Focused tests for normalized backend report aggregation."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.schemas.report import (
    ReportCompletion,
    ReportSectionName,
    ScoreStatus,
    SectionEnvelope,
    SectionState,
)
from app.services.report_service import ReportService


def _result(*, rows=None, scalar=None):
    """Build a minimal async SQLAlchemy result double."""
    value = MagicMock()
    value.scalars.return_value.all.return_value = rows or []
    value.scalar_one_or_none.return_value = scalar
    return value


def _session():
    """Build a persisted session-shaped object."""
    return SimpleNamespace(
        id=uuid4(),
        status="completed",
        scenario_id=uuid4(),
        campaign_id=None,
        created_at=datetime.now(UTC),
        ended_at=datetime.now(UTC),
        negotiation_standard_version=None,
    )


@pytest.mark.asyncio
async def test_report_aggregates_loaded_and_empty_artifacts():
    """A report retains loaded sections and marks missing optional artifacts empty."""
    session = _session()
    evaluation = SimpleNamespace(
        session_id=session.id,
        overall_score=82.0,
        category_scores=[],
        strengths=[],
        weaknesses=[],
        weighted_total=82.0,
        passing_score=70,
        passed=True,
        is_too_short=False,
        negotiation_standard_version_id=None,
        standard_snapshot={"historical": True},
        rubric_result=None,
    )
    db = AsyncMock()
    db.execute.side_effect = [
        _result(
            rows=[
                SimpleNamespace(
                    speaker="agent",
                    utterance_text="Hello",
                    timestamp_ms=session.created_at,
                    sequence_number=0,
                )
            ]
        ),
        _result(scalar=evaluation),
        _result(scalar=None),
        _result(scalar=None),
    ]

    with patch(
        "app.services.report_service.get_authorized_session", AsyncMock(return_value=session)
    ):
        report = await ReportService(db).get_report(session.id, SimpleNamespace())

    states = {section.name: section.state for section in report.sections}
    assert tuple(states) == ReportService._SECTION_NAMES
    assert states[ReportSectionName.TRANSCRIPT] is SectionState.LOADED
    assert states[ReportSectionName.EVALUATION] is SectionState.LOADED
    assert states[ReportSectionName.COACHING] is SectionState.EMPTY
    assert report.report_status is ReportCompletion.COMPLETE
    assert report.score_status is ScoreStatus.EVALUATED


@pytest.mark.asyncio
async def test_evaluation_mapping_uses_persisted_version_and_hides_too_short_score():
    """Historical version fields survive while too-short scores are unavailable."""
    version_id = uuid4()
    item = SimpleNamespace(
        session_id=uuid4(),
        overall_score=0.0,
        category_scores=[{"score": 0}],
        strengths=[],
        weaknesses=[],
        weighted_total=0.0,
        passing_score=70,
        passed=False,
        is_too_short=True,
        negotiation_standard_version_id=version_id,
        negotiation_standard_version=SimpleNamespace(
            version_number=4,
            standard=SimpleNamespace(name="Pinned standard"),
        ),
        standard_snapshot={"historical": True},
        rubric_result={"status": "not_applicable"},
    )
    db = AsyncMock()
    db.execute.return_value = _result(scalar=item)
    service = ReportService(db)

    evaluation = await service._evaluation(item.session_id)
    version = service._version(_session(), evaluation)

    assert evaluation["overall_score"] is None
    assert evaluation["passed"] is None
    assert version.kind.value == "current"
    assert version.id == version_id
    assert version.number == 4
    assert version.name == "Pinned standard"

    """A persisted too-short evaluation never becomes a passing/failing score."""
    db = AsyncMock()
    service = ReportService(db)
    evaluation = {
        "is_too_short": True,
        "rubric_result": {"status": "not_applicable"},
        "overall_score": 0,
        "passed": False,
    }

    assert service._score_status(evaluation) is ScoreStatus.NOT_APPLICABLE
    assert (
        service._completion([service._loaded(ReportSectionName.SUMMARY, {})], evaluation)
        is ReportCompletion.NOT_APPLICABLE
    )


@pytest.mark.asyncio
async def test_section_database_failure_is_safe_and_local():
    """A section query failure becomes a redacted failed envelope."""
    service = ReportService(AsyncMock())
    service._transcript = AsyncMock(side_effect=RuntimeError("SELECT secret FROM users"))

    section = await service._load_section(
        ReportSectionName.TRANSCRIPT, uuid4(), service._transcript
    )

    assert section.state is SectionState.FAILED
    assert section.failure is not None
    assert (
        section.failure.safe_message
        == "The report is temporarily unavailable. Please try again later."
    )
    assert "SELECT" not in section.failure.safe_message


@pytest.mark.asyncio
async def test_authorization_and_not_found_errors_remain_route_safe() -> None:
    """The service delegates access failures without exposing persisted internals."""
    service = ReportService(AsyncMock())
    for status_code, detail in ((403, "Session access denied"), (404, "Session not found")):
        with (
            patch(
                "app.services.report_service.get_authorized_session",
                AsyncMock(side_effect=HTTPException(status_code=status_code, detail=detail)),
            ),
            pytest.raises(HTTPException) as error,
        ):
            await service.get_report(uuid4(), SimpleNamespace())

        assert error.value.status_code == status_code
        assert "SELECT" not in str(error.value.detail)
        assert "token" not in str(error.value.detail).lower()


@pytest.mark.asyncio
async def test_section_loader_exposes_loaded_empty_and_failed_terminal_states() -> None:
    """Each section response maps independently to one terminal state."""
    service = ReportService(AsyncMock())
    session_id = uuid4()

    loaded = await service._load_section(
        ReportSectionName.TRANSCRIPT, session_id, AsyncMock(return_value=[{"text": "ok"}])
    )
    empty = await service._load_section(
        ReportSectionName.COACHING, session_id, AsyncMock(return_value=None)
    )
    failed = await service._load_section(
        ReportSectionName.LEARNING_PLAN,
        session_id,
        AsyncMock(side_effect=RuntimeError("database details")),
    )

    assert loaded.state is SectionState.LOADED
    assert loaded.data == [{"text": "ok"}]
    assert empty.state is SectionState.EMPTY
    assert empty.data is None
    assert failed.state is SectionState.FAILED
    assert failed.failure is not None
    assert failed.data is None
    assert failed.unavailable_reason is None
    assert "database details" not in failed.failure.safe_message


@pytest.mark.asyncio
async def test_failed_section_does_not_replace_successful_retry_data() -> None:
    """Retry-compatible section loading preserves unrelated successful data."""
    service = ReportService(AsyncMock())
    session_id = uuid4()
    transcript = service._loaded(ReportSectionName.TRANSCRIPT, [{"text": "kept"}])

    failed_evaluation = await service._load_section(
        ReportSectionName.EVALUATION,
        session_id,
        AsyncMock(side_effect=RuntimeError("temporary failure")),
    )
    retried_evaluation = await service._load_section(
        ReportSectionName.EVALUATION,
        session_id,
        AsyncMock(return_value={"overall_score": 91}),
    )

    assert transcript.state is SectionState.LOADED
    assert transcript.data == [{"text": "kept"}]
    assert failed_evaluation.state is SectionState.FAILED
    assert retried_evaluation.state is SectionState.LOADED
    assert retried_evaluation.data == {"overall_score": 91}


def test_completion_aggregation_distinguishes_partial_complete_and_failed() -> None:
    """Aggregate status reflects section failures while retaining terminal successes."""
    loaded = ReportService._loaded(ReportSectionName.TRANSCRIPT, {"count": 1})
    failed_section = SectionEnvelope(
        name=ReportSectionName.EVALUATION,
        state=SectionState.FAILED,
    )

    assert ReportService._completion([loaded], None) is ReportCompletion.COMPLETE
    assert ReportService._completion([loaded, failed_section], None) is ReportCompletion.PARTIAL
    assert ReportService._completion([failed_section], None) is ReportCompletion.FAILED


@pytest.mark.asyncio
async def test_legacy_report_has_legacy_version_and_too_short_status() -> None:
    """Legacy metadata and not-applicable score semantics remain explicit."""
    session = _session()
    evaluation = SimpleNamespace(
        session_id=session.id,
        overall_score=0,
        category_scores=[{"score": 0}],
        strengths=[],
        weaknesses=[],
        weighted_total=0,
        passing_score=70,
        passed=False,
        is_too_short=True,
        negotiation_standard_version_id=None,
        standard_snapshot={"legacy": True},
        rubric_result={"status": "not_applicable"},
    )
    db = AsyncMock()
    db.execute.side_effect = [
        _result(rows=[]),
        _result(scalar=evaluation),
        _result(scalar=None),
        _result(scalar=None),
    ]

    with patch(
        "app.services.report_service.get_authorized_session", AsyncMock(return_value=session)
    ):
        report = await ReportService(db).get_report(session.id, SimpleNamespace())

    assert report.evaluation_version.kind.value == "legacy"
    assert report.score_status is ScoreStatus.NOT_APPLICABLE
    assert report.report_status is ReportCompletion.NOT_APPLICABLE
    evaluation_section = next(
        section for section in report.sections if section.name is ReportSectionName.EVALUATION
    )
    assert evaluation_section.data["passed"] is None

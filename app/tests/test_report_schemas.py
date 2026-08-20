"""Contract tests for normalized report schemas."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas.report import (
    CsvReportStatus,
    EvaluationKind,
    EvaluationVersionMetadata,
    FailureClass,
    ReportCompletion,
    ReportFailure,
    ReportResponse,
    ReportSectionName,
    ReportSessionMetadata,
    ScoreStatus,
    SectionEnvelope,
    SectionState,
)


def _session() -> ReportSessionMetadata:
    return ReportSessionMetadata(
        id=uuid4(),
        status="completed",
        created_at=datetime.now(UTC),
    )


def test_report_response_accepts_loaded_and_explicit_empty_sections() -> None:
    """Loaded data and unavailable data use explicit terminal states."""
    response = ReportResponse(
        session=_session(),
        report_status=ReportCompletion.PARTIAL,
        score_status=ScoreStatus.UNAVAILABLE,
        evaluation_version=EvaluationVersionMetadata(kind=EvaluationKind.LEGACY),
        sections=[
            SectionEnvelope(
                name=ReportSectionName.METADATA,
                state=SectionState.LOADED,
                data={"label": "completed"},
            ),
            SectionEnvelope(
                name=ReportSectionName.EVALUATION,
                state=SectionState.EMPTY,
                unavailable_reason="Evaluation is not available.",
            ),
        ],
    )

    assert response.sections[0].data == {"label": "completed"}
    assert response.sections[1].unavailable_reason == "Evaluation is not available."
    assert response.evaluation_version.kind is EvaluationKind.LEGACY


def test_report_failure_serializes_stable_class_alias() -> None:
    """Failure class is exposed under the contract's JSON key ``class``."""
    failure = ReportFailure(
        class_=FailureClass.DATA_CONTRACT,
        code="malformed_section",
        safe_message="This report section is unavailable.",
    )

    assert failure.model_dump(by_alias=True)["class"] == "data_contract"


def test_report_schemas_reject_unexpected_fields() -> None:
    """Strict contracts reject accidental or unrecognized response fields."""
    with pytest.raises(ValidationError):
        SectionEnvelope(
            name=ReportSectionName.SUMMARY,
            state=SectionState.EMPTY,
            unexpected="must be rejected",
        )


def test_report_schemas_reject_invalid_state_and_version_values() -> None:
    """Stable enums and positive version numbers reject malformed values."""
    with pytest.raises(ValidationError):
        SectionEnvelope(name="unknown", state="loaded")
    with pytest.raises(ValidationError):
        EvaluationVersionMetadata(kind=EvaluationKind.CURRENT, number=0)


def test_section_envelope_preserves_explicit_empty_and_failed_details() -> None:
    """Empty and failed sections retain actionable, contract-safe availability data."""
    empty = SectionEnvelope(
        name=ReportSectionName.COACHING,
        state=SectionState.EMPTY,
        unavailable_reason="No coaching was generated.",
    )
    failed = SectionEnvelope(
        name=ReportSectionName.LEARNING_PLAN,
        state=SectionState.FAILED,
        failure=ReportFailure(
            class_=FailureClass.BACKEND,
            code="service_unavailable",
            safe_message="This report section is unavailable.",
            correlation_id="opaque-correlation-id",
        ),
    )

    assert empty.data is None
    assert empty.unavailable_reason == "No coaching was generated."
    assert failed.failure is not None
    assert failed.failure.class_ is FailureClass.BACKEND


def test_csv_status_preserves_unavailable_and_failure_metadata() -> None:
    """CSV rows retain explicit section availability and failure classification."""
    status = CsvReportStatus(
        session_id=uuid4(),
        report_status=ReportCompletion.FAILED,
        section_name=ReportSectionName.COACHING,
        section_state=SectionState.FAILED,
        evaluation_kind=EvaluationKind.CURRENT,
        score_status=ScoreStatus.FAILED,
        unavailable_reason="Coaching is temporarily unavailable.",
        failure_class=FailureClass.BACKEND,
    )

    assert status.section_state is SectionState.FAILED
    assert status.failure_class is FailureClass.BACKEND


def test_report_failure_rejects_empty_sensitive_contract_fields() -> None:
    """Failure records require bounded code and safe message values."""
    with pytest.raises(ValidationError):
        ReportFailure(class_=FailureClass.BACKEND, code="", safe_message="safe")
    with pytest.raises(ValidationError):
        ReportFailure(class_=FailureClass.BACKEND, code="failure", safe_message="")

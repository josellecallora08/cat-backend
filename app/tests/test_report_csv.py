"""Tests for deterministic and safe report CSV serialization."""

import csv
import io
from uuid import UUID, uuid4

from app.schemas.report import (
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
from app.services.report_csv import CSV_COLUMNS, report_csv_filename, serialize_report_csv


SESSION_ID = UUID("11111111-1111-1111-1111-111111111111")


def _report() -> ReportResponse:
    return ReportResponse(
        session=ReportSessionMetadata(id=SESSION_ID, status="completed"),
        report_status=ReportCompletion.PARTIAL,
        score_status=ScoreStatus.EVALUATED,
        evaluation_version=EvaluationVersionMetadata(
            kind=EvaluationKind.CURRENT,
            id=uuid4(),
            number=3,
            name="Negotiation standard",
        ),
        sections=[
            SectionEnvelope(
                name=ReportSectionName.EVALUATION,
                state=SectionState.LOADED,
                data={
                    "overall_score": 82,
                    "passing_score": 75,
                    "passed": True,
                    "strengths": ['Comma, quote "and" newline\nUnicode ✓'],
                },
            ),
            SectionEnvelope(
                name=ReportSectionName.COACHING,
                state=SectionState.EMPTY,
                unavailable_reason="No coaching is available.",
            ),
            SectionEnvelope(
                name=ReportSectionName.LEARNING_PLAN,
                state=SectionState.FAILED,
                unavailable_reason="Section unavailable.",
                failure=ReportFailure(
                    class_=FailureClass.BACKEND,
                    code="service_error",
                    safe_message="The section is unavailable.",
                ),
            ),
        ],
    )


def test_csv_has_stable_columns_and_round_trips_special_values() -> None:
    rows = list(csv.DictReader(io.StringIO(serialize_report_csv(_report()))))

    assert tuple(rows[0]) == CSV_COLUMNS
    assert rows[0]["section_name"] == "evaluation"
    assert rows[0]["score"] == "82"
    assert "Comma, quote" in rows[0]["evidence"]
    assert "Unicode ✓" in rows[0]["evidence"]
    assert rows[1]["section_state"] == "empty"
    assert rows[1]["unavailable_reason"] == "No coaching is available."
    assert rows[2]["section_state"] == "failed"
    assert rows[2]["failure_class"] == "backend"


def test_csv_neutralizes_formula_like_values_and_keeps_empty_fields() -> None:
    report = _report()
    report.sections[0].data = {"score": "=SUM(A1:A2)", "evidence": "@user"}

    row = next(csv.DictReader(io.StringIO(serialize_report_csv(report))))

    assert row["score"] == "'=SUM(A1:A2)"
    assert row["evidence"] == "'@user"
    assert row["unavailable_reason"] == ""


def test_csv_output_is_deterministic_and_filename_uses_only_session_id() -> None:
    report = _report()

    assert serialize_report_csv(report) == serialize_report_csv(report)
    assert report_csv_filename(SESSION_ID) == f"report-{SESSION_ID}.csv"

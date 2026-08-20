"""Deterministic, safe CSV serialization for normalized session reports."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from uuid import UUID

    from app.schemas.report import ReportResponse, SectionEnvelope


CSV_COLUMNS = (
    "session_id",
    "report_status",
    "section_name",
    "section_state",
    "evaluation_kind",
    "evaluation_version_id",
    "evaluation_version_number",
    "score_status",
    "score",
    "passing_score",
    "passed",
    "evidence",
    "data",
    "unavailable_reason",
    "failure_class",
)
_FORMULA_PREFIXES = ("=", "+", "-", "@")


def serialize_report_csv(report: ReportResponse) -> str:
    """Serialize a normalized report into deterministic UTF-8 CSV text.

    One row is emitted for every report section, including empty and failed
    sections. Structured section data is represented as stable JSON so nested
    scores and evidence are not silently discarded.
    """
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, lineterminator="\r\n")
    writer.writeheader()
    for section in report.sections:
        writer.writerow(_section_row(report, section))
    return output.getvalue()


def report_csv_filename(session_id: UUID) -> str:
    """Return a safe, deterministic attachment filename for a session report."""
    return f"report-{session_id}.csv"


def _section_row(report: ReportResponse, section: SectionEnvelope) -> dict[str, str]:
    version = report.evaluation_version
    data = _as_mapping(section.data)
    failure_class = section.failure.class_.value if section.failure else ""
    return {
        "session_id": str(report.session.id),
        "report_status": report.report_status.value,
        "section_name": section.name.value,
        "section_state": section.state.value,
        "evaluation_kind": version.kind.value,
        "evaluation_version_id": _stringify(version.id),
        "evaluation_version_number": _stringify(version.number),
        "score_status": report.score_status.value,
        "score": _safe_cell(data.get("score", data.get("overall_score"))),
        "passing_score": _safe_cell(data.get("passing_score")),
        "passed": _safe_cell(data.get("passed")),
        "evidence": _safe_json(data.get("evidence", data.get("strengths"))),
        "data": _safe_json(section.data),
        "unavailable_reason": _safe_cell(section.unavailable_reason),
        "failure_class": failure_class,
    }


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _safe_json(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str | int | float | bool):
        return _safe_cell(value)
    return _safe_cell(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _safe_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.startswith(_FORMULA_PREFIXES):
        return "'" + text
    return text


def _stringify(value: Any) -> str:
    return "" if value is None else str(value)

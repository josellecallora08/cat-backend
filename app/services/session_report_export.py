"""Render a stored SessionReport payload into JSON, CSV, or PDF bytes.

CSV reuses the exact formula-neutralization helper already implemented in
app.services.upload_extractor rather than duplicating that security rule.
PDF uses the pinned pure-Python reportlab renderer.
"""

import csv
import io
import json
import re
from datetime import UTC, datetime
from xml.sax.saxutils import escape as escape_xml

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.models import SessionReport
from app.services.upload_extractor import _neutralize_csv_cell


SUPPORTED_FORMATS = frozenset({"json", "csv", "pdf"})
_FILENAME_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_-]")


class UnsupportedExportFormatError(ValueError):
    """Raised when an export format value is not one of SUPPORTED_FORMATS."""


class ExportFormatNotImplementedError(NotImplementedError):
    """Raised for a supported format whose renderer is not yet implemented."""


def sanitize_filename_component(value: str) -> str:
    """Return a conservative Content-Disposition filename component.

    Only ASCII letters, digits, underscores, and hyphens are retained.  This
    removes path separators, control characters, quotes, CR/LF, and other
    header syntax in one bounded transformation.  Empty/untrusted values are
    represented by ``unknown`` so the resulting fallback remains useful.
    """
    text = str(value) if value is not None else ""
    safe = _FILENAME_COMPONENT_RE.sub("_", text)
    return safe[:128] or "unknown"


def build_export_filename(session_id: str, report_version: int, export_format: str) -> str:
    """Build a deterministic, sanitized export filename.

    Validate the extension before constructing the header so callers cannot
    smuggle header syntax through the filename suffix.
    """
    if export_format not in SUPPORTED_FORMATS:
        raise UnsupportedExportFormatError("Unsupported export format")
    safe_session_id = sanitize_filename_component(session_id)
    safe_version = sanitize_filename_component(report_version)
    return f"session-report_{safe_session_id}_v{safe_version}.{export_format}"


def render_json(report: SessionReport) -> bytes:
    """Return the stored canonical payload, unchanged, as UTF-8 JSON bytes."""
    return json.dumps(report.payload, sort_keys=True, ensure_ascii=True).encode("utf-8")


def _iso(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return str(value)


def render_csv(report: SessionReport) -> bytes:
    """Flatten all stored sections into a CRLF, spreadsheet-safe CSV.

    The shared neutralizer is applied to every serialized cell, including
    headers, reason/terminal rows, and empty-row values.  A terminal or
    unavailable section remains visible rather than being silently omitted.
    """
    payload = report.payload or {}
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\r\n")

    def write_row(*cells: object) -> None:
        safe_cells = [_neutralize_csv_cell(str(cell)) for cell in cells]
        writer.writerow(safe_cells)

    def reason(section: dict) -> str:
        return str(section.get("reason_code") or section.get("reason") or "Not available")

    summary = payload.get("summary", {}) or {}
    write_row("Section", "Field", "Value")
    for field in (
        "session_id",
        "scenario_id",
        "campaign_name",
        "status",
        "created_at",
        "ended_at",
        "duration_seconds",
        "standard_name",
        "standard_version_number",
    ):
        value = (
            _iso(summary.get(field))
            if field in {"created_at", "ended_at"}
            else summary.get(field) or ""
        )
        write_row("summary", field, value)
    write_row()

    evaluation = payload.get("evaluation", {}) or {}
    write_row("Section", "Category", "Raw Score", "Weighted Contribution", "Passed")
    evaluation_rows = 0
    if evaluation.get("mode") == "canonical" and evaluation.get("canonical"):
        for category in evaluation["canonical"].get("categories", []):
            evaluation_rows += 1
            write_row(
                "evaluation",
                category.get("category", ""),
                category.get("raw_score", ""),
                category.get("weighted_contribution", ""),
                category.get("passed", ""),
            )
    elif evaluation.get("mode") == "legacy" and evaluation.get("legacy"):
        for category in evaluation["legacy"].get("category_scores", []):
            evaluation_rows += 1
            write_row("evaluation", category.get("category", ""), category.get("score", ""), "", "")
    if not evaluation_rows:
        write_row("evaluation", "terminal", reason(evaluation), "", "")
    write_row()

    coaching = payload.get("coaching", {}) or {}
    write_row("Section", "Block", "Criterion", "Recommended Response")
    coaching_rows = 0
    if coaching.get("mode") == "canonical":
        for block in coaching.get("blocks", []):
            for recommendation in block.get("recommendations", []):
                coaching_rows += 1
                write_row(
                    "coaching",
                    block.get("block_name", ""),
                    recommendation.get("criterion_id", ""),
                    recommendation.get("recommended_response", ""),
                )
    elif coaching.get("mode") == "legacy":
        for category, mistakes in coaching.get("legacy_mistakes_by_category", {}).items():
            for mistake in mistakes:
                coaching_rows += 1
                write_row("coaching", category, "", mistake.get("recommended_alternative", ""))
    if not coaching_rows:
        write_row("coaching", "terminal", reason(coaching), "")
    write_row()

    learning_plan = payload.get("learning_plan", {}) or {}
    write_row("Section", "Block", "Criterion", "Practice Focus", "Score", "Scenario")
    plan_rows = 0
    for item in learning_plan.get("items", []):
        plan_rows += 1
        write_row(
            "learning_plan",
            item.get("rubric_block_id", ""),
            item.get("criterion_id", ""),
            item.get("practice_focus") or item.get("recommended_scenario", ""),
            item.get("score", ""),
            item.get("scenario_id", ""),
        )
    if not plan_rows:
        terminal_value = "all_passing" if learning_plan.get("available") else "terminal"
        write_row("learning_plan", terminal_value, reason(learning_plan), "", "", "")
    write_row()

    transcript = payload.get("transcript", {}) or {}
    write_row("Section", "Sequence", "Speaker", "Text")
    entries = transcript.get("entries", []) or []
    for entry in entries:
        write_row(
            "transcript",
            entry.get("sequence_number", ""),
            entry.get("speaker", ""),
            entry.get("text", ""),
        )
    if not entries:
        write_row("transcript", "terminal", reason(transcript), "")

    return buf.getvalue().encode("utf-8")


def render_pdf(report: SessionReport) -> bytes:
    """Render a readable paginated PDF containing report sections."""
    payload = report.payload or {}
    summary = payload.get("summary", {})
    evaluation = payload.get("evaluation", {})
    coaching = payload.get("coaching", {}) or {}
    learning_plan = payload.get("learning_plan", {}) or {}
    transcript = payload.get("transcript", {}) or {}

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title=f"Session Report {summary.get('session_id', report.session_id)}",
    )
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]
    body_style.leading = 12
    story = [
        Paragraph("Session Report", title_style),
        Paragraph(
            f"Session: {escape_xml(str(summary.get('session_id', '')))} · Version {report.report_version}",
            body_style,
        ),
        Spacer(1, 10),
        Paragraph("Summary", heading_style),
    ]

    summary_rows = [
        ["Scenario", str(summary.get("scenario_id", ""))],
        ["Campaign", str(summary.get("campaign_name") or "—")],
        ["Status", str(summary.get("status", ""))],
        ["Created", _iso(summary.get("created_at"))],
        ["Ended", _iso(summary.get("ended_at")) or "—"],
        [
            "Duration",
            f"{summary.get('duration_seconds')} seconds"
            if summary.get("duration_seconds") is not None
            else "—",
        ],
        ["Standard", str(summary.get("standard_name") or "—")],
        ["Standard version", str(summary.get("standard_version_number") or "—")],
    ]
    summary_table = Table(summary_rows, colWidths=[1.45 * inch, 5.8 * inch], repeatRows=0)
    summary_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f3f4f6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend([summary_table, Spacer(1, 12), Paragraph("Evaluation", heading_style)])

    evaluation_rows = [["Category", "Raw", "Penalized", "Weight", "Weighted", "Result"]]
    canonical = evaluation.get("canonical") or {}
    if evaluation.get("mode") == "canonical":
        for category in canonical.get("categories", []):
            evaluation_rows.append(
                [
                    str(category.get("category", "")),
                    str(
                        category.get("raw_score")
                        if category.get("raw_score") is not None
                        else "N/A"
                    ),
                    str(
                        category.get("penalized_score")
                        if category.get("penalized_score") is not None
                        else "N/A"
                    ),
                    str(category.get("weight", "")),
                    str(category.get("weighted_contribution", "")),
                    "Passed" if category.get("passed") else "Needs practice",
                ]
            )
    elif evaluation.get("mode") == "legacy":
        for category in (evaluation.get("legacy") or {}).get("category_scores", []):
            evaluation_rows.append(
                [
                    str(category.get("category", "")),
                    str(category.get("score", "")),
                    "—",
                    "—",
                    "—",
                    "—",
                ]
            )
    else:
        evaluation_rows.append(
            [
                evaluation.get("reason_code") or evaluation.get("reason") or "Not available",
                "—",
                "—",
                "—",
                "—",
                "—",
            ]
        )
    evaluation_table = Table(
        evaluation_rows,
        repeatRows=1,
        colWidths=[2.1 * inch, 0.65 * inch, 0.75 * inch, 0.6 * inch, 0.8 * inch, 1.2 * inch],
    )
    evaluation_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.extend([evaluation_table, Spacer(1, 12), Paragraph("Coaching", heading_style)])

    coaching_rows = [["Block", "Criterion", "Recommendation", "Advice"]]
    if coaching.get("mode") == "canonical":
        for block in coaching.get("blocks", []):
            for recommendation in block.get("recommendations", []):
                coaching_rows.append(
                    [
                        str(block.get("block_name", "")),
                        str(
                            recommendation.get("criterion_name")
                            or recommendation.get("criterion_id", "")
                        ),
                        str(recommendation.get("recommended_response", "")),
                        str(recommendation.get("coaching_advice", "")),
                    ]
                )
    elif coaching.get("mode") == "legacy":
        for category, mistakes in coaching.get("legacy_mistakes_by_category", {}).items():
            for mistake in mistakes:
                coaching_rows.append(
                    [
                        str(category),
                        "—",
                        str(mistake.get("recommended_alternative", "")),
                        str(mistake.get("explanation", "")),
                    ]
                )
    else:
        coaching_rows.append(
            [
                coaching.get("reason_code") or coaching.get("reason") or "Not available",
                "—",
                "—",
                "—",
            ]
        )
    coaching_table = Table(
        coaching_rows, repeatRows=1, colWidths=[1.35 * inch, 1.25 * inch, 2.35 * inch, 2.15 * inch]
    )
    coaching_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.extend([coaching_table, Spacer(1, 12), Paragraph("Learning Plan", heading_style)])

    learning_plan_rows = [["Block", "Criterion", "Practice Focus", "Score", "Scenario"]]
    for item in learning_plan.get("items", []):
        learning_plan_rows.append(
            [
                str(item.get("rubric_block_id", "")),
                str(item.get("criterion_id", "")),
                str(item.get("practice_focus") or item.get("recommended_scenario", "")),
                str(item.get("score", "")),
                str(item.get("scenario_id", "")),
            ]
        )
    if len(learning_plan_rows) == 1:
        learning_plan_rows.append(
            [
                "all passing" if learning_plan.get("available") else "terminal",
                str(
                    learning_plan.get("reason_code")
                    or learning_plan.get("reason")
                    or "Not available"
                ),
                "",
                "",
                "",
            ]
        )
    learning_plan_table = Table(
        learning_plan_rows,
        repeatRows=1,
        colWidths=[1.25 * inch, 1.25 * inch, 2.55 * inch, 0.65 * inch, 1.25 * inch],
    )
    learning_plan_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.extend([learning_plan_table, Spacer(1, 12), Paragraph("Transcript", heading_style)])

    transcript_rows = [["Sequence", "Speaker", "Text"]]
    for entry in transcript.get("entries", []):
        transcript_rows.append(
            [
                str(entry.get("sequence_number", "")),
                str(entry.get("speaker", "")),
                str(entry.get("text", "")),
            ]
        )
    if len(transcript_rows) == 1:
        transcript_rows.append(
            [
                "terminal",
                "",
                str(
                    transcript.get("reason_code")
                    or transcript.get("reason")
                    or "No transcript recorded"
                ),
            ]
        )
    transcript_table = Table(
        transcript_rows, repeatRows=1, colWidths=[0.7 * inch, 0.9 * inch, 5.5 * inch]
    )
    transcript_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(transcript_table)
    document.build(story)
    return buffer.getvalue()


def render_export(report: SessionReport, export_format: str) -> tuple[bytes, str]:
    """Render `report` in `export_format`. Returns (body_bytes, media_type).

    Raises:
        UnsupportedExportFormatError: for any value outside SUPPORTED_FORMATS.
        ExportFormatNotImplementedError: for a supported but unimplemented format.
    """
    if export_format not in SUPPORTED_FORMATS:
        raise UnsupportedExportFormatError(f"Unsupported export format: {export_format}")

    if export_format == "json":
        return render_json(report), "application/json"
    if export_format == "csv":
        return render_csv(report), "text/csv"
    if export_format == "pdf":
        return render_pdf(report), "application/pdf"

    raise ExportFormatNotImplementedError(f"Export format '{export_format}' is not implemented")

"""Focused safety and completeness tests for session-report exports."""

import io
import json
import re
from email.message import Message
from types import SimpleNamespace

import pytest
from pypdf import PdfReader

from app.services import session_report_export as export
from app.services.upload_extractor import _neutralize_csv_cell


def _report(payload=None):
    return SimpleNamespace(
        session_id="stored-session-id",
        report_version=7,
        payload=payload or _payload(),
    )


def _payload():
    return {
        "summary": {
            "session_id": "session/with\\unsafe\r\n",
            "scenario_id": "scenario-1",
            "campaign_name": "Campaign",
            "status": "completed",
            "created_at": "2026-01-01T00:00:00+00:00",
            "ended_at": "2026-01-01T00:01:00+00:00",
            "duration_seconds": 60,
            "standard_name": "Pinned standard",
            "standard_version_number": 2,
        },
        "evaluation": {
            "available": True,
            "mode": "canonical",
            "canonical": {
                "categories": [{
                    "category": "=formula-category",
                    "raw_score": 80,
                    "weighted_contribution": 40,
                    "passed": True,
                    "penalized_score": 80,
                    "weight": 50,
                }],
            },
        },
        "coaching": {
            "available": True,
            "mode": "canonical",
            "blocks": [{
                "block_name": "Coaching block",
                "recommendations": [{
                    "criterion_id": "criterion-1",
                    "recommended_response": "+safe response",
                    "coaching_advice": "Advice",
                }],
            }],
        },
        "learning_plan": {
            "available": True,
            "items": [{
                "rubric_block_id": "block-1",
                "criterion_id": "criterion-1",
                "practice_focus": "Practice focus",
                "score": 50,
                "scenario_id": "scenario-1",
            }],
        },
        "transcript": {
            "available": True,
            "entries": [{
                "sequence_number": 0,
                "speaker": "agent",
                "text": "- transcript text",
            }],
        },
    }


@pytest.mark.parametrize("value", ["=formula", "+formula", "-formula", "@formula", "|formula", "\tformula"])
def test_csv_helper_neutralizes_every_formula_leading_character(value):
    result = _neutralize_csv_cell(value)
    assert result == "'" + value


def test_filename_is_deterministic_and_safe_for_content_disposition_parsing():
    filename = export.build_export_filename(
        'id/..\\"\r\n;:<>|?*', 7, "csv"
    )
    assert filename == export.build_export_filename(
        'id/..\\"\r\n;:<>|?*', 7, "csv"
    )
    assert not re.search(r'[\\/\x00-\x1f\x7f";:<>|?*]', filename)

    message = Message()
    message["Content-Disposition"] = f'attachment; filename="{filename}"'
    assert message.get_filename() == filename
    assert filename.endswith(".csv")


def test_json_export_is_semantically_identical_to_stored_payload():
    payload = _payload()
    assert json.loads(export.render_json(_report(payload))) == payload


def test_csv_export_has_crlf_and_all_sections_and_neutralizes_every_cell(monkeypatch):
    calls = []
    original = export._neutralize_csv_cell

    def recording_neutralizer(cell):
        calls.append(cell)
        return original(cell)

    monkeypatch.setattr(export, "_neutralize_csv_cell", recording_neutralizer)
    body = export.render_csv(_report())
    text = body.decode("utf-8")

    assert body
    assert "\r\n" in text
    assert "evaluation" in text
    assert "coaching" in text
    assert "learning_plan" in text
    assert "transcript" in text
    assert "'=formula-category" in text
    assert "'+safe response" in text
    assert "'- transcript text" in text
    assert len(calls) > 20


def test_pdf_export_is_non_empty_and_contains_paginated_report_sections():
    body = export.render_pdf(_report())
    reader = PdfReader(io.BytesIO(body))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    assert body.startswith(b"%PDF")
    assert len(body) > 500
    assert len(reader.pages) >= 1
    assert "Summary" in text
    assert "Evaluation" in text
    assert "Coaching" in text
    assert "Learning Plan" in text
    assert "Transcript" in text


def test_pdf_export_includes_terminal_state_and_empty_transcript():
    payload = _payload()
    payload["evaluation"] = {
        "available": True,
        "mode": "too_short",
        "reason_code": "session_too_short",
        "reason": "Too short",
    }
    payload["transcript"] = {
        "available": True,
        "reason_code": "empty_transcript",
        "entries": [],
    }
    body = export.render_pdf(_report(payload))
    text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(body)).pages)
    assert "session_too_short" in text
    assert "empty_transcript" in text


@pytest.mark.parametrize(
    ("format_name", "media_type"),
    [("json", "application/json"), ("csv", "text/csv"), ("pdf", "application/pdf")],
)
def test_render_export_supports_all_media_types(format_name, media_type):
    body, actual_media_type = export.render_export(_report(), format_name)
    assert actual_media_type == media_type
    assert body

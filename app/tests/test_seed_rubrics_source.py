"""Tests for strict rubric seed source loading."""

import json
from uuid import UUID

import pytest
from hypothesis import given
from hypothesis import strategies as st
from scripts.seed_rubrics import (
    ClassifiedSeedError,
    DefinitionOutcome,
    RubricSourceError,
    SeedRunResult,
    canonical_rubric_content,
    load_rubric_source,
    normalize_rubric_content,
    rubric_content_hash,
)

from app.config import settings
from app.services.negotiation_standard_service import canonical_content_hash


CAMPAIGN_ID = "00000000-0000-0000-0000-000000000001"
CONTENT = {"schema_version": 1, "overall_passing_score": 70, "blocks": []}


def _document() -> dict:
    return {
        "source_id": " approved-rubrics-v1 ",
        "definitions": [
            {
                "campaign_id": CAMPAIGN_ID,
                "name": "  Collections   Quality Rubric ",
                "draft_content": CONTENT,
            }
        ],
    }


def test_load_rubric_source_validates_and_normalizes(tmp_path) -> None:
    path = tmp_path / "rubrics.json"
    path.write_text(json.dumps(_document()), encoding="utf-8")

    document = load_rubric_source(str(path))

    assert document.source_id == "approved-rubrics-v1"
    assert document.definitions[0].name == "Collections Quality Rubric"
    assert document.definitions[0].campaign_id == UUID(CAMPAIGN_ID)


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        ({"source_id": "source", "definitions": [{"campaign_id": CAMPAIGN_ID}]}, "malformed"),
        ({"source_id": "source", "definitions": [], "unexpected": True}, "malformed"),
    ],
)
def test_load_rubric_source_rejects_invalid_documents(
    tmp_path, payload: dict, expected_code: str
) -> None:
    path = tmp_path / "rubrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RubricSourceError) as error:
        load_rubric_source(str(path))

    assert error.value.code == expected_code
    assert str(path) not in error.value.message


def test_load_rubric_source_rejects_oversized_input(tmp_path) -> None:
    path = tmp_path / "rubrics.json"
    path.write_text(json.dumps(_document()), encoding="utf-8")

    with pytest.raises(RubricSourceError, match="size limit") as error:
        load_rubric_source(str(path), max_bytes=1)

    assert error.value.code == "unsupported"


def test_load_rubric_source_rejects_remote_source() -> None:
    with pytest.raises(RubricSourceError) as error:
        load_rubric_source("https://example.invalid/rubrics.json")

    assert error.value.code == "unsupported"


def test_load_rubric_source_uses_safe_source_id_override(tmp_path) -> None:
    path = tmp_path / "rubrics.json"
    path.write_text(json.dumps(_document()), encoding="utf-8")

    document = load_rubric_source(str(path), source_id_override="  operator-source  ")

    assert document.source_id == "operator-source"


def test_normalize_rubric_content_returns_canonical_snapshot() -> None:
    content = {"blocks": [], "overall_passing_score": 70, "schema_version": 1}

    normalized = normalize_rubric_content(content)

    assert normalized.model_dump(mode="json") == content
    assert canonical_rubric_content(normalized) == content


def test_rubric_content_hash_reuses_service_semantics() -> None:
    first = {"schema_version": 1, "overall_passing_score": 70, "blocks": []}
    equivalent = {"blocks": [], "schema_version": 1, "overall_passing_score": 70}

    assert rubric_content_hash(first) == rubric_content_hash(equivalent)
    assert rubric_content_hash(first) == canonical_content_hash(first)


def test_seed_result_aggregates_safe_definition_outcomes() -> None:
    result = SeedRunResult.new(" approved-rubrics-v1 ")
    result.add_outcome(
        DefinitionOutcome(identity="quality", status="created", published=True),
        rubric_created=True,
        version_created=True,
    )
    result.add_outcome(
        DefinitionOutcome(
            identity="broken",
            status="rejected",
            error=ClassifiedSeedError(code="validation", message="draft_content is invalid"),
        )
    )

    assert result.status == "partial_success"
    assert result.source_id == "approved-rubrics-v1"
    assert result.created_rubrics == 1
    assert result.created_versions == 1
    assert result.published_versions == 1
    assert result.rejected_definitions == 1
    assert "draft_content" in result.definition_outcomes[1].error.message


def test_disabled_seed_result_has_no_definition_details() -> None:
    result = SeedRunResult.new("configured-source", disabled=True)

    assert result.status == "disabled"
    assert result.definition_outcomes == []
    assert result.run_id is not None


@st.composite
def valid_content_dicts(draw: st.DrawFn) -> dict:
    """Generate valid minimal rubric content for canonical round-trip testing."""
    return {
        "schema_version": draw(st.integers(min_value=1, max_value=5)),
        "overall_passing_score": draw(st.integers(min_value=0, max_value=100)),
        "blocks": [],
    }


@given(valid_content_dicts())
def test_canonical_content_round_trip_preserves_fingerprint(content: dict) -> None:
    """Property 1: canonical serialization preserves content and its fingerprint."""
    normalized = normalize_rubric_content(content)
    serialized = canonical_rubric_content(normalized)
    reparsed = normalize_rubric_content(serialized)

    assert reparsed == normalized
    assert rubric_content_hash(reparsed) == rubric_content_hash(normalized)


@pytest.mark.parametrize(
    ("source_id", "expected"),
    [
        ("  source   with   spaces  ", "source with spaces"),
        ("x" * 200, "x" * 120),
        ("\t\n", "unknown"),
    ],
)
def test_seed_result_safe_source_identifier(source_id: str, expected: str) -> None:
    """Result source IDs are normalized, bounded, and never empty."""
    assert SeedRunResult.new(source_id).source_id == expected


def test_source_error_redacts_path_and_secret(tmp_path) -> None:
    """Source failures expose remediation guidance but no sensitive source details."""
    sensitive_marker = "testkey"
    path = tmp_path / f"rubrics-{sensitive_marker}.json"
    path.write_bytes(b"not-json")

    with pytest.raises(RubricSourceError) as error:
        load_rubric_source(str(path))

    assert error.value.code == "malformed"
    assert sensitive_marker not in error.value.message
    assert str(path) not in error.value.message


def test_source_validation_rejects_invalid_uuid_and_unexpected_fields(tmp_path) -> None:
    """Strict source validation rejects invalid identifiers and unknown fields."""
    payload = _document()
    payload["definitions"][0]["campaign_id"] = "not-a-uuid"
    payload["definitions"][0]["private_marker"] = "do-not-return"
    path = tmp_path / "rubrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RubricSourceError) as error:
        load_rubric_source(str(path))

    assert error.value.code == "malformed"
    assert "not-a-uuid" not in error.value.message
    assert "do-not-return" not in error.value.message


def test_source_validation_rejects_non_object_and_empty_source_id(tmp_path) -> None:
    """Top-level JSON must be an object with a non-whitespace source ID."""
    path = tmp_path / "rubrics.json"
    path.write_text(json.dumps({"source_id": " ", "definitions": []}), encoding="utf-8")

    with pytest.raises(RubricSourceError) as error:
        load_rubric_source(str(path))

    assert error.value.code == "malformed"


def test_load_source_requires_configuration(monkeypatch) -> None:
    """Missing configured source is classified without exposing settings details."""
    monkeypatch.setattr(settings, "rubric_source", None)

    with pytest.raises(RubricSourceError) as error:
        load_rubric_source()

    assert error.value.code == "configuration"
    assert "CAT_" not in error.value.message


def test_seed_result_aggregates_reused_and_failed_outcomes() -> None:
    """Counter aggregation distinguishes reuse from persistence failure."""
    result = SeedRunResult.new("source")
    result.add_outcome(
        DefinitionOutcome(identity="existing", status="reused", published=False),
        rubric_created=False,
        version_created=False,
    )
    result.add_outcome(
        DefinitionOutcome(
            identity="failed",
            status="failed",
            error=ClassifiedSeedError(code="persistence", message="Persistence failed safely"),
        )
    )

    assert result.status == "failed"
    assert result.reused_rubrics == 1
    assert result.reused_versions == 1
    assert result.created_rubrics == 0
    assert result.created_versions == 0
    assert result.rejected_definitions == 0
    serialized = result.model_dump_json()
    assert "Persistence failed safely" in serialized
    assert "password" not in serialized.lower()
    assert "token" not in serialized.lower()


def test_rubric_seed_audit_contains_safe_run_summary(caplog) -> None:
    """Run audit records contain identity, principal, timestamp, and counters only."""
    from app.services.audit import log_rubric_seed_completed

    run_id = UUID("00000000-0000-0000-0000-000000000002")
    principal = UUID("00000000-0000-0000-0000-000000000003")
    with caplog.at_level("INFO", logger="cats.audit"):
        log_rubric_seed_completed(
            run_id,
            "approved-rubrics-v1",
            principal,
            "partial_success",
            1,
            2,
            3,
            4,
            1,
            1,
        )

    record = next(record for record in caplog.records if record.message == "RUBRIC_SEED_COMPLETED")
    assert record.run_id == str(run_id)
    assert record.source_id == "approved-rubrics-v1"
    assert record.initiated_by == str(principal)
    assert record.status == "partial_success"
    assert record.created_versions == 4
    assert record.rejected_definitions == 1
    assert hasattr(record, "timestamp")
    assert "password" not in record.__dict__
    assert "token" not in record.__dict__
    assert "snapshot" not in record.__dict__

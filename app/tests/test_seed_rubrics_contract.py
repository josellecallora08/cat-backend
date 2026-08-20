"""End-to-end contract tests for documented rubric seed examples and output."""

from __future__ import annotations

import json
from uuid import UUID

from scripts import seed_rubrics as seed_module


PLACEHOLDER_CAMPAIGN_ID = "00000000-0000-0000-0000-000000000000"
PLACEHOLDER_SOURCE_ID = "approved-rubrics-v1"


def _documented_source() -> dict:
    """Return the placeholder source document from the seed-flow contract."""
    return {
        "source_id": PLACEHOLDER_SOURCE_ID,
        "definitions": [
            {
                "campaign_id": PLACEHOLDER_CAMPAIGN_ID,
                "name": "Collections Quality Rubric",
                "description": "Placeholder rubric for documentation and contract tests.",
                "draft_content": {
                    "schema_version": 1,
                    "overall_passing_score": 70,
                    "blocks": [],
                },
                "publish": True,
                "publication_note": "Initial approved version",
            }
        ],
    }


def test_documented_placeholder_source_matches_strict_contract(tmp_path) -> None:
    """The documented placeholder JSON loads without a database or external service."""
    source_path = tmp_path / "documented-rubrics.json"
    source_path.write_text(json.dumps(_documented_source()), encoding="utf-8")

    document = seed_module.load_rubric_source(str(source_path))
    definition = document.definitions[0]

    assert document.source_id == PLACEHOLDER_SOURCE_ID
    assert definition.campaign_id == UUID(PLACEHOLDER_CAMPAIGN_ID)
    assert definition.name == "Collections Quality Rubric"
    assert definition.publish is True
    assert definition.draft_content.overall_passing_score == 70


def test_safe_result_contract_serializes_without_source_secrets() -> None:
    """Operational output contains the documented fields but no credentials or payloads."""
    result = seed_module.SeedRunResult.new(PLACEHOLDER_SOURCE_ID)
    result.add_outcome(
        seed_module.DefinitionOutcome(
            identity="approved-rubrics-v1:collections quality rubric",
            status="created",
            version_id=UUID(PLACEHOLDER_CAMPAIGN_ID),
            published=True,
        ),
        rubric_created=True,
        version_created=True,
    )

    payload = json.loads(result.model_dump_json())
    expected_fields = {
        "run_id",
        "source_id",
        "status",
        "reused_rubrics",
        "created_rubrics",
        "reused_versions",
        "created_versions",
        "published_versions",
        "rejected_definitions",
        "warnings",
        "definition_outcomes",
        "campaign_id",
        "scenario_id",
        "agent_id",
        "rubric_version_id",
        "session_id",
    }

    assert set(payload) == expected_fields
    assert payload["status"] == "success"
    assert payload["source_id"] == PLACEHOLDER_SOURCE_ID
    assert payload["created_rubrics"] == 1
    assert payload["created_versions"] == 1
    assert payload["published_versions"] == 1
    serialized = json.dumps(payload).lower()
    assert "password" not in serialized
    assert "token" not in serialized
    assert "credential" not in serialized
    assert "draft_content" not in serialized
    assert "publication_note" not in serialized


def test_cli_contract_emits_safe_json_without_database(monkeypatch, capsys) -> None:
    """CLI output follows the result contract when the runner is supplied independently."""
    result = seed_module.SeedRunResult.new(PLACEHOLDER_SOURCE_ID)
    monkeypatch.setattr(seed_module, "run_seed", _async_result(result))

    assert seed_module.main() == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert captured.err == ""
    assert payload["status"] == "success"
    assert payload["source_id"] == PLACEHOLDER_SOURCE_ID
    assert "Traceback" not in captured.out
    assert "secret" not in captured.out.lower()


def _async_result(result: seed_module.SeedRunResult):
    """Return an async runner replacement for the service-free CLI contract test."""

    async def runner() -> seed_module.SeedRunResult:
        return result

    return runner

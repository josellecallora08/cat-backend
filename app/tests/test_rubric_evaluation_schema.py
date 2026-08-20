"""Tests for the strict AI rubric observation contract."""

import pytest
from pydantic import ValidationError

from app.schemas.rubric_evaluation import (
    RubricAIObservation,
    RubricAppliedTechniques,
)


def _category(raw_score: int | None = 80) -> dict:
    return {
        "rubric_block_id": "opening",
        "raw_score": raw_score,
        "evidence": [
            {
                "sequence_number": 1,
                "speaker": "agent",
                "excerpt": "Hello, I am calling about your account.",
                "explanation": "The agent states the purpose of the call.",
            }
        ],
        "strengths": [],
        "violations": [],
        "failed_criteria": [],
        "recommendation_inputs": [],
    }


def _observation(**overrides: object) -> dict:
    value: dict = {
        "status": "evaluated",
        "summary": "The agent used an evidence-supported opening.",
        "categories": [_category()],
        "applied_techniques": {"techniques_used": [], "reason_if_empty": "None observed."},
        "missed_opportunities": {
            "missed_techniques": [],
            "reason_if_empty": "None identified.",
        },
    }
    value.update(overrides)
    return value


def test_valid_evaluated_observation_requires_all_contract_sections() -> None:
    observation = RubricAIObservation.model_validate(_observation())

    assert observation.status == "evaluated"
    assert observation.categories[0].raw_score == 80
    assert observation.applied_techniques.reason_if_empty == "None observed."


def test_blank_reason_if_empty_gets_fallback_for_empty_lists() -> None:
    observation = RubricAIObservation.model_validate(
        _observation(
            applied_techniques={"techniques_used": [], "reason_if_empty": ""},
            missed_opportunities={"missed_techniques": [], "reason_if_empty": "   "},
        )
    )

    assert observation.applied_techniques.reason_if_empty == "No techniques were evidenced."
    assert (
        observation.missed_opportunities.reason_if_empty
        == "No missed opportunities were evidenced."
    )


def test_blank_reason_if_empty_is_valid_when_corresponding_list_has_entries() -> None:
    applied = {
        "techniques_used": [
            {
                "technique_name": "Move Forward",
                "execution_type": "Executed Properly",
                "execution_description": "The agent redirected the discussion.",
                "evidence_sequence_numbers": [1],
            }
        ],
        "reason_if_empty": "",
    }
    missed = {
        "missed_techniques": [
            {"technique_name": "Summarize", "reason": "The agent did not recap the plan."}
        ],
        "reason_if_empty": "",
    }

    observation = RubricAIObservation.model_validate(
        _observation(applied_techniques=applied, missed_opportunities=missed)
    )

    assert observation.applied_techniques.reason_if_empty == ""
    assert observation.missed_opportunities.reason_if_empty == ""


def test_not_applicable_uses_explicit_null_scores() -> None:
    observation = RubricAIObservation.model_validate(
        _observation(status="not_applicable", categories=[_category(None)])
    )

    assert observation.categories[0].raw_score is None


def test_evaluated_observation_rejects_missing_score() -> None:
    with pytest.raises(ValidationError, match="require a raw_score"):
        RubricAIObservation.model_validate(_observation(categories=[_category(None)]))


def test_not_applicable_observation_rejects_a_score() -> None:
    with pytest.raises(ValidationError, match="must not contain category scores"):
        RubricAIObservation.model_validate(
            _observation(status="not_applicable", categories=[_category(80)])
        )


def test_unknown_properties_are_rejected_at_every_level() -> None:
    category = _category()
    category["unexpected"] = True

    with pytest.raises(ValidationError):
        RubricAIObservation.model_validate(_observation(categories=[category]))


def test_applied_and_missed_techniques_are_unique_and_disjoint() -> None:
    applied = {
        "techniques_used": [
            {
                "technique_name": "Move Forward",
                "execution_type": "Executed Properly",
                "execution_description": "The agent redirected the discussion.",
                "evidence_sequence_numbers": [1],
            }
        ],
        "reason_if_empty": "Not empty.",
    }
    missed = {
        "missed_techniques": [{"technique_name": "Move Forward", "reason": "Already used."}],
        "reason_if_empty": "Not empty.",
    }

    with pytest.raises(ValidationError, match="disjoint"):
        RubricAIObservation.model_validate(
            _observation(applied_techniques=applied, missed_opportunities=missed)
        )


def test_missed_opportunity_is_not_an_applied_execution_type() -> None:
    with pytest.raises(ValidationError):
        RubricAppliedTechniques(
            techniques_used=[
                {
                    "technique_name": "Move Forward",
                    "execution_type": "Missed Opportunity",
                    "execution_description": "Not attempted.",
                    "evidence_sequence_numbers": [1],
                }
            ],
            reason_if_empty="Not empty.",
        )

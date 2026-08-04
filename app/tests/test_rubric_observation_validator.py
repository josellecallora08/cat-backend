"""Focused validation tests for TASK-032 AI observations."""

import math

import pytest

from app.services.rubric_observation_validator import (
    ObservationValidationError,
    validate_observation,
)


SNAPSHOT = {
    "schema_version": 1,
    "overall_passing_score": 70,
    "blocks": [
        {
            "id": "opening",
            "category": "Call Opening",
            "weight": 100,
            "passing_score": 70,
            "scoring_instructions": "Use evidence.",
            "positive_behaviors": [{"id": "greeting", "name": "Greeting", "description": "Greets.", "evidence_instructions": "Quote it."}],
            "violations": [{"id": "rude-tone", "name": "Rude tone", "description": "Uses a rude tone.", "evidence_instructions": "Quote it."}],
            "penalties": [{"violation_id": "rude-tone", "deduction": 20, "max_occurrences": 1}],
            "recommendation_guidance": "Be respectful.",
            "display_order": 0,
        }
    ],
}


def _category(**overrides: object) -> dict:
    value = {
        "rubric_block_id": "opening",
        "raw_score": 80,
        "evidence": [{"sequence_number": 1, "speaker": "agent", "excerpt": "Hello there", "explanation": "Greeting."}],
        "strengths": [],
        "violations": [],
        "failed_criteria": [],
        "recommendation_inputs": [],
    }
    value.update(overrides)
    return value


def _observation(**overrides: object) -> dict:
    value = {
        "status": "evaluated",
        "summary": "Grounded result.",
        "categories": [_category()],
        "applied_techniques": {"techniques_used": [], "reason_if_empty": "None."},
        "missed_opportunities": {"missed_techniques": [], "reason_if_empty": "None."},
    }
    value.update(overrides)
    return value


def _assert_invalid(observation: dict) -> set[str]:
    with pytest.raises(ObservationValidationError) as caught:
        validate_observation(observation, SNAPSHOT, [{"sequence_number": 1, "speaker": "agent", "text": "Hello there"}])
    return {error.code for error in caught.value.errors}


def test_requires_exact_category_coverage_and_rejects_unknown_or_duplicate() -> None:
    duplicate = _observation(categories=[_category(), _category()])
    codes = _assert_invalid(duplicate)
    assert "duplicate" in codes

    unknown = _category(rubric_block_id="unknown")
    codes = _assert_invalid(_observation(categories=[unknown]))
    assert "unknown_reference" in codes


@pytest.mark.parametrize("score", [-1, 101, math.nan, math.inf, -math.inf])
def test_rejects_non_finite_or_out_of_range_scores(score: float) -> None:
    assert "invalid" in _assert_invalid(_observation(categories=[_category(raw_score=score)]))


def test_rejects_missing_transcript_evidence_and_mismatched_speaker_or_excerpt() -> None:
    missing = _category(evidence=[{"sequence_number": 99, "speaker": "agent", "excerpt": "Hello", "explanation": "x"}])
    assert "unknown_reference" in _assert_invalid(_observation(categories=[missing]))

    wrong_speaker = _category(evidence=[{"sequence_number": 1, "speaker": "debtor", "excerpt": "Hello there", "explanation": "x"}])
    assert "invalid_evidence" in _assert_invalid(_observation(categories=[wrong_speaker]))

    wrong_excerpt = _category(evidence=[{"sequence_number": 1, "speaker": "agent", "excerpt": "Not present", "explanation": "x"}])
    assert "invalid_evidence" in _assert_invalid(_observation(categories=[wrong_excerpt]))


def test_rejects_unknown_references_and_findings_without_category_evidence() -> None:
    category = _category(
        evidence=[],
        strengths=[{"criterion_id": "greeting", "explanation": "x", "evidence_sequence_numbers": [1]}],
        violations=[{"violation_id": "unknown", "explanation": "x", "evidence_sequence_numbers": [1]}],
        failed_criteria=["greeting"],
    )
    codes = _assert_invalid(_observation(categories=[category]))
    assert "required" in codes
    assert "unknown_reference" in codes


def test_whitespace_normalization_is_not_fuzzy_matching() -> None:
    category = _category(evidence=[{"sequence_number": 1, "speaker": "agent", "excerpt": "Hello   there", "explanation": "x"}])
    validate_observation(_observation(categories=[category]), SNAPSHOT, [{"sequence_number": 1, "speaker": "agent", "text": " Hello there "}])
    assert "invalid_evidence" in _assert_invalid(_observation(categories=[_category(evidence=[{"sequence_number": 1, "speaker": "agent", "excerpt": "Hello world", "explanation": "x"}])]))

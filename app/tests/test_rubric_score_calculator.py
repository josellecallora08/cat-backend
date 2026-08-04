"""Focused deterministic scoring tests for TASK-032."""

from decimal import Decimal

import pytest

from app.services.rubric_observation_validator import validate_observation
from app.services.rubric_score_calculator import (
    ScoreInvariantError,
    _money,
    calculate_rubric_score,
)



def _block(block_id: str, weight: int, order: int, passing: int = 50) -> dict:
    return {
        "id": block_id,
        "category": block_id.title(),
        "weight": weight,
        "passing_score": passing,
        "scoring_instructions": "Use evidence.",
        "positive_behaviors": [{"id": f"{block_id}-good", "name": "Good", "description": "Good.", "evidence_instructions": "Quote."}],
        "violations": [{"id": f"{block_id}-bad", "name": "Bad", "description": "Bad.", "evidence_instructions": "Quote."}],
        "penalties": [{"violation_id": f"{block_id}-bad", "deduction": 10, "max_occurrences": 2}],
        "recommendation_guidance": "Improve.",
        "display_order": order,
    }


SNAPSHOT = {"schema_version": 1, "overall_passing_score": 70, "blocks": [_block("opening", 60, 0), _block("resolution", 40, 1)]}


def _category(block_id: str, score: int, sequence: int, violations: int = 0) -> dict:
    return {
        "rubric_block_id": block_id,
        "raw_score": score,
        "evidence": [{"sequence_number": sequence + i, "speaker": "agent", "excerpt": "Line", "explanation": "Evidence."} for i in range(max(1, violations))],
        "strengths": [],
        "violations": [{"violation_id": f"{block_id}-bad", "explanation": "Bad finding.", "evidence_sequence_numbers": [sequence + i]} for i in range(violations)],
        "failed_criteria": [],
        "recommendation_inputs": [],
    }


def _validated(categories: list[dict]):
    observation = {
        "status": "evaluated",
        "summary": "Grounded result.",
        "categories": categories,
        "applied_techniques": {"techniques_used": [], "reason_if_empty": "None."},
        "missed_opportunities": {"missed_techniques": [], "reason_if_empty": "None."},
    }
    transcript = [{"sequence_number": i, "speaker": "agent", "text": "Line"} for i in range(1, 6)]
    return validate_observation(observation, SNAPSHOT, transcript)


def test_penalties_are_capped_and_scores_floor_at_zero() -> None:
    validated = _validated([_category("resolution", 20, 4, 0), _category("opening", 15, 1, 3)])
    result = calculate_rubric_score(validated)

    opening = result.categories[0]
    assert opening.raw_score == 15
    assert opening.penalty_total == 20
    assert opening.penalized_score == 0
    assert opening.weighted_contribution == Decimal("0.00")
    assert result.categories[1].weighted_contribution == Decimal("8.00")
    assert result.weighted_total == Decimal("8.00")
    assert result.passed is False


def test_total_and_category_order_are_independent_of_observation_order() -> None:
    first = calculate_rubric_score(_validated([_category("opening", 80, 1), _category("resolution", 90, 2)]))
    second = calculate_rubric_score(_validated([_category("resolution", 90, 2), _category("opening", 80, 1)]))

    assert first.weighted_total == second.weighted_total == Decimal("84.00")
    assert [item.rubric_block_id for item in first.categories] == ["opening", "resolution"]
    assert first.passed is True


def test_snapshot_weights_are_authoritative_and_must_total_one_hundred() -> None:
    validated = _validated([_category("opening", 100, 1), _category("resolution", 100, 2)])
    altered = {**SNAPSHOT, "blocks": [{**SNAPSHOT["blocks"][0], "weight": 50}, {**SNAPSHOT["blocks"][1], "weight": 40}]}
    with pytest.raises(ScoreInvariantError, match="does not match"):
        calculate_rubric_score(validated, altered)


def test_rounding_uses_decimal_round_half_up() -> None:
    assert _money(Decimal("69.995")) == Decimal("70.00")


def test_not_applicable_is_safe_and_not_passing() -> None:
    observation = {
        "status": "not_applicable",
        "summary": "Transcript was insufficient.",
        "categories": [_category("opening", None, 1), _category("resolution", None, 2)],
        "applied_techniques": {"techniques_used": [], "reason_if_empty": "None."},
        "missed_opportunities": {"missed_techniques": [], "reason_if_empty": "None."},
    }
    transcript = [{"sequence_number": i, "speaker": "agent", "text": "Line"} for i in (1, 2)]
    validated = validate_observation(observation, SNAPSHOT, transcript)
    result = calculate_rubric_score(validated)
    assert result.weighted_total == Decimal("0.00")
    assert result.passed is False
    assert all(item.raw_score is None for item in result.categories)

"""Hypothesis release-gate properties for deterministic rubric scoring."""

from decimal import Decimal

import pytest
from hypothesis import given, strategies as st

from app.services.rubric_observation_validator import validate_observation
from app.services.rubric_score_calculator import _money, calculate_rubric_score


SNAPSHOT = {
    "schema_version": 1,
    "overall_passing_score": 70,
    "blocks": [
        {
            "id": "opening",
            "category": "Opening",
            "weight": 60,
            "passing_score": 70,
            "scoring_instructions": "Use evidence.",
            "positive_behaviors": [],
            "violations": [{"id": "rude-tone", "name": "Rude tone", "description": "Rude.", "evidence_instructions": "Quote."}],
            "penalties": [{"violation_id": "rude-tone", "deduction": 10, "max_occurrences": 2}],
            "recommendation_guidance": "Improve.",
            "display_order": 0,
        },
        {
            "id": "resolution",
            "category": "Resolution",
            "weight": 40,
            "passing_score": 70,
            "scoring_instructions": "Use evidence.",
            "positive_behaviors": [],
            "violations": [],
            "penalties": [],
            "recommendation_guidance": "Improve.",
            "display_order": 1,
        },
    ],
}


def _category(block_id: str, score: int, sequence: int, violation_count: int = 0) -> dict:
    sequences = [sequence + index for index in range(max(1, violation_count))]
    return {
        "rubric_block_id": block_id,
        "raw_score": score,
        "evidence": [{"sequence_number": item, "speaker": "agent", "excerpt": "Line", "explanation": "Evidence."} for item in sequences],
        "strengths": [],
        "violations": [{"violation_id": "rude-tone", "explanation": "Rude finding.", "evidence_sequence_numbers": [item]} for item in sequences[:violation_count]],
        "failed_criteria": [],
        "recommendation_inputs": [],
    }


def _validated(opening_score: int, resolution_score: int, violation_count: int = 0, order: tuple[str, str] = ("opening", "resolution")):
    categories = {
        "opening": _category("opening", opening_score, 1, violation_count),
        "resolution": _category("resolution", resolution_score, 5),
    }
    observation = {
        "status": "evaluated",
        "summary": "Grounded result.",
        "categories": [categories[item] for item in order],
        "applied_techniques": {"techniques_used": [], "reason_if_empty": "None."},
        "missed_opportunities": {"missed_techniques": [], "reason_if_empty": "None."},
    }
    transcript = [{"sequence_number": item, "speaker": "agent", "text": "Line"} for item in range(1, 6)]
    return validate_observation(observation, SNAPSHOT, transcript)


@given(st.integers(min_value=0, max_value=100), st.integers(min_value=0, max_value=100), st.integers(min_value=0, max_value=5))
def test_penalized_scores_and_weighted_totals_stay_in_bounds(raw_opening: int, raw_resolution: int, violation_count: int) -> None:
    result = calculate_rubric_score(_validated(raw_opening, raw_resolution, violation_count))
    assert 0 <= result.weighted_total <= 100
    assert all(0 <= category.penalized_score <= 100 for category in result.categories if category.penalized_score is not None)
    assert all(0 <= category.weighted_contribution <= 100 for category in result.categories)


@given(st.integers(min_value=0, max_value=100), st.integers(min_value=0, max_value=100))
def test_category_input_permutation_does_not_change_serialized_score(opening_score: int, resolution_score: int) -> None:
    first = calculate_rubric_score(_validated(opening_score, resolution_score, order=("opening", "resolution")))
    second = calculate_rubric_score(_validated(opening_score, resolution_score, order=("resolution", "opening")))
    assert first.weighted_total == second.weighted_total
    assert first.model_dump_json() == second.model_dump_json()


@given(st.integers(min_value=0, max_value=100), st.integers(min_value=0, max_value=5))
def test_penalties_never_increase_a_score(raw_opening: int, violation_count: int) -> None:
    without_penalty = calculate_rubric_score(_validated(raw_opening, 80, 0))
    with_penalty = calculate_rubric_score(_validated(raw_opening, 80, violation_count))
    assert with_penalty.weighted_total <= without_penalty.weighted_total
    assert with_penalty.categories[0].penalized_score <= without_penalty.categories[0].penalized_score


@given(st.integers(min_value=0, max_value=100), st.integers(min_value=0, max_value=5))
def test_same_snapshot_and_observation_are_serialization_deterministic(raw_opening: int, violation_count: int) -> None:
    first = calculate_rubric_score(_validated(raw_opening, 80, violation_count))
    second = calculate_rubric_score(_validated(raw_opening, 80, violation_count))
    assert first.model_dump_json() == second.model_dump_json()


def test_exact_decimal_rounding_is_round_half_up() -> None:
    assert _money(Decimal("69.995")) == Decimal("70.00")
    assert _money(Decimal("69.994")) == Decimal("69.99")

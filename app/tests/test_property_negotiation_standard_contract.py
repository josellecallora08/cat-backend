"""Property tests for deterministic negotiation standard validation."""

from hypothesis import given, strategies as st

from app.schemas.negotiation_standard import NegotiationStandardContent
from app.services.negotiation_standard_validator import validate_standard


def _block(index: int, weight: int) -> dict:
    return {
        "id": f"block-{index}",
        "category": f"Category {index}",
        "weight": weight,
        "passing_score": 70,
        "scoring_instructions": "Score only transcript-supported behavior.",
        "positive_behaviors": [
            {
                "id": f"positive-{index}",
                "name": "Positive behavior",
                "description": "A supported positive behavior.",
                "evidence_instructions": "Cite the supporting turn.",
            }
        ],
        "violations": [
            {
                "id": f"violation-{index}",
                "name": "Violation",
                "description": "A supported violation.",
                "evidence_instructions": "Cite the exact turn.",
            }
        ],
        "penalties": [
            {
                "violation_id": f"violation-{index}",
                "deduction": 10,
                "max_occurrences": 1,
            }
        ],
        "recommendation_guidance": "Give a grounded alternative.",
        "display_order": index,
    }


@given(st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=5))
def test_publish_validity_matches_weight_sum(weights: list[int]) -> None:
    content = NegotiationStandardContent(
        overall_passing_score=70,
        blocks=[_block(index, weight) for index, weight in enumerate(weights)],
    )

    result = validate_standard(content, for_publish=True)

    assert result.valid is (sum(weights) == 100)
    assert result.weight_total == sum(weights)


@given(st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=5))
def test_validation_is_deterministic(weights: list[int]) -> None:
    content = NegotiationStandardContent(
        overall_passing_score=70,
        blocks=[_block(index, weight) for index, weight in enumerate(weights)],
    )

    first = validate_standard(content, for_publish=True).model_dump()
    second = validate_standard(content, for_publish=True).model_dump()

    assert first == second

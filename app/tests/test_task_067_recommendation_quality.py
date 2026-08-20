"""Contract tests for TASK-067 recommendation quality and safety."""

import pytest

from app.services.evaluation_compatibility import (
    RecommendationValidationError,
    build_rubric_recommendations,
)
from app.services.rubric_observation_validator import (
    ObservationValidationError,
    validate_observation,
)
from app.services.rubric_score_calculator import calculate_rubric_score


SNAPSHOT = {
    "schema_version": 1,
    "overall_passing_score": 70,
    "blocks": [
        {
            "id": "negotiation",
            "category": "Negotiation",
            "weight": 100,
            "passing_score": 70,
            "scoring_instructions": "Use evidence.",
            "positive_behaviors": [
                {
                    "id": "move-forward",
                    "name": "Move Forward",
                    "description": "Redirect the discussion toward a resolution.",
                    "evidence_instructions": "Cite the redirect.",
                }
            ],
            "violations": [
                {
                    "id": "legal-threat",
                    "name": "Legal Threat",
                    "description": "States an unsupported legal consequence.",
                    "evidence_instructions": "Cite the claim.",
                }
            ],
            "penalties": [],
            "recommendation_guidance": "Offer a compliant alternative and confirm the next step.",
            "display_order": 0,
        }
    ],
}


def _observation(**category_changes: object) -> dict:
    category = {
        "rubric_block_id": "negotiation",
        "raw_score": 60,
        "evidence": [
            {
                "sequence_number": 1,
                "speaker": "agent",
                "excerpt": "We can discuss options",
                "explanation": "Redirect.",
            },
            {
                "sequence_number": 2,
                "speaker": "agent",
                "excerpt": "You will face legal action",
                "explanation": "Claim.",
            },
        ],
        "strengths": [
            {
                "criterion_id": "move-forward",
                "explanation": "The agent redirected.",
                "evidence_sequence_numbers": [1],
            }
        ],
        "violations": [
            {
                "violation_id": "legal-threat",
                "explanation": "The agent made a threat.",
                "evidence_sequence_numbers": [2],
            }
        ],
        "failed_criteria": ["legal-threat"],
        "recommendation_inputs": [
            {
                "criterion_id": "legal-threat",
                "transcript_sequence_number": 2,
                "need": "Use a compliant alternative.",
            }
        ],
    }
    category.update(category_changes)
    return {
        "status": "evaluated",
        "summary": "Grounded.",
        "categories": [category],
        "applied_techniques": {"techniques_used": [], "reason_if_empty": "None."},
        "missed_opportunities": {"missed_techniques": [], "reason_if_empty": "None."},
    }


def _canonical(observation: dict | None = None):
    validated = validate_observation(
        observation or _observation(),
        SNAPSHOT,
        [
            {"sequence_number": 1, "speaker": "agent", "text": "We can discuss options"},
            {"sequence_number": 2, "speaker": "agent", "text": "You will face legal action"},
        ],
    )
    return calculate_rubric_score(validated)


def test_recommendation_evidence_must_belong_to_cited_criterion() -> None:
    observation = _observation(
        recommendation_inputs=[
            {
                "criterion_id": "legal-threat",
                "transcript_sequence_number": 1,
                "need": "Use a compliant alternative.",
            }
        ]
    )

    with pytest.raises(ObservationValidationError):
        validate_observation(
            observation,
            SNAPSHOT,
            [
                {"sequence_number": 1, "speaker": "agent", "text": "We can discuss options"},
                {"sequence_number": 2, "speaker": "agent", "text": "You will face legal action"},
            ],
        )

    canonical = _canonical()
    canonical.categories[0].recommendation_inputs[0].transcript_sequence_number = 1
    with pytest.raises(RecommendationValidationError):
        build_rubric_recommendations(canonical, SNAPSHOT)


def test_failed_positive_criterion_can_use_a_validated_recommendation_input() -> None:
    observation = _observation(
        strengths=[],
        violations=[],
        failed_criteria=["move-forward"],
        recommendation_inputs=[
            {
                "criterion_id": "move-forward",
                "transcript_sequence_number": 1,
                "need": "Add a clear redirect.",
            }
        ],
    )

    canonical = _canonical(observation)
    recommendations = build_rubric_recommendations(canonical, SNAPSHOT)

    assert len(recommendations) == 1
    assert recommendations[0].criterion_id == "move-forward"
    assert recommendations[0].evidence_sequence_number == 1


def test_equal_priority_recommendations_have_stable_rubric_order() -> None:
    observation = _observation(
        strengths=[
            {
                "criterion_id": "move-forward",
                "explanation": "The agent redirected.",
                "evidence_sequence_numbers": [2],
            }
        ],
        violations=[
            {
                "violation_id": "legal-threat",
                "explanation": "The agent made a threat.",
                "evidence_sequence_numbers": [2],
            }
        ],
        failed_criteria=["move-forward", "legal-threat"],
        recommendation_inputs=[
            {
                "criterion_id": "move-forward",
                "transcript_sequence_number": 2,
                "need": "Add a clear redirect.",
            },
            {
                "criterion_id": "legal-threat",
                "transcript_sequence_number": 2,
                "need": "Use a compliant alternative.",
            },
        ],
    )
    canonical = _canonical(observation)

    recommendations = build_rubric_recommendations(canonical, SNAPSHOT)

    assert [item.criterion_id for item in recommendations] == ["legal-threat", "move-forward"]


def test_all_recommendation_text_is_criterion_guided_and_redacted() -> None:
    snapshot = {
        **SNAPSHOT,
        "blocks": [
            {
                **SNAPSHOT["blocks"][0],
                "recommendation_guidance": "ACME_CORP contact jane [at] example [dot] com at 555-123-4567.",
            }
        ],
    }
    canonical = _canonical()
    canonical.categories[0].violations[
        0
    ].explanation = "Dr. Jane from ACME_CORP owes PHP 5,000; jane [at] example [dot] com."

    recommendation = build_rubric_recommendations(canonical, snapshot)[0]
    text = " ".join(
        [
            recommendation.explanation,
            recommendation.recommended_response,
            recommendation.coaching_advice,
        ]
    )

    assert "Legal Threat" in recommendation.recommended_response
    assert "ACME_CORP" not in text
    assert "jane [at] example [dot] com" not in text
    assert "555-123-4567" not in text
    assert "PHP 5,000" not in text
    assert "Dr. Jane" not in text

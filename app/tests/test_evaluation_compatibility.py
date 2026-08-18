"""Tests for canonical-to-legacy rendering and grounded rubric recommendations."""

import pytest

from app.services.evaluation_compatibility import (
    RecommendationValidationError,
    build_rubric_recommendations,
    render_legacy_review,
    to_jinja_context,
    to_legacy_review,
)
from app.services.rubric_observation_validator import validate_observation
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
                    "description": "Redirects.",
                    "evidence_instructions": "Quote.",
                }
            ],
            "violations": [
                {
                    "id": "legal-threat",
                    "name": "Legal Threat",
                    "description": "Threatens.",
                    "evidence_instructions": "Quote.",
                }
            ],
            "penalties": [{"violation_id": "legal-threat", "deduction": 20, "max_occurrences": 1}],
            "recommendation_guidance": "Recommend a compliant alternative.",
            "display_order": 0,
        }
    ],
}


def _canonical():
    observation = {
        "status": "evaluated",
        "summary": "The result is grounded in the call.",
        "categories": [
            {
                "rubric_block_id": "negotiation",
                "raw_score": 60,
                "evidence": [
                    {
                        "sequence_number": 2,
                        "speaker": "agent",
                        "excerpt": "We can discuss options",
                        "explanation": "The agent discussed options.",
                    }
                ],
                "strengths": [
                    {
                        "criterion_id": "move-forward",
                        "explanation": "The agent redirected the discussion.",
                        "evidence_sequence_numbers": [2],
                    }
                ],
                "violations": [
                    {
                        "violation_id": "legal-threat",
                        "explanation": "The agent made a legal threat.",
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
        ],
        "applied_techniques": {
            "techniques_used": [
                {
                    "technique_name": "Move Forward",
                    "execution_type": "Executed Properly",
                    "execution_description": "The agent redirected the discussion.",
                    "evidence_sequence_numbers": [2],
                }
            ],
            "reason_if_empty": "None.",
        },
        "missed_opportunities": {
            "missed_techniques": [
                {"technique_name": "Legal Threat", "reason": "A compliant alternative was needed."}
            ],
            "reason_if_empty": "None.",
        },
    }
    validated = validate_observation(
        observation,
        SNAPSHOT,
        [{"sequence_number": 2, "speaker": "agent", "text": "We can discuss options"}],
    )
    return calculate_rubric_score(validated)


def test_legacy_context_contains_only_derived_template_variables() -> None:
    canonical = _canonical()
    review = to_legacy_review(canonical)
    context = to_jinja_context(review)

    assert set(context) == {"summary", "applied_techniques", "missed_opportunities"}
    assert context["applied_techniques"]["techniques_used"][0]["technique_name"] == "Move Forward"
    assert (
        context["missed_opportunities"]["missed_techniques"][0]["technique_name"] == "Legal Threat"
    )
    assert "Applied Technique Delivery" in render_legacy_review(canonical)
    assert "Missed Opportunity" in render_legacy_review(canonical)
    assert "Summary" in render_legacy_review(canonical)


def test_recommendations_are_grounded_prioritized_and_redacted() -> None:
    canonical = _canonical()
    recommendations = build_rubric_recommendations(canonical, SNAPSHOT)

    assert len(recommendations) == 1
    recommendation = recommendations[0]
    assert recommendation.rubric_block_id == "negotiation"
    assert recommendation.criterion_id == "legal-threat"
    assert recommendation.evidence_sequence_number == 2
    assert "compliant alternative" in recommendation.coaching_advice.lower()
    assert "@" not in recommendation.coaching_advice


def test_recommendations_redact_names_organizations_amounts_emails_and_locations() -> None:
    canonical = _canonical()
    canonical.categories[0].violations[
        0
    ].explanation = "John from Acme in Manila owes $5,000; contact john@example.com."
    recommendation = build_rubric_recommendations(canonical, SNAPSHOT)[0]

    text = f"{recommendation.explanation} {recommendation.coaching_advice}"
    assert "John" not in text
    assert "Acme" not in text
    assert "Manila" not in text
    assert "$5,000" not in text
    assert "john@example.com" not in text


def test_unsupported_recommendation_reference_is_rejected() -> None:
    canonical = _canonical()
    canonical.categories[0].recommendation_inputs[0].criterion_id = "unknown-criterion"

    with pytest.raises(RecommendationValidationError):
        build_rubric_recommendations(canonical, SNAPSHOT)

"""Contract tests for TASK-066 canonical rubric coaching metadata."""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.schemas import EvaluationCategory, EvaluationResult, StrengthItem, WeaknessItem
from app.services.coaching_engine import CoachingEngine
from app.services.evaluation_compatibility import build_rubric_recommendations


VERSION_ID = uuid4()

SNAPSHOT = {
    "schema_version": 1,
    "overall_passing_score": 70,
    "blocks": [
        {
            "id": "custom-resolution",
            "category": "Custom Resolution",
            "weight": 50,
            "passing_score": 70,
            "scoring_instructions": "Use evidence.",
            "positive_behaviors": [
                {
                    "id": "offer-plan",
                    "name": "Offer a plan",
                    "description": "Offers a plan.",
                    "evidence_instructions": "Quote it.",
                }
            ],
            "violations": [],
            "penalties": [],
            "recommendation_guidance": "Offer a realistic plan.",
            "display_order": 1,
        },
        {
            "id": "custom-opening",
            "category": "Custom Opening",
            "weight": 50,
            "passing_score": 70,
            "scoring_instructions": "Use evidence.",
            "positive_behaviors": [],
            "violations": [
                {
                    "id": "skip-greeting",
                    "name": "Skip greeting",
                    "description": "Skips greeting.",
                    "evidence_instructions": "Quote it.",
                }
            ],
            "penalties": [],
            "recommendation_guidance": "Start respectfully.",
            "display_order": 0,
        },
    ],
}


def _evaluation() -> EvaluationResult:
    return EvaluationResult(
        session_id=uuid4(),
        category_scores=[],
        overall_score=50,
        strengths=[
            StrengthItem(
                description="x", category=EvaluationCategory.COMPLIANCE, transcript_excerpt="x"
            )
        ],
        weaknesses=[
            WeaknessItem(
                description="x", category=EvaluationCategory.COMPLIANCE, transcript_excerpt="x"
            )
        ],
        negotiation_standard_version_id=VERSION_ID,
        standard_version_number=7,
        standard_snapshot=SNAPSHOT,
        rubric_result={
            "status": "evaluated",
            "summary": "Grounded.",
            "categories": [
                {
                    "rubric_block_id": "custom-opening",
                    "category": "Custom Opening",
                    "raw_score": 50,
                    "penalty_total": 0,
                    "penalized_score": 50,
                    "weight": 50,
                    "weighted_contribution": 25,
                    "passing_score": 70,
                    "passed": False,
                    "evidence": [
                        {
                            "sequence_number": 2,
                            "speaker": "agent",
                            "excerpt": "No greeting",
                            "explanation": "Missing greeting.",
                        }
                    ],
                    "strengths": [],
                    "violations": [],
                    "failed_criteria": ["skip-greeting"],
                    "recommendation_inputs": [
                        {
                            "criterion_id": "skip-greeting",
                            "transcript_sequence_number": 2,
                            "need": "Add a greeting.",
                        }
                    ],
                },
                {
                    "rubric_block_id": "custom-resolution",
                    "category": "Custom Resolution",
                    "raw_score": 50,
                    "penalty_total": 0,
                    "penalized_score": 50,
                    "weight": 50,
                    "weighted_contribution": 25,
                    "passing_score": 70,
                    "passed": False,
                    "evidence": [
                        {
                            "sequence_number": 4,
                            "speaker": "agent",
                            "excerpt": "No plan",
                            "explanation": "No plan.",
                        }
                    ],
                    "strengths": [],
                    "violations": [],
                    "failed_criteria": ["offer-plan"],
                    "recommendation_inputs": [
                        {
                            "criterion_id": "offer-plan",
                            "transcript_sequence_number": 4,
                            "need": "Offer a plan.",
                        }
                    ],
                },
            ],
            "weighted_total": 50,
            "passing_score": 70,
            "passed": False,
            "applied_techniques": {"techniques_used": [], "reason_if_empty": "None."},
            "missed_opportunities": {"missed_techniques": [], "reason_if_empty": "None."},
            "recommendations": [
                {
                    "rubric_block_id": "custom-opening",
                    "criterion_id": "skip-greeting",
                    "evidence_sequence_number": 2,
                    "explanation": "Missing greeting.",
                    "recommended_response": "Hello.",
                    "coaching_advice": "Start respectfully.",
                },
                {
                    "rubric_block_id": "custom-resolution",
                    "criterion_id": "offer-plan",
                    "evidence_sequence_number": 4,
                    "explanation": "No plan.",
                    "recommended_response": "Let us plan.",
                    "coaching_advice": "Offer a realistic plan.",
                },
            ],
        },
    )


def test_custom_multi_block_recommendations_preserve_rubric_metadata() -> None:
    recommendations = build_rubric_recommendations(
        _evaluation().rubric_result,
        SNAPSHOT,
        VERSION_ID,
        7,
    )

    assert [(item.rubric_block_id, item.display_order) for item in recommendations] == [
        ("custom-opening", 0),
        ("custom-resolution", 1),
    ]
    assert recommendations[0].block_name == "Custom Opening"
    assert recommendations[0].criterion_name == "Skip greeting"
    assert recommendations[0].standard_version_id == VERSION_ID
    assert recommendations[0].standard_version_number == 7


@pytest.mark.asyncio
async def test_canonical_coaching_groups_by_custom_block_without_legacy_categories() -> None:
    report = await CoachingEngine(AsyncMock()).generate_report(uuid4(), [], _evaluation())

    assert report.mistakes_by_category == {}
    assert report.rubric_coaching is not None
    assert [block.rubric_block_id for block in report.rubric_coaching.blocks] == [
        "custom-opening",
        "custom-resolution",
    ]
    assert report.rubric_coaching.standard_version_id == VERSION_ID
    assert report.rubric_coaching.standard_version_number == 7
    assert report.rubric_coaching.blocks[0].recommendations[0].criterion_name == "Skip greeting"

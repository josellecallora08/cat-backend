"""End-to-end canonical coaching-flow coverage for TASK-070."""

from uuid import uuid4

import pytest

from app.schemas import EvaluationCategory, EvaluationResult
from app.services.coaching_engine import CoachingEngine
from app.services.evaluation_compatibility import build_rubric_recommendations
from app.services.learning_plan_generator import LearningPlanGenerator
from app.services.rubric_observation_validator import validate_observation
from app.services.rubric_score_calculator import calculate_rubric_score


SNAPSHOT = {
    "schema_version": 1,
    "overall_passing_score": 70,
    "blocks": [
        {
            "id": "opening-quality",
            "category": "Opening Quality",
            "weight": 40,
            "passing_score": 70,
            "scoring_instructions": "Use exact transcript evidence.",
            "positive_behaviors": [],
            "violations": [
                {
                    "id": "missing-identity",
                    "name": "Missing identity",
                    "description": "The agent does not identify the organization.",
                    "evidence_instructions": "Quote the opening.",
                }
            ],
            "penalties": [
                {"violation_id": "missing-identity", "deduction": 20, "max_occurrences": 1}
            ],
            "recommendation_guidance": "Identify the organization before discussing the account.",
            "display_order": 0,
        },
        {
            "id": "resolution-quality",
            "category": "Resolution Quality",
            "weight": 60,
            "passing_score": 70,
            "scoring_instructions": "Use exact transcript evidence.",
            "positive_behaviors": [
                {
                    "id": "offer-plan",
                    "name": "Offer a plan",
                    "description": "Offer a realistic payment plan.",
                    "evidence_instructions": "Quote the proposed plan or its absence.",
                }
            ],
            "violations": [],
            "penalties": [],
            "recommendation_guidance": "Offer a realistic plan and confirm the next step.",
            "display_order": 1,
        },
    ],
}

TRANSCRIPT = [
    {"sequence_number": 0, "speaker": "agent", "text": "We need to discuss your account."},
    {"sequence_number": 1, "speaker": "agent", "text": "The full balance is due today."},
]


def _observation() -> dict:
    return {
        "status": "evaluated",
        "summary": "The agent needs a stronger opening and resolution plan.",
        "categories": [
            {
                "rubric_block_id": "opening-quality",
                "raw_score": 80,
                "evidence": [
                    {
                        "sequence_number": 0,
                        "speaker": "agent",
                        "excerpt": "We need to discuss your account.",
                        "explanation": "The opening does not identify the organization.",
                    }
                ],
                "strengths": [],
                "violations": [
                    {
                        "violation_id": "missing-identity",
                        "explanation": "The organization was not identified.",
                        "evidence_sequence_numbers": [0],
                    }
                ],
                "failed_criteria": ["missing-identity"],
                "recommendation_inputs": [
                    {
                        "criterion_id": "missing-identity",
                        "transcript_sequence_number": 0,
                        "need": "Identify the organization before discussing the account.",
                    }
                ],
            },
            {
                "rubric_block_id": "resolution-quality",
                "raw_score": 50,
                "evidence": [
                    {
                        "sequence_number": 1,
                        "speaker": "agent",
                        "excerpt": "The full balance is due today.",
                        "explanation": "No realistic payment plan was offered.",
                    }
                ],
                "strengths": [],
                "violations": [],
                "failed_criteria": ["offer-plan"],
                "recommendation_inputs": [
                    {
                        "criterion_id": "offer-plan",
                        "transcript_sequence_number": 1,
                        "need": "Offer a realistic payment plan.",
                    }
                ],
            },
        ],
        "applied_techniques": {"techniques_used": [], "reason_if_empty": "None observed."},
        "missed_opportunities": {"missed_techniques": [], "reason_if_empty": "None missed."},
    }


@pytest.mark.asyncio
async def test_two_block_canonical_flow_keeps_scores_coaching_and_practice_aligned():
    session_id = uuid4()
    version_id = uuid4()
    validated = validate_observation(_observation(), SNAPSHOT, TRANSCRIPT)
    canonical = calculate_rubric_score(validated)
    canonical = canonical.model_copy(
        update={"recommendations": build_rubric_recommendations(canonical, SNAPSHOT, version_id, 4)}
    )
    evaluation = EvaluationResult(
        session_id=session_id,
        category_scores=[],
        overall_score=float(canonical.weighted_total),
        strengths=[
            {
                "description": "The transcript was reviewed.",
                "category": EvaluationCategory.CALL_OPENING,
                "transcript_excerpt": TRANSCRIPT[0]["text"],
            }
        ],
        weaknesses=[
            {
                "description": "The transcript needs rubric-grounded improvement.",
                "category": EvaluationCategory.CALL_OPENING,
                "transcript_excerpt": TRANSCRIPT[0]["text"],
            }
        ],
        is_too_short=False,
        negotiation_standard_version_id=version_id,
        standard_version_number=4,
        standard_snapshot=SNAPSHOT,
        rubric_result=canonical,
    )

    coaching = await CoachingEngine(None).generate_report(session_id, TRANSCRIPT, evaluation)
    learning_plan = LearningPlanGenerator().generate(evaluation, session_id, uuid4())

    assert [
        (category.rubric_block_id, category.penalty_total, category.penalized_score)
        for category in canonical.categories
    ] == [
        ("opening-quality", 20, 60),
        ("resolution-quality", 0, 50),
    ]
    assert [(item.rubric_block_id, item.criterion_id) for item in canonical.recommendations] == [
        ("resolution-quality", "offer-plan"),
        ("opening-quality", "missing-identity"),
    ]
    assert [block.rubric_block_id for block in coaching.rubric_coaching.blocks] == [
        "opening-quality",
        "resolution-quality",
    ]
    assert coaching.mistakes_by_category == {}
    assert coaching.total_mistakes == 2
    assert [
        (item.rubric_block_id, item.criterion_id) for item in learning_plan.weak_competencies
    ] == [
        ("resolution-quality", "offer-plan"),
        ("opening-quality", "missing-identity"),
    ]
    assert all(item.practice_focus for item in learning_plan.weak_competencies)

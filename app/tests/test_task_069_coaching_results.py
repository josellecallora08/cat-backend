"""Contract tests for TASK-069 coaching and terminal result behavior."""

from uuid import uuid4

from app.services.evaluation_compatibility import build_rubric_recommendations
from app.services.learning_plan_generator import LearningPlanGenerator
from app.tests.test_task_066_coaching_contract import SNAPSHOT, _evaluation


def test_recommendation_retains_source_speaker_and_excerpt():
    recommendations = build_rubric_recommendations(
        _evaluation().rubric_result,
        SNAPSHOT,
        uuid4(),
        7,
    )

    recommendation = recommendations[0]
    assert recommendation.source_speaker == "agent"
    assert recommendation.source_excerpt == "No greeting"


def test_not_applicable_rubric_has_no_coaching_recommendations():
    evaluation = _evaluation()
    evaluation.rubric_result = evaluation.rubric_result.model_dump()
    evaluation.rubric_result["status"] = "not_applicable"
    for category in evaluation.rubric_result["categories"]:
        category["raw_score"] = None
        category["penalized_score"] = None
        category["failed_criteria"] = []
        category["recommendation_inputs"] = []

    recommendations = build_rubric_recommendations(
        evaluation.rubric_result,
        SNAPSHOT,
        evaluation.negotiation_standard_version_id,
        evaluation.standard_version_number,
    )
    plan = LearningPlanGenerator().generate(
        evaluation, evaluation.session_id, uuid4()
    )

    assert recommendations == []
    assert plan.weak_competencies == []
    assert plan.all_passing is False

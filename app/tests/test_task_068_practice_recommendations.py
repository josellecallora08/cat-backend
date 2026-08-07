"""Contract tests for TASK-068 rubric-linked practice recommendations."""

import uuid
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.schemas import EvaluationCategory, EvaluationResult, LearningPlanItem
from app.services.learning_plan_generator import LearningPlanGenerator


def _evaluation(categories, *, version_id=None):
    """Build a canonical evaluation fixture from compact category dictionaries."""
    result = {
        "status": "evaluated",
        "summary": "Evidence-grounded result.",
        "categories": categories,
        "weighted_total": "40.00",
        "passing_score": 70,
        "passed": False,
        "applied_techniques": {"techniques_used": [], "reason_if_empty": "None."},
        "missed_opportunities": {"missed_techniques": [], "reason_if_empty": "None."},
        "recommendations": [],
    }
    return EvaluationResult(
        session_id=uuid.uuid4(),
        category_scores=[],
        overall_score=40,
        strengths=[{"description": "Good", "category": EvaluationCategory.CALL_OPENING, "transcript_excerpt": "Hi"}],
        weaknesses=[{"description": "Improve", "category": EvaluationCategory.CALL_OPENING, "transcript_excerpt": "No"}],
        negotiation_standard_version_id=version_id,
        rubric_result=result,
        standard_snapshot={"blocks": []},
    )


def _category(block_id, name, score, *, passed=False, penalty=0, failed=None, violations=None):
    return {
        "rubric_block_id": block_id,
        "category": name,
        "raw_score": score,
        "penalty_total": penalty,
        "penalized_score": score - penalty,
        "weight": 50,
        "weighted_contribution": str(Decimal(score - penalty) / 2),
        "passing_score": 70,
        "passed": passed,
        "evidence": [],
        "strengths": [],
        "violations": [
            {"violation_id": item, "explanation": "Observed.", "evidence_sequence_numbers": [1]}
            for item in (violations or [])
        ],
        "failed_criteria": failed or [],
        "recommendation_inputs": [],
    }


def test_custom_rubric_category_and_criterion_are_preserved_without_scenario():
    session_id = uuid.uuid4()
    evaluation = _evaluation([_category("custom-block", "De-escalation", 40, failed=["calm-tone"])])
    evaluation.standard_snapshot = {"blocks": [{
        "id": "custom-block",
        "category": "De-escalation",
        "display_order": 0,
        "positive_behaviors": [{
            "id": "calm-tone",
            "name": "Calm tone",
            "description": "Use calm language.",
            "evidence_instructions": "Ask one clarifying question.",
        }],
    }]}

    plan = LearningPlanGenerator().generate(evaluation, session_id, uuid.uuid4())

    item = plan.weak_competencies[0]
    assert item.category == "De-escalation"
    assert item.category != EvaluationCategory.CALL_OPENING
    assert item.rubric_block_id == "custom-block"
    assert item.criterion_id == "calm-tone"
    assert item.scenario_id is None
    assert item.recommended_scenario is None
    assert item.practice_focus


def test_multiple_criteria_are_deduplicated_and_ranked_deterministically():
    evaluation = _evaluation([
        _category("z-block", "Z", 30, penalty=10, failed=["b", "a", "a"], violations=["b"]),
        _category("a-block", "A", 30, penalty=10, failed=["c"]),
    ])
    evaluation.standard_snapshot = {"blocks": [
        {"id": "z-block", "category": "Z", "display_order": 0},
        {"id": "a-block", "category": "A", "display_order": 0},
    ]}

    items = LearningPlanGenerator().generate(evaluation, uuid.uuid4(), uuid.uuid4()).weak_competencies

    assert [(item.rubric_block_id, item.criterion_id) for item in items] == [
        ("a-block", "c"), ("z-block", "a"), ("z-block", "b")
    ]


def test_passing_category_without_findings_is_not_remediation():
    evaluation = _evaluation([_category("good", "Good", 80, passed=True)])
    evaluation.standard_snapshot = {"blocks": [{"id": "good", "category": "Good"}]}

    plan = LearningPlanGenerator().generate(evaluation, uuid.uuid4(), uuid.uuid4())

    assert plan.all_passing is True
    assert plan.weak_competencies == []


def test_passing_category_with_failed_criterion_remains_remediation():
    evaluation = _evaluation([_category("good", "Good", 80, passed=True, failed=["missed-step"])])
    evaluation.standard_snapshot = {"blocks": [{"id": "good", "category": "Good"}]}

    plan = LearningPlanGenerator().generate(evaluation, uuid.uuid4(), uuid.uuid4())

    assert plan.all_passing is False
    assert plan.weak_competencies[0].criterion_id == "missed-step"


def test_learning_plan_item_allows_optional_authorized_scenario_id():
    scenario_id = uuid.uuid4()
    item = LearningPlanItem(
        category="Custom Category",
        score=55,
        recommended_scenario=None,
        scenario_id=scenario_id,
        rubric_block_id="custom-block",
        criterion_id="custom-criterion",
        practice_focus="Practice the custom criterion.",
    )

    assert item.scenario_id == scenario_id
    assert item.recommended_scenario is None


@pytest.mark.asyncio
async def test_authorized_active_campaign_scenario_is_resolved():
    scenario_id = uuid.uuid4()
    version_id = uuid.uuid4()
    evaluation = _evaluation([_category("custom", "Custom", 40, failed=["criterion"])], version_id=version_id)
    evaluation.standard_snapshot = {"blocks": [{"id": "custom", "category": "Custom"}]}
    db = AsyncMock()
    result = SimpleNamespace(scalar_one_or_none=lambda: SimpleNamespace(id=scenario_id, name="Authorized Practice"))
    db.execute.return_value = result

    with patch("app.services.learning_plan_generator.retry_db_operation", new_callable=AsyncMock):
        plan = await LearningPlanGenerator().generate_and_persist(
            evaluation, uuid.uuid4(), uuid.uuid4(), db=db
        )

    assert plan.weak_competencies[0].scenario_id == scenario_id
    assert plan.weak_competencies[0].recommended_scenario == "Authorized Practice"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["inactive", "out-of-campaign", "unauthorized", "unavailable"])
async def test_invalid_scenario_resolution_keeps_focus_without_false_cta(reason):
    """Any scenario that fails the authorization query must produce no CTA."""
    evaluation = _evaluation([_category("custom", "Custom", 40, failed=["criterion"])])
    evaluation.standard_snapshot = {"blocks": [{"id": "custom", "category": "Custom"}]}
    db = AsyncMock()
    generator = LearningPlanGenerator()

    with (
        patch.object(generator, "_resolve_authorized_scenario", new_callable=AsyncMock, return_value=None),
        patch("app.services.learning_plan_generator.retry_db_operation", new_callable=AsyncMock),
    ):
        plan = await generator.generate_and_persist(
            evaluation, uuid.uuid4(), uuid.uuid4(), db=db
        )

    item = plan.weak_competencies[0]
    assert reason
    assert item.scenario_id is None
    assert item.recommended_scenario is None
    assert item.practice_focus

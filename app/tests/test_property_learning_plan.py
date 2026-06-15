"""Property-based tests for learning plan competency-to-scenario mapping.

Feature: collection-agent-trainer, Property 13: Learning plan competency-to-scenario mapping

**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7**

Property 13: For any evaluation result, the Learning Plan Generator SHALL include
in weak_competencies exactly those categories with scores below 70, and SHALL map
each to the correct recommended scenario: Empathy and Communication → "Financial Hardship",
Negotiation and Resolution → "Payment Arrangement", Compliance → "Compliance Fundamentals",
Call Opening → "Call Opening Basics". When all scores are ≥ 70, weak_competencies SHALL
be empty and all_passing SHALL be true.
"""

import uuid

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.schemas import (
    CompetencyScore,
    EvaluationCategory,
    EvaluationResult,
    StrengthItem,
    WeaknessItem,
)
from app.services.learning_plan_generator import (
    COMPETENCY_SCENARIO_MAP,
    WEAKNESS_THRESHOLD,
    LearningPlanGenerator,
)


# --- Constants ---

EXPECTED_SCENARIO_MAP = {
    EvaluationCategory.EMPATHY_COMMUNICATION: "Financial Hardship",
    EvaluationCategory.NEGOTIATION_RESOLUTION: "Payment Arrangement",
    EvaluationCategory.COMPLIANCE: "Compliance Fundamentals",
    EvaluationCategory.CALL_OPENING: "Call Opening Basics",
}

ALL_CATEGORIES = [
    EvaluationCategory.CALL_OPENING,
    EvaluationCategory.COMPLIANCE,
    EvaluationCategory.EMPATHY_COMMUNICATION,
    EvaluationCategory.NEGOTIATION_RESOLUTION,
]


# --- Strategies ---

scores = st.integers(min_value=0, max_value=100)

category_scores_strategy = st.fixed_dictionaries({
    EvaluationCategory.CALL_OPENING: scores,
    EvaluationCategory.COMPLIANCE: scores,
    EvaluationCategory.EMPATHY_COMMUNICATION: scores,
    EvaluationCategory.NEGOTIATION_RESOLUTION: scores,
})


# --- Helpers ---

def build_evaluation(category_score_map: dict) -> EvaluationResult:
    """Build an EvaluationResult from a map of category → score."""
    session_id = uuid.uuid4()
    competency_scores = [
        CompetencyScore(category=cat, score=score)
        for cat, score in category_score_map.items()
    ]
    return EvaluationResult(
        session_id=session_id,
        category_scores=competency_scores,
        overall_score=50.0,
        strengths=[
            StrengthItem(
                description="Good effort",
                category=EvaluationCategory.CALL_OPENING,
                transcript_excerpt="Hello, this is agent calling.",
            ),
        ],
        weaknesses=[
            WeaknessItem(
                description="Needs improvement",
                category=EvaluationCategory.COMPLIANCE,
                transcript_excerpt="You must pay now.",
            ),
        ],
        is_too_short=False,
    )


# --- Property Tests ---


class TestLearningPlanCompetencyMapping:
    """Property 13: Learning plan competency-to-scenario mapping.

    Feature: collection-agent-trainer, Property 13: Learning plan competency-to-scenario mapping
    """

    @given(score_map=category_scores_strategy)
    @settings(max_examples=100)
    def test_weak_competencies_contains_exactly_categories_below_threshold(
        self, score_map: dict
    ):
        """**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7**

        weak_competencies SHALL contain exactly those categories with
        scores below 70.
        """
        evaluation = build_evaluation(score_map)
        generator = LearningPlanGenerator()
        session_id = uuid.uuid4()
        agent_id = uuid.uuid4()

        plan = generator.generate(evaluation, session_id, agent_id)

        # Determine expected weak categories
        expected_weak = {
            cat for cat, score in score_map.items() if score < WEAKNESS_THRESHOLD
        }
        actual_weak = {item.category for item in plan.weak_competencies}

        assert actual_weak == expected_weak, (
            f"Expected weak categories {expected_weak} but got {actual_weak}. "
            f"Scores: {score_map}, threshold: {WEAKNESS_THRESHOLD}"
        )

    @given(score_map=category_scores_strategy)
    @settings(max_examples=100)
    def test_weak_competencies_map_to_correct_scenarios(self, score_map: dict):
        """**Validates: Requirements 7.2, 7.3, 7.4, 7.5**

        Each weak category SHALL map to the correct recommended scenario:
        empathy → "Financial Hardship", negotiation → "Payment Arrangement",
        compliance → "Compliance Fundamentals", call_opening → "Call Opening Basics".
        """
        evaluation = build_evaluation(score_map)
        generator = LearningPlanGenerator()
        session_id = uuid.uuid4()
        agent_id = uuid.uuid4()

        plan = generator.generate(evaluation, session_id, agent_id)

        for item in plan.weak_competencies:
            expected_scenario = EXPECTED_SCENARIO_MAP[item.category]
            assert item.recommended_scenario == expected_scenario, (
                f"Category '{item.category.value}' should map to "
                f"'{expected_scenario}' but got '{item.recommended_scenario}'"
            )

    @given(score_map=category_scores_strategy)
    @settings(max_examples=100)
    def test_all_passing_flag_correctness(self, score_map: dict):
        """**Validates: Requirements 7.7**

        all_passing SHALL be True when all scores >= 70, False otherwise.
        """
        evaluation = build_evaluation(score_map)
        generator = LearningPlanGenerator()
        session_id = uuid.uuid4()
        agent_id = uuid.uuid4()

        plan = generator.generate(evaluation, session_id, agent_id)

        all_above_threshold = all(
            score >= WEAKNESS_THRESHOLD for score in score_map.values()
        )

        assert plan.all_passing == all_above_threshold, (
            f"Expected all_passing={all_above_threshold} but got "
            f"all_passing={plan.all_passing}. Scores: {score_map}"
        )

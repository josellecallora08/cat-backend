"""Unit tests for the LearningPlanGenerator service.

Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas import (
    CompetencyScore,
    EvaluationCategory,
    EvaluationResult,
    LearningPlanItem,
    StrengthItem,
    WeaknessItem,
)
from app.services.learning_plan_generator import (
    COMPETENCY_SCENARIO_MAP,
    WEAKNESS_THRESHOLD,
    LearningPlanGenerator,
)


@pytest.fixture
def generator():
    """Provide a LearningPlanGenerator instance."""
    return LearningPlanGenerator()


@pytest.fixture
def session_id():
    return uuid.uuid4()


@pytest.fixture
def agent_id():
    return uuid.uuid4()


def _make_evaluation(session_id: uuid.UUID, scores: dict[EvaluationCategory, int]) -> EvaluationResult:
    """Helper to build an EvaluationResult with given category scores."""
    category_scores = [
        CompetencyScore(category=cat, score=score, strengths=[], weaknesses=[])
        for cat, score in scores.items()
    ]
    return EvaluationResult(
        session_id=session_id,
        category_scores=category_scores,
        overall_score=50.0,
        strengths=[
            StrengthItem(
                description="Good opening",
                category=EvaluationCategory.CALL_OPENING,
                transcript_excerpt="Hello, this is...",
            )
        ],
        weaknesses=[
            WeaknessItem(
                description="Lacking empathy",
                category=EvaluationCategory.EMPATHY_COMMUNICATION,
                transcript_excerpt="Pay now.",
            )
        ],
        is_too_short=False,
    )


class TestLearningPlanGeneratorGenerate:
    """Tests for the generate() method."""

    def test_all_scores_above_threshold_sets_all_passing(
        self, generator, session_id, agent_id
    ):
        """When all scores are >= 70, all_passing should be True and no weak competencies."""
        evaluation = _make_evaluation(
            session_id,
            {
                EvaluationCategory.CALL_OPENING: 80,
                EvaluationCategory.COMPLIANCE: 90,
                EvaluationCategory.EMPATHY_COMMUNICATION: 75,
                EvaluationCategory.NEGOTIATION_RESOLUTION: 70,
            },
        )

        plan = generator.generate(evaluation, session_id, agent_id)

        assert plan.all_passing is True
        assert plan.weak_competencies == []
        assert plan.session_id == session_id

    def test_all_scores_at_threshold_sets_all_passing(
        self, generator, session_id, agent_id
    ):
        """Scores exactly at 70 should not be considered weak."""
        evaluation = _make_evaluation(
            session_id,
            {
                EvaluationCategory.CALL_OPENING: 70,
                EvaluationCategory.COMPLIANCE: 70,
                EvaluationCategory.EMPATHY_COMMUNICATION: 70,
                EvaluationCategory.NEGOTIATION_RESOLUTION: 70,
            },
        )

        plan = generator.generate(evaluation, session_id, agent_id)

        assert plan.all_passing is True
        assert plan.weak_competencies == []

    def test_single_weak_empathy_maps_to_financial_hardship(
        self, generator, session_id, agent_id
    ):
        """Empathy below 70 should recommend Financial Hardship scenario."""
        evaluation = _make_evaluation(
            session_id,
            {
                EvaluationCategory.CALL_OPENING: 80,
                EvaluationCategory.COMPLIANCE: 90,
                EvaluationCategory.EMPATHY_COMMUNICATION: 60,
                EvaluationCategory.NEGOTIATION_RESOLUTION: 75,
            },
        )

        plan = generator.generate(evaluation, session_id, agent_id)

        assert plan.all_passing is False
        assert len(plan.weak_competencies) == 1
        item = plan.weak_competencies[0]
        assert item.category == EvaluationCategory.EMPATHY_COMMUNICATION
        assert item.score == 60
        assert item.recommended_scenario == "Financial Hardship"

    def test_single_weak_negotiation_maps_to_payment_arrangement(
        self, generator, session_id, agent_id
    ):
        """Negotiation below 70 should recommend Payment Arrangement scenario."""
        evaluation = _make_evaluation(
            session_id,
            {
                EvaluationCategory.CALL_OPENING: 80,
                EvaluationCategory.COMPLIANCE: 90,
                EvaluationCategory.EMPATHY_COMMUNICATION: 75,
                EvaluationCategory.NEGOTIATION_RESOLUTION: 50,
            },
        )

        plan = generator.generate(evaluation, session_id, agent_id)

        assert plan.all_passing is False
        assert len(plan.weak_competencies) == 1
        item = plan.weak_competencies[0]
        assert item.category == EvaluationCategory.NEGOTIATION_RESOLUTION
        assert item.score == 50
        assert item.recommended_scenario == "Payment Arrangement"

    def test_single_weak_compliance_maps_to_compliance_fundamentals(
        self, generator, session_id, agent_id
    ):
        """Compliance below 70 should recommend Compliance Fundamentals scenario."""
        evaluation = _make_evaluation(
            session_id,
            {
                EvaluationCategory.CALL_OPENING: 80,
                EvaluationCategory.COMPLIANCE: 40,
                EvaluationCategory.EMPATHY_COMMUNICATION: 75,
                EvaluationCategory.NEGOTIATION_RESOLUTION: 80,
            },
        )

        plan = generator.generate(evaluation, session_id, agent_id)

        assert plan.all_passing is False
        assert len(plan.weak_competencies) == 1
        item = plan.weak_competencies[0]
        assert item.category == EvaluationCategory.COMPLIANCE
        assert item.score == 40
        assert item.recommended_scenario == "Compliance Fundamentals"

    def test_single_weak_call_opening_maps_to_call_opening_basics(
        self, generator, session_id, agent_id
    ):
        """Call Opening below 70 should recommend Call Opening Basics scenario."""
        evaluation = _make_evaluation(
            session_id,
            {
                EvaluationCategory.CALL_OPENING: 30,
                EvaluationCategory.COMPLIANCE: 90,
                EvaluationCategory.EMPATHY_COMMUNICATION: 75,
                EvaluationCategory.NEGOTIATION_RESOLUTION: 80,
            },
        )

        plan = generator.generate(evaluation, session_id, agent_id)

        assert plan.all_passing is False
        assert len(plan.weak_competencies) == 1
        item = plan.weak_competencies[0]
        assert item.category == EvaluationCategory.CALL_OPENING
        assert item.score == 30
        assert item.recommended_scenario == "Call Opening Basics"

    def test_multiple_weak_categories(self, generator, session_id, agent_id):
        """Multiple categories below 70 should all appear in weak_competencies."""
        evaluation = _make_evaluation(
            session_id,
            {
                EvaluationCategory.CALL_OPENING: 50,
                EvaluationCategory.COMPLIANCE: 30,
                EvaluationCategory.EMPATHY_COMMUNICATION: 20,
                EvaluationCategory.NEGOTIATION_RESOLUTION: 10,
            },
        )

        plan = generator.generate(evaluation, session_id, agent_id)

        assert plan.all_passing is False
        assert len(plan.weak_competencies) == 4

        # Check all mappings are present
        categories_in_plan = {item.category for item in plan.weak_competencies}
        assert categories_in_plan == set(EvaluationCategory)

        for item in plan.weak_competencies:
            assert item.recommended_scenario == COMPETENCY_SCENARIO_MAP[item.category]

    def test_score_at_zero_is_weak(self, generator, session_id, agent_id):
        """A score of 0 should be considered weak."""
        evaluation = _make_evaluation(
            session_id,
            {
                EvaluationCategory.CALL_OPENING: 0,
                EvaluationCategory.COMPLIANCE: 90,
                EvaluationCategory.EMPATHY_COMMUNICATION: 75,
                EvaluationCategory.NEGOTIATION_RESOLUTION: 80,
            },
        )

        plan = generator.generate(evaluation, session_id, agent_id)

        assert plan.all_passing is False
        assert len(plan.weak_competencies) == 1
        assert plan.weak_competencies[0].score == 0

    def test_score_at_69_is_weak(self, generator, session_id, agent_id):
        """A score of 69 (just below threshold) should be considered weak."""
        evaluation = _make_evaluation(
            session_id,
            {
                EvaluationCategory.CALL_OPENING: 69,
                EvaluationCategory.COMPLIANCE: 70,
                EvaluationCategory.EMPATHY_COMMUNICATION: 71,
                EvaluationCategory.NEGOTIATION_RESOLUTION: 100,
            },
        )

        plan = generator.generate(evaluation, session_id, agent_id)

        assert plan.all_passing is False
        assert len(plan.weak_competencies) == 1
        assert plan.weak_competencies[0].category == EvaluationCategory.CALL_OPENING
        assert plan.weak_competencies[0].score == 69


class TestLearningPlanGeneratorPersistence:
    """Tests for the generate_and_persist() method with db persistence."""

    @pytest.mark.asyncio
    async def test_persist_called_when_db_provided(
        self, generator, session_id, agent_id
    ):
        """When db is provided, should persist the learning plan."""
        evaluation = _make_evaluation(
            session_id,
            {
                EvaluationCategory.CALL_OPENING: 80,
                EvaluationCategory.COMPLIANCE: 90,
                EvaluationCategory.EMPATHY_COMMUNICATION: 60,
                EvaluationCategory.NEGOTIATION_RESOLUTION: 75,
            },
        )

        mock_db = AsyncMock()

        with patch(
            "app.services.learning_plan_generator.retry_db_operation",
            new_callable=AsyncMock,
        ) as mock_retry:
            plan = await generator.generate_and_persist(
                evaluation, session_id, agent_id, db=mock_db
            )

            # retry_db_operation should have been called
            mock_retry.assert_called_once()
            # Plan should still be correctly generated
            assert plan.all_passing is False
            assert len(plan.weak_competencies) == 1

    @pytest.mark.asyncio
    async def test_no_persist_when_db_not_provided(
        self, generator, session_id, agent_id
    ):
        """When db is None, should not attempt persistence."""
        evaluation = _make_evaluation(
            session_id,
            {
                EvaluationCategory.CALL_OPENING: 80,
                EvaluationCategory.COMPLIANCE: 90,
                EvaluationCategory.EMPATHY_COMMUNICATION: 75,
                EvaluationCategory.NEGOTIATION_RESOLUTION: 75,
            },
        )

        with patch(
            "app.services.learning_plan_generator.retry_db_operation",
            new_callable=AsyncMock,
        ) as mock_retry:
            plan = await generator.generate_and_persist(
                evaluation, session_id, agent_id, db=None
            )

            # retry_db_operation should NOT have been called
            mock_retry.assert_not_called()
            assert plan.all_passing is True


class TestWeaknessThreshold:
    """Tests verifying the threshold constant is correct."""

    def test_weakness_threshold_is_70(self):
        """The weakness threshold should be 70 per requirements."""
        assert WEAKNESS_THRESHOLD == 70

    def test_competency_scenario_map_has_all_categories(self):
        """The mapping should cover all evaluation categories."""
        for category in EvaluationCategory:
            assert category in COMPETENCY_SCENARIO_MAP

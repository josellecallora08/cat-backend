"""Learning Plan Generator service.

Maps weak competencies to recommended training scenarios based on
evaluation results. Categories scoring below the weakness threshold (70)
are identified and mapped to specific remedial scenarios.

Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8
"""

import logging
from uuid import UUID

from app.schemas import (
    EvaluationCategory,
    EvaluationResult,
    LearningPlanItem,
    LearningPlanSchema,
)
from app.schemas.rubric_evaluation import CanonicalEvaluationResult
from app.services.db_retry import retry_db_operation
from app.services.evaluation_compatibility import redact_recommendation_text

logger = logging.getLogger(__name__)

# Threshold below which a competency is considered weak
WEAKNESS_THRESHOLD = 70

# Mapping from weak competency categories to recommended scenarios
COMPETENCY_SCENARIO_MAP: dict[EvaluationCategory, str] = {
    EvaluationCategory.EMPATHY_COMMUNICATION: "Financial Hardship",
    EvaluationCategory.NEGOTIATION_RESOLUTION: "Payment Arrangement",
    EvaluationCategory.COMPLIANCE: "Compliance Fundamentals",
    EvaluationCategory.CALL_OPENING: "Call Opening Basics",
}


class LearningPlanGenerator:
    """Generates personalized learning plans from evaluation results.

    Identifies categories where the agent scored below the weakness
    threshold and maps each to a recommended training scenario.
    """

    def generate(
        self,
        evaluation: EvaluationResult,
        session_id: UUID,
        agent_id: UUID,
    ) -> LearningPlanSchema:
        """Generate a learning plan based on evaluation category scores.

        For each category score below the weakness threshold (70), adds
        the category to weak_competencies with the mapped recommended
        scenario. Sets all_passing=True when no weak competencies exist.

        Args:
            evaluation: The completed EvaluationResult with category_scores.
            session_id: The UUID of the session this plan belongs to.
            agent_id: The UUID of the agent this plan is for.

        Returns:
            LearningPlanSchema with weak competencies and all_passing flag.
        """
        if evaluation.rubric_result is not None and evaluation.standard_snapshot is not None:
            return self._generate_rubric_plan(evaluation, session_id)

        weak_competencies: list[LearningPlanItem] = []

        for competency_score in evaluation.category_scores:
            if competency_score.score < WEAKNESS_THRESHOLD:
                recommended_scenario = COMPETENCY_SCENARIO_MAP.get(
                    competency_score.category, "General Practice"
                )
                weak_competencies.append(
                    LearningPlanItem(
                        category=competency_score.category,
                        score=competency_score.score,
                        recommended_scenario=recommended_scenario,
                    )
                )

        all_passing = len(weak_competencies) == 0

        return LearningPlanSchema(
            session_id=session_id,
            weak_competencies=weak_competencies,
            all_passing=all_passing,
        )

    def _generate_rubric_plan(
        self, evaluation: EvaluationResult, session_id: UUID
    ) -> LearningPlanSchema:
        """Create criterion-linked practice items from canonical rubric results."""
        canonical = CanonicalEvaluationResult.model_validate(evaluation.rubric_result)
        display_order = {
            block.get("id"): block.get("display_order", 0)
            for block in evaluation.standard_snapshot.get("blocks", [])
        }
        ranked = sorted(
            canonical.categories,
            key=lambda category: (
                -(max(0, category.passing_score - (category.penalized_score or 0))),
                -category.penalty_total,
                display_order.get(category.rubric_block_id, 0),
            ),
        )
        items: list[LearningPlanItem] = []
        for category in ranked:
            criteria = list(dict.fromkeys(
                category.failed_criteria
                + [item.violation_id for item in category.violations]
            ))
            if category.passed and not criteria:
                continue
            targets = criteria or [None]
            score = category.penalized_score or 0
            for criterion_id in targets:
                focus = (
                    f"Practice criterion {criterion_id} using the pinned rubric guidance."
                    if criterion_id
                    else "Practice this rubric category using the pinned rubric guidance."
                )
                focus = redact_recommendation_text(focus)
                items.append(
                    LearningPlanItem(
                        category=EvaluationCategory.CALL_OPENING,
                        score=score,
                        recommended_scenario=redact_recommendation_text(
                            f"Rubric practice: {category.category}"
                        ),
                        rubric_block_id=category.rubric_block_id,
                        criterion_id=criterion_id,
                        practice_focus=focus,
                    )
                )
        return LearningPlanSchema(
            session_id=session_id,
            weak_competencies=items,
            all_passing=not items,
            standard_version_id=evaluation.negotiation_standard_version_id,
        )

    async def generate_and_persist(
        self,
        evaluation: EvaluationResult,
        session_id: UUID,
        agent_id: UUID,
        db=None,
    ) -> LearningPlanSchema:
        """Generate a learning plan and persist it to the database.

        Calls generate() to build the plan, then persists with retry
        wrapper when a database session is provided.

        Args:
            evaluation: The completed EvaluationResult with category_scores.
            session_id: The UUID of the session this plan belongs to.
            agent_id: The UUID of the agent this plan is for.
            db: Optional database session for persistence.

        Returns:
            LearningPlanSchema with weak competencies and all_passing flag.
        """
        plan = self.generate(evaluation, session_id, agent_id)

        if db is not None:
            await self._persist_plan(session_id, agent_id, plan, db)

        return plan

    async def _persist_plan(
        self, session_id: UUID, agent_id: UUID, plan: LearningPlanSchema, db
    ) -> None:
        """Persist the learning plan to the database with retry logic.

        Args:
            session_id: The session UUID.
            agent_id: The agent UUID.
            plan: The LearningPlanSchema to persist.
            db: The database session.
        """
        from app.models import LearningPlan

        async def _do_persist():
            learning_plan = LearningPlan(
                session_id=session_id,
                agent_id=agent_id,
                weak_competencies=[
                    item.model_dump() for item in plan.weak_competencies
                ],
                all_passing=plan.all_passing,
            )
            db.add(learning_plan)
            await db.commit()

        await retry_db_operation(
            _do_persist,
            session_id=str(session_id),
            data=plan.model_dump(),
        )

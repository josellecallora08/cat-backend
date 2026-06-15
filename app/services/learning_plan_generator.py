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
from app.services.db_retry import retry_db_operation

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

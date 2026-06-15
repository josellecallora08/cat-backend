"""Evaluation Pipeline orchestrator for post-session processing.

Connects the end-of-session flow: transcript retrieval → evaluation →
coaching → learning plan generation. All results are persisted with the
retry wrapper and associated to the session_id.

In production this would run as a background task (e.g., via Celery or
FastAPI BackgroundTasks). Currently exposed as a callable from the
end_session endpoint.

Validates: Requirements 5.1, 6.1, 7.8, 8.2, 8.3
"""

import logging
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Transcript
from app.schemas import CoachingReportSchema, EvaluationResult, LearningPlanSchema
from app.services.coaching_engine import CoachingEngine
from app.services.evaluation_engine import EvaluationEngine
from app.services.learning_plan_generator import LearningPlanGenerator
from app.services.llm_service import LLMServiceProtocol

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Contains all artifacts produced by the evaluation pipeline."""

    session_id: UUID
    transcript: list[dict]
    evaluation: EvaluationResult
    coaching_report: CoachingReportSchema
    learning_plan: LearningPlanSchema


class EvaluationPipeline:
    """Orchestrates the full post-session evaluation flow.

    Pipeline stages:
    1. Retrieve transcript from database
    2. Run evaluation engine (scoring + strengths/weaknesses)
    3. Run coaching engine (mistake identification + recommendations)
    4. Run learning plan generator (weak competency mapping)

    All persistence is handled by the individual engines using the
    retry wrapper, ensuring resilience against transient DB failures.

    NOTE: In production, this pipeline should be invoked as a background
    task to avoid blocking the HTTP response. The 30s completion target
    (Requirement 5.1) is met by the LLM inference budget, not by
    synchronous request handling.
    """

    def __init__(self, llm_service: LLMServiceProtocol):
        """Initialize the pipeline with required services.

        Args:
            llm_service: LLM service for evaluation and coaching engines.
        """
        self._evaluation_engine = EvaluationEngine(llm_service=llm_service)
        self._coaching_engine = CoachingEngine(llm_service=llm_service)
        self._learning_plan_generator = LearningPlanGenerator()

    async def get_transcript(
        self, session_id: UUID, db: AsyncSession
    ) -> list[dict]:
        """Retrieve the transcript for a session as a list of dicts.

        Args:
            session_id: The UUID of the session.
            db: Async database session.

        Returns:
            List of transcript entry dicts with 'speaker' and 'text' keys,
            ordered by sequence_number ascending.
        """
        stmt = (
            select(Transcript)
            .where(Transcript.session_id == session_id)
            .order_by(Transcript.sequence_number.asc())
        )
        result = await db.execute(stmt)
        entries = result.scalars().all()

        return [
            {"speaker": entry.speaker, "text": entry.utterance_text}
            for entry in entries
        ]

    async def run_evaluation(
        self, session_id: UUID, transcript: list[dict], db: AsyncSession
    ) -> EvaluationResult:
        """Run the evaluation engine on the transcript.

        Args:
            session_id: The UUID of the session.
            transcript: List of transcript entry dicts.
            db: Async database session for persistence.

        Returns:
            EvaluationResult with scores, strengths, and weaknesses.
        """
        return await self._evaluation_engine.evaluate(
            session_id=session_id,
            transcript=transcript,
            db=db,
        )

    async def run_coaching(
        self,
        session_id: UUID,
        transcript: list[dict],
        evaluation: EvaluationResult,
        db: AsyncSession,
    ) -> CoachingReportSchema:
        """Run the coaching engine to identify mistakes and recommendations.

        Args:
            session_id: The UUID of the session.
            transcript: List of transcript entry dicts.
            evaluation: The evaluation result to guide coaching.
            db: Async database session for persistence.

        Returns:
            CoachingReportSchema with mistakes grouped by category.
        """
        return await self._coaching_engine.generate_report(
            session_id=session_id,
            transcript=transcript,
            evaluation=evaluation,
            db=db,
        )

    async def run_learning_plan(
        self,
        session_id: UUID,
        agent_id: UUID,
        evaluation: EvaluationResult,
        db: AsyncSession,
    ) -> LearningPlanSchema:
        """Generate and persist the learning plan from evaluation results.

        Args:
            session_id: The UUID of the session.
            agent_id: The UUID of the agent.
            evaluation: The evaluation result to derive weak competencies.
            db: Async database session for persistence.

        Returns:
            LearningPlanSchema with weak competencies and scenario mappings.
        """
        return await self._learning_plan_generator.generate_and_persist(
            evaluation=evaluation,
            session_id=session_id,
            agent_id=agent_id,
            db=db,
        )

    async def run(
        self, session_id: UUID, agent_id: UUID, db: AsyncSession
    ) -> PipelineResult:
        """Execute the full evaluation pipeline end-to-end.

        Orchestrates: transcript → evaluation → coaching → learning plan.
        All artifacts are persisted by the individual engines with retry
        logic and associated to the session_id.

        Args:
            session_id: The UUID of the completed session.
            agent_id: The UUID of the agent who completed the session.
            db: Async database session.

        Returns:
            PipelineResult containing all generated artifacts.

        Raises:
            ValueError: If no transcript is found for the session.
        """
        logger.info("Starting evaluation pipeline for session %s", session_id)

        # Step 1: Retrieve transcript
        transcript = await self.get_transcript(session_id, db)
        if not transcript:
            logger.warning(
                "No transcript found for session %s, running with empty transcript",
                session_id,
            )

        # Step 2: Run evaluation
        evaluation = await self.run_evaluation(session_id, transcript, db)
        logger.info(
            "Evaluation complete for session %s: overall_score=%.1f, is_too_short=%s",
            session_id,
            evaluation.overall_score,
            evaluation.is_too_short,
        )

        # Step 3: Run coaching (even for too-short sessions, produces empty report)
        coaching_report = await self.run_coaching(
            session_id, transcript, evaluation, db
        )
        logger.info(
            "Coaching complete for session %s: %d mistakes identified",
            session_id,
            coaching_report.total_mistakes,
        )

        # Step 4: Generate learning plan
        learning_plan = await self.run_learning_plan(
            session_id, agent_id, evaluation, db
        )
        logger.info(
            "Learning plan generated for session %s: all_passing=%s",
            session_id,
            learning_plan.all_passing,
        )

        logger.info("Evaluation pipeline completed for session %s", session_id)

        return PipelineResult(
            session_id=session_id,
            transcript=transcript,
            evaluation=evaluation,
            coaching_report=coaching_report,
            learning_plan=learning_plan,
        )

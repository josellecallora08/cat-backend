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
from types import SimpleNamespace
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Session, Transcript
from app.schemas import (
    CoachingReportSchema,
    EvaluationCategory,
    EvaluationResult,
    LearningPlanSchema,
)
from app.schemas.event import EventMetadata
from app.schemas.rubric_evaluation import CanonicalEvaluationResult
from app.services.coaching_engine import CoachingEngine
from app.services.evaluation_compatibility import build_rubric_recommendations
from app.services.evaluation_engine import EvaluationEngine
from app.services.event_instances import event_broadcaster
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

    async def get_transcript(self, session_id: UUID, db: AsyncSession) -> list[dict]:
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

        return [{"speaker": entry.speaker, "text": entry.utterance_text} for entry in entries]

    async def run_evaluation(
        self, session_id: UUID, transcript: list[dict], db: AsyncSession
    ) -> EvaluationResult:
        """Run legacy-compatible or pinned rubric evaluation for a session."""
        if isinstance(db, AsyncSession):
            result = await db.execute(select(Session).where(Session.id == session_id))
            session = result.scalar_one_or_none()
            if session is None or session.negotiation_standard_version is None:
                raise ValueError("Session has no pinned published negotiation standard version")
            standard_version = session.negotiation_standard_version
            pinned_version = SimpleNamespace(
                id=standard_version.id,
                version_number=standard_version.version_number,
                snapshot=standard_version.snapshot,
                standard=SimpleNamespace(name=standard_version.standard.name),
            )
            # The session lookup starts a transaction. The rubric evaluation
            # performs external LLM calls, so release the connection first
            # without closing the request-owned session that later stages use.
            await db.rollback()
            standard_version = pinned_version
            rubric_transcript = [
                {**entry, "sequence_number": entry.get("sequence_number", index)}
                for index, entry in enumerate(transcript)
            ]
            canonical = await self._evaluation_engine.evaluate_rubric(
                session_id=session_id,
                transcript=rubric_transcript,
                standard_version=standard_version,
                db=None,
            )
            return self._canonical_to_legacy_result(session_id, canonical, standard_version)

        # Existing unit callers use lightweight mocks and the legacy contract.
        return await self._evaluation_engine.evaluate(
            session_id=session_id,
            transcript=transcript,
            db=db,
        )

    @staticmethod
    def _canonical_to_legacy_result(
        session_id: UUID,
        canonical: CanonicalEvaluationResult,
        version,
    ) -> EvaluationResult:
        """Expose a safe compatibility view while canonical data remains authoritative."""
        canonical = canonical.model_copy(
            update={
                "recommendations": build_rubric_recommendations(
                    canonical,
                    version.snapshot,
                    getattr(version, "id", None),
                    getattr(version, "version_number", None),
                )
            }
        )

        def legacy_category(value: str) -> EvaluationCategory:
            try:
                return EvaluationCategory(value)
            except ValueError:
                return EvaluationCategory.CALL_OPENING

        evidence_by_sequence = {
            item.sequence_number: item.excerpt
            for category in canonical.categories
            for item in category.evidence
        }
        strengths = [
            {
                "description": item.explanation,
                "category": legacy_category(category.category.lower().replace(" ", "_")),
                "transcript_excerpt": evidence_by_sequence.get(
                    item.evidence_sequence_numbers[0], "Evidence-backed finding"
                ),
            }
            for category in canonical.categories
            for item in category.strengths
        ][:5]
        weaknesses = [
            {
                "description": item.explanation,
                "category": legacy_category(category.category.lower().replace(" ", "_")),
                "transcript_excerpt": evidence_by_sequence.get(
                    item.evidence_sequence_numbers[0], "Evidence-backed finding"
                ),
            }
            for category in canonical.categories
            for item in category.violations
        ][:5]
        if not strengths:
            strengths = [
                {
                    "description": canonical.summary,
                    "category": EvaluationCategory.CALL_OPENING,
                    "transcript_excerpt": canonical.summary,
                }
            ]
        if not weaknesses:
            weaknesses = [
                {
                    "description": canonical.summary,
                    "category": EvaluationCategory.CALL_OPENING,
                    "transcript_excerpt": canonical.summary,
                }
            ]

        return EvaluationResult(
            session_id=session_id,
            category_scores=[],
            overall_score=float(canonical.weighted_total),
            strengths=strengths,
            weaknesses=weaknesses,
            is_too_short=canonical.status == "not_applicable",
            negotiation_standard_version_id=version.id,
            standard_name=getattr(getattr(version, "standard", None), "name", None),
            standard_version_number=version.version_number,
            standard_snapshot=version.snapshot,
            rubric_result=canonical.model_dump(mode="json"),
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
        *,
        persist: bool = True,
    ) -> LearningPlanSchema:
        """Generate and persist the learning plan from evaluation results."""
        return await self._learning_plan_generator.generate_and_persist(
            evaluation=evaluation,
            session_id=session_id,
            agent_id=agent_id,
            db=db,
            persist=persist,
        )

    async def _persist_rubric_artifacts(
        self,
        evaluation: EvaluationResult,
        coaching_report: CoachingReportSchema,
        learning_plan: LearningPlanSchema,
        agent_id: UUID,
        db: AsyncSession,
    ) -> None:
        """Commit canonical evaluation and derived artifacts as one unit."""
        from app.models import CoachingReport, Evaluation, LearningPlan
        from app.services.db_retry import retry_db_operation

        if evaluation.rubric_result is None or evaluation.standard_snapshot is None:
            raise ValueError("Pinned rubric evaluation is missing canonical persistence data")

        canonical = CanonicalEvaluationResult.model_validate(evaluation.rubric_result)
        serialized_coaching = {
            category.value: [item.model_dump(mode="json") for item in items]
            for category, items in coaching_report.mistakes_by_category.items()
        }

        if coaching_report.rubric_coaching is not None:
            serialized_coaching["_rubric_coaching"] = coaching_report.rubric_coaching.model_dump(
                mode="json"
            )
        if coaching_report.rubric_recommendations:
            serialized_coaching["_rubric_recommendations"] = [
                item.model_dump(mode="json") for item in coaching_report.rubric_recommendations
            ]
            serialized_coaching["_rubric_recommendations_by_block"] = {
                block_id: [item.model_dump(mode="json") for item in items]
                for block_id, items in coaching_report.rubric_recommendations_by_block.items()
            }

        async def _do_persist() -> None:
            # Evaluation, coaching, and learning-plan rows are one-per-session
            # artifacts. Update existing rows so a failed pipeline can be
            # retried safely without violating their unique session_id keys.
            evaluation_row = (
                await db.execute(
                    select(Evaluation).where(Evaluation.session_id == evaluation.session_id)
                )
            ).scalar_one_or_none()
            if evaluation_row is None:
                evaluation_row = Evaluation(session_id=evaluation.session_id)
                db.add(evaluation_row)
            evaluation_row.overall_score = evaluation.overall_score
            evaluation_row.category_scores = [
                item.model_dump(mode="json") for item in canonical.categories
            ]
            evaluation_row.strengths = [
                item.model_dump(mode="json") for item in evaluation.strengths
            ]
            evaluation_row.weaknesses = [
                item.model_dump(mode="json") for item in evaluation.weaknesses
            ]
            evaluation_row.negotiation_standard_version_id = (
                evaluation.negotiation_standard_version_id
            )
            evaluation_row.standard_snapshot = evaluation.standard_snapshot
            evaluation_row.weighted_total = float(canonical.weighted_total)
            evaluation_row.passing_score = canonical.passing_score
            evaluation_row.passed = canonical.passed
            evaluation_row.rubric_result = canonical.model_dump(mode="json")
            evaluation_row.is_too_short = evaluation.is_too_short

            coaching_row = (
                await db.execute(
                    select(CoachingReport).where(CoachingReport.session_id == evaluation.session_id)
                )
            ).scalar_one_or_none()
            if coaching_row is None:
                coaching_row = CoachingReport(session_id=evaluation.session_id)
                db.add(coaching_row)
            coaching_row.mistakes_by_category = serialized_coaching
            coaching_row.total_mistakes = coaching_report.total_mistakes
            coaching_row.no_mistakes = coaching_report.no_mistakes

            learning_plan_row = (
                await db.execute(
                    select(LearningPlan).where(LearningPlan.session_id == evaluation.session_id)
                )
            ).scalar_one_or_none()
            if learning_plan_row is None:
                learning_plan_row = LearningPlan(
                    session_id=evaluation.session_id,
                    agent_id=agent_id,
                )
                db.add(learning_plan_row)
            learning_plan_row.agent_id = agent_id
            learning_plan_row.weak_competencies = [
                item.model_dump(mode="json") for item in learning_plan.weak_competencies
            ]
            learning_plan_row.all_passing = learning_plan.all_passing
            try:
                await db.commit()
            except Exception:
                await db.rollback()
                raise

        try:
            await retry_db_operation(
                _do_persist,
                session_id=str(evaluation.session_id),
                data={"rubric_result": canonical.model_dump(mode="json")},
            )
        except Exception:
            await db.rollback()
            raise

    async def run(self, session_id: UUID, agent_id: UUID, db: AsyncSession) -> PipelineResult:
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

        rubric_transaction = isinstance(db, AsyncSession)

        # Step 2: Run evaluation
        evaluation = await self.run_evaluation(
            session_id,
            transcript,
            db,
        )
        logger.info(
            "Evaluation complete for session %s: overall_score=%.1f, is_too_short=%s",
            session_id,
            evaluation.overall_score,
            evaluation.is_too_short,
        )

        # Step 3: Run coaching (even for too-short sessions, produces empty report)
        coaching_report = await self.run_coaching(
            session_id,
            transcript,
            evaluation,
            None if rubric_transaction else db,
        )
        logger.info(
            "Coaching complete for session %s: %d mistakes identified",
            session_id,
            coaching_report.total_mistakes,
        )

        # Step 4: Generate learning plan
        learning_plan = await self.run_learning_plan(
            session_id,
            agent_id,
            evaluation,
            db,
            persist=not rubric_transaction,
        )
        logger.info(
            "Learning plan generated for session %s: all_passing=%s",
            session_id,
            learning_plan.all_passing,
        )

        logger.info("Evaluation pipeline completed for session %s", session_id)

        if rubric_transaction:
            await self._persist_rubric_artifacts(
                evaluation, coaching_report, learning_plan, agent_id, db
            )

        # Event delivery is observability, not artifact persistence. A broken
        # subscriber must not make a successful evaluation look failed or force
        # a retry that collides with the one-row artifact constraints.
        try:
            await event_broadcaster.emit(
                "session.evaluated",
                session_id,
                EventMetadata(agent_id=agent_id),
            )
            await event_broadcaster.emit(
                "dashboard.updated",
                session_id,
                EventMetadata(),
            )
        except Exception:
            logger.exception(
                "Evaluation pipeline event delivery failed for session %s",
                session_id,
            )

        return PipelineResult(
            session_id=session_id,
            transcript=transcript,
            evaluation=evaluation,
            coaching_report=coaching_report,
            learning_plan=learning_plan,
        )

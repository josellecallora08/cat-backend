"""Aggregate persisted session artifacts into the normalized report contract."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from app.models import CoachingReport, Evaluation, LearningPlan, Session, Transcript
from app.models.user import User


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.report import (
    EvaluationKind,
    EvaluationVersionMetadata,
    ReportCompletion,
    ReportFailure,
    ReportResponse,
    ReportSectionName,
    ReportSessionMetadata,
    ScoreStatus,
    SectionEnvelope,
    SectionState,
)
from app.services.report_failures import FailureClass, build_failure_context
from app.services.session_access import get_authorized_session


logger = logging.getLogger(__name__)


class ReportService:
    """Build a report from immutable session artifacts and isolated section queries."""

    _SECTION_NAMES = (
        ReportSectionName.METADATA,
        ReportSectionName.TRANSCRIPT,
        ReportSectionName.EVALUATION,
        ReportSectionName.COACHING,
        ReportSectionName.LEARNING_PLAN,
        ReportSectionName.SUMMARY,
    )

    def __init__(self, db: AsyncSession) -> None:
        """Initialize the service with the request-scoped database session."""
        self.db = db

    async def get_report(self, session_id: UUID, current_user: User) -> ReportResponse:
        """Return the authorized session's normalized aggregate report."""
        session = await get_authorized_session(self.db, session_id, current_user)
        sections = [
            self._loaded(ReportSectionName.METADATA, self._metadata(session)),
            await self._load_section(ReportSectionName.TRANSCRIPT, session.id, self._transcript),
            await self._load_section(ReportSectionName.EVALUATION, session.id, self._evaluation),
            await self._load_section(ReportSectionName.COACHING, session.id, self._coaching),
            await self._load_section(
                ReportSectionName.LEARNING_PLAN, session.id, self._learning_plan
            ),
        ]
        evaluation = next(
            (section.data for section in sections if section.name == ReportSectionName.EVALUATION),
            None,
        )
        agent_id = getattr(session, "agent_id", None)
        participant_name = (
            await self.db.scalar(select(User.full_name).where(User.id == agent_id))
            if agent_id is not None
            else None
        )
        if not isinstance(participant_name, str):
            participant_name = None
        scenario = getattr(session, "scenario", None)
        campaign = getattr(session, "campaign", None)
        sections.append(self._summary(evaluation))
        return ReportResponse(
            session=ReportSessionMetadata(
                id=session.id,
                status=session.status,
                created_at=session.created_at,
                ended_at=session.ended_at,
                scenario_id=session.scenario_id,
                scenario_name=getattr(scenario, "name", None),
                campaign_id=session.campaign_id,
                campaign_name=getattr(campaign, "name", None),
                participant_name=participant_name,
            ),
            report_status=self._completion(sections, evaluation),
            score_status=self._score_status(evaluation),
            evaluation_version=self._version(session, evaluation),
            sections=sections,
        )

    def _metadata(self, session: Session) -> dict[str, Any]:
        """Return safe session metadata for the metadata section."""
        return {
            "id": session.id,
            "status": session.status,
            "created_at": session.created_at,
            "ended_at": session.ended_at,
            "scenario_id": session.scenario_id,
            "scenario_name": getattr(getattr(session, "scenario", None), "name", None),
            "campaign_id": session.campaign_id,
            "campaign_name": getattr(getattr(session, "campaign", None), "name", None),
        }

    async def _load_section(
        self,
        name: ReportSectionName,
        session_id: UUID,
        loader: Callable[[UUID], Awaitable[Any | None]],
    ) -> SectionEnvelope:
        """Load one artifact and convert expected absence/errors to terminal state."""
        try:
            data = await loader(session_id)
        except Exception:
            logger.exception("Report section load failed: %s", name.value)
            context = build_failure_context(
                component="report",
                method="GET",
                route=f"/api/sessions/{session_id}/report",
                failure_class=FailureClass.BACKEND,
                status=500,
                code=f"{name.value}_unavailable",
            )
            return SectionEnvelope(
                name=name,
                state=SectionState.FAILED,
                failure=ReportFailure(
                    class_=context.failure_class,
                    code=context.code or "section_unavailable",
                    safe_message=context.safe_message,
                    correlation_id=context.correlation_id,
                ),
                updated_at=datetime.now(UTC),
            )
        if data is None or data == []:
            return SectionEnvelope(
                name=name,
                state=SectionState.EMPTY,
                unavailable_reason="No persisted data is available for this section.",
                updated_at=datetime.now(UTC),
            )
        return self._loaded(name, data)

    async def _transcript(self, session_id: UUID) -> list[dict[str, Any]]:
        """Load transcript entries in their persisted order."""
        result = await self.db.execute(
            select(Transcript)
            .where(Transcript.session_id == session_id)
            .order_by(Transcript.sequence_number.asc())
        )
        return [
            {
                "speaker": item.speaker,
                "text": item.utterance_text,
                "timestamp": item.timestamp_ms,
                "sequence_number": item.sequence_number,
            }
            for item in result.scalars().all()
        ]

    async def _evaluation(self, session_id: UUID) -> dict[str, Any] | None:
        """Load the persisted evaluation without recalculating it."""
        result = await self.db.execute(
            select(Evaluation).where(Evaluation.session_id == session_id)
        )
        item = result.scalar_one_or_none()
        if item is None:
            return None
        is_too_short = bool(item.is_too_short) or (
            isinstance(item.rubric_result, dict)
            and item.rubric_result.get("status") == "not_applicable"
        )
        version = getattr(item, "negotiation_standard_version", None)
        standard = getattr(version, "standard", None)
        return {
            "session_id": item.session_id,
            "overall_score": None if is_too_short else item.overall_score,
            "category_scores": [] if is_too_short else (item.category_scores or []),
            "strengths": item.strengths or [],
            "weaknesses": item.weaknesses or [],
            "weighted_total": None if is_too_short else item.weighted_total,
            "passing_score": None if is_too_short else item.passing_score,
            "passed": None if is_too_short else item.passed,
            "is_too_short": is_too_short,
            "negotiation_standard_version_id": item.negotiation_standard_version_id,
            "standard_version_number": getattr(version, "version_number", None),
            "standard_name": getattr(standard, "name", None),
            "standard_snapshot": item.standard_snapshot,
            "rubric_result": item.rubric_result,
        }

    async def _coaching(self, session_id: UUID) -> dict[str, Any] | None:
        """Load the persisted coaching report."""
        result = await self.db.execute(
            select(CoachingReport).where(CoachingReport.session_id == session_id)
        )
        item = result.scalar_one_or_none()
        return (
            None
            if item is None
            else {
                "session_id": item.session_id,
                "mistakes_by_category": item.mistakes_by_category or {},
                "total_mistakes": item.total_mistakes,
                "no_mistakes": item.no_mistakes,
            }
        )

    async def _learning_plan(self, session_id: UUID) -> dict[str, Any] | None:
        """Load the persisted learning plan."""
        result = await self.db.execute(
            select(LearningPlan).where(LearningPlan.session_id == session_id)
        )
        item = result.scalar_one_or_none()
        return (
            None
            if item is None
            else {
                "session_id": item.session_id,
                "weak_competencies": item.weak_competencies or [],
                "all_passing": item.all_passing,
            }
        )

    def _summary(self, evaluation: Any) -> SectionEnvelope:
        """Build a summary from persisted evaluation values only."""
        if evaluation is None:
            return SectionEnvelope(
                name=ReportSectionName.SUMMARY,
                state=SectionState.EMPTY,
                unavailable_reason="An evaluation is not available yet.",
            )
        return self._loaded(
            ReportSectionName.SUMMARY,
            {
                "overall_score": evaluation.get("overall_score"),
                "passed": evaluation.get("passed"),
                "is_too_short": evaluation.get("is_too_short", False),
            },
        )

    def _version(self, session: Session, evaluation: Any) -> EvaluationVersionMetadata:
        """Map the pinned version relationship, never the current configuration."""
        version = getattr(session, "negotiation_standard_version", None)
        version_id = evaluation.get("negotiation_standard_version_id") if evaluation else None
        if evaluation is None or version_id is None:
            return EvaluationVersionMetadata(kind=EvaluationKind.LEGACY)
        if version is None and evaluation.get("standard_version_number") is None:
            return EvaluationVersionMetadata(kind=EvaluationKind.LEGACY)
        standard = getattr(version, "standard", None)
        number = getattr(version, "version_number", None) or evaluation.get(
            "standard_version_number"
        )
        name = getattr(standard, "name", None) or evaluation.get("standard_name")
        if number is None:
            return EvaluationVersionMetadata(kind=EvaluationKind.LEGACY)
        return EvaluationVersionMetadata(
            kind=EvaluationKind.CURRENT,
            id=version_id,
            number=number,
            name=name,
        )

    @staticmethod
    def _loaded(name: ReportSectionName, data: Any) -> SectionEnvelope:
        """Create a loaded section envelope."""
        return SectionEnvelope(
            name=name, state=SectionState.LOADED, data=data, updated_at=datetime.now(UTC)
        )

    @staticmethod
    def _score_status(evaluation: Any) -> ScoreStatus:
        """Classify score semantics without treating too-short as pass/fail."""
        if evaluation is None:
            return ScoreStatus.UNAVAILABLE
        rubric_result = evaluation.get("rubric_result") or {}
        if evaluation.get("is_too_short") or rubric_result.get("status") == "not_applicable":
            return ScoreStatus.NOT_APPLICABLE
        return ScoreStatus.EVALUATED

    @staticmethod
    def _completion(sections: list[SectionEnvelope], evaluation: Any) -> ReportCompletion:
        """Aggregate terminal section states into the report completion state."""
        states = {section.state for section in sections}
        if states == {SectionState.FAILED}:
            return ReportCompletion.FAILED
        if (
            evaluation is not None
            and ReportService._score_status(evaluation) == ScoreStatus.NOT_APPLICABLE
        ):
            return ReportCompletion.NOT_APPLICABLE
        if SectionState.FAILED in states or SectionState.EMPTY in states:
            return ReportCompletion.PARTIAL
        return ReportCompletion.COMPLETE

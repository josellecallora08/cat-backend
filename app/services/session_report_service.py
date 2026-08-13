"""Persist versioned session reports and expose current-report lookup.

A report is generated only for a `completed` session. Generation never
mutates a previously stored payload: each call inserts a new row at the next
version under a transaction-safe session-row lock. A pending row is promoted
to ready only after structural validation, or to failed with a typed reason;
non-ready rows never contain readable payload or hash data.
"""

import logging
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    CoachingReport,
    Evaluation,
    LearningPlan,
    Session,
    SessionReport,
    SessionReportStatus,
    Transcript,
)
from app.models.session_report import SessionReportReasonCode
from app.schemas.session_report import (
    ReadyReport,
    ReportAttemptMetadata,
    ReportEmptyTranscriptStatus,
    ReportFailedStatus,
    ReportGeneratingStatus,
    ReportIncompleteStatus,
    ReportLegacyOnlyStatus,
    ReportMissingStatus,
    ReportNoEvidenceStatus,
    ReportNotApplicableStatus,
    ReportReadyStatus,
    ReportReason,
    ReportStatusEnvelope,
    ReportTooShortStatus,
    SessionReportPayload,
)
from app.services.audit import (
    log_session_report_generated,
    log_session_report_generation_failed,
)
from app.services.session_report_assembler import (
    assemble_report_payload,
    compute_content_hash,
)


logger = logging.getLogger(__name__)


class SessionNotCompletedError(ValueError):
    """Raised when report generation is requested for a non-completed session."""

    def __init__(self, session_id: UUID, status: str) -> None:
        super().__init__(
            f"Session {session_id} must be completed to generate a report "
            f"(current status: {status})"
        )
        self.session_id = session_id
        self.status = status


class SessionReportConflictError(RuntimeError):
    """Raised when concurrent report generation loses its allocation boundary."""

    def __init__(self, session_id: UUID) -> None:
        super().__init__("A report generation conflict occurred")
        self.session_id = session_id


async def _next_report_version(db: AsyncSession, session_id: UUID) -> int:
    """Return the next version while holding the session allocation lock.

    PostgreSQL honors ``FOR UPDATE`` on the selected session row.  The
    pending-marker check and version allocation occur before the marker is
    committed, so an overlapping request observes the marker and conflicts.
    The unique session/version constraint remains the final guard for dialects
    that do not provide row locks (including SQLite test databases).
    """
    lock_stmt = select(Session.id).where(Session.id == session_id).with_for_update()
    locked_session = (await db.execute(lock_stmt)).scalar_one_or_none()
    if locked_session is None:
        raise ValueError(f"Session with id {session_id} not found")

    pending_stmt = (
        select(SessionReport.id)
        .where(
            SessionReport.session_id == session_id,
            SessionReport.status == SessionReportStatus.PENDING,
        )
        .limit(1)
    )
    if (await db.execute(pending_stmt)).scalar_one_or_none() is not None:
        raise SessionReportConflictError(session_id)

    stmt = select(func.max(SessionReport.report_version)).where(
        SessionReport.session_id == session_id
    )
    current_max = (await db.execute(stmt)).scalar_one_or_none()
    return (current_max or 0) + 1


def _is_serialization_conflict(exc: Exception) -> bool:
    """Classify database errors that must become the public 409 outcome."""
    return isinstance(exc, IntegrityError | OperationalError)


async def _commit_report_transaction(db: AsyncSession, session_id: UUID) -> None:
    """Commit a generation transaction and normalize allocation conflicts."""
    try:
        await db.commit()
    except Exception as exc:
        if _is_serialization_conflict(exc):
            await db.rollback()
            raise SessionReportConflictError(session_id) from exc
        await db.rollback()
        raise


async def generate_report(
    db: AsyncSession,
    session: Session,
    generated_by: UUID | None = None,
) -> SessionReport:
    """Assemble and persist a new versioned report for a completed session.

    A pending row is allocated under the session row lock before assembly. It
    is then promoted to ready only after full payload validation, or to failed
    with null payload/hash and a finite generation reason. Existing rows are
    never updated, so regeneration preserves every prior snapshot value.

    Raises:
        SessionNotCompletedError: if the selected session is not completed.
        SessionReportConflictError: if concurrent allocation/serialization is
            rejected by the database.
    """
    if session.status != "completed":
        raise SessionNotCompletedError(session.id, session.status)

    try:
        next_version = await _next_report_version(db, session.id)
    except SessionReportConflictError:
        await db.rollback()
        raise
    except Exception as exc:
        if _is_serialization_conflict(exc):
            await db.rollback()
            raise SessionReportConflictError(session.id) from exc
        raise

    pending_row = SessionReport(
        session_id=session.id,
        agent_id=session.agent_id,
        campaign_id=session.campaign_id,
        negotiation_standard_version_id=session.negotiation_standard_version_id,
        status=SessionReportStatus.PENDING,
        report_version=next_version,
        payload=None,
        content_hash=None,
        generated_by=generated_by,
        failure_reason="Report generation is pending",
        reason_code=SessionReportReasonCode.GENERATION_PENDING,
    )
    db.add(pending_row)

    try:
        # Commit the pending marker before doing artifact work. This keeps the
        # allocation transaction short and lets overlapping requests detect
        # the in-flight attempt instead of silently becoming version 2.
        await db.flush()
        await _commit_report_transaction(db, session.id)
    except SessionReportConflictError:
        raise
    except Exception as exc:
        if _is_serialization_conflict(exc):
            await db.rollback()
            raise SessionReportConflictError(session.id) from exc
        await db.rollback()
        raise

    try:
        payload = await assemble_report_payload(db, session)
        # The assembler validates today, but this second boundary is required
        # before any row can become readable and protects alternate assemblers.
        payload = SessionReportPayload.model_validate(payload)
        content_hash = compute_content_hash(payload)
    except Exception as exc:
        logger.error(
            "Session report generation failed for session %s",
            session.id,
        )
        pending_row.status = SessionReportStatus.FAILED
        pending_row.payload = None
        pending_row.content_hash = None
        pending_row.reason_code = SessionReportReasonCode.GENERATION_FAILED
        pending_row.failure_reason = _safe_failure_reason(exc)
        try:
            await _commit_report_transaction(db, session.id)
        except SessionReportConflictError:
            raise
        log_session_report_generation_failed(
            str(session.id),
            next_version,
            SessionReportReasonCode.GENERATION_FAILED,
        )
        raise

    pending_row.status = SessionReportStatus.READY
    pending_row.payload = payload.model_dump(mode="json")
    pending_row.content_hash = content_hash
    pending_row.reason_code = None
    pending_row.failure_reason = None
    await _commit_report_transaction(db, session.id)
    await db.refresh(pending_row)

    logger.info(
        "Session report generated: session=%s version=%d hash=%s",
        session.id,
        next_version,
        content_hash,
    )
    log_session_report_generated(
        str(session.id), next_version, str(generated_by) if generated_by else None
    )
    return pending_row


def _safe_failure_reason(exc: Exception) -> str:
    """Return safe display text without exception arguments or traceback data."""
    # Keep the argument for callers that pass the original exception, but do
    # not inspect it: exception text can contain SQL details or artifact data.
    del exc
    return "Report generation failed"


async def get_current_report(db: AsyncSession, session_id: UUID) -> SessionReport | None:
    """Return the highest-versioned `ready` report for a session, or None."""
    stmt = (
        select(SessionReport)
        .where(
            SessionReport.session_id == session_id,
            SessionReport.status == SessionReportStatus.READY,
        )
        .order_by(SessionReport.report_version.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


_STATUS_REASON_MESSAGES = {
    "artifact_missing": "Required report artifacts are not complete",
    "generation_pending": "Report generation is in progress",
    "generation_failed": "Report generation failed",
    "not_applicable": "Evaluation is not applicable",
    "session_too_short": "The session was too short to evaluate",
    "legacy_only": "This report contains legacy evaluation data",
    "empty_transcript": "No transcript was recorded for this session",
    "no_evidence": "The evaluation contains no evidence",
}


def _status_reason(code: str) -> ReportReason:
    """Build a safe typed status reason without copying artifact text."""
    return ReportReason(code=code, message=_STATUS_REASON_MESSAGES.get(code))


def _attempt_metadata(row: object) -> ReportAttemptMetadata:
    """Build attempt metadata from a query row that never selected payload."""
    status = getattr(row, "status", None)
    expected = (
        SessionReportReasonCode.GENERATION_PENDING
        if status == SessionReportStatus.PENDING
        else SessionReportReasonCode.GENERATION_FAILED
    )
    # Rows created before the typed reason column was introduced, or rows with
    # an invalid legacy value, remain safe to expose: lifecycle status is
    # authoritative and supplies the finite typed code.
    return ReportAttemptMetadata(
        status="pending" if status == SessionReportStatus.PENDING else "failed",
        report_version=row.report_version,
        reason=_status_reason(expected),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _ready_envelope(row: object) -> ReadyReport:
    """Validate one ready snapshot before putting it in a status envelope."""
    payload = SessionReportPayload.model_validate(row.payload)
    return ReadyReport(
        session_id=row.session_id,
        report_version=row.report_version,
        status="ready",
        content_hash=row.content_hash,
        created_at=row.created_at,
        payload=payload,
    )


def _resolve_ready_status(
    ready: ReadyReport,
    latest_attempt: ReportAttemptMetadata | None,
) -> ReportStatusEnvelope:
    """Resolve a readable snapshot, preserving failed regeneration metadata."""
    # A failed regeneration is intentionally represented as ready content plus
    # failure metadata. This preserves the last readable snapshot and follows
    # the selected older-ready-plus-failed contract.
    if latest_attempt is not None:
        return ReportReadyStatus(
            session_id=ready.session_id,
            status="ready",
            report=ready,
            latest_attempt=latest_attempt,
        )

    payload = ready.payload
    evaluation = payload.evaluation
    if evaluation.mode == "not_applicable":
        return ReportNotApplicableStatus(
            session_id=ready.session_id,
            status="not_applicable",
            reason=_status_reason(SessionReportReasonCode.NOT_APPLICABLE),
            report=ready,
        )
    if evaluation.mode == "too_short":
        return ReportTooShortStatus(
            session_id=ready.session_id,
            status="too_short",
            reason=_status_reason(SessionReportReasonCode.SESSION_TOO_SHORT),
            report=ready,
        )
    if evaluation.mode == "legacy":
        return ReportLegacyOnlyStatus(
            session_id=ready.session_id,
            status="legacy_only",
            reason=_status_reason(SessionReportReasonCode.LEGACY_ONLY),
            report=ready,
        )
    if not payload.transcript.entries:
        return ReportEmptyTranscriptStatus(
            session_id=ready.session_id,
            status="empty_transcript",
            reason=_status_reason(SessionReportReasonCode.EMPTY_TRANSCRIPT),
            report=ready,
        )
    if evaluation.reason_code == SessionReportReasonCode.NO_EVIDENCE:
        return ReportNoEvidenceStatus(
            session_id=ready.session_id,
            status="no_evidence",
            reason=_status_reason(SessionReportReasonCode.NO_EVIDENCE),
            report=ready,
        )
    return ReportReadyStatus(
        session_id=ready.session_id,
        status="ready",
        report=ready,
    )


async def _probe_report_artifacts(db: AsyncSession, session_id: UUID) -> list[str]:
    """Probe required artifact presence with a fixed, payload-free query bound."""
    missing = False
    for model in (Transcript, Evaluation, CoachingReport, LearningPlan):
        stmt = select(model.id).where(model.session_id == session_id).limit(1)
        if (await db.execute(stmt)).scalar_one_or_none() is None:
            missing = True
    return [SessionReportReasonCode.ARTIFACT_MISSING] if missing else []


async def get_report_status(db: AsyncSession, session_id: UUID) -> ReportStatusEnvelope:
    """Resolve the approved report-status union for an authorized session.

    Ready and attempt metadata are selected separately. Attempt queries select
    no payload or hash columns, so pending and failed bytes cannot enter the
    status response. Artifact probes run only for a session with no report
    rows, and have a fixed four-query bound independent of artifact size.
    Authorization is deliberately owned by the API layer and must complete
    before this function is called.
    """
    ready_stmt = (
        select(
            SessionReport.session_id,
            SessionReport.report_version,
            SessionReport.status,
            SessionReport.content_hash,
            SessionReport.created_at,
            SessionReport.payload,
        )
        .where(
            SessionReport.session_id == session_id,
            SessionReport.status == SessionReportStatus.READY,
        )
        .order_by(SessionReport.report_version.desc())
        .limit(1)
    )
    ready_row = (await db.execute(ready_stmt)).first()

    attempt_stmt = (
        select(
            SessionReport.session_id,
            SessionReport.report_version,
            SessionReport.status,
            SessionReport.reason_code,
            SessionReport.created_at,
            SessionReport.updated_at,
        )
        .where(
            SessionReport.session_id == session_id,
            SessionReport.status.in_((SessionReportStatus.PENDING, SessionReportStatus.FAILED)),
        )
        .order_by(SessionReport.report_version.desc())
        .limit(1)
    )
    attempt_row = (await db.execute(attempt_stmt)).first()

    if ready_row is not None:
        ready = _ready_envelope(ready_row)
        latest_attempt = None
        if attempt_row is not None and attempt_row.report_version > ready.report_version:
            latest_attempt = _attempt_metadata(attempt_row)
        return _resolve_ready_status(ready, latest_attempt)

    if attempt_row is not None:
        attempt = _attempt_metadata(attempt_row)
        if attempt.status == "pending":
            return ReportGeneratingStatus(
                session_id=session_id,
                status="generating",
                reason=_status_reason(SessionReportReasonCode.GENERATION_PENDING),
                latest_attempt=attempt,
            )
        return ReportFailedStatus(
            session_id=session_id,
            status="failed",
            reason=_status_reason(SessionReportReasonCode.GENERATION_FAILED),
            latest_attempt=attempt,
        )

    missing_sections = await _probe_report_artifacts(db, session_id)
    if missing_sections:
        return ReportIncompleteStatus(
            session_id=session_id,
            status="incomplete",
            reason=_status_reason(SessionReportReasonCode.ARTIFACT_MISSING),
            missing_sections=missing_sections,
        )
    return ReportMissingStatus(
        session_id=session_id,
        status="missing",
        reason=_status_reason(SessionReportReasonCode.ARTIFACT_MISSING),
    )

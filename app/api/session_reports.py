"""Session report retrieval and export endpoints.

Authorization always runs `require_auth` -> `get_authorized_session` before
any report query, matching the policy already established in
app/api/sessions.py. Routes are registered under the same
`/api/sessions` prefix as the sessions router but do not collide with its
`/{session_id}` paths (see app/tests/test_session_reports_api.py for a
route-table assertion).

See .kiro/specs/session-report-generation/tasks.md, task 4.1 and 4.2.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session as get_db_session
from app.models.user import User
from app.schemas.report import ReportResponse
from app.services.auth import require_admin, require_auth
from app.services.report_service import ReportService
from app.services.session_access import get_authorized_session
from app.services.session_report_export import (
    SUPPORTED_FORMATS,
    ExportFormatNotImplementedError,
    UnsupportedExportFormatError,
    build_export_filename,
    render_export,
)
from app.services.session_report_service import (
    SessionNotCompletedError,
    SessionReportConflictError,
    generate_report,
    get_current_report,
    get_report_status,
)


logger = logging.getLogger(__name__)

router = APIRouter()


async def _get_authorized_session(db: AsyncSession, session_id: UUID, current_user: User):
    """Apply the shared policy while keeping report errors non-sensitive."""
    try:
        return await get_authorized_session(db, session_id, current_user)
    except HTTPException as exc:
        if exc.status_code == 404:
            raise HTTPException(status_code=404, detail="Session not found") from None
        if exc.status_code == 403:
            raise HTTPException(status_code=403, detail="Session access denied") from None
        raise


@router.get("/{session_id}/report/status")
async def get_session_report_status(
    session_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_auth),
):
    """Return the authorized session report status without changing GET /report."""
    # Keep the authorization boundary ahead of every report/status query.  In
    # particular, the status resolver performs report and artifact probes.
    session = await _get_authorized_session(db, session_id, current_user)

    try:
        return await get_report_status(db, session.id)
    except Exception:
        logger.error("Session report status resolution failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to determine report status. Please try again.",
        ) from None


@router.get("/{session_id}/report", response_model=ReportResponse)
async def get_session_report(
    session_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_auth),
) -> ReportResponse:
    """Return the stable six-section report for an authorized session."""
    try:
        return await ReportService(db).get_report(session_id, current_user)
    except HTTPException:
        raise
    except Exception:
        logger.error("Session report retrieval failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve the session report. Please try again.",
        ) from None


@router.post("/{session_id}/report", status_code=201)
async def create_session_report(
    session_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_auth),
):
    """Generate (or regenerate) the report for an authorized, completed session."""
    session = await _get_authorized_session(db, session_id, current_user)

    try:
        report = await generate_report(db, session, generated_by=current_user.id)
    except SessionNotCompletedError:
        raise HTTPException(
            status_code=409,
            detail="Session must be completed before a report can be generated",
        ) from None
    except SessionReportConflictError:
        raise HTTPException(
            status_code=409,
            detail="A report generation conflict occurred. Please try again.",
        ) from None
    except Exception:
        # Do not leak assembly internals; generate_report already recorded
        # a safe failure_reason on the persisted failed row.
        raise HTTPException(
            status_code=500,
            detail="Report generation failed. Please try again.",
        ) from None

    return {
        "session_id": str(report.session_id),
        "report_version": report.report_version,
        "status": report.status,
        "content_hash": report.content_hash,
        "created_at": report.created_at.isoformat() if report.created_at else None,
        "payload": report.payload,
    }


@router.get("/{session_id}/report/export")
async def export_session_report(
    session_id: UUID,
    export_format: str = Query(default="json", alias="format"),
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_admin),
):
    session = await _get_authorized_session(db, session_id, current_user)

    # Reject unsupported formats before report lookup/rendering. Authorization
    # still completes first, so denied requests preserve the policy boundary.
    if export_format not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail="Unsupported export format")

    try:
        report = await get_current_report(db, session.id)
    except Exception:
        logger.error("Session report export lookup failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve the session report for export. Please try again.",
        ) from None
    if report is None:
        raise HTTPException(
            status_code=404,
            detail="No report is available for this session",
        )

    try:
        body, media_type = render_export(report, export_format)
    except UnsupportedExportFormatError:
        raise HTTPException(
            status_code=400,
            detail="Unsupported export format",
        ) from None
    except ExportFormatNotImplementedError:
        raise HTTPException(
            status_code=501,
            detail="The requested export format is not available",
        ) from None
    except Exception:
        logger.error("Session report export rendering failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to export the session report. Please try again.",
        ) from None

    filename = build_export_filename(str(session.id), report.report_version, export_format)
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

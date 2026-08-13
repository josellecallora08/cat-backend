"""S1-10: Admin review API endpoints.

Provides:
  GET  /api/scripts/uploads/{upload_id}/review          — review detail
  PATCH /api/scripts/uploads/{upload_id}/review         — edit contract
  POST /api/scripts/uploads/{upload_id}/review/retry    — retry processing
  POST /api/scripts/uploads/{upload_id}/review/reject   — reject upload
  POST /api/scripts/uploads/{upload_id}/review/publish  — publish script

All endpoints require admin authorization.
Endpoints are intentionally thin — business logic lives in review_service.
"""

import json
import logging
from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_session
from app.models.script import Script, ScriptStatus
from app.models.script_upload import ScriptUpload, UploadStatus
from app.models.user import User
from app.schemas.review import (
    ReviewDetailResponse,
    ReviewEditRequest,
    ReviewEditResponse,
    ReviewPublishResponse,
    ReviewRejectRequest,
    ReviewRejectResponse,
    ReviewRetryRequest,
    ReviewRetryResponse,
)
from app.services.auth import require_admin
from app.services.review_service import (
    build_review_detail,
    build_warnings,
    derive_review_status,
    get_published_at,
    load_upload_with_script,
)


router = APIRouter()
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# GET /uploads/{upload_id}/review
# ──────────────────────────────────────────────────────────────────────────────


@router.get(
    "/uploads/{upload_id}/review",
    response_model=ReviewDetailResponse,
)
async def get_review_detail(
    upload_id: UUID,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> ReviewDetailResponse:
    """Return the full admin review detail for an upload."""
    upload, script = await load_upload_with_script(db, upload_id)
    if upload is None:
        raise HTTPException(status_code=404, detail="Upload not found")

    response = build_review_detail(upload, script)

    # Enrich published_at if applicable
    if script and script.status == ScriptStatus.PUBLISHED.value:
        published_at = await get_published_at(db, script)
        response.published_at = published_at

    return response


# ──────────────────────────────────────────────────────────────────────────────
# PATCH /uploads/{upload_id}/review — edit
# ──────────────────────────────────────────────────────────────────────────────


@router.patch(
    "/uploads/{upload_id}/review",
    response_model=ReviewEditResponse,
)
async def edit_review(
    upload_id: UUID,
    body: ReviewEditRequest,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> ReviewEditResponse:
    """Edit the script contract linked to an upload.

    Accepts a complete ScriptContract replacement. Validates with full
    configurable pipeline. Row locking prevents lost updates.
    """
    from app.services.script_registry import update_draft
    from app.services.script_validator import (
        ScriptFormatError,
        ScriptLimits,
        ScriptValidationError,
        validate_script,
    )

    # 1. Load and lock upload row
    stmt = select(ScriptUpload).where(ScriptUpload.id == upload_id).with_for_update()
    result = await db.execute(stmt)
    upload = result.scalar_one_or_none()
    if upload is None:
        raise HTTPException(status_code=404, detail="Upload not found")

    # 2. Must have a linked script
    if upload.script_id is None:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "no_linked_script",
                "message": "This upload has not been converted to a script yet.",
            },
        )

    # 3. Load and lock script
    script_stmt = select(Script).where(Script.id == upload.script_id).with_for_update()
    script_result = await db.execute(script_stmt)
    script = script_result.scalar_one_or_none()

    if script is None or script.is_deleted:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "script_unavailable",
                "message": "Linked script has been deleted.",
            },
        )

    # 4. Verify review status allows editing
    review_status = derive_review_status(upload, script)
    if review_status != "ready_for_review":
        raise HTTPException(
            status_code=422,
            detail={
                "error": "edit_not_allowed",
                "message": f"Cannot edit in current state: {review_status}.",
            },
        )

    # 5. Optimistic concurrency check
    if body.expected_updated_at is not None and script.updated_at:
        if abs((script.updated_at - body.expected_updated_at).total_seconds()) > 1:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "concurrent_modification",
                    "message": "Script has been modified since you loaded it.",
                    "current_updated_at": script.updated_at.isoformat(),
                },
            )

    # 6. Validate submitted contract against full pipeline
    raw_definition = json.dumps(body.script_contract)
    limits = ScriptLimits(
        max_definition_size_bytes=settings.script_max_definition_size_bytes,
        max_trigger_phrases=settings.script_max_trigger_phrases,
        max_expected_replies=settings.script_max_expected_replies,
        max_escalation_conditions=settings.script_max_escalation_conditions,
        max_field_text_length=settings.script_max_field_text_length,
    )

    try:
        validated = validate_script(raw_definition, "json", limits)
    except ScriptFormatError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "format_error", "message": str(exc)},
        )
    except ScriptValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "validation_failed",
                "message": f"Contract validation failed ({len(exc.errors)} error(s)).",
                "errors": exc.errors,
            },
        )

    # 7. Persist via registry service (reuse, don't duplicate)
    try:
        updated_script = await update_draft(
            db,
            script_id=script.id,
            raw_definition=raw_definition,
            format="json",
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (ScriptFormatError, ScriptValidationError) as exc:
        await db.rollback()
        raise HTTPException(
            status_code=422,
            detail={"error": "validation_failed", "message": str(exc)},
        )

    # 8. Audit log (after successful commit by update_draft)
    logger.info(
        "upload_draft_edited",
        extra={
            "upload_id": str(upload_id),
            "script_id": str(script.id),
            "user_id": str(admin.id),
            "action": "edit",
        },
    )

    # 9. Return normalized contract + warnings
    warnings = build_warnings(upload, updated_script, "ready_for_review")
    contract_out = validated.model_dump(mode="json")

    return ReviewEditResponse(
        upload_id=upload.id,
        script_id=updated_script.id,
        script_contract=contract_out,
        review_status="ready_for_review",
        updated_at=updated_script.updated_at,
        warnings=warnings,
        can_publish=len([w for w in warnings if w.severity == "error"]) == 0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# POST /uploads/{upload_id}/review/retry
# ──────────────────────────────────────────────────────────────────────────────


@router.post(
    "/uploads/{upload_id}/review/retry",
    response_model=ReviewRetryResponse,
)
async def retry_review(
    upload_id: UUID,
    body: ReviewRetryRequest = ReviewRetryRequest(),
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> ReviewRetryResponse:
    """Retry a failed processing step for an upload.

    Targets: conversion, scan, extraction.
    """
    # 1. Load and lock upload
    stmt = select(ScriptUpload).where(ScriptUpload.id == upload_id).with_for_update(nowait=True)
    try:
        result = await db.execute(stmt)
    except Exception:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "processing_in_progress",
                "message": "Another retry is already in progress.",
            },
        )
    upload = result.scalar_one_or_none()
    if upload is None:
        raise HTTPException(status_code=404, detail="Upload not found")

    # 2. Global rejection for unsafe states
    if upload.scan_status == "infected":
        raise HTTPException(
            status_code=422,
            detail={
                "error": "retry_not_allowed",
                "message": "Cannot retry infected files.",
            },
        )
    if upload.status == UploadStatus.DELETED.value:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "retry_not_allowed",
                "message": "Cannot retry deleted uploads.",
            },
        )
    if upload.status == UploadStatus.REJECTED.value:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "retry_not_allowed",
                "message": "Cannot retry rejected uploads.",
            },
        )

    # Load linked script if any
    script = None
    if upload.script_id:
        script_stmt = select(Script).where(Script.id == upload.script_id)
        script_result = await db.execute(script_stmt)
        script = script_result.scalar_one_or_none()

    review_status = derive_review_status(upload, script)

    if review_status == "published":
        raise HTTPException(
            status_code=422,
            detail={
                "error": "retry_not_allowed",
                "message": "Cannot retry already-published uploads.",
            },
        )

    # 3. Dispatch by target
    if body.target == "conversion":
        return await _retry_conversion(db, upload, admin)
    if body.target == "scan":
        return await _retry_scan(db, upload, admin)
    if body.target == "extraction":
        return await _retry_extraction(db, upload, admin)
    raise HTTPException(
        status_code=422,
        detail={
            "error": "invalid_target",
            "message": f"Unknown retry target: '{body.target}'. "
            "Valid: conversion, scan, extraction.",
        },
    )


async def _retry_conversion(
    db: AsyncSession, upload: ScriptUpload, admin: User
) -> ReviewRetryResponse:
    """Retry conversion for clean, extracted content."""
    if upload.scan_status != "clean":
        raise HTTPException(
            status_code=422,
            detail={
                "error": "retry_not_allowed",
                "message": f"Scan status '{upload.scan_status}' is not eligible for conversion.",
            },
        )
    if upload.extraction_status != "completed":
        raise HTTPException(
            status_code=422,
            detail={
                "error": "retry_not_allowed",
                "message": f"Extraction status '{upload.extraction_status}' is not eligible.",
            },
        )
    if not upload.extracted_content or not upload.extracted_content.strip():
        raise HTTPException(
            status_code=422,
            detail={
                "error": "retry_not_allowed",
                "message": "No extracted content available for conversion.",
            },
        )
    if upload.scenario_id is None:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "retry_not_allowed",
                "message": "No scenario_id. A valid scenario is required.",
            },
        )
    if upload.script_id is not None:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "already_converted",
                "message": "This upload already has a linked script. Use edit instead.",
            },
        )

    from app.services.conversion_service import (
        ConversionConflict,
        ConversionInternalError,
        ConversionRejection,
        ConversionSuccess,
        convert_upload_to_script_draft,
    )

    try:
        conv_result = await convert_upload_to_script_draft(db, upload.id, admin.id)
    except ConversionInternalError:
        logger.warning(
            "upload_retry_failed",
            extra={
                "upload_id": str(upload.id),
                "user_id": str(admin.id),
                "action": "retry",
                "target": "conversion",
            },
        )
        raise HTTPException(
            status_code=500,
            detail="Conversion retry failed internally. No changes were made.",
        )

    if isinstance(conv_result, ConversionSuccess):
        logger.info(
            "upload_retry_completed",
            extra={
                "upload_id": str(upload.id),
                "script_id": str(conv_result.script_id),
                "user_id": str(admin.id),
                "action": "retry",
                "target": "conversion",
            },
        )
        return ReviewRetryResponse(
            upload_id=upload.id,
            script_id=conv_result.script_id,
            review_status="ready_for_review",
        )

    if isinstance(conv_result, ConversionConflict):
        raise HTTPException(
            status_code=409,
            detail={"error": conv_result.error, "message": conv_result.message},
        )

    if isinstance(conv_result, ConversionRejection):
        detail: dict = {"error": conv_result.error, "message": conv_result.message}
        if conv_result.details:
            detail["details"] = conv_result.details
        raise HTTPException(status_code=422, detail=detail)

    raise HTTPException(status_code=500, detail="Unexpected conversion result")


async def _retry_scan(db: AsyncSession, upload: ScriptUpload, admin: User) -> ReviewRetryResponse:
    """Retry malware scan if quarantine source still exists."""
    from app.services.upload_quarantine import get_quarantine_path
    from app.services.upload_scanner import scan_file

    quarantine_dir = get_quarantine_path()
    source_path = quarantine_dir / upload.storage_key

    if not source_path.exists():
        raise HTTPException(
            status_code=410,
            detail={
                "error": "source_expired",
                "message": "Quarantine source file no longer exists.",
            },
        )

    if upload.scan_status != "error":
        raise HTTPException(
            status_code=422,
            detail={
                "error": "retry_not_allowed",
                "message": f"Scan status '{upload.scan_status}' is not eligible for scan retry.",
            },
        )

    scan_result = scan_file(source_path)

    if not scan_result.clean:
        if scan_result.signature:
            upload.scan_status = "infected"
            upload.scan_signature = scan_result.signature
            upload.status = UploadStatus.FAILED.value
            source_path.unlink(missing_ok=True)
            upload.deleted_at = datetime.now(UTC)
        else:
            upload.scan_status = "error"

        await db.commit()
        logger.warning(
            "upload_retry_failed",
            extra={
                "upload_id": str(upload.id),
                "user_id": str(admin.id),
                "action": "retry",
                "target": "scan",
                "scan_status": upload.scan_status,
            },
        )

        if scan_result.error and "unavailable" in scan_result.error.lower():
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "scanner_unavailable",
                    "message": "Malware scanner is temporarily unavailable.",
                },
            )
        raise HTTPException(
            status_code=422,
            detail={
                "error": "scan_failed",
                "message": "Scan retry detected an issue.",
            },
        )

    upload.scan_status = "clean"
    upload.scan_signature = None
    await db.commit()

    logger.info(
        "upload_retry_completed",
        extra={
            "upload_id": str(upload.id),
            "user_id": str(admin.id),
            "action": "retry",
            "target": "scan",
        },
    )

    await db.refresh(upload)
    script = None
    if upload.script_id:
        sr = await db.execute(select(Script).where(Script.id == upload.script_id))
        script = sr.scalar_one_or_none()
    new_status = derive_review_status(upload, script)

    return ReviewRetryResponse(
        upload_id=upload.id,
        script_id=upload.script_id,
        review_status=new_status,
    )


async def _retry_extraction(
    db: AsyncSession, upload: ScriptUpload, admin: User
) -> ReviewRetryResponse:
    """Retry content extraction if quarantine source still exists."""
    import os

    from app.services.upload_extractor import compute_content_hash, extract_content
    from app.services.upload_quarantine import get_quarantine_path

    quarantine_dir = get_quarantine_path()
    source_path = quarantine_dir / upload.storage_key

    if not source_path.exists():
        raise HTTPException(
            status_code=410,
            detail={
                "error": "source_expired",
                "message": "Quarantine source file no longer exists.",
            },
        )

    if upload.scan_status != "clean":
        raise HTTPException(
            status_code=422,
            detail={
                "error": "retry_not_allowed",
                "message": f"Cannot retry extraction: scan status is '{upload.scan_status}'.",
            },
        )

    ext = os.path.splitext(upload.filename_original)[1].lower()
    try:
        content = extract_content(source_path, ext)
    except Exception:
        upload.extraction_status = "failed"
        upload.extraction_error = "Extraction retry failed"
        await db.commit()
        logger.warning(
            "upload_retry_failed",
            extra={
                "upload_id": str(upload.id),
                "user_id": str(admin.id),
                "action": "retry",
                "target": "extraction",
            },
        )
        raise HTTPException(
            status_code=422,
            detail={
                "error": "extraction_failed",
                "message": "Content extraction failed on retry.",
            },
        )

    content_hash = compute_content_hash(content)
    upload.extraction_status = "completed"
    upload.extraction_error = None
    upload.extracted_content = content
    upload.content_hash = content_hash
    upload.status = UploadStatus.COMPLETED.value
    await db.commit()

    logger.info(
        "upload_retry_completed",
        extra={
            "upload_id": str(upload.id),
            "user_id": str(admin.id),
            "action": "retry",
            "target": "extraction",
        },
    )

    await db.refresh(upload)
    new_status = derive_review_status(upload, None)

    return ReviewRetryResponse(
        upload_id=upload.id,
        script_id=upload.script_id,
        review_status=new_status,
    )


# ──────────────────────────────────────────────────────────────────────────────
# POST /uploads/{upload_id}/review/reject
# ──────────────────────────────────────────────────────────────────────────────


@router.post(
    "/uploads/{upload_id}/review/reject",
    response_model=ReviewRejectResponse,
)
async def reject_review(
    upload_id: UUID,
    body: ReviewRejectRequest,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> ReviewRejectResponse:
    """Reject an upload with a required human-readable reason.

    Preserves scan and extraction history. Marks upload as rejected.
    Preserves the linked draft script (unpublished) for auditability.
    """
    # 1. Load and lock
    stmt = select(ScriptUpload).where(ScriptUpload.id == upload_id).with_for_update()
    result = await db.execute(stmt)
    upload = result.scalar_one_or_none()
    if upload is None:
        raise HTTPException(status_code=404, detail="Upload not found")

    # 2. Check current state allows rejection
    script = None
    if upload.script_id:
        script_stmt = select(Script).where(Script.id == upload.script_id).with_for_update()
        sr = await db.execute(script_stmt)
        script = sr.scalar_one_or_none()

    review_status = derive_review_status(upload, script)

    # Published cannot be rejected, already-rejected is idempotent
    if review_status == "published":
        raise HTTPException(
            status_code=422,
            detail={
                "error": "reject_not_allowed",
                "message": "Cannot reject a published upload.",
            },
        )
    if review_status == "rejected":
        # Idempotent: return current state
        return ReviewRejectResponse(
            upload_id=upload.id,
            review_status="rejected",
            rejection_reason=upload.rejection_reason or body.reason,
            rejected_by=upload.rejected_by or admin.id,
            rejected_at=upload.rejected_at or datetime.now(UTC),
        )

    # 3. Mark upload as rejected with metadata. Keep the quarantine file until
    # the database commit succeeds so a rollback cannot lose the source.
    now = datetime.now(UTC)
    upload.status = UploadStatus.REJECTED.value
    upload.rejected_at = now
    upload.rejected_by = admin.id
    upload.rejection_reason = body.reason

    # 5. Preserve linked script as draft (for auditability) — do NOT delete
    # Script remains as an unpublished draft, but can no longer be published
    # via this upload's review workflow since the upload is now rejected.

    # 5. Commit
    try:
        await db.commit()
    except Exception:
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to persist rejection")

    # Clean up the external source only after durable persistence. Cleanup is
    # best-effort and must not turn a successful rejection into an HTTP 500.
    try:
        from app.services.upload_quarantine import get_quarantine_path

        source_path = get_quarantine_path() / upload.storage_key
        source_path.unlink(missing_ok=True)
    except Exception:
        logger.warning(
            "upload_rejection_source_cleanup_failed",
            extra={"upload_id": str(upload_id), "user_id": str(admin.id)},
            exc_info=True,
        )

    # 6. Audit (after successful commit)
    logger.info(
        "upload_rejected",
        extra={
            "upload_id": str(upload_id),
            "script_id": str(upload.script_id) if upload.script_id else None,
            "user_id": str(admin.id),
            "action": "reject",
            "reason_code": body.reason_code or "manual_rejection",
        },
    )

    return ReviewRejectResponse(
        upload_id=upload.id,
        review_status="rejected",
        rejection_reason=body.reason,
        rejected_by=admin.id,
        rejected_at=now,
    )


# ──────────────────────────────────────────────────────────────────────────────
# POST /uploads/{upload_id}/review/publish
# ──────────────────────────────────────────────────────────────────────────────


@router.post(
    "/uploads/{upload_id}/review/publish",
    response_model=ReviewPublishResponse,
)
async def publish_review(
    upload_id: UUID,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> ReviewPublishResponse:
    """Publish the script linked to an upload via the review workflow.

    Delegates to script_registry.publish() — no duplication of publication logic.
    """
    from app.services.script_registry import publish
    from app.services.script_validator import (
        ScriptFormatError,
        ScriptLimitError,
        ScriptValidationError,
    )

    # 1. Load upload
    stmt = select(ScriptUpload).where(ScriptUpload.id == upload_id)
    result = await db.execute(stmt)
    upload = result.scalar_one_or_none()
    if upload is None:
        raise HTTPException(status_code=404, detail="Upload not found")

    # 2. Must have linked script
    if upload.script_id is None:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "no_linked_script",
                "message": "This upload has not been converted to a script.",
            },
        )

    # 3. Pre-publication eligibility (security gates)
    if upload.scan_status != "clean":
        raise HTTPException(
            status_code=422,
            detail={
                "error": "publish_not_allowed",
                "message": f"Cannot publish: scan status is '{upload.scan_status}'.",
            },
        )
    if upload.extraction_status != "completed":
        raise HTTPException(
            status_code=422,
            detail={
                "error": "publish_not_allowed",
                "message": f"Cannot publish: extraction status is '{upload.extraction_status}'.",
            },
        )
    if upload.status == UploadStatus.REJECTED.value:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "publish_not_allowed",
                "message": "Cannot publish a rejected upload.",
            },
        )

    # 4. Load script and verify state
    script_stmt = select(Script).where(
        Script.id == upload.script_id,
        Script.is_deleted == False,  # noqa: E712
    )
    script_result = await db.execute(script_stmt)
    script = script_result.scalar_one_or_none()
    if script is None:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "script_unavailable",
                "message": "Linked script has been deleted.",
            },
        )

    review_status = derive_review_status(upload, script)
    if review_status != "ready_for_review":
        if review_status == "published":
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "already_published",
                    "message": "Script is already published.",
                },
            )
        raise HTTPException(
            status_code=422,
            detail={
                "error": "publish_not_allowed",
                "message": f"Cannot publish in state: {review_status}.",
            },
        )

    # 5. Delegate to registry publish (atomic, creates ScriptVersion)
    try:
        version = await publish(db, script.id, admin_id=admin.id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (ScriptFormatError, ScriptValidationError) as exc:
        errors = getattr(exc, "errors", None)
        detail: dict = {"error": "validation_failed", "message": str(exc)}
        if errors:
            detail["errors"] = errors
        raise HTTPException(status_code=422, detail=detail)
    except ScriptLimitError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "limit_exceeded",
                "message": str(exc),
                "violations": exc.violations,
            },
        )

    # 6. Audit (after successful commit inside publish())
    logger.info(
        "upload_script_published",
        extra={
            "upload_id": str(upload_id),
            "script_id": str(script.id),
            "version_number": version.version_number,
            "user_id": str(admin.id),
            "action": "publish",
        },
    )

    return ReviewPublishResponse(
        upload_id=upload.id,
        script_id=script.id,
        script_status=ScriptStatus.PUBLISHED.value,
        version_number=version.version_number,
        review_status="published",
        published_at=version.published_at,
    )

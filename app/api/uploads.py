"""Training document upload endpoint (admin only).

Extracted text is stored as sanitized pending content. ScriptContract conversion
is handled separately by S1-09.
"""

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID as PyUUID

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_session
from app.models.script_upload import ScriptUpload, UploadStatus
from app.models.user import User
from app.schemas.conversion import ConversionResponse
from app.schemas.upload import (
    UploadListItem,
    UploadRejectionResponse,
    UploadStatusResponse,
    UploadSuccessResponse,
)
from app.services.auth import require_admin
from app.services.script_converter import ConversionError, convert_extracted_to_contract
from app.services.script_registry import create_draft_in_transaction
from app.services.script_validator import (
    ScriptFormatError,
    ScriptLimitError,
    ScriptLimits,
    ScriptValidationError,
    validate_limits,
    validate_conflicts,
    validate_script,
)
from app.services.upload_extractor import compute_content_hash, extract_content
from app.services.upload_quarantine import sanitize_filename, store_in_quarantine
from app.services.upload_rate_limiter import (
    get_retry_after,
    is_rate_limited,
    record_rejection,
)
from app.services.upload_scanner import scan_file
from app.services.upload_validator import (
    UploadRejectionReason,
    validate_docx_archive,
    validate_extension,
    validate_file_signature,
    validate_file_size_streaming,
    validate_mime_type,
    validate_pdf_not_encrypted,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _reject(
    reason: UploadRejectionReason,
    message: str,
    details: dict | None = None,
    *,
    status_code: int = 422,
    admin: User,
    filename: str,
    file_size: int,
    request: Request,
    extra_audit: dict | None = None,
) -> HTTPException:
    """Build rejection response with structured audit logging.

    Args:
        status_code: HTTP status (422 for validation, 503 for service unavailable).
        extra_audit: Additional structured fields merged into the audit log entry.
    """
    audit_fields = {
        "user_id": str(admin.id),
        "upload_filename": filename,
        "file_size": file_size,
        "reason_code": reason.value,
        "ip_address": request.client.host if request.client else "unknown",
    }
    if extra_audit:
        audit_fields.update(extra_audit)

    logger.warning("upload_rejected", extra=audit_fields)
    record_rejection(str(admin.id))
    return HTTPException(
        status_code=status_code,
        detail=UploadRejectionResponse(
            reason_code=reason.value,
            message=message,
            details=details,
        ).model_dump(),
    )


class ScanMetadataPersistenceError(Exception):
    """Raised when scan-failure metadata cannot be persisted to the database."""
    pass


async def _persist_failed_scan(
    db: AsyncSession,
    *,
    upload_id,
    original_filename: str,
    mime_type: str,
    file_size_bytes: int,
    storage_key: str,
    admin_id,
    scan_status: str,
    scan_signature: str | None,
    scenario_id,
    quarantine_expires_at: datetime,
    deleted_at: datetime,
) -> None:
    """Persist a failed-scan upload record with truthful metadata.

    On persistence failure: rolls back, logs ERROR, raises
    ScanMetadataPersistenceError so endpoint returns HTTP 500.
    """
    extraction_error = (
        f"Skipped: malware detected ({scan_signature})"
        if scan_status == "infected"
        else "Skipped: scanner unavailable"
    )
    record = ScriptUpload(
        id=upload_id,
        filename_original=original_filename,
        mime_type=mime_type,
        file_size_bytes=file_size_bytes,
        content_hash=None,
        storage_key=storage_key,
        uploaded_by=admin_id,
        scan_status=scan_status,
        scan_signature=scan_signature,
        extraction_status="failed",
        extraction_error=extraction_error,
        extracted_content=None,
        scenario_id=scenario_id,
        status=UploadStatus.FAILED.value,
        script_id=None,
        quarantine_expires_at=quarantine_expires_at,
        deleted_at=deleted_at,
    )
    try:
        db.add(record)
        await db.commit()
    except Exception:
        await db.rollback()
        logger.error(
            "scan_metadata_persistence_failed",
            exc_info=True,
            extra={
                "upload_id": str(upload_id),
                "user_id": str(admin_id),
                "upload_filename": original_filename,
                "scan_status": scan_status,
                "reason": "database_persistence_failure",
            },
        )
        raise ScanMetadataPersistenceError("Failed to persist scan result")


@router.post("/upload", status_code=201, response_model=UploadSuccessResponse)
async def upload_training_document(
    request: Request,
    file: UploadFile,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
    scenario_id: Optional[PyUUID] = Form(None),
) -> UploadSuccessResponse:
    """Upload a training document for AI debtor script creation."""
    original_filename = sanitize_filename(file.filename or "unnamed")
    user_id_str = str(admin.id)

    # 1. Rate limit check
    if is_rate_limited(user_id_str):
        retry_after = get_retry_after(user_id_str)
        raise HTTPException(
            status_code=429,
            detail=UploadRejectionResponse(
                reason_code=UploadRejectionReason.RATE_LIMITED.value,
                message="Too many rejected uploads. Please wait before trying again.",
                details={"retry_after_seconds": retry_after},
            ).model_dump(),
            headers={"Retry-After": str(retry_after)},
        )

    # 2. Validate scenario_id
    if scenario_id:
        from app.models import Scenario
        scenario_result = await db.execute(
            select(Scenario).where(Scenario.id == scenario_id)
        )
        if scenario_result.scalar_one_or_none() is None:
            raise HTTPException(
                status_code=422,
                detail={"error": "invalid_scenario",
                        "message": f"Scenario with id '{scenario_id}' does not exist."},
            )

    # 3. Validate extension
    valid, reason = validate_extension(original_filename)
    if not valid:
        raise _reject(reason, "File extension not allowed. Accepted: .pdf, .docx, .txt, .csv, .md",
                      admin=admin, filename=original_filename, file_size=0, request=request)

    # 4. Validate MIME type
    content_type = file.content_type or "application/octet-stream"
    valid, reason = validate_mime_type(original_filename, content_type)
    if not valid:
        raise _reject(reason, f"MIME type '{content_type}' does not match expected type.",
                      admin=admin, filename=original_filename, file_size=0, request=request)

    # 5. Streaming size validation
    file_bytes, reason = await validate_file_size_streaming(file, settings.upload_max_file_size)
    if reason is not None:
        raise _reject(reason, f"File exceeds maximum size of {settings.upload_max_file_size} bytes.",
                      details={"limit": settings.upload_max_file_size, "actual": len(file_bytes)},
                      admin=admin, filename=original_filename, file_size=len(file_bytes), request=request)

    # 6. Validate binary signature
    header_bytes = file_bytes[:8] if len(file_bytes) >= 8 else file_bytes
    valid, reason = validate_file_signature(original_filename, header_bytes)
    if not valid:
        raise _reject(reason, "File content does not match the declared file type.",
                      admin=admin, filename=original_filename, file_size=len(file_bytes), request=request)

    # 7. PDF encryption check
    ext = os.path.splitext(original_filename)[1].lower()
    if ext == ".pdf":
        valid, reason = validate_pdf_not_encrypted(file_bytes)
        if not valid:
            raise _reject(reason, "Encrypted or password-protected PDFs are not accepted.",
                          admin=admin, filename=original_filename, file_size=len(file_bytes), request=request)

    # 8. Store in quarantine
    quarantine_path = store_in_quarantine(file_bytes, ext)
    storage_key = quarantine_path.name  # Capture before potential deletion

    try:
        # 9. DOCX archive safety check
        if ext == ".docx":
            valid, reason = validate_docx_archive(quarantine_path)
            if not valid:
                quarantine_path.unlink(missing_ok=True)
                raise _reject(reason, "DOCX file failed archive safety check.",
                              admin=admin, filename=original_filename,
                              file_size=len(file_bytes), request=request)

        # 10. Malware scan
        scan_result = scan_file(quarantine_path)
        if not scan_result.clean:
            # Delete quarantined file immediately — infected files must not be retained
            quarantine_path.unlink(missing_ok=True)
            deleted_at = datetime.now(timezone.utc)

            # Determine scan outcome
            scan_status = "infected" if scan_result.signature else "error"
            quarantine_expires = datetime.now(timezone.utc) + timedelta(
                hours=settings.upload_quarantine_retention_hours
            )

            # Determine rejection parameters
            if scan_result.error:
                reason_code = UploadRejectionReason.SCANNER_UNAVAILABLE
                http_status = 503
                message = "Malware scanner is unavailable. Upload rejected (fail-closed)."
                details = {"scan_error": scan_result.error}
                extra_audit = {"scan_status": "error"}
            else:
                reason_code = UploadRejectionReason.MALWARE_DETECTED
                http_status = 422
                message = f"Malware detected: {scan_result.signature}"
                details = {"scan_signature": scan_result.signature}
                extra_audit = {"scan_status": "infected", "scan_signature": scan_result.signature}

            # Emit rejection audit BEFORE persistence (ensures audit even if DB fails)
            audit_fields = {
                "user_id": str(admin.id),
                "upload_filename": original_filename,
                "file_size": len(file_bytes),
                "reason_code": reason_code.value,
                "ip_address": request.client.host if request.client else "unknown",
            }
            audit_fields.update(extra_audit)
            logger.warning("upload_rejected", extra=audit_fields)
            record_rejection(str(admin.id))

            # Persist truthful failed-scan metadata (may raise ScanMetadataPersistenceError)
            failed_id = uuid.uuid4()
            await _persist_failed_scan(
                db,
                upload_id=failed_id,
                original_filename=original_filename,
                mime_type=content_type,
                file_size_bytes=len(file_bytes),
                storage_key=storage_key,
                admin_id=admin.id,
                scan_status=scan_status,
                scan_signature=scan_result.signature,
                scenario_id=scenario_id,
                quarantine_expires_at=quarantine_expires,
                deleted_at=deleted_at,
            )

            # Return the rejection response
            raise HTTPException(
                status_code=http_status,
                detail=UploadRejectionResponse(
                    reason_code=reason_code.value,
                    message=message,
                    details=details,
                ).model_dump(),
            )

        # 11. Content security validation (before extraction)
        if ext == ".pdf":
            from app.services.pdf_security import validate_pdf_security
            valid, reason = validate_pdf_security(file_bytes)
            if not valid:
                quarantine_path.unlink(missing_ok=True)
                raise _reject(reason, f"PDF rejected: {reason.value}",
                              admin=admin, filename=original_filename,
                              file_size=len(file_bytes), request=request)

        if ext == ".docx":
            from app.services.docx_security import validate_docx_security
            valid, reason = validate_docx_security(quarantine_path)
            if not valid:
                quarantine_path.unlink(missing_ok=True)
                raise _reject(reason, f"DOCX rejected: {reason.value}",
                              admin=admin, filename=original_filename,
                              file_size=len(file_bytes), request=request)

        # 12. Content extraction (only reached for validated clean files)
        from app.services.upload_extractor import ExtractionError
        try:
            content = extract_content(quarantine_path, ext)
        except ExtractionError:
            quarantine_path.unlink(missing_ok=True)
            raise _reject(
                UploadRejectionReason.EXTRACTION_FAILED,
                "Document content could not be safely extracted.",
                admin=admin, filename=original_filename,
                file_size=len(file_bytes), request=request,
            )
        except Exception:
            quarantine_path.unlink(missing_ok=True)
            raise _reject(
                UploadRejectionReason.EXTRACTION_FAILED,
                "Document content could not be safely extracted.",
                admin=admin, filename=original_filename,
                file_size=len(file_bytes), request=request,
            )
        content_hash = compute_content_hash(content)

    except HTTPException:
        raise
    except ScanMetadataPersistenceError:
        # File already deleted, audit log already emitted by _reject before _persist
        raise HTTPException(status_code=500, detail="Failed to persist scan result")
    except Exception as e:
        quarantine_path.unlink(missing_ok=True)
        logger.error("Unexpected error during upload processing: %s", e)
        raise HTTPException(status_code=500, detail="Internal processing error")

    # 12. Persist successful upload metadata
    upload_id = uuid.uuid4()
    quarantine_expires = datetime.now(timezone.utc) + timedelta(
        hours=settings.upload_quarantine_retention_hours
    )
    upload_record = ScriptUpload(
        id=upload_id,
        filename_original=original_filename,
        mime_type=content_type,
        file_size_bytes=len(file_bytes),
        content_hash=content_hash,
        storage_key=storage_key,
        uploaded_by=admin.id,
        scan_status="clean",
        scan_signature=None,
        extraction_status="completed",
        extraction_error=None,
        extracted_content=content,
        scenario_id=scenario_id,
        status=UploadStatus.COMPLETED.value,
        script_id=None,
        quarantine_expires_at=quarantine_expires,
    )

    try:
        db.add(upload_record)
        await db.commit()
        await db.refresh(upload_record)
    except Exception as e:
        await db.rollback()
        logger.error("Database error persisting upload record: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to persist upload record")

    # 13. Log success ONLY after successful database commit
    logger.info(
        "upload_success",
        extra={
            "user_id": str(admin.id),
            "upload_filename": original_filename,
            "file_size": len(file_bytes),
            "content_hash": content_hash,
            "scan_status": "clean",
            "ip_address": request.client.host if request.client else "unknown",
        },
    )

    return UploadSuccessResponse(
        id=upload_record.id,
        filename_original=original_filename,
        mime_type=content_type,
        file_size_bytes=len(file_bytes),
        content_hash=content_hash,
        storage_key=storage_key,
        scan_result="clean",
        extraction_status="completed",
        status=UploadStatus.COMPLETED.value,
        quarantine_expires_at=quarantine_expires,
        created_at=upload_record.created_at or datetime.now(timezone.utc),
        script_id=None,
        scenario_id=scenario_id,
        processing_notes="Content extracted and stored as pending. ScriptContract conversion available via S1-09.",
    )


@router.get("/uploads/{upload_id}/status", response_model=UploadStatusResponse)
async def get_upload_status(
    upload_id: PyUUID,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> UploadStatusResponse:
    """Get the processing status of an upload (admin only)."""
    stmt = select(ScriptUpload).where(ScriptUpload.id == upload_id)
    result = await db.execute(stmt)
    upload = result.scalar_one_or_none()
    if upload is None:
        raise HTTPException(status_code=404, detail="Upload not found")
    return UploadStatusResponse(
        id=upload.id,
        filename_original=upload.filename_original,
        mime_type=upload.mime_type,
        file_size_bytes=upload.file_size_bytes,
        content_hash=upload.content_hash,
        storage_key=upload.storage_key,
        uploaded_by=upload.uploaded_by,
        scan_status=upload.scan_status,
        scan_signature=upload.scan_signature,
        extraction_status=upload.extraction_status,
        extraction_error=upload.extraction_error,
        status=upload.status,
        script_id=upload.script_id,
        scenario_id=upload.scenario_id,
        created_at=upload.created_at,
        updated_at=upload.updated_at,
        quarantine_expires_at=upload.quarantine_expires_at,
        deleted_at=upload.deleted_at,
    )


@router.get("/uploads", response_model=list[UploadListItem])
async def list_uploads(
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[UploadListItem]:
    """List recent uploads (admin only)."""
    stmt = (
        select(ScriptUpload)
        .order_by(ScriptUpload.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(stmt)
    uploads = result.scalars().all()
    return [
        UploadListItem(
            id=u.id, filename_original=u.filename_original, mime_type=u.mime_type,
            file_size_bytes=u.file_size_bytes, status=u.status,
            script_id=u.script_id, scenario_id=u.scenario_id, created_at=u.created_at,
        )
        for u in uploads
    ]


@router.post(
    "/uploads/{upload_id}/convert",
    response_model=ConversionResponse,
    status_code=201,
)
async def convert_upload_to_script(
    upload_id: PyUUID,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> ConversionResponse:
    """Convert a completed, clean ScriptUpload into a Script draft (S1-09).

    Uses a single atomic transaction with row-level locking:
    1. SELECT ... FOR UPDATE on the ScriptUpload row to prevent concurrent duplicates.
    2. Validate eligibility (scan clean, extraction completed, overall completed,
       extracted_content present, scenario_id valid).
    3. Convert extracted_content into a ScriptContract.
    4. Run full configurable limit validation (publication-readiness check).
    5. Create the Script draft via flush (not commit).
    6. Assign upload.script_id.
    7. Commit once (both Script creation and upload linkage atomically).

    If any step fails, rollback removes both the draft and linkage.
    Converted contracts are always persisted as normalized JSON.
    """
    import json as json_mod

    # 1. Load and lock upload row (SELECT ... FOR UPDATE)
    stmt = (
        select(ScriptUpload)
        .where(ScriptUpload.id == upload_id)
        .with_for_update()
    )
    result = await db.execute(stmt)
    upload = result.scalar_one_or_none()

    if upload is None:
        raise HTTPException(status_code=404, detail="Upload not found")

    # 2. Duplicate conversion check (under lock — no race condition)
    if upload.script_id is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "already_converted",
                "message": "This upload has already been converted to a script.",
                "script_id": str(upload.script_id),
            },
        )

    # 3. Eligibility checks
    if upload.status == UploadStatus.DELETED.value:
        raise HTTPException(
            status_code=422,
            detail={"error": "upload_ineligible", "message": "Upload has been deleted."},
        )

    if upload.scan_status == "infected":
        raise HTTPException(
            status_code=422,
            detail={"error": "upload_ineligible", "message": "Upload is infected and cannot be converted."},
        )

    if upload.scan_status == "error":
        raise HTTPException(
            status_code=422,
            detail={"error": "upload_ineligible", "message": "Upload scan failed and cannot be converted."},
        )

    if upload.scan_status == "pending":
        raise HTTPException(
            status_code=422,
            detail={"error": "upload_ineligible", "message": "Upload scan is still pending."},
        )

    if upload.scan_status != "clean":
        raise HTTPException(
            status_code=422,
            detail={"error": "upload_ineligible", "message": f"Upload scan status '{upload.scan_status}' is not eligible for conversion."},
        )

    if upload.extraction_status != "completed":
        raise HTTPException(
            status_code=422,
            detail={"error": "upload_ineligible", "message": f"Extraction status '{upload.extraction_status}' is not eligible for conversion."},
        )

    if upload.status != UploadStatus.COMPLETED.value:
        raise HTTPException(
            status_code=422,
            detail={"error": "upload_ineligible", "message": f"Upload status '{upload.status}' is not eligible for conversion. Must be 'completed'."},
        )

    if not upload.extracted_content or not upload.extracted_content.strip():
        raise HTTPException(
            status_code=422,
            detail={"error": "upload_ineligible", "message": "Upload has no extracted content available."},
        )

    if upload.scenario_id is None:
        raise HTTPException(
            status_code=422,
            detail={"error": "upload_ineligible", "message": "Upload has no scenario_id. A valid scenario is required for conversion."},
        )

    # Validate scenario still exists
    from app.models import Scenario
    scenario_result = await db.execute(
        select(Scenario).where(Scenario.id == upload.scenario_id)
    )
    if scenario_result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=422,
            detail={"error": "upload_ineligible", "message": f"Scenario '{upload.scenario_id}' does not exist."},
        )

    # Check if scenario already has a Script (unique constraint on Script.scenario_id)
    from app.models.script import Script
    existing_script_result = await db.execute(
        select(Script).where(
            Script.scenario_id == upload.scenario_id,
            Script.is_deleted == False,  # noqa: E712
        )
    )
    existing_script = existing_script_result.scalar_one_or_none()
    if existing_script is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "scenario_has_script",
                "message": f"Scenario '{upload.scenario_id}' already has a script.",
                "existing_script_id": str(existing_script.id),
            },
        )

    # 4. Convert extracted_content into ScriptContract
    try:
        contract_data = convert_extracted_to_contract(upload.extracted_content)
    except ConversionError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "conversion_failed",
                "message": str(exc),
                "details": exc.details,
            },
        )

    # 5. Full configurable limit validation (publication-readiness)
    raw_definition = json_mod.dumps(contract_data)
    limits = ScriptLimits(
        max_definition_size_bytes=settings.script_max_definition_size_bytes,
        max_trigger_phrases=settings.script_max_trigger_phrases,
        max_expected_replies=settings.script_max_expected_replies,
        max_escalation_conditions=settings.script_max_escalation_conditions,
        max_field_text_length=settings.script_max_field_text_length,
    )

    # Run the full validation pipeline (parse + structure + conflicts + limits)
    try:
        validate_script(raw_definition, "json", limits)
    except ScriptFormatError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "conversion_failed",
                "message": f"Converted contract failed format validation: {exc}",
            },
        )
    except ScriptValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "conversion_failed",
                "message": f"Converted contract does not meet publication requirements ({len(exc.errors)} issue(s)).",
                "details": {"validation_errors": exc.errors},
            },
        )

    # 6. Create draft via flush (no commit yet) — atomic with linkage
    try:
        script = await create_draft_in_transaction(
            db,
            admin_id=admin.id,
            name=f"Converted from upload {upload.filename_original}",
            scenario_id=upload.scenario_id,
            format="json",
            raw_definition=raw_definition,
        )
    except (ScriptFormatError, ScriptValidationError) as exc:
        await db.rollback()
        raise HTTPException(
            status_code=422,
            detail={
                "error": "conversion_failed",
                "message": f"Script registry validation failed: {exc}",
                "details": getattr(exc, "errors", None) or getattr(exc, "violations", None),
            },
        )
    except Exception as exc:
        await db.rollback()
        logger.error(
            "conversion_registry_failure",
            extra={
                "upload_id": str(upload_id),
                "user_id": str(admin.id),
                "error": str(exc),
            },
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to create script draft. No changes were made.",
        )

    # 7. Link upload to created script (same transaction)
    upload.script_id = script.id

    # 8. Single atomic commit — both Script and linkage
    try:
        await db.commit()
    except Exception as exc:
        await db.rollback()
        logger.error(
            "conversion_commit_failure",
            extra={
                "upload_id": str(upload_id),
                "user_id": str(admin.id),
                "error": str(exc),
            },
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to commit conversion. No changes were made.",
        )

    # 9. Audit log — only after successful commit
    logger.info(
        "upload_converted",
        extra={
            "upload_id": str(upload_id),
            "script_id": str(script.id),
            "scenario_id": str(upload.scenario_id),
            "user_id": str(admin.id),
            "format": "json",
        },
    )

    return ConversionResponse(
        upload_id=upload.id,
        script_id=script.id,
        scenario_id=upload.scenario_id,
        status="converted",
        format="json",
    )

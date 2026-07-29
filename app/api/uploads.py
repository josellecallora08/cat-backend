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

    Delegates to the conversion service which handles:
    - SELECT ... FOR UPDATE row locking
    - Eligibility and scenario checks
    - Content conversion and full limit validation
    - Transaction-aware draft creation (flush)
    - Upload linkage and single atomic commit
    - IntegrityError classification for scenario uniqueness races

    Converted contracts are always persisted as normalized JSON.
    """
    from app.services.conversion_service import (
        ConversionConflict,
        ConversionInternalError,
        ConversionRejection,
        ConversionSuccess,
        convert_upload_to_script_draft,
    )

    try:
        result = await convert_upload_to_script_draft(db, upload_id, admin.id)
    except ConversionInternalError:
        raise HTTPException(
            status_code=500,
            detail="Failed to complete conversion. No changes were made.",
        )

    if isinstance(result, ConversionSuccess):
        return ConversionResponse(
            upload_id=result.upload_id,
            script_id=result.script_id,
            scenario_id=result.scenario_id,
            status="converted",
            format="json",
        )

    if isinstance(result, ConversionConflict):
        detail = {
            "error": result.error,
            "message": result.message,
        }
        if result.existing_script_id is not None:
            detail["script_id" if result.error == "already_converted" else "existing_script_id"] = str(result.existing_script_id)
        raise HTTPException(status_code=409, detail=detail)

    if isinstance(result, ConversionRejection):
        if result.error == "not_found":
            raise HTTPException(status_code=404, detail="Upload not found")
        detail: dict = {"error": result.error, "message": result.message}
        if result.details:
            detail["details"] = result.details
        raise HTTPException(status_code=422, detail=detail)

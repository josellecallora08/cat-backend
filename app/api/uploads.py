"""Training document upload endpoint (admin only).

Administrators upload documents that define how the AI debtor should respond
during training calls. This endpoint enforces the complete validation pipeline:
rate limiting → extension → MIME → streaming size → binary signature →
quarantine → DOCX archive check → malware scan → content extraction → hash.

Extracted text is stored as sanitized pending content. ScriptContract conversion
is handled separately by S1-09; this endpoint does NOT attempt to parse
extracted document text as JSON/YAML ScriptContract data.
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
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _reject(
    reason: UploadRejectionReason,
    message: str,
    details: dict | None = None,
    *,
    admin: User,
    filename: str,
    file_size: int,
    request: Request,
) -> HTTPException:
    """Build rejection response and log the event."""
    logger.warning(
        "upload_rejected",
        extra={
            "user_id": str(admin.id),
            "upload_filename": filename,
            "file_size": file_size,
            "reason_code": reason.value,
            "ip_address": request.client.host if request.client else "unknown",
        },
    )
    record_rejection(str(admin.id))
    return HTTPException(
        status_code=422,
        detail=UploadRejectionResponse(
            reason_code=reason.value,
            message=message,
            details=details,
        ).model_dump(),
    )


@router.post("/upload", status_code=201, response_model=UploadSuccessResponse)
async def upload_training_document(
    request: Request,
    file: UploadFile,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
    scenario_id: Optional[PyUUID] = Form(None),
) -> UploadSuccessResponse:
    """Upload a training document for AI debtor script creation.

    Only administrators can access this endpoint. The file goes through
    a multi-stage validation pipeline before being accepted.

    Optionally accepts a scenario_id to link the upload to a scenario.
    Extracted text is stored as pending content for future ScriptContract
    conversion (S1-09). This endpoint does NOT parse extracted text as
    JSON/YAML.
    """
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

    # 2. Validate scenario_id exists (if provided)
    if scenario_id:
        from app.models import Scenario
        scenario_stmt = select(Scenario).where(Scenario.id == scenario_id)
        scenario_result = await db.execute(scenario_stmt)
        scenario = scenario_result.scalar_one_or_none()
        if scenario is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_scenario",
                    "message": f"Scenario with id '{scenario_id}' does not exist.",
                },
            )

    # 3. Validate extension
    valid, reason = validate_extension(original_filename)
    if not valid:
        raise _reject(
            reason,
            "File extension not allowed. Accepted: .pdf, .docx, .txt, .csv, .md",
            admin=admin,
            filename=original_filename,
            file_size=0,
            request=request,
        )

    # 4. Validate MIME type
    content_type = file.content_type or "application/octet-stream"
    valid, reason = validate_mime_type(original_filename, content_type)
    if not valid:
        raise _reject(
            reason,
            f"MIME type '{content_type}' does not match expected type for this file extension.",
            admin=admin,
            filename=original_filename,
            file_size=0,
            request=request,
        )

    # 5. Streaming size validation
    file_bytes, reason = await validate_file_size_streaming(
        file, settings.upload_max_file_size
    )
    if reason is not None:
        raise _reject(
            reason,
            f"File exceeds maximum size of {settings.upload_max_file_size} bytes.",
            details={
                "limit": settings.upload_max_file_size,
                "actual": len(file_bytes),
            },
            admin=admin,
            filename=original_filename,
            file_size=len(file_bytes),
            request=request,
        )

    # 6. Validate binary signature
    header_bytes = file_bytes[:8] if len(file_bytes) >= 8 else file_bytes
    valid, reason = validate_file_signature(original_filename, header_bytes)
    if not valid:
        raise _reject(
            reason,
            "File content does not match the declared file type (binary signature mismatch).",
            admin=admin,
            filename=original_filename,
            file_size=len(file_bytes),
            request=request,
        )

    # 7. Store in quarantine
    ext = os.path.splitext(original_filename)[1].lower()
    quarantine_path = store_in_quarantine(file_bytes, ext)

    try:
        # 8. DOCX archive safety check
        if ext == ".docx":
            valid, reason = validate_docx_archive(quarantine_path)
            if not valid:
                quarantine_path.unlink(missing_ok=True)
                raise _reject(
                    reason,
                    "DOCX file failed archive safety check (possible zip bomb).",
                    admin=admin,
                    filename=original_filename,
                    file_size=len(file_bytes),
                    request=request,
                )

        # 9. Malware scan
        scan_result = scan_file(quarantine_path)
        if not scan_result.clean:
            quarantine_path.unlink(missing_ok=True)
            if scan_result.error:
                raise _reject(
                    UploadRejectionReason.SCANNER_UNAVAILABLE,
                    "Malware scanner is unavailable. Upload rejected (fail-closed).",
                    admin=admin,
                    filename=original_filename,
                    file_size=len(file_bytes),
                    request=request,
                )
            raise _reject(
                UploadRejectionReason.MALWARE_DETECTED,
                f"Malware detected: {scan_result.signature}",
                admin=admin,
                filename=original_filename,
                file_size=len(file_bytes),
                request=request,
            )

        # 10. Content extraction
        content = extract_content(quarantine_path, ext)
        content_hash = compute_content_hash(content)

    except HTTPException:
        raise
    except Exception as e:
        quarantine_path.unlink(missing_ok=True)
        logger.error("Unexpected error during upload processing: %s", e)
        raise HTTPException(status_code=500, detail="Internal processing error")

    # 11. Log success
    logger.info(
        "upload_success",
        extra={
            "user_id": str(admin.id),
            "upload_filename": original_filename,
            "file_size": len(file_bytes),
            "content_hash": content_hash,
            "ip_address": request.client.host if request.client else "unknown",
        },
    )

    # 12. Persist upload metadata to database
    # Extracted content is stored as pending text. ScriptContract conversion
    # is a separate step (S1-09). We do NOT attempt JSON/YAML parsing here.
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
        storage_key=quarantine_path.name,
        uploaded_by=admin.id,
        scan_status="clean",
        scan_signature=None,
        extraction_status="completed",
        extraction_error=None,
        extracted_content=content,
        scenario_id=scenario_id,
        status=UploadStatus.COMPLETED.value,
        script_id=None,  # Not linked until S1-09 ScriptContract conversion
        quarantine_expires_at=quarantine_expires,
    )

    try:
        db.add(upload_record)
        await db.commit()
        await db.refresh(upload_record)
    except Exception as e:
        await db.rollback()
        logger.error("Database error persisting upload record: %s", e)
        raise HTTPException(status_code=500, detail="Failed to persist upload record")

    return UploadSuccessResponse(
        id=upload_record.id,
        filename_original=original_filename,
        mime_type=content_type,
        file_size_bytes=len(file_bytes),
        content_hash=content_hash,
        storage_key=quarantine_path.name,
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
            id=u.id,
            filename_original=u.filename_original,
            mime_type=u.mime_type,
            file_size_bytes=u.file_size_bytes,
            status=u.status,
            script_id=u.script_id,
            scenario_id=u.scenario_id,
            created_at=u.created_at,
        )
        for u in uploads
    ]

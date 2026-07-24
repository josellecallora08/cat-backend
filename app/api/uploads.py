"""Training document upload endpoint (admin only).

Administrators upload documents that define how the AI debtor should respond
during training calls. This endpoint enforces the complete validation pipeline:
rate limiting → extension → MIME → streaming size → binary signature →
quarantine → DOCX archive check → malware scan → content extraction → hash.
"""

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_session
from app.models.user import User
from app.schemas.upload import UploadRejectionResponse, UploadSuccessResponse
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
) -> UploadSuccessResponse:
    """Upload a training document for AI debtor script creation.

    Only administrators can access this endpoint. The file goes through
    a multi-stage validation pipeline before being accepted.
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

    # 2. Validate extension
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

    # 3. Validate MIME type
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

    # 4. Streaming size validation
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

    # 5. Validate binary signature
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

    # 6. Store in quarantine
    ext = os.path.splitext(original_filename)[1].lower()
    quarantine_path = store_in_quarantine(file_bytes, ext)

    try:
        # 7. DOCX archive safety check
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

        # 8. Malware scan
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

        # 9. Content extraction
        content = extract_content(quarantine_path, ext)
        content_hash = compute_content_hash(content)

    except HTTPException:
        raise
    except Exception as e:
        quarantine_path.unlink(missing_ok=True)
        logger.error("Unexpected error during upload processing: %s", e)
        raise HTTPException(status_code=500, detail="Internal processing error")

    # 10. Log success
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

    # 11. Build response
    upload_id = uuid.uuid4()
    quarantine_expires = datetime.now(timezone.utc) + timedelta(
        hours=settings.upload_quarantine_retention_hours
    )

    return UploadSuccessResponse(
        id=upload_id,
        filename_original=original_filename,
        content_hash=content_hash,
        extracted_size_bytes=len(content.encode("utf-8")),
        scan_result="clean",
        quarantine_expires_at=quarantine_expires,
    )

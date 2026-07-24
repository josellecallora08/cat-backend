"""Upload validation response schemas."""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel


class UploadRejectionResponse(BaseModel):
    """Response returned when a file upload is rejected."""

    error: str = "upload_rejected"
    reason_code: str
    message: str
    details: Optional[dict[str, Any]] = None


class UploadValidationResult(BaseModel):
    """Internal validation result for upload checks."""

    valid: bool
    reason_code: Optional[str] = None
    message: Optional[str] = None


class UploadSuccessResponse(BaseModel):
    """Response returned when a file upload succeeds."""

    id: UUID
    filename_original: str
    content_hash: str
    extracted_size_bytes: int
    scan_result: str  # "clean"
    quarantine_expires_at: datetime

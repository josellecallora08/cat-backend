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
    mime_type: str
    file_size_bytes: int
    content_hash: str
    storage_key: str
    scan_result: str
    extraction_status: str
    status: str
    quarantine_expires_at: datetime
    created_at: datetime
    script_id: Optional[UUID] = None
    scenario_id: Optional[UUID] = None
    processing_notes: Optional[str] = None


class UploadStatusResponse(BaseModel):
    """Response for GET /uploads/{id}/status."""

    id: UUID
    filename_original: str
    mime_type: str
    file_size_bytes: int
    content_hash: str
    storage_key: str
    uploaded_by: UUID
    scan_status: str
    scan_signature: Optional[str] = None
    extraction_status: str
    extraction_error: Optional[str] = None
    status: str
    script_id: Optional[UUID] = None
    scenario_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime
    quarantine_expires_at: datetime
    deleted_at: Optional[datetime] = None


class UploadListItem(BaseModel):
    """Summary item for the uploads list."""

    id: UUID
    filename_original: str
    mime_type: str
    file_size_bytes: int
    status: str
    script_id: Optional[UUID] = None
    scenario_id: Optional[UUID] = None
    created_at: datetime

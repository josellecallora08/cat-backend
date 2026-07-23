"""Upload validation response schemas."""

from typing import Any, Optional

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

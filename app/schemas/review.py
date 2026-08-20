"""Admin review API schemas for S1-10.

Defines request/response models for the upload review lifecycle:
GET review detail, PATCH edit, POST retry, POST reject, POST publish.

Review status is DERIVED from persisted upload + script state, never
stored as a separate column. State transitions are documented inline.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ReviewWarning(BaseModel):
    """A structured warning surfaced during review.

    Codes are stable, machine-readable identifiers. Severity follows
    standard levels: info, warning, error.
    """

    code: str
    message: str
    field: str | None = None
    severity: str = "warning"  # info | warning | error


class ReviewScanResult(BaseModel):
    """Sanitized scan result for admin review (never exposes internals)."""

    status: str  # clean | infected | error | pending
    signature: str | None = None
    error: str | None = None  # safe, summarized error message


class ReviewDetailResponse(BaseModel):
    """Full admin review detail for an upload.

    Never returns:
    - storage keys / filesystem paths
    - scanner connection details
    - stack traces or internal exception strings
    - database errors
    """

    # Upload metadata
    upload_id: UUID
    filename_original: str
    mime_type: str
    file_size_bytes: int
    scenario_id: UUID | None = None
    uploaded_by: UUID

    # Content (omitted for infected/unsafe uploads)
    sanitized_content: str | None = None
    script_contract: dict[str, Any] | None = None
    contract_format: str | None = None

    # Warnings
    warnings: list[ReviewWarning] = Field(default_factory=list)

    # Scan
    scan_result: ReviewScanResult

    # Statuses
    upload_status: str
    scan_status: str
    extraction_status: str
    script_id: UUID | None = None
    script_status: str | None = None
    review_status: str

    # Actions
    can_edit: bool = False
    can_retry: bool = False
    can_reject: bool = False
    can_publish: bool = False

    # Rejection metadata
    rejection_reason: str | None = None
    rejected_by: UUID | None = None
    rejected_at: datetime | None = None

    # Timestamps
    created_at: datetime
    updated_at: datetime | None = None
    published_at: datetime | None = None


class ReviewEditRequest(BaseModel):
    """Request body for editing the script contract via review.

    Accepts a complete ScriptContract replacement. Partial patching is
    not supported (no existing partial-patch semantics in the project).
    """

    model_config = {"extra": "forbid"}

    script_contract: dict[str, Any]
    expected_updated_at: datetime | None = None  # Optimistic concurrency


class ReviewEditResponse(BaseModel):
    """Response after a successful edit."""

    upload_id: UUID
    script_id: UUID
    script_contract: dict[str, Any]
    review_status: str
    updated_at: datetime
    warnings: list[ReviewWarning] = Field(default_factory=list)
    can_publish: bool = True


class ReviewRetryRequest(BaseModel):
    """Request body for retry action."""

    model_config = {"extra": "forbid"}

    target: str = "conversion"  # conversion | scan | extraction


class ReviewRetryResponse(BaseModel):
    """Response after a successful retry."""

    upload_id: UUID
    script_id: UUID | None = None
    review_status: str
    warnings: list[ReviewWarning] = Field(default_factory=list)


class ReviewRejectRequest(BaseModel):
    """Request body for rejecting an upload.

    Reason is required and must not be empty or whitespace-only.
    """

    model_config = {"extra": "forbid"}

    reason: str = Field(min_length=1, max_length=2000)
    reason_code: str | None = Field(default=None, max_length=100)

    @field_validator("reason")
    @classmethod
    def reason_not_whitespace(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Rejection reason must not be empty or whitespace-only.")
        return stripped


class ReviewRejectResponse(BaseModel):
    """Response after rejecting an upload."""

    upload_id: UUID
    review_status: str
    rejection_reason: str
    rejected_by: UUID
    rejected_at: datetime


class ReviewPublishResponse(BaseModel):
    """Response after publishing via review."""

    upload_id: UUID
    script_id: UUID
    script_status: str
    version_number: int
    review_status: str
    published_at: datetime

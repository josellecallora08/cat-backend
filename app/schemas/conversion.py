"""Schemas for S1-09 upload-to-script conversion responses."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ConversionResponse(BaseModel):
    """Response returned when an upload is successfully converted to a script draft.

    Converted contracts are always persisted as normalized JSON regardless of
    the original extracted_content format.
    """

    upload_id: UUID
    script_id: UUID
    scenario_id: UUID
    status: str = "converted"
    format: str
    review_warnings: list[str] = Field(default_factory=list)


class ConversionErrorResponse(BaseModel):
    """Response returned when conversion fails."""

    error: str = "conversion_failed"
    message: str
    details: dict[str, Any] | None = None

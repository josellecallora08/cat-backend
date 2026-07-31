"""Pydantic schemas for Profile API request/response models."""

from pydantic import BaseModel, Field


class ProfileResponse(BaseModel):
    """User profile data."""

    model_config = {"extra": "forbid"}

    id: str
    email: str
    full_name: str
    department: str | None
    avatar_url: str | None


class ProfileUpdateRequest(BaseModel):
    """Request payload for updating profile fields."""

    model_config = {"extra": "forbid"}

    full_name: str | None = Field(default=None, min_length=1, max_length=255)
    department: str | None = Field(default=None, max_length=255)


class PasswordChangeRequest(BaseModel):
    """Request payload for changing password."""

    model_config = {"extra": "forbid"}

    current_password: str = Field(min_length=1)
    new_password: str = Field(min_length=8, max_length=128)

"""Pydantic schemas for Admin User Management API request/response models."""

import re
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


VALID_ROLES = {"admin", "user"}
VALID_USER_TYPES = {"trainer", "agent"}
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


# --- Request Schemas ---


class AdminUserCreate(BaseModel):
    """Payload for creating a new user via admin."""

    model_config = {"extra": "forbid"}

    email: str = Field(max_length=255)
    full_name: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=8)
    role: str
    user_type: str | None = None

    @model_validator(mode="after")
    def validate_role_and_user_type(self) -> "AdminUserCreate":
        """Validate role/user_type consistency and email format."""
        if not EMAIL_REGEX.match(self.email):
            raise ValueError("Invalid email format")

        if self.role not in VALID_ROLES:
            raise ValueError(f"role must be one of: {', '.join(sorted(VALID_ROLES))}")

        if self.role == "user":
            if self.user_type is None or self.user_type not in VALID_USER_TYPES:
                raise ValueError(
                    "user_type is required when role is 'user' "
                    f"and must be one of: {', '.join(sorted(VALID_USER_TYPES))}"
                )
        elif self.role == "admin":
            if self.user_type is not None:
                self.user_type = None

        return self


class AdminUserUpdate(BaseModel):
    """Payload for updating a user via admin."""

    model_config = {"extra": "forbid"}

    full_name: str = Field(min_length=1, max_length=255)
    role: str
    user_type: str | None = None

    @model_validator(mode="after")
    def validate_role_and_user_type(self) -> "AdminUserUpdate":
        """Validate role/user_type consistency."""
        if self.role not in VALID_ROLES:
            raise ValueError(f"role must be one of: {', '.join(sorted(VALID_ROLES))}")

        if self.role == "user":
            if self.user_type is None or self.user_type not in VALID_USER_TYPES:
                raise ValueError(
                    "user_type is required when role is 'user' "
                    f"and must be one of: {', '.join(sorted(VALID_USER_TYPES))}"
                )
        elif self.role == "admin":
            if self.user_type is not None:
                self.user_type = None

        return self


class AdminUserStatusUpdate(BaseModel):
    """Payload for activating/deactivating a user."""

    model_config = {"extra": "forbid"}

    is_active: bool


class AdminPasswordReset(BaseModel):
    """Payload for resetting a user's password."""

    model_config = {"extra": "forbid"}

    new_password: str = Field(min_length=8)


# --- Response Schemas ---


class AdminUserResponse(BaseModel):
    """Response schema for user in admin context."""

    id: UUID
    email: str
    full_name: str
    role: str
    user_type: str | None
    is_active: bool
    auth_provider: str
    created_at: datetime

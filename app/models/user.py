"""User model for authentication and authorization."""

import uuid
from enum import Enum

from sqlalchemy import Boolean, Column, DateTime, String, Uuid
from sqlalchemy.sql import func

from app.database import Base


class UserRole(str, Enum):
    ADMIN = "admin"
    AGENT = "agent"


class AuthProvider(str, Enum):
    LOCAL = "local"
    LARK = "lark"
    GOOGLE = "google"


class User(Base):
    """User account for agents and administrators."""

    __tablename__ = "users"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(
        String(255), nullable=True
    )  # Nullable for OAuth-only users
    full_name = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default=UserRole.AGENT.value)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # OAuth fields
    auth_provider = Column(String(20), nullable=False, default=AuthProvider.LOCAL.value)
    lark_open_id = Column(String(255), unique=True, nullable=True, index=True)
    lark_union_id = Column(String(255), unique=True, nullable=True, index=True)
    google_sub = Column(String(255), unique=True, nullable=True, index=True)

    # Lark profile data
    avatar_url = Column(String(512), nullable=True)
    employee_id = Column(String(100), nullable=True)
    department = Column(String(255), nullable=True)

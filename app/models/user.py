"""User model for authentication and authorization."""

import uuid
from enum import Enum

from sqlalchemy import Boolean, Column, DateTime, String, Uuid
from sqlalchemy.sql import func

from app.database import Base


class UserRole(str, Enum):
    ADMIN = "admin"
    AGENT = "agent"


class User(Base):
    """User account for agents and administrators."""

    __tablename__ = "users"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default=UserRole.AGENT.value)
    is_active = Column(Boolean, default=True, nullable=False)
    password_changed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class PasswordResetToken(Base):
    """Stores hashed password reset tokens for secure single-use verification."""

    __tablename__ = "password_reset_tokens"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id = Column(Uuid, nullable=False, index=True)
    token_hash = Column(String(64), nullable=False, unique=True, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

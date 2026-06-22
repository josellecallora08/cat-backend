"""Authentication API endpoints — register, login, me, manage users, password reset."""

import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models.user import User, UserRole, PasswordResetToken
from app.config import settings
from app.services.auth import (
    create_access_token,
    hash_password,
    verify_password,
    require_auth,
    require_admin,
)
from app.services.email import send_password_reset_email
from app.services.rate_limiter import (
    rate_limiter,
    FORGOT_PASSWORD_IP_LIMIT,
    FORGOT_PASSWORD_IP_WINDOW,
    FORGOT_PASSWORD_EMAIL_LIMIT,
    FORGOT_PASSWORD_EMAIL_WINDOW,
    RESET_PASSWORD_IP_LIMIT,
    RESET_PASSWORD_IP_WINDOW,
    RESET_PASSWORD_TOKEN_LIMIT,
    RESET_PASSWORD_TOKEN_WINDOW,
)
from app.services.audit import (
    log_reset_requested,
    log_reset_success,
    log_reset_invalid_token,
    log_reset_rate_limited,
    log_reset_weak_password,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Schemas ---


class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str
    role: str = "agent"  # "admin" or "agent"


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserResponse"


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    is_active: bool


class UpdateUserRequest(BaseModel):
    full_name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


# --- Endpoints ---


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_session)):
    """Register a new user account."""
    # Check if email already exists
    stmt = select(User).where(User.email == body.email)
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Validate role
    if body.role not in (UserRole.ADMIN.value, UserRole.AGENT.value):
        raise HTTPException(status_code=400, detail="Invalid role. Must be 'admin' or 'agent'")

    # Create user
    user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
        role=body.role,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    # Generate token
    token = create_access_token(user.id, user.role)

    logger.info("New user registered: %s (%s)", user.email, user.role)

    return TokenResponse(
        access_token=token,
        user=UserResponse(
            id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
        ),
    )


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_session)):
    """Authenticate and get an access token."""
    stmt = select(User).where(User.email == body.email)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated")

    token = create_access_token(user.id, user.role)

    return TokenResponse(
        access_token=token,
        user=UserResponse(
            id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
        ),
    )


@router.get("/me", response_model=UserResponse)
async def get_me(user: User = Depends(require_auth)):
    """Get the current authenticated user's info."""
    return UserResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        is_active=user.is_active,
    )


@router.get("/users", response_model=list[UserResponse])
async def list_users(
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
):
    """List all users (admin only)."""
    stmt = select(User).order_by(User.created_at.desc())
    result = await db.execute(stmt)
    users = result.scalars().all()

    return [
        UserResponse(
            id=str(u.id),
            email=u.email,
            full_name=u.full_name,
            role=u.role,
            is_active=u.is_active,
        )
        for u in users
    ]


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: UUID,
    body: UpdateUserRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
):
    """Update a user's role or active status (admin only)."""
    stmt = select(User).where(User.id == user_id)
    result = await db.execute(stmt)
    target_user = result.scalar_one_or_none()

    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")

    if body.full_name is not None:
        target_user.full_name = body.full_name
    if body.role is not None:
        if body.role not in (UserRole.ADMIN.value, UserRole.AGENT.value):
            raise HTTPException(status_code=400, detail="Invalid role")
        target_user.role = body.role
    if body.is_active is not None:
        target_user.is_active = body.is_active

    await db.commit()
    await db.refresh(target_user)

    return UserResponse(
        id=str(target_user.id),
        email=target_user.email,
        full_name=target_user.full_name,
        role=target_user.role,
        is_active=target_user.is_active,
    )


# --- Password Reset Schemas ---


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


class MessageResponse(BaseModel):
    message: str


# --- Helpers ---


def _get_client_ip(request: Request) -> str:
    """Get client IP from request, respecting X-Forwarded-For."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _hash_token(token: str) -> str:
    """SHA-256 hash a token for secure storage."""
    return hashlib.sha256(token.encode()).hexdigest()


def _normalize_email(email: str) -> str:
    """Normalize email for consistent rate limiting."""
    return email.strip().lower()


def _validate_password_strength(password: str) -> Optional[str]:
    """Validate password meets minimum strength requirements.

    Returns error message if invalid, None if valid.
    """
    if not password:
        return "Password cannot be empty."
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if not any(c.isupper() for c in password):
        return "Password must contain at least one uppercase letter."
    if not any(c.isdigit() for c in password):
        return "Password must contain at least one number."
    return None


# --- Password Reset Endpoints ---


@router.post("/forgot-password", response_model=MessageResponse)
async def forgot_password(
    body: ForgotPasswordRequest,
    request: Request,
    db: AsyncSession = Depends(get_session),
):
    """Request a password reset email.

    Always returns a generic success message regardless of whether the email exists.
    Rate limited by IP and normalized email.
    """
    ip = _get_client_ip(request)
    normalized_email = _normalize_email(body.email)

    # Rate limit by IP
    if rate_limiter.is_rate_limited(
        f"forgot:ip:{ip}", FORGOT_PASSWORD_IP_LIMIT, FORGOT_PASSWORD_IP_WINDOW
    ):
        log_reset_rate_limited(ip, "ip")
        # Still return generic success (don't reveal rate limiting to attacker)
        return MessageResponse(
            message="If an account exists with this email, we'll send a reset link."
        )

    # Rate limit by email
    if rate_limiter.is_rate_limited(
        f"forgot:email:{normalized_email}",
        FORGOT_PASSWORD_EMAIL_LIMIT,
        FORGOT_PASSWORD_EMAIL_WINDOW,
    ):
        log_reset_rate_limited(ip, "email")
        return MessageResponse(
            message="If an account exists with this email, we'll send a reset link."
        )

    # Look up user
    stmt = select(User).where(User.email == normalized_email)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    user_found = user is not None and user.is_active
    log_reset_requested(normalized_email, ip, user_found)

    if user and user.is_active:
        # Invalidate all existing unused tokens for this user
        invalidate_stmt = (
            update(PasswordResetToken)
            .where(
                PasswordResetToken.user_id == user.id,
                PasswordResetToken.used_at.is_(None),
            )
            .values(used_at=datetime.now(timezone.utc))
        )
        await db.execute(invalidate_stmt)

        # Generate cryptographically secure token
        raw_token = secrets.token_urlsafe(32)
        token_hash = _hash_token(raw_token)

        # Store only the hash
        reset_record = PasswordResetToken(
            user_id=user.id,
            token_hash=token_hash,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=settings.reset_token_expiry_minutes),
        )
        db.add(reset_record)
        await db.commit()

        # Send email with the raw token (never stored/logged)
        send_password_reset_email(user.email, raw_token)

    # Always return the same generic response
    return MessageResponse(
        message="If an account exists with this email, we'll send a reset link."
    )


@router.post("/reset-password", response_model=MessageResponse)
async def reset_password(
    body: ResetPasswordRequest,
    request: Request,
    db: AsyncSession = Depends(get_session),
):
    """Reset password using a valid, single-use reset token.

    Token is verified by hashing the provided token and looking up the hash in the DB.
    """
    ip = _get_client_ip(request)
    token_hash = _hash_token(body.token)

    # Rate limit by IP
    if rate_limiter.is_rate_limited(
        f"reset:ip:{ip}", RESET_PASSWORD_IP_LIMIT, RESET_PASSWORD_IP_WINDOW
    ):
        log_reset_rate_limited(ip, "ip")
        raise HTTPException(
            status_code=429,
            detail="Too many attempts. Please wait before trying again.",
        )

    # Rate limit by token hash (prevents brute-forcing a specific token)
    if rate_limiter.is_rate_limited(
        f"reset:token:{token_hash[:16]}", RESET_PASSWORD_TOKEN_LIMIT, RESET_PASSWORD_TOKEN_WINDOW
    ):
        log_reset_rate_limited(ip, "token")
        raise HTTPException(
            status_code=429,
            detail="Too many attempts. Please wait before trying again.",
        )

    # Look up the token by its hash
    stmt = select(PasswordResetToken).where(PasswordResetToken.token_hash == token_hash)
    result = await db.execute(stmt)
    reset_record = result.scalar_one_or_none()

    if not reset_record:
        log_reset_invalid_token(ip, "not_found")
        raise HTTPException(
            status_code=400,
            detail="This reset link is invalid. Please request a new one.",
        )

    # Check if already used
    if reset_record.used_at is not None:
        log_reset_invalid_token(ip, "already_used")
        raise HTTPException(
            status_code=400,
            detail="This reset link has already been used. Please request a new one.",
        )

    # Check if expired
    if datetime.now(timezone.utc) > reset_record.expires_at:
        log_reset_invalid_token(ip, "expired")
        raise HTTPException(
            status_code=400,
            detail="This reset link has expired. Please request a new one.",
        )

    # Find the user
    user_stmt = select(User).where(User.id == reset_record.user_id, User.is_active == True)
    user_result = await db.execute(user_stmt)
    user = user_result.scalar_one_or_none()

    if not user:
        log_reset_invalid_token(ip, "user_not_found")
        raise HTTPException(
            status_code=400,
            detail="This reset link is invalid. Please request a new one.",
        )

    # Validate password strength
    password_error = _validate_password_strength(body.new_password)
    if password_error:
        log_reset_weak_password(str(user.id), ip, password_error)
        raise HTTPException(status_code=400, detail=password_error)

    # Reject if new password is the same as the old one
    if verify_password(body.new_password, user.hashed_password):
        log_reset_weak_password(str(user.id), ip, "same_as_old")
        raise HTTPException(
            status_code=400,
            detail="New password cannot be the same as your current password.",
        )

    # Update password and set password_changed_at (invalidates existing JWTs)
    now = datetime.now(timezone.utc)
    user.hashed_password = hash_password(body.new_password)
    user.password_changed_at = now

    # Mark this token as used
    reset_record.used_at = now

    # Invalidate ALL remaining unused tokens for this user
    invalidate_stmt = (
        update(PasswordResetToken)
        .where(
            PasswordResetToken.user_id == user.id,
            PasswordResetToken.used_at.is_(None),
        )
        .values(used_at=now)
    )
    await db.execute(invalidate_stmt)

    await db.commit()

    log_reset_success(str(user.id), ip)

    return MessageResponse(message="Password has been reset successfully. You can now log in.")

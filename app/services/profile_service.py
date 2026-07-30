"""Service layer for user profile operations.

Provides functions to read, update profile fields, and change passwords
for authenticated users.
"""

from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.schemas.profile import (
    PasswordChangeRequest,
    ProfileResponse,
    ProfileUpdateRequest,
)
from app.services.auth import hash_password, verify_password


async def get_profile(db: AsyncSession, user_id: UUID) -> ProfileResponse:
    """Fetch the profile data for a given user.

    Args:
        db: Async database session.
        user_id: The authenticated user's ID.

    Returns:
        ProfileResponse with the user's current profile fields.

    Raises:
        HTTPException: 404 if the user does not exist.
    """
    user = await _get_user_or_404(db, user_id)
    return _build_profile_response(user)


async def update_profile(
    db: AsyncSession,
    user_id: UUID,
    data: ProfileUpdateRequest,
) -> ProfileResponse:
    """Update the user's full_name and/or department.

    Only fields that are not None in the request payload are updated.

    Args:
        db: Async database session.
        user_id: The authenticated user's ID.
        data: Validated update request with optional full_name and department.

    Returns:
        ProfileResponse with the updated profile fields.

    Raises:
        HTTPException: 404 if the user does not exist.
    """
    user = await _get_user_or_404(db, user_id)

    if data.full_name is not None:
        user.full_name = data.full_name
    if data.department is not None:
        user.department = data.department

    await db.commit()
    await db.refresh(user)
    return _build_profile_response(user)


async def change_password(
    db: AsyncSession,
    user_id: UUID,
    data: PasswordChangeRequest,
) -> None:
    """Verify the current password and update to a new password.

    Args:
        db: Async database session.
        user_id: The authenticated user's ID.
        data: Validated request with current_password and new_password.

    Raises:
        HTTPException: 404 if the user does not exist.
        HTTPException: 400 if the current password is invalid.
    """
    user = await _get_user_or_404(db, user_id)

    if not verify_password(data.current_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is invalid")

    user.hashed_password = hash_password(data.new_password)
    await db.commit()


async def _get_user_or_404(db: AsyncSession, user_id: UUID) -> User:
    """Fetch a user by ID or raise HTTP 404."""
    stmt = select(User).where(User.id == user_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


def _build_profile_response(user: User) -> ProfileResponse:
    """Construct a ProfileResponse from a User model instance."""
    return ProfileResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        department=user.department,
        avatar_url=user.avatar_url,
    )

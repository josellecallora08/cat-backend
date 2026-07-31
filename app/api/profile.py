"""Profile API endpoints for authenticated users.

Provides routes to view and update the current user's profile,
and to change their password.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models.user import User
from app.schemas.profile import (
    PasswordChangeRequest,
    ProfileResponse,
    ProfileUpdateRequest,
)
from app.services.auth import require_auth
from app.services.profile_service import change_password, get_profile, update_profile


router = APIRouter()


@router.get("/", response_model=ProfileResponse)
async def get_my_profile(
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_session),
) -> ProfileResponse:
    """Return the authenticated user's profile information.

    Returns the user's full name, email, department, and avatar URL.
    Email is read-only metadata and cannot be changed via the update endpoint.
    """
    return await get_profile(db, user.id)


@router.patch("/", response_model=ProfileResponse)
async def update_my_profile(
    body: ProfileUpdateRequest,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_session),
) -> ProfileResponse:
    """Update the authenticated user's name and/or department.

    Only non-null fields in the request body are updated. Email cannot be
    changed through this endpoint.
    """
    return await update_profile(db, user.id, body)


@router.post("/password")
async def change_my_password(
    body: PasswordChangeRequest,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    """Change the authenticated user's password.

    Requires the correct current password and a new password of at least
    8 characters. Returns HTTP 400 if the current password is invalid.
    """
    await change_password(db, user.id, body)
    return {"message": "Password updated successfully"}

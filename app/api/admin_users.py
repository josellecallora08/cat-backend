"""Admin User Management API endpoints.

Provides CRUD operations for user accounts, accessible only to authenticated admins.
Endpoints cover listing, creating, updating, deactivating, deleting users,
and resetting passwords.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models.user import User
from app.schemas.admin_user import (
    AdminPasswordReset,
    AdminUserCreate,
    AdminUserResponse,
    AdminUserStatusUpdate,
    AdminUserUpdate,
)
from app.services.auth import require_admin
from app.services.user_service import UserService


router = APIRouter()


@router.get("/", response_model=list[AdminUserResponse])
async def list_users(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
) -> list[AdminUserResponse]:
    """List all users ordered by creation date descending."""
    service = UserService(db)
    users = await service.list_users()
    return [
        AdminUserResponse.model_validate(user, from_attributes=True) for user in users
    ]


@router.post("/", response_model=AdminUserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: AdminUserCreate,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
) -> AdminUserResponse:
    """Create a new user account with hashed password."""
    service = UserService(db)
    user = await service.create_user(payload)
    return AdminUserResponse.model_validate(user, from_attributes=True)


@router.put("/{user_id}", response_model=AdminUserResponse)
async def update_user(
    user_id: UUID,
    payload: AdminUserUpdate,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
) -> AdminUserResponse:
    """Update user details. Prevents self-demotion."""
    service = UserService(db)
    user = await service.update_user(user_id, payload, admin.id)
    return AdminUserResponse.model_validate(user, from_attributes=True)


@router.patch("/{user_id}/status", response_model=AdminUserResponse)
async def update_user_status(
    user_id: UUID,
    payload: AdminUserStatusUpdate,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
) -> AdminUserResponse:
    """Activate or deactivate a user. Prevents self-deactivation."""
    service = UserService(db)
    user = await service.set_user_status(user_id, payload.is_active, admin.id)
    return AdminUserResponse.model_validate(user, from_attributes=True)


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_user(
    user_id: UUID,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
) -> None:
    """Permanently delete a user. Nullifies session references first."""
    service = UserService(db)
    await service.delete_user(user_id, admin.id)


@router.post("/{user_id}/reset-password")
async def reset_password(
    user_id: UUID,
    payload: AdminPasswordReset,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    """Reset a user's password to a new value."""
    service = UserService(db)
    await service.reset_password(user_id, payload.new_password)
    return {"message": "Password reset successfully"}

"""Service layer for admin user management operations.

Encapsulates business logic for creating, updating, listing, deactivating,
deleting users, and resetting passwords.
"""

import secrets
import string
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Session
from app.models.user import User, UserRole
from app.schemas.admin_user import AdminUserCreate, AdminUserUpdate
from app.services.auth import hash_password


class UserService:
    """Business logic for admin user management operations."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def list_users(self) -> list[User]:
        """Return all users ordered by creation date descending."""
        stmt = select(User).order_by(User.created_at.desc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_user_by_id(self, user_id: UUID) -> User | None:
        """Fetch a single user by ID."""
        stmt = select(User).where(User.id == user_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def create_user(self, payload: AdminUserCreate) -> User:
        """Create a new user with hashed password.

        Args:
            payload: Validated creation data including email, name, password, role.

        Returns:
            The newly created User record.

        Raises:
            HTTPException: 409 if email is already registered.
        """
        existing = await self._get_user_by_email(payload.email)
        if existing:
            raise HTTPException(status_code=409, detail="Email already registered")

        user = User(
            email=payload.email,
            full_name=payload.full_name,
            hashed_password=hash_password(payload.password),
            role=payload.role,
            user_type=payload.user_type,
        )
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def update_user(
        self, user_id: UUID, payload: AdminUserUpdate, admin_id: UUID
    ) -> User:
        """Update user details. Prevents self-demotion.

        Args:
            user_id: The ID of the user to update.
            payload: Validated update data (full_name, role, user_type).
            admin_id: The ID of the admin performing the update.

        Returns:
            The updated User record.

        Raises:
            HTTPException: 404 if user not found, 400 if self-demotion attempted.
        """
        user = await self._get_user_or_404(user_id)

        if user_id == admin_id and user.role == UserRole.ADMIN.value:
            if payload.role != UserRole.ADMIN.value:
                raise HTTPException(
                    status_code=400, detail="Cannot demote your own account"
                )

        user.full_name = payload.full_name
        user.role = payload.role
        user.user_type = payload.user_type
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def set_user_status(
        self, user_id: UUID, is_active: bool, admin_id: UUID
    ) -> User:
        """Activate or deactivate a user. Prevents self-deactivation.

        Args:
            user_id: The ID of the user to update.
            is_active: Whether the user should be active.
            admin_id: The ID of the admin performing the action.

        Returns:
            The updated User record.

        Raises:
            HTTPException: 404 if user not found, 400 if self-deactivation attempted.
        """
        if user_id == admin_id and not is_active:
            raise HTTPException(
                status_code=400, detail="Cannot deactivate your own account"
            )

        user = await self._get_user_or_404(user_id)
        user.is_active = is_active
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def delete_user(self, user_id: UUID, admin_id: UUID) -> None:
        """Permanently delete a user. Nullifies session references first.

        Args:
            user_id: The ID of the user to delete.
            admin_id: The ID of the admin performing the deletion.

        Raises:
            HTTPException: 404 if user not found, 400 if self-deletion attempted.
        """
        if user_id == admin_id:
            raise HTTPException(
                status_code=400, detail="Cannot delete your own account"
            )

        user = await self._get_user_or_404(user_id)

        # Nullify agent_id references in sessions before deleting
        stmt = update(Session).where(Session.agent_id == user_id).values(agent_id=None)
        await self.db.execute(stmt)

        await self.db.delete(user)
        await self.db.commit()

    async def reset_password(self, user_id: UUID, new_password: str) -> None:
        """Hash and update the user's password.

        Args:
            user_id: The ID of the user whose password to reset.
            new_password: The new plaintext password to hash and store.

        Raises:
            HTTPException: 404 if user not found.
        """
        user = await self._get_user_or_404(user_id)
        user.hashed_password = hash_password(new_password)
        await self.db.commit()

    def generate_random_password(self) -> str:
        """Generate a cryptographically secure 12-char password.

        Ensures at least one uppercase letter, one lowercase letter,
        one digit, and one special character are present.

        Returns:
            A 12-character random password string.
        """
        special_chars = "!@#$%^&*"
        alphabet = string.ascii_letters + string.digits + special_chars

        # Guarantee at least one character from each required category
        password_chars = [
            secrets.choice(string.ascii_uppercase),
            secrets.choice(string.ascii_lowercase),
            secrets.choice(string.digits),
            secrets.choice(special_chars),
        ]

        # Fill remaining 8 characters from the full alphabet
        for _ in range(8):
            password_chars.append(secrets.choice(alphabet))

        # Shuffle to avoid predictable positions
        # Use Fisher-Yates shuffle with secrets for cryptographic randomness
        for i in range(len(password_chars) - 1, 0, -1):
            j = secrets.randbelow(i + 1)
            password_chars[i], password_chars[j] = password_chars[j], password_chars[i]

        return "".join(password_chars)

    async def _get_user_by_email(self, email: str) -> User | None:
        """Look up a user by email address."""
        stmt = select(User).where(User.email == email)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_user_or_404(self, user_id: UUID) -> User:
        """Fetch a user by ID or raise 404."""
        user = await self.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user

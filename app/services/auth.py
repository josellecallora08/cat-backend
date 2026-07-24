"""Authentication and authorization service.

Handles JWT token creation/verification, password hashing, and role-based access control.
"""

from datetime import datetime, timedelta, timezone
from uuid import UUID

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_session
from app.models.user import User, UserRole, UserType

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"), hashed_password.encode("utf-8")
        )
    except Exception:
        return False


def create_access_token(
    user_id: UUID,
    role: str,
    user_type: str | None = None,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a JWT access token for the given user.

    Args:
        user_id: The user's unique identifier.
        role: User role for claims (e.g., "user", "admin").
        user_type: User type for claims (e.g., "agent", "trainer").
            Included in token only when role is "user".
        expires_delta: Custom token lifetime. Defaults to configured expiry.

    Returns:
        Encoded JWT string.
    """
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(hours=settings.jwt_expiry_hours)
    )
    payload = {
        "sub": str(user_id),
        "role": role,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    if role == UserRole.USER.value and user_type is not None:
        payload["user_type"] = user_type
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


async def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_session),
) -> User | None:
    """Extract current user from JWT token. Returns None if no token or invalid."""
    if not token:
        return None

    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        user_id = payload.get("sub")
        if not user_id:
            return None
    except JWTError:
        return None

    stmt = select(User).where(User.id == user_id, User.is_active.is_(True))
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def require_auth(
    user: User | None = Depends(get_current_user),
) -> User:
    """Require a valid authenticated user."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def require_admin(
    user: User = Depends(require_auth),
) -> User:
    """Require admin role."""
    if user.role != UserRole.ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


async def require_agent(
    user: User = Depends(require_auth),
) -> User:
    """Require agent access (role=user+user_type=agent, or admin).

    Admins have full access. Non-admin users must have user_type="agent".
    All other combinations are rejected with 403.
    """
    if user.role == UserRole.ADMIN.value:
        return user
    if user.role == UserRole.USER.value and user.user_type == UserType.AGENT.value:
        return user
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Agent access required",
    )

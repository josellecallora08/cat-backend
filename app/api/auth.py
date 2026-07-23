"""Authentication API endpoints — register, login, me, manage users, Lark OAuth, Google OAuth."""

import logging
import secrets
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_session
from app.models.user import User, UserRole
from app.services.auth import (
    create_access_token,
    hash_password,
    verify_password,
    require_auth,
    require_admin,
)
from app.services.google_oauth import (
    exchange_code_for_tokens as google_exchange_code,
    fetch_google_user_info,
    get_authorize_url as google_get_authorize_url,
    get_or_create_google_user,
)
from app.services.lark_oauth import (
    exchange_code_for_user_token,
    fetch_lark_user_info,
    fetch_lark_user_department,
    get_authorize_url,
    get_or_create_lark_user,
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
        raise HTTPException(
            status_code=400, detail="Invalid role. Must be 'admin' or 'agent'"
        )

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
async def login(
    request: Request,
    db: AsyncSession = Depends(get_session),
):
    """Authenticate and get an access token.

    Accepts both JSON body ({"email": "...", "password": "..."}) and
    OAuth2 form data (username=...&password=...) for Swagger compatibility.
    """
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        body = await request.json()
        email = body.get("email", "")
        password = body.get("password", "")
    else:
        form = await request.form()
        email = form.get("username", "")
        password = form.get("password", "")
        if not email or not password:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Provide JSON body or form data with username and password",
            )

    stmt = select(User).where(User.email == email)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if (
        not user
        or not user.hashed_password
        or not verify_password(password, user.hashed_password)
    ):
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


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: UUID,
    body: UpdateUserRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
):
    """Update a user's details (admin only)."""
    stmt = select(User).where(User.id == user_id)
    result = await db.execute(stmt)
    target = result.scalar_one_or_none()

    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    if body.full_name is not None:
        target.full_name = body.full_name
    if body.role is not None:
        if body.role not in (UserRole.ADMIN.value, UserRole.AGENT.value):
            raise HTTPException(status_code=400, detail="Invalid role")
        target.role = body.role
    if body.is_active is not None:
        target.is_active = body.is_active

    await db.commit()
    await db.refresh(target)

    return UserResponse(
        id=str(target.id),
        email=target.email,
        full_name=target.full_name,
        role=target.role,
        is_active=target.is_active,
    )


# --- Lark OAuth Schemas ---


class LarkAuthorizeResponse(BaseModel):
    """Response containing the Lark OAuth authorization URL."""

    authorize_url: str
    state: str


class LarkCallbackRequest(BaseModel):
    """Request body for the Lark OAuth callback."""

    code: str = Field(min_length=1, max_length=512)
    state: str = Field(min_length=1, max_length=128)


# --- Lark OAuth Endpoints ---


@router.get("/lark/authorize", response_model=LarkAuthorizeResponse)
async def lark_authorize():
    """Generate the Lark OAuth authorization URL.

    The frontend should redirect the user to the returned `authorize_url`.
    The `state` parameter should be stored client-side and verified on callback.
    """
    if not settings.lark_app_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Lark OAuth is not configured",
        )

    state = secrets.token_urlsafe(32)
    authorize_url = get_authorize_url(state)

    return LarkAuthorizeResponse(authorize_url=authorize_url, state=state)


@router.post("/lark/callback", response_model=TokenResponse, status_code=200)
async def lark_callback(
    body: LarkCallbackRequest,
    db: AsyncSession = Depends(get_session),
):
    """Handle the Lark OAuth callback.

    Exchanges the authorization code for user info, creates or links
    the user account, and returns a CAT JWT access token.
    """
    if not settings.lark_app_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Lark OAuth is not configured",
        )

    try:
        # Exchange code for Lark user_access_token
        user_token = await exchange_code_for_user_token(body.code)

        # Fetch user profile from Lark
        lark_info = await fetch_lark_user_info(user_token)

        # Fetch department (best-effort, requires scope)
        department = await fetch_lark_user_department(user_token)

        # Create or link user in our database
        user, access_token = await get_or_create_lark_user(lark_info, db)

        # Save department if fetched (separate from get_or_create to keep it simple)
        if department and user.department != department:
            user.department = department
            await db.commit()
            await db.refresh(user)

    except ValueError as e:
        logger.warning("Lark OAuth failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Lark authentication failed: {e}",
        )
    except Exception as e:
        logger.error("Unexpected error during Lark OAuth: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to communicate with Lark",
        )

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated")

    return TokenResponse(
        access_token=access_token,
        user=UserResponse(
            id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
        ),
    )


# --- Google OAuth Schemas ---


class GoogleAuthorizeResponse(BaseModel):
    """Response containing the Google OAuth authorization URL."""

    authorize_url: str
    state: str


class GoogleCallbackRequest(BaseModel):
    """Request body for the Google OAuth callback."""

    code: str = Field(min_length=1, max_length=2048)
    state: str = Field(min_length=1, max_length=128)


# --- Google OAuth Endpoints ---


@router.get("/google/authorize", response_model=GoogleAuthorizeResponse)
async def google_authorize():
    """Generate the Google OAuth authorization URL.

    The frontend should redirect the user to the returned `authorize_url`.
    The `state` parameter should be stored client-side and verified on callback.
    """
    if not settings.google_client_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth is not configured",
        )

    state = secrets.token_urlsafe(32)
    authorize_url = google_get_authorize_url(state)

    return GoogleAuthorizeResponse(authorize_url=authorize_url, state=state)


@router.post("/google/callback", response_model=TokenResponse, status_code=200)
async def google_callback(
    body: GoogleCallbackRequest,
    db: AsyncSession = Depends(get_session),
):
    """Handle the Google OAuth callback.

    Exchanges the authorization code for user info, creates or links
    the user account, and returns a CAT JWT access token.
    """
    if not settings.google_client_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth is not configured",
        )

    try:
        # Exchange code for Google access token
        access_token = await google_exchange_code(body.code)

        # Fetch user profile from Google
        google_info = await fetch_google_user_info(access_token)

        # Create or link user in our database
        user, cat_token = await get_or_create_google_user(google_info, db)

    except ValueError as e:
        logger.warning("Google OAuth failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Google authentication failed: {e}",
        )
    except Exception as e:
        logger.error("Unexpected error during Google OAuth: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to communicate with Google",
        )

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated")

    return TokenResponse(
        access_token=cat_token,
        user=UserResponse(
            id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
        ),
    )

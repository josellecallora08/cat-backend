"""Google OAuth 2.0 service.

Handles the OAuth 2.0 flow with Google:
1. Generate authorize URL for frontend redirect
2. Exchange authorization code for tokens
3. Fetch user profile from Google's userinfo endpoint
4. Create or link user accounts
"""

import logging
from dataclasses import dataclass
from urllib.parse import urlencode

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.user import AuthProvider, User, UserRole
from app.services.auth import create_access_token

logger = logging.getLogger(__name__)

# Google OAuth endpoints
GOOGLE_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

# Scopes needed for basic profile + email
GOOGLE_SCOPES = "openid email profile"


@dataclass
class GoogleUserInfo:
    """User information retrieved from Google."""

    sub: str
    name: str
    email: str
    email_verified: bool
    picture: str


def get_authorize_url(state: str) -> str:
    """Build the Google OAuth authorization URL.

    Args:
        state: Anti-CSRF state parameter (should be verified on callback).

    Returns:
        Full authorization URL for frontend redirect.
    """
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": GOOGLE_SCOPES,
        "state": state,
        "access_type": "offline",
        "prompt": "select_account",
    }
    return f"{GOOGLE_AUTHORIZE_URL}?{urlencode(params)}"


async def exchange_code_for_tokens(code: str) -> str:
    """Exchange an authorization code for an access token.

    Args:
        code: The authorization code from Google's redirect callback.

    Returns:
        The access_token string.

    Raises:
        ValueError: If token exchange fails.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": settings.google_redirect_uri,
            },
        )

    if resp.status_code != 200:
        error_data = resp.json()
        msg = error_data.get(
            "error_description", error_data.get("error", "Unknown error")
        )
        logger.error("Google token exchange failed: %s", msg)
        raise ValueError(f"Google token exchange error: {msg}")

    data = resp.json()
    return data["access_token"]


async def fetch_google_user_info(access_token: str) -> GoogleUserInfo:
    """Fetch the authenticated user's profile from Google.

    Args:
        access_token: Token from the code exchange step.

    Returns:
        GoogleUserInfo with the user's profile data.

    Raises:
        ValueError: If the API call fails.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )

    if resp.status_code != 200:
        logger.error("Failed to fetch Google user info: %d", resp.status_code)
        raise ValueError("Failed to fetch Google user info")

    data = resp.json()
    return GoogleUserInfo(
        sub=data["sub"],
        name=data.get("name", "Google User"),
        email=data.get("email", ""),
        email_verified=data.get("email_verified", False),
        picture=data.get("picture", ""),
    )


async def get_or_create_google_user(
    google_info: GoogleUserInfo,
    db: AsyncSession,
) -> tuple[User, str]:
    """Find an existing user by Google sub or email, or create a new one.

    If a user with the same email already exists (registered via local auth or Lark),
    their account is linked to Google. Otherwise a new user is created.

    Args:
        google_info: Profile data from Google.
        db: Async database session.

    Returns:
        Tuple of (User instance, JWT access_token).
    """
    # First, try to find by google_sub
    stmt = select(User).where(User.google_sub == google_info.sub)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user:
        token = create_access_token(user.id, user.role)
        return user, token

    # Try to find by email and link accounts
    if google_info.email and google_info.email_verified:
        stmt = select(User).where(User.email == google_info.email)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if user:
            # Link existing account to Google
            user.google_sub = google_info.sub
            if user.auth_provider == AuthProvider.LOCAL.value:
                user.auth_provider = AuthProvider.GOOGLE.value
            await db.commit()
            await db.refresh(user)
            token = create_access_token(user.id, user.role)
            logger.info("Linked Google account to existing user: %s", user.email)
            return user, token

    # Create new user from Google profile
    if not google_info.email:
        raise ValueError("Google account does not have an email address")

    user = User(
        email=google_info.email,
        hashed_password=None,
        full_name=google_info.name,
        role=UserRole.AGENT.value,
        auth_provider=AuthProvider.GOOGLE.value,
        google_sub=google_info.sub,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    token = create_access_token(user.id, user.role)
    logger.info("Created new Google user: %s (%s)", user.email, google_info.sub)
    return user, token

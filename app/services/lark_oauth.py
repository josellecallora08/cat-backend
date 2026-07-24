"""Lark (Feishu) OAuth service.

Handles the OAuth 2.0 flow with Lark Open Platform:
1. Generate authorize URL for frontend redirect
2. Exchange authorization code for user_access_token
3. Fetch user profile information
4. Create or link user accounts
"""

import logging
from dataclasses import dataclass

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.user import AuthProvider, User, UserRole, UserType
from app.services.auth import create_access_token

logger = logging.getLogger(__name__)

# Lark API endpoints
LARK_AUTHORIZE_URL = "https://open.larksuite.com/open-apis/authen/v1/authorize"
LARK_TOKEN_URL = "https://open.larksuite.com/open-apis/authen/v1/oidc/access_token"
LARK_USER_INFO_URL = "https://open.larksuite.com/open-apis/authen/v1/user_info"
LARK_APP_TOKEN_URL = (
    "https://open.larksuite.com/open-apis/auth/v3/app_access_token/internal"
)


@dataclass
class LarkUserInfo:
    """User information retrieved from Lark."""

    open_id: str
    union_id: str
    name: str
    email: str
    avatar_url: str
    employee_id: str


def get_authorize_url(state: str) -> str:
    """Build the Lark OAuth authorization URL.

    Args:
        state: Anti-CSRF state parameter (should be verified on callback).

    Returns:
        Full authorization URL for frontend redirect.
    """
    params = {
        "app_id": settings.lark_app_id,
        "redirect_uri": settings.lark_redirect_uri,
        "response_type": "code",
        "state": state,
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{LARK_AUTHORIZE_URL}?{query}"


async def _get_app_access_token() -> str:
    """Obtain an app_access_token from Lark for internal apps.

    Returns:
        The app_access_token string.

    Raises:
        httpx.HTTPStatusError: If the Lark API returns an error.
        ValueError: If the response indicates failure.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            LARK_APP_TOKEN_URL,
            json={
                "app_id": settings.lark_app_id,
                "app_secret": settings.lark_app_secret,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    if data.get("code") != 0:
        msg = data.get("msg", "Unknown error")
        logger.error("Failed to get Lark app_access_token: %s", msg)
        raise ValueError(f"Lark app token error: {msg}")

    return data["app_access_token"]


async def exchange_code_for_user_token(code: str) -> str:
    """Exchange an authorization code for a user_access_token.

    Args:
        code: The authorization code from Lark's redirect callback.

    Returns:
        The user_access_token string.

    Raises:
        ValueError: If token exchange fails.
    """
    app_token = await _get_app_access_token()

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            LARK_TOKEN_URL,
            headers={"Authorization": f"Bearer {app_token}"},
            json={
                "grant_type": "authorization_code",
                "code": code,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    if data.get("code") != 0:
        msg = data.get("msg", "Unknown error")
        logger.error("Lark token exchange failed: %s", msg)
        raise ValueError(f"Lark token exchange error: {msg}")

    return data["data"]["access_token"]


async def fetch_lark_user_info(user_access_token: str) -> LarkUserInfo:
    """Fetch the authenticated user's profile from Lark.

    Args:
        user_access_token: Token from the code exchange step.

    Returns:
        LarkUserInfo with the user's profile data.

    Raises:
        ValueError: If the API call fails.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            LARK_USER_INFO_URL,
            headers={"Authorization": f"Bearer {user_access_token}"},
        )
        resp.raise_for_status()
        data = resp.json()

    if data.get("code") != 0:
        msg = data.get("msg", "Unknown error")
        logger.error("Failed to fetch Lark user info: %s", msg)
        raise ValueError(f"Lark user info error: {msg}")

    info = data["data"]
    return LarkUserInfo(
        open_id=info["open_id"],
        union_id=info.get("union_id", ""),
        name=info.get("name", "Lark User"),
        email=info.get("email", ""),
        avatar_url=info.get("avatar_url", ""),
        employee_id=info.get("employee_no", ""),
    )


async def fetch_lark_user_department(user_access_token: str) -> str:
    """Fetch the user's department name from Lark Contact API.

    This requires the contact:user.department:readonly scope.
    Returns empty string if the scope is not granted or the API call fails.

    Args:
        user_access_token: Token from the code exchange step.

    Returns:
        Department name string, or empty string on failure.
    """
    try:
        app_token = await _get_app_access_token()
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://open.larksuite.com/open-apis/contact/v3/users/me",
                headers={"Authorization": f"Bearer {user_access_token}"},
                params={"user_id_type": "open_id"},
            )
            if resp.status_code != 200:
                return ""
            data = resp.json()

        if data.get("code") != 0:
            return ""

        dept_ids = data.get("data", {}).get("user", {}).get("department_ids", [])
        if not dept_ids:
            return ""

        # Fetch the first department's name
        dept_resp_data = None
        async with httpx.AsyncClient(timeout=10.0) as client:
            dept_resp = await client.get(
                f"https://open.larksuite.com/open-apis/contact/v3/departments/{dept_ids[0]}",
                headers={"Authorization": f"Bearer {app_token}"},
                params={"department_id_type": "department_id"},
            )
            if dept_resp.status_code == 200:
                dept_resp_data = dept_resp.json()

        if dept_resp_data and dept_resp_data.get("code") == 0:
            return dept_resp_data.get("data", {}).get("department", {}).get("name", "")

        return ""
    except Exception as e:
        logger.warning("Failed to fetch department from Lark: %s", e)
        return ""


async def get_or_create_lark_user(
    lark_info: LarkUserInfo,
    db: AsyncSession,
) -> tuple[User, str]:
    """Find an existing user by Lark open_id or email, or create a new one.

    If a user with the same email already exists (registered via local auth),
    their account is linked to Lark. Otherwise a new user is created.

    Args:
        lark_info: Profile data from Lark.
        db: Async database session.

    Returns:
        Tuple of (User instance, JWT access_token).
    """
    # First, try to find by lark_open_id
    stmt = select(User).where(User.lark_open_id == lark_info.open_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user:
        # Update profile data from Lark (in case it changed)
        user.avatar_url = lark_info.avatar_url or user.avatar_url
        user.full_name = lark_info.name or user.full_name
        if lark_info.employee_id:
            user.employee_id = lark_info.employee_id
        await db.commit()
        await db.refresh(user)
        token = create_access_token(user.id, user.role, user_type=user.user_type)
        return user, token

    # Try to find by email and link accounts
    if lark_info.email:
        stmt = select(User).where(User.email == lark_info.email)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if user:
            # Link existing account to Lark
            user.lark_open_id = lark_info.open_id
            user.lark_union_id = lark_info.union_id or None
            user.auth_provider = AuthProvider.LARK.value
            user.avatar_url = lark_info.avatar_url or user.avatar_url
            if lark_info.employee_id:
                user.employee_id = lark_info.employee_id
            await db.commit()
            await db.refresh(user)
            token = create_access_token(user.id, user.role, user_type=user.user_type)
            logger.info("Linked Lark account to existing user: %s", user.email)
            return user, token

    # Create new user from Lark profile
    email = lark_info.email or f"{lark_info.open_id}@lark.oauth"
    user = User(
        email=email,
        hashed_password=None,
        full_name=lark_info.name,
        role=UserRole.USER.value,
        user_type=UserType.AGENT.value,
        auth_provider=AuthProvider.LARK.value,
        lark_open_id=lark_info.open_id,
        lark_union_id=lark_info.union_id or None,
        avatar_url=lark_info.avatar_url or None,
        employee_id=lark_info.employee_id or None,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    token = create_access_token(user.id, user.role, user_type=user.user_type)
    logger.info("Created new Lark user: %s (%s)", user.email, lark_info.open_id)
    return user, token

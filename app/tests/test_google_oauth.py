"""Tests for Google OAuth endpoints and service."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.google_oauth import (
    GoogleUserInfo,
    get_authorize_url,
    exchange_code_for_tokens,
    fetch_google_user_info,
)


@pytest.fixture
async def client():
    """Async test client for the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# --- Unit Tests: Service Functions ---


class TestGetAuthorizeUrl:
    """Tests for Google authorize URL generation."""

    @patch("app.services.google_oauth.settings")
    def test_builds_correct_url(self, mock_settings):
        """Authorize URL includes client_id, redirect_uri, scope, and state."""
        mock_settings.google_client_id = "test-client-id.apps.googleusercontent.com"
        mock_settings.google_redirect_uri = "http://localhost:3000/auth/google/callback"

        url = get_authorize_url("test-state-123")

        assert "https://accounts.google.com/o/oauth2/v2/auth" in url
        assert "client_id=test-client-id.apps.googleusercontent.com" in url
        assert "state=test-state-123" in url
        assert "response_type=code" in url
        assert "scope=openid" in url
        assert "access_type=offline" in url

    @patch("app.services.google_oauth.settings")
    def test_includes_prompt_select_account(self, mock_settings):
        """URL includes prompt=select_account for account chooser."""
        mock_settings.google_client_id = "client-id"
        mock_settings.google_redirect_uri = "http://localhost:3000/callback"

        url = get_authorize_url("state")

        assert "prompt=select_account" in url


class TestExchangeCodeForTokens:
    """Tests for token exchange with Google."""

    @pytest.mark.asyncio
    @patch("app.services.google_oauth.httpx.AsyncClient")
    async def test_successful_token_exchange(self, mock_client_class):
        """Successful code exchange returns access_token."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "google-access-token-123",
            "token_type": "Bearer",
            "expires_in": 3600,
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        token = await exchange_code_for_tokens("valid-auth-code")

        assert token == "google-access-token-123"

    @pytest.mark.asyncio
    @patch("app.services.google_oauth.httpx.AsyncClient")
    async def test_invalid_code_raises_value_error(self, mock_client_class):
        """When Google rejects the code, raises ValueError."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": "invalid_grant",
            "error_description": "Code was already redeemed.",
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        with pytest.raises(ValueError, match="Code was already redeemed"):
            await exchange_code_for_tokens("used-code")


class TestFetchGoogleUserInfo:
    """Tests for fetching user info from Google."""

    @pytest.mark.asyncio
    @patch("app.services.google_oauth.httpx.AsyncClient")
    async def test_successful_user_info_fetch(self, mock_client_class):
        """Successful fetch returns GoogleUserInfo dataclass."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "sub": "google-sub-12345",
            "name": "Test Google User",
            "email": "test@gmail.com",
            "email_verified": True,
            "picture": "https://lh3.googleusercontent.com/photo.jpg",
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        info = await fetch_google_user_info("google-access-token")

        assert info.sub == "google-sub-12345"
        assert info.name == "Test Google User"
        assert info.email == "test@gmail.com"
        assert info.email_verified is True

    @pytest.mark.asyncio
    @patch("app.services.google_oauth.httpx.AsyncClient")
    async def test_invalid_token_raises_value_error(self, mock_client_class):
        """When token is invalid, raises ValueError."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        with pytest.raises(ValueError, match="Failed to fetch Google user info"):
            await fetch_google_user_info("expired-token")


# --- Integration Tests: API Endpoints ---


class TestGoogleAuthorizeEndpoint:
    """Tests for GET /api/auth/google/authorize."""

    @pytest.mark.asyncio
    @patch("app.services.google_oauth.settings")
    @patch("app.api.auth.settings")
    async def test_returns_authorize_url_and_state(
        self, mock_api_settings, mock_svc_settings, client
    ):
        """Returns a valid authorize URL and state when Google is configured."""
        mock_api_settings.google_client_id = "test-client-id.apps.googleusercontent.com"
        mock_api_settings.lark_app_id = ""
        mock_svc_settings.google_client_id = "test-client-id.apps.googleusercontent.com"
        mock_svc_settings.google_redirect_uri = (
            "http://localhost:3000/auth/google/callback"
        )

        resp = await client.get("/api/auth/google/authorize")

        assert resp.status_code == 200
        data = resp.json()
        assert "authorize_url" in data
        assert "state" in data
        assert "accounts.google.com" in data["authorize_url"]
        assert "test-client-id" in data["authorize_url"]
        assert len(data["state"]) > 20

    @pytest.mark.asyncio
    @patch("app.api.auth.settings")
    async def test_returns_503_when_not_configured(self, mock_settings, client):
        """Returns 503 when Google OAuth is not configured."""
        mock_settings.google_client_id = ""
        mock_settings.lark_app_id = ""

        resp = await client.get("/api/auth/google/authorize")

        assert resp.status_code == 503
        assert "not configured" in resp.json()["detail"]


class TestGoogleCallbackEndpoint:
    """Tests for POST /api/auth/google/callback."""

    @pytest.mark.asyncio
    @patch("app.api.auth.settings")
    async def test_returns_503_when_not_configured(self, mock_settings, client):
        """Returns 503 when Google OAuth is not configured."""
        mock_settings.google_client_id = ""
        mock_settings.lark_app_id = ""

        resp = await client.post(
            "/api/auth/google/callback",
            json={"code": "test-code", "state": "test-state"},
        )

        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_rejects_empty_code(self, client):
        """Returns 422 for empty code parameter."""
        resp = await client.post(
            "/api/auth/google/callback",
            json={"code": "", "state": "test-state"},
        )

        assert resp.status_code == 422

    @pytest.mark.asyncio
    @patch("app.api.auth.settings")
    @patch("app.api.auth.get_or_create_google_user")
    @patch("app.api.auth.fetch_google_user_info")
    @patch("app.api.auth.google_exchange_code")
    async def test_successful_callback_returns_token(
        self,
        mock_exchange,
        mock_fetch_info,
        mock_get_or_create,
        mock_settings,
        client,
    ):
        """Successful callback returns access_token and user info."""
        mock_settings.google_client_id = "test-client-id.apps.googleusercontent.com"
        mock_settings.google_client_secret = "test-secret"  # pragma: allowlist secret
        mock_settings.google_redirect_uri = "http://localhost:3000/callback"
        mock_settings.lark_app_id = ""

        mock_exchange.return_value = "google-access-token"
        mock_fetch_info.return_value = GoogleUserInfo(
            sub="google-sub-12345",
            name="Test Google User",
            email="test@gmail.com",
            email_verified=True,
            picture="",
        )

        mock_user = AsyncMock()
        mock_user.id = uuid4()
        mock_user.email = "test@gmail.com"
        mock_user.full_name = "Test Google User"
        mock_user.role = "agent"
        mock_user.user_type = None
        mock_user.is_active = True

        mock_get_or_create.return_value = (mock_user, "cat-jwt-token")

        resp = await client.post(
            "/api/auth/google/callback",
            json={"code": "valid-code", "state": "valid-state"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["access_token"] == "cat-jwt-token"
        assert data["user"]["email"] == "test@gmail.com"
        assert data["user"]["full_name"] == "Test Google User"

    @pytest.mark.asyncio
    @patch("app.api.auth.settings")
    @patch("app.api.auth.google_exchange_code")
    async def test_invalid_code_returns_401(self, mock_exchange, mock_settings, client):
        """Returns 401 when Google rejects the authorization code."""
        mock_settings.google_client_id = "test-client-id"
        mock_settings.google_client_secret = "test-secret"  # pragma: allowlist secret
        mock_settings.google_redirect_uri = "http://localhost:3000/callback"
        mock_settings.lark_app_id = ""

        mock_exchange.side_effect = ValueError(
            "Google token exchange error: invalid_grant"
        )

        resp = await client.post(
            "/api/auth/google/callback",
            json={"code": "invalid-code", "state": "test-state"},
        )

        assert resp.status_code == 401
        assert "Google authentication failed" in resp.json()["detail"]

    @pytest.mark.asyncio
    @patch("app.api.auth.settings")
    @patch("app.api.auth.get_or_create_google_user")
    @patch("app.api.auth.fetch_google_user_info")
    @patch("app.api.auth.google_exchange_code")
    async def test_deactivated_user_returns_403(
        self,
        mock_exchange,
        mock_fetch_info,
        mock_get_or_create,
        mock_settings,
        client,
    ):
        """Returns 403 when the linked user account is deactivated."""
        mock_settings.google_client_id = "test-client-id"
        mock_settings.google_client_secret = "test-secret"  # pragma: allowlist secret
        mock_settings.google_redirect_uri = "http://localhost:3000/callback"
        mock_settings.lark_app_id = ""

        mock_exchange.return_value = "token"
        mock_fetch_info.return_value = GoogleUserInfo(
            sub="sub-123",
            name="Inactive User",
            email="inactive@gmail.com",
            email_verified=True,
            picture="",
        )

        mock_user = AsyncMock()
        mock_user.id = uuid4()
        mock_user.email = "inactive@gmail.com"
        mock_user.full_name = "Inactive User"
        mock_user.role = "agent"
        mock_user.user_type = None
        mock_user.is_active = False

        mock_get_or_create.return_value = (mock_user, "token")

        resp = await client.post(
            "/api/auth/google/callback",
            json={"code": "valid-code", "state": "valid-state"},
        )

        assert resp.status_code == 403
        assert "deactivated" in resp.json()["detail"]

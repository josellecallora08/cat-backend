"""Tests for Lark OAuth endpoints and service."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.lark_oauth import (
    LarkUserInfo,
    get_authorize_url,
    exchange_code_for_user_token,
    fetch_lark_user_info,
)


@pytest.fixture
async def client():
    """Async test client for the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# --- Unit Tests: Service Functions ---


class TestGetAuthorizeUrl:
    """Tests for authorize URL generation."""

    @patch("app.services.lark_oauth.settings")
    def test_builds_correct_url(self, mock_settings):
        """Authorize URL includes app_id, redirect_uri, response_type, and state."""
        mock_settings.lark_app_id = "test-app-id"
        mock_settings.lark_redirect_uri = "http://localhost:3000/auth/lark/callback"

        url = get_authorize_url("test-state-123")

        assert "https://open.larksuite.com/open-apis/authen/v1/authorize" in url
        assert "app_id=test-app-id" in url
        assert "redirect_uri=http://localhost:3000/auth/lark/callback" in url
        assert "response_type=code" in url
        assert "state=test-state-123" in url

    @patch("app.services.lark_oauth.settings")
    def test_includes_state_parameter(self, mock_settings):
        """State parameter is included for CSRF protection."""
        mock_settings.lark_app_id = "app-id"
        mock_settings.lark_redirect_uri = "http://example.com/callback"

        url = get_authorize_url("random-state")

        assert "state=random-state" in url


class TestExchangeCodeForUserToken:
    """Tests for token exchange with Lark."""

    @pytest.mark.asyncio
    @patch("app.services.lark_oauth._get_app_access_token")
    @patch("app.services.lark_oauth.httpx.AsyncClient")
    async def test_successful_token_exchange(
        self, mock_client_class, mock_get_app_token
    ):
        """Successful code exchange returns user_access_token."""
        mock_get_app_token.return_value = "app-token-123"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "code": 0,
            "data": {"access_token": "user-token-456"},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        token = await exchange_code_for_user_token("auth-code-789")

        assert token == "user-token-456"

    @pytest.mark.asyncio
    @patch("app.services.lark_oauth._get_app_access_token")
    @patch("app.services.lark_oauth.httpx.AsyncClient")
    async def test_failed_token_exchange_raises_value_error(
        self, mock_client_class, mock_get_app_token
    ):
        """When Lark rejects the code, raises ValueError."""
        mock_get_app_token.return_value = "app-token-123"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "code": 10014,
            "msg": "Invalid authorization code",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        with pytest.raises(ValueError, match="Lark token exchange error"):
            await exchange_code_for_user_token("bad-code")


class TestFetchLarkUserInfo:
    """Tests for fetching user info from Lark."""

    @pytest.mark.asyncio
    @patch("app.services.lark_oauth.httpx.AsyncClient")
    async def test_successful_user_info_fetch(self, mock_client_class):
        """Successful fetch returns LarkUserInfo dataclass."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "code": 0,
            "data": {
                "open_id": "ou_abc123",
                "union_id": "on_def456",
                "name": "Test User",
                "email": "test@company.com",
                "avatar_url": "https://example.com/avatar.png",
            },
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        info = await fetch_lark_user_info("user-token-456")

        assert info.open_id == "ou_abc123"
        assert info.union_id == "on_def456"
        assert info.name == "Test User"
        assert info.email == "test@company.com"

    @pytest.mark.asyncio
    @patch("app.services.lark_oauth.httpx.AsyncClient")
    async def test_failed_user_info_raises_value_error(self, mock_client_class):
        """When user info fetch fails, raises ValueError."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "code": 10012,
            "msg": "Invalid token",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        with pytest.raises(ValueError, match="Lark user info error"):
            await fetch_lark_user_info("invalid-token")


# --- Integration Tests: API Endpoints ---


class TestLarkAuthorizeEndpoint:
    """Tests for GET /api/auth/lark/authorize."""

    @pytest.mark.asyncio
    @patch("app.services.lark_oauth.settings")
    @patch("app.api.auth.settings")
    async def test_returns_authorize_url_and_state(
        self, mock_api_settings, mock_svc_settings, client
    ):
        """Returns a valid authorize URL and state when Lark is configured."""
        mock_api_settings.lark_app_id = "test-app-id"
        mock_api_settings.google_client_id = ""
        mock_svc_settings.lark_app_id = "test-app-id"
        mock_svc_settings.lark_redirect_uri = "http://localhost:3000/auth/lark/callback"

        resp = await client.get("/api/auth/lark/authorize")

        assert resp.status_code == 200
        data = resp.json()
        assert "authorize_url" in data
        assert "state" in data
        assert "test-app-id" in data["authorize_url"]
        assert len(data["state"]) > 20

    @pytest.mark.asyncio
    @patch("app.api.auth.settings")
    async def test_returns_503_when_not_configured(self, mock_settings, client):
        """Returns 503 when Lark OAuth is not configured."""
        mock_settings.lark_app_id = ""
        mock_settings.google_client_id = ""

        resp = await client.get("/api/auth/lark/authorize")

        assert resp.status_code == 503
        assert "not configured" in resp.json()["detail"]


class TestLarkCallbackEndpoint:
    """Tests for POST /api/auth/lark/callback."""

    @pytest.mark.asyncio
    @patch("app.api.auth.settings")
    async def test_returns_503_when_not_configured(self, mock_settings, client):
        """Returns 503 when Lark OAuth is not configured."""
        mock_settings.lark_app_id = ""
        mock_settings.google_client_id = ""

        resp = await client.post(
            "/api/auth/lark/callback",
            json={"code": "test-code", "state": "test-state"},
        )

        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_rejects_empty_code(self, client):
        """Returns 422 for empty code parameter."""
        resp = await client.post(
            "/api/auth/lark/callback",
            json={"code": "", "state": "test-state"},
        )

        assert resp.status_code == 422

    @pytest.mark.asyncio
    @patch("app.api.auth.settings")
    @patch("app.api.auth.get_or_create_lark_user")
    @patch("app.api.auth.fetch_lark_user_info")
    @patch("app.api.auth.exchange_code_for_user_token")
    async def test_successful_callback_returns_token(
        self,
        mock_exchange,
        mock_fetch_info,
        mock_get_or_create,
        mock_settings,
        client,
    ):
        """Successful callback returns access_token and user info."""
        mock_settings.lark_app_id = "test-app-id"
        mock_settings.lark_app_secret = "test-secret"  # pragma: allowlist secret
        mock_settings.lark_redirect_uri = "http://localhost:3000/callback"
        mock_settings.google_client_id = ""

        mock_exchange.return_value = "user-access-token"
        mock_fetch_info.return_value = LarkUserInfo(
            open_id="ou_abc123",
            union_id="on_def456",
            name="Test Lark User",
            email="lark@company.com",
            avatar_url="",
        )

        mock_user = AsyncMock()
        mock_user.id = uuid4()
        mock_user.email = "lark@company.com"
        mock_user.full_name = "Test Lark User"
        mock_user.role = "agent"
        mock_user.user_type = None
        mock_user.is_active = True

        mock_get_or_create.return_value = (mock_user, "cat-jwt-token")

        resp = await client.post(
            "/api/auth/lark/callback",
            json={"code": "valid-code", "state": "valid-state"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["access_token"] == "cat-jwt-token"
        assert data["user"]["email"] == "lark@company.com"
        assert data["user"]["full_name"] == "Test Lark User"

    @pytest.mark.asyncio
    @patch("app.api.auth.settings")
    @patch("app.api.auth.exchange_code_for_user_token")
    async def test_invalid_code_returns_401(self, mock_exchange, mock_settings, client):
        """Returns 401 when Lark rejects the authorization code."""
        mock_settings.lark_app_id = "test-app-id"
        mock_settings.lark_app_secret = "test-secret"  # pragma: allowlist secret
        mock_settings.lark_redirect_uri = "http://localhost:3000/callback"
        mock_settings.google_client_id = ""

        mock_exchange.side_effect = ValueError(
            "Lark token exchange error: invalid code"
        )

        resp = await client.post(
            "/api/auth/lark/callback",
            json={"code": "invalid-code", "state": "test-state"},
        )

        assert resp.status_code == 401
        assert "Lark authentication failed" in resp.json()["detail"]

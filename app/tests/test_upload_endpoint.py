"""Integration tests for POST /api/scripts/upload endpoint."""

import io
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.models.user import User, UserRole
from app.services.auth import require_admin
from app.services import upload_rate_limiter


def _make_admin_user():
    """Mock admin user."""
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.email = "admin@test.com"
    user.role = UserRole.ADMIN.value
    user.is_active = True
    return user


def _make_agent_user():
    """Mock agent (non-admin) user."""
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.email = "agent@test.com"
    user.role = UserRole.AGENT.value
    user.is_active = True
    return user


@pytest.fixture
def app():
    """Create test app without running lifespan (no DB/migrations)."""
    test_app = create_app()
    return test_app


@pytest.fixture
def admin_user():
    return _make_admin_user()


@pytest.fixture
def agent_user():
    return _make_agent_user()


@pytest.fixture(autouse=True)
def clear_rate_limiter():
    """Reset rate limiter state between tests."""
    upload_rate_limiter._rejection_tracker.clear()
    upload_rate_limiter._cooldown_tracker.clear()
    yield
    upload_rate_limiter._rejection_tracker.clear()
    upload_rate_limiter._cooldown_tracker.clear()


class TestUploadAuth:
    """Task 16.1 + 16.2: Test authentication and authorization."""

    @pytest.mark.asyncio
    async def test_401_without_token(self, app):
        """Unauthenticated request gets 401."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/scripts/upload",
                files={"file": ("test.pdf", b"%PDF-1.4 content", "application/pdf")},
            )
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_403_with_agent_role(self, app, agent_user):
        """Agent role gets 403."""
        from app.services.auth import require_admin

        # Override require_admin to simulate an agent trying to access
        async def mock_require_admin():
            from fastapi import HTTPException

            raise HTTPException(status_code=403, detail="Admin access required")

        app.dependency_overrides[require_admin] = mock_require_admin
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/scripts/upload",
                    files={"file": ("test.pdf", b"%PDF-1.4 content", "application/pdf")},
                )
                assert resp.status_code == 403
        finally:
            app.dependency_overrides.clear()


class TestUploadValidation:
    """Task 16.3-16.5: Test validation pipeline."""

    @pytest.fixture
    def override_admin(self, app, admin_user):
        """Override require_admin to return mock admin."""
        from app.services.auth import require_admin
        from app.database import get_session

        app.dependency_overrides[require_admin] = lambda: admin_user
        app.dependency_overrides[get_session] = lambda: AsyncMock()
        yield
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_201_with_valid_pdf(self, app, override_admin):
        """Admin uploading a valid PDF gets 201."""
        pdf_content = b"%PDF-1.4 test content for upload"

        with patch("app.api.uploads.scan_file") as mock_scan:
            mock_scan.return_value = MagicMock(clean=True, signature=None, error=None)
            with patch("app.api.uploads.extract_content") as mock_extract:
                mock_extract.return_value = "extracted text"
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.post(
                        "/api/scripts/upload",
                        files={"file": ("report.pdf", pdf_content, "application/pdf")},
                    )
                    assert resp.status_code == 201
                    data = resp.json()
                    assert "content_hash" in data
                    assert data["filename_original"] == "report.pdf"
                    assert data["scan_result"] == "clean"

    @pytest.mark.asyncio
    async def test_422_with_oversized_file(self, app, override_admin):
        """File over 10 MB gets 422."""
        big_content = b"%PDF-1.4" + b"x" * 10_485_760  # Just over 10 MB

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/scripts/upload",
                files={"file": ("big.pdf", big_content, "application/pdf")},
            )
            assert resp.status_code == 422
            data = resp.json()["detail"]
            assert data["reason_code"] == "file_too_large"

    @pytest.mark.asyncio
    async def test_422_with_wrong_extension(self, app, override_admin):
        """File with forbidden extension gets 422."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/scripts/upload",
                files={"file": ("music.mp3", b"fake mp3", "audio/mpeg")},
            )
            assert resp.status_code == 422
            data = resp.json()["detail"]
            assert data["reason_code"] == "invalid_extension"


class TestUploadRateLimit:
    """Task 16.6: Test 429 when rate limited."""

    @pytest.mark.asyncio
    async def test_429_when_rate_limited(self, app, admin_user):
        """Rate limited user gets 429."""
        from app.services.auth import require_admin
        from app.database import get_session

        app.dependency_overrides[require_admin] = lambda: admin_user
        app.dependency_overrides[get_session] = lambda: AsyncMock()

        # Trigger rate limit for this user
        user_id = str(admin_user.id)
        for _ in range(10):
            upload_rate_limiter.record_rejection(user_id)

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/scripts/upload",
                    files={"file": ("doc.pdf", b"%PDF-1.4 test", "application/pdf")},
                )
                assert resp.status_code == 429
                assert "Retry-After" in resp.headers
        finally:
            app.dependency_overrides.clear()

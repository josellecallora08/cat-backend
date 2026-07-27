"""Integration tests for the upload API endpoints.

Covers: auth (401/403), validation pipeline, scenario_id validation,
extracted content storage (not ScriptContract), rate limiting,
GET status, GET list, and database error handling.
"""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.models.user import User, UserRole
from app.services import upload_rate_limiter


def _make_admin_user():
    """Mock admin user."""
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.email = "admin@test.com"
    user.role = UserRole.ADMIN.value
    user.is_active = True
    return user


@pytest.fixture
def app():
    """Create test app without running lifespan."""
    return create_app()


@pytest.fixture
def admin_user():
    return _make_admin_user()


@pytest.fixture(autouse=True)
def clear_rate_limiter():
    """Reset rate limiter state between tests."""
    upload_rate_limiter._rejection_tracker.clear()
    upload_rate_limiter._cooldown_tracker.clear()
    yield
    upload_rate_limiter._rejection_tracker.clear()
    upload_rate_limiter._cooldown_tracker.clear()


def _mock_db_session():
    """Create a properly mocked AsyncSession.

    db.add() is synchronous (MagicMock), while commit/refresh/execute are async.
    """
    mock_db = MagicMock()
    mock_db.add = MagicMock()  # synchronous
    mock_db.commit = AsyncMock()
    mock_db.refresh = AsyncMock()
    mock_db.rollback = AsyncMock()
    mock_db.execute = AsyncMock()
    return mock_db


class TestUploadAuth:
    """Authentication and authorization for all upload endpoints."""

    @pytest.mark.asyncio
    async def test_upload_401_without_token(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/scripts/upload",
                files={"file": ("test.pdf", b"%PDF-1.4 content", "application/pdf")},
            )
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_upload_403_non_admin(self, app):
        from app.services.auth import require_admin

        async def mock_reject():
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail="Admin access required")

        app.dependency_overrides[require_admin] = mock_reject
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

    @pytest.mark.asyncio
    async def test_get_status_401_without_token(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/api/scripts/uploads/{uuid.uuid4()}/status")
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_get_list_401_without_token(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/scripts/uploads")
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_get_status_403_non_admin(self, app):
        from app.services.auth import require_admin

        async def mock_reject():
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail="Admin access required")

        app.dependency_overrides[require_admin] = mock_reject
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(f"/api/scripts/uploads/{uuid.uuid4()}/status")
                assert resp.status_code == 403
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_get_list_403_non_admin(self, app):
        from app.services.auth import require_admin

        async def mock_reject():
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail="Admin access required")

        app.dependency_overrides[require_admin] = mock_reject
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/scripts/uploads")
                assert resp.status_code == 403
        finally:
            app.dependency_overrides.clear()


class TestUploadValidation:
    """Validation pipeline tests."""

    @pytest.fixture
    def override_admin_and_db(self, app, admin_user):
        from app.services.auth import require_admin
        from app.database import get_session

        mock_db = _mock_db_session()
        # Make refresh set created_at on the record
        async def _refresh(obj):
            if not hasattr(obj, 'created_at') or obj.created_at is None:
                obj.created_at = datetime.now(timezone.utc)
        mock_db.refresh = AsyncMock(side_effect=_refresh)

        app.dependency_overrides[require_admin] = lambda: admin_user
        app.dependency_overrides[get_session] = lambda: mock_db
        yield mock_db
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_201_valid_pdf_no_scenario(self, app, override_admin_and_db):
        """Valid PDF without scenario_id succeeds with pending status."""
        pdf_content = b"%PDF-1.4 test content for upload"

        with patch("app.api.uploads.validate_pdf_not_encrypted") as mock_pdf_check:
            mock_pdf_check.return_value = (True, None)
            with patch("app.api.uploads.scan_file") as mock_scan:
                mock_scan.return_value = MagicMock(clean=True, signature=None, error=None)
                with patch("app.api.uploads.extract_content") as mock_extract:
                    mock_extract.return_value = "extracted text from pdf"
                    transport = ASGITransport(app=app)
                    async with AsyncClient(transport=transport, base_url="http://test") as client:
                        resp = await client.post(
                            "/api/scripts/upload",
                            files={"file": ("report.pdf", pdf_content, "application/pdf")},
                        )
                        assert resp.status_code == 201
                        data = resp.json()
                        assert data["filename_original"] == "report.pdf"
                        assert data["scan_result"] == "clean"
                        assert data["extraction_status"] == "completed"
                        assert data["script_id"] is None
                        assert data["scenario_id"] is None
                        assert "pending" in data["processing_notes"].lower()

    @pytest.mark.asyncio
    async def test_201_extracted_text_not_parsed_as_json(self, app, override_admin_and_db):
        """Ordinary text content is stored as-is, NOT parsed as ScriptContract."""
        txt_content = b"Hello this is a normal training document with no JSON structure"

        with patch("app.api.uploads.scan_file") as mock_scan:
            mock_scan.return_value = MagicMock(clean=True, signature=None, error=None)
            with patch("app.api.uploads.extract_content") as mock_extract:
                mock_extract.return_value = "Hello this is a normal training document"
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.post(
                        "/api/scripts/upload",
                        files={"file": ("doc.txt", txt_content, "text/plain")},
                    )
                    assert resp.status_code == 201
                    data = resp.json()
                    # Verify it does NOT attempt ScriptContract parsing
                    assert data["script_id"] is None
                    assert data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_422_invalid_scenario_id(self, app, admin_user):
        """Non-existent scenario_id returns 422."""
        from app.services.auth import require_admin
        from app.database import get_session

        mock_db = _mock_db_session()
        # execute returns None for scenario lookup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        app.dependency_overrides[require_admin] = lambda: admin_user
        app.dependency_overrides[get_session] = lambda: mock_db

        try:
            transport = ASGITransport(app=app)
            fake_scenario = str(uuid.uuid4())
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/scripts/upload",
                    data={"scenario_id": fake_scenario},
                    files={"file": ("report.pdf", b"%PDF-1.4 test", "application/pdf")},
                )
                assert resp.status_code == 422
                data = resp.json()["detail"]
                assert "does not exist" in data["message"]
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_201_valid_scenario_id_stores_link(self, app, admin_user):
        """Valid scenario_id is stored on the upload record."""
        from app.services.auth import require_admin
        from app.database import get_session

        mock_db = _mock_db_session()
        fake_scenario_id = uuid.uuid4()

        # First execute call = scenario lookup (found)
        mock_scenario = MagicMock()
        mock_scenario.id = fake_scenario_id
        mock_scenario_result = MagicMock()
        mock_scenario_result.scalar_one_or_none.return_value = mock_scenario
        mock_db.execute = AsyncMock(return_value=mock_scenario_result)

        async def _refresh(obj):
            if not hasattr(obj, 'created_at') or obj.created_at is None:
                obj.created_at = datetime.now(timezone.utc)
        mock_db.refresh = AsyncMock(side_effect=_refresh)

        app.dependency_overrides[require_admin] = lambda: admin_user
        app.dependency_overrides[get_session] = lambda: mock_db

        try:
            with patch("app.api.uploads.validate_pdf_not_encrypted") as mock_pdf:
                mock_pdf.return_value = (True, None)
                with patch("app.api.uploads.scan_file") as mock_scan:
                    mock_scan.return_value = MagicMock(clean=True, signature=None, error=None)
                    with patch("app.api.uploads.extract_content") as mock_extract:
                        mock_extract.return_value = "extracted content"
                        transport = ASGITransport(app=app)
                        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            resp = await client.post(
                                "/api/scripts/upload",
                                data={"scenario_id": str(fake_scenario_id)},
                                files={"file": ("report.pdf", b"%PDF-1.4 test", "application/pdf")},
                            )
                            assert resp.status_code == 201
                            data = resp.json()
                            assert data["scenario_id"] == str(fake_scenario_id)
                            # Verify db.add was called with the upload record
                            mock_db.add.assert_called_once()
                            record = mock_db.add.call_args[0][0]
                            assert record.scenario_id == fake_scenario_id
                            assert record.extracted_content == "extracted content"
                            assert record.script_id is None
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_422_wrong_extension(self, app, override_admin_and_db):
        """Forbidden extension gets 422."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/scripts/upload",
                files={"file": ("music.mp3", b"fake mp3", "audio/mpeg")},
            )
            assert resp.status_code == 422
            data = resp.json()["detail"]
            assert data["reason_code"] == "invalid_extension"

    @pytest.mark.asyncio
    async def test_422_oversized_file(self, app, override_admin_and_db):
        """File over 10 MB gets 422."""
        big_content = b"%PDF-1.4" + b"x" * 10_485_760
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/scripts/upload",
                files={"file": ("big.pdf", big_content, "application/pdf")},
            )
            assert resp.status_code == 422
            data = resp.json()["detail"]
            assert data["reason_code"] == "file_too_large"


class TestDatabaseErrorHandling:
    """Test db.rollback() on failure and proper 500 responses."""

    @pytest.mark.asyncio
    async def test_500_on_db_commit_failure(self, app, admin_user):
        """Database commit failure returns 500 and calls rollback."""
        from app.services.auth import require_admin
        from app.database import get_session

        mock_db = _mock_db_session()
        mock_db.commit = AsyncMock(side_effect=Exception("DB connection lost"))

        async def _refresh(obj):
            obj.created_at = datetime.now(timezone.utc)
        mock_db.refresh = AsyncMock(side_effect=_refresh)

        app.dependency_overrides[require_admin] = lambda: admin_user
        app.dependency_overrides[get_session] = lambda: mock_db

        try:
            with patch("app.api.uploads.validate_pdf_not_encrypted") as mock_pdf:
                mock_pdf.return_value = (True, None)
                with patch("app.api.uploads.scan_file") as mock_scan:
                    mock_scan.return_value = MagicMock(clean=True, signature=None, error=None)
                    with patch("app.api.uploads.extract_content") as mock_extract:
                        mock_extract.return_value = "text"
                        transport = ASGITransport(app=app)
                        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            resp = await client.post(
                                "/api/scripts/upload",
                                files={"file": ("doc.pdf", b"%PDF-1.4 x", "application/pdf")},
                            )
                            assert resp.status_code == 500
                            assert "persist" in resp.json()["detail"].lower()
                            mock_db.rollback.assert_awaited_once()
        finally:
            app.dependency_overrides.clear()


class TestRateLimit:
    """Rate limiting returns 429 with Retry-After."""

    @pytest.mark.asyncio
    async def test_429_when_rate_limited(self, app, admin_user):
        from app.services.auth import require_admin
        from app.database import get_session

        app.dependency_overrides[require_admin] = lambda: admin_user
        app.dependency_overrides[get_session] = lambda: _mock_db_session()

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


class TestGetUploadStatus:
    """GET /uploads/{id}/status endpoint."""

    @pytest.mark.asyncio
    async def test_404_nonexistent_upload(self, app, admin_user):
        from app.services.auth import require_admin
        from app.database import get_session

        mock_db = _mock_db_session()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        app.dependency_overrides[require_admin] = lambda: admin_user
        app.dependency_overrides[get_session] = lambda: mock_db

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(f"/api/scripts/uploads/{uuid.uuid4()}/status")
                assert resp.status_code == 404
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_200_existing_upload(self, app, admin_user):
        from app.services.auth import require_admin
        from app.database import get_session

        mock_db = _mock_db_session()
        upload_id = uuid.uuid4()
        now = datetime.now(timezone.utc)

        mock_upload = MagicMock()
        mock_upload.id = upload_id
        mock_upload.filename_original = "report.pdf"
        mock_upload.mime_type = "application/pdf"
        mock_upload.file_size_bytes = 1024
        mock_upload.content_hash = "abc123"
        mock_upload.storage_key = "uuid-file.pdf"
        mock_upload.uploaded_by = admin_user.id
        mock_upload.scan_status = "clean"
        mock_upload.scan_signature = None
        mock_upload.extraction_status = "completed"
        mock_upload.extraction_error = None
        mock_upload.status = "completed"
        mock_upload.script_id = None
        mock_upload.scenario_id = None
        mock_upload.created_at = now
        mock_upload.updated_at = now
        mock_upload.quarantine_expires_at = now + timedelta(hours=24)
        mock_upload.deleted_at = None

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_upload
        mock_db.execute = AsyncMock(return_value=mock_result)

        app.dependency_overrides[require_admin] = lambda: admin_user
        app.dependency_overrides[get_session] = lambda: mock_db

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(f"/api/scripts/uploads/{upload_id}/status")
                assert resp.status_code == 200
                data = resp.json()
                assert data["id"] == str(upload_id)
                assert data["filename_original"] == "report.pdf"
                assert data["status"] == "completed"
                assert data["scan_status"] == "clean"
        finally:
            app.dependency_overrides.clear()


class TestListUploads:
    """GET /uploads list with pagination."""

    @pytest.mark.asyncio
    async def test_200_empty_list(self, app, admin_user):
        from app.services.auth import require_admin
        from app.database import get_session

        mock_db = _mock_db_session()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute = AsyncMock(return_value=mock_result)

        app.dependency_overrides[require_admin] = lambda: admin_user
        app.dependency_overrides[get_session] = lambda: mock_db

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/scripts/uploads")
                assert resp.status_code == 200
                assert resp.json() == []
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_200_with_items_and_pagination(self, app, admin_user):
        from app.services.auth import require_admin
        from app.database import get_session

        mock_db = _mock_db_session()
        now = datetime.now(timezone.utc)

        items = []
        for i in range(3):
            item = MagicMock()
            item.id = uuid.uuid4()
            item.filename_original = f"file{i}.pdf"
            item.mime_type = "application/pdf"
            item.file_size_bytes = 1000 + i
            item.status = "completed"
            item.script_id = None
            item.scenario_id = None
            item.created_at = now
            items.append(item)

        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = items
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute = AsyncMock(return_value=mock_result)

        app.dependency_overrides[require_admin] = lambda: admin_user
        app.dependency_overrides[get_session] = lambda: mock_db

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/scripts/uploads?limit=10&offset=0")
                assert resp.status_code == 200
                data = resp.json()
                assert len(data) == 3
                assert data[0]["filename_original"] == "file0.pdf"
        finally:
            app.dependency_overrides.clear()

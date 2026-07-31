"""Tests for the Script Registry management API endpoints."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_session as get_db_session
from app.main import app
from app.services.auth import get_current_user, require_admin
from app.services.script_validator import ScriptValidationError


def _mock_admin_user():
    """Return a mock admin user for dependency override."""
    user = MagicMock()
    user.id = uuid.uuid4()
    user.email = "admin@test.com"
    user.full_name = "Test Admin"
    user.role = "admin"
    return user


def _mock_non_admin_user():
    """Return a mock non-admin (agent) user for dependency override."""
    user = MagicMock()
    user.id = uuid.uuid4()
    user.email = "agent@test.com"
    user.full_name = "Test Agent"
    user.role = "agent"
    return user


def _make_script(
    name: str = "Test Script",
    status: str = "draft",
    format: str = "json",
    scenario_id: uuid.UUID | None = None,
    draft_content: dict | None = None,
    current_version_id: uuid.UUID | None = None,
) -> MagicMock:
    """Create a mock Script ORM object with required attributes."""
    script = MagicMock()
    script.id = uuid.uuid4()
    script.name = name
    script.scenario_id = scenario_id or uuid.uuid4()
    script.status = status
    script.format = format
    script.draft_content = (
        draft_content if draft_content is not None else {"opening_response": "Hello"}
    )
    script.current_version_id = current_version_id
    script.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    script.updated_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return script


def _make_version(
    script_id: uuid.UUID | None = None,
    version_number: int = 1,
    content: dict | None = None,
    published_by: uuid.UUID | None = None,
) -> MagicMock:
    """Create a mock ScriptVersion ORM object with required attributes."""
    version = MagicMock()
    version.id = uuid.uuid4()
    version.script_id = script_id or uuid.uuid4()
    version.version_number = version_number
    version.content = content if content is not None else {"opening_response": "Hello"}
    version.published_by = published_by or uuid.uuid4()
    version.published_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return version


def _mock_db_with_count(total: int):
    """Create a mock AsyncSession whose execute() returns a scalar count."""
    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar.return_value = total
    mock_db.execute = AsyncMock(return_value=mock_result)
    return mock_db


def _mock_db_with_scalars(values):
    """Create a mock AsyncSession whose execute() returns scalars().all() of values."""
    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = values
    mock_result.scalars.return_value = mock_scalars
    mock_db.execute = AsyncMock(return_value=mock_result)
    return mock_db


@pytest.fixture
def admin_override():
    """Override require_admin dependency for the duration of the test."""
    mock_user = _mock_admin_user()
    app.dependency_overrides[require_admin] = lambda: mock_user
    yield mock_user
    app.dependency_overrides.clear()


@pytest.fixture
async def client(admin_override):
    """Async test client with admin auth overridden."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def unauth_client():
    """Async test client without auth overrides (for auth tests)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestCreateScript:
    """Tests for POST /api/scripts."""

    async def test_creates_script_successfully(self, client):
        script = _make_script(name="New Script")
        with patch(
            "app.api.scripts.create_draft",
            new_callable=AsyncMock,
            return_value=script,
        ):
            response = await client.post(
                "/api/scripts",
                json={
                    "name": "New Script",
                    "scenario_id": str(script.scenario_id),
                    "format": "json",
                    "raw_definition": '{"opening_response": "Hello"}',
                },
            )
            assert response.status_code == 201
            data = response.json()
            assert data["id"] == str(script.id)
            assert data["name"] == "New Script"
            assert data["status"] == "draft"
            assert data["format"] == "json"
            assert data["scenario_id"] == str(script.scenario_id)
            assert data["draft_content"] == script.draft_content
            assert data["current_version_id"] is None

    async def test_returns_422_for_invalid_definition(self, client):
        with patch(
            "app.api.scripts.create_draft",
            new_callable=AsyncMock,
            side_effect=ScriptValidationError(
                [{"loc": ("debtor_persona",), "msg": "field required"}]
            ),
        ):
            response = await client.post(
                "/api/scripts",
                json={
                    "name": "Bad Script",
                    "scenario_id": str(uuid.uuid4()),
                    "format": "json",
                    "raw_definition": "{}",
                },
            )
            assert response.status_code == 422

    async def test_requires_admin_auth(self, unauth_client):
        response = await unauth_client.post(
            "/api/scripts",
            json={
                "name": "Unauthorized",
                "scenario_id": str(uuid.uuid4()),
                "format": "json",
                "raw_definition": "{}",
            },
        )
        assert response.status_code in (401, 403)

    async def test_returns_403_for_non_admin_user(self, unauth_client):
        # Override the underlying auth dependency (not require_admin itself)
        # so require_admin's real role check executes and rejects.
        mock_user = _mock_non_admin_user()
        app.dependency_overrides[get_current_user] = lambda: mock_user
        try:
            response = await unauth_client.post(
                "/api/scripts",
                json={
                    "name": "Non Admin",
                    "scenario_id": str(uuid.uuid4()),
                    "format": "json",
                    "raw_definition": "{}",
                },
            )
            assert response.status_code == 403
        finally:
            app.dependency_overrides.clear()


class TestListScripts:
    """Tests for GET /api/scripts."""

    async def test_returns_paginated_list(self, client):
        script = _make_script(name="Script 1")
        mock_db = _mock_db_with_count(1)
        app.dependency_overrides[get_db_session] = lambda: mock_db
        try:
            with patch(
                "app.api.scripts.list_scripts",
                new_callable=AsyncMock,
                return_value=[script],
            ):
                response = await client.get("/api/scripts")
                assert response.status_code == 200
                data = response.json()
                assert data["total"] == 1
                assert data["page"] == 1
                assert data["page_size"] == 15
                assert data["total_pages"] == 1
                assert len(data["items"]) == 1
                assert data["items"][0]["name"] == "Script 1"
        finally:
            app.dependency_overrides.pop(get_db_session, None)


class TestGetScript:
    """Tests for GET /api/scripts/{script_id}."""

    async def test_returns_script_detail(self, client):
        script = _make_script(name="Detail Script")
        with patch(
            "app.api.scripts.get_script",
            new_callable=AsyncMock,
            return_value=script,
        ):
            response = await client.get(f"/api/scripts/{script.id}")
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Detail Script"
            assert data["id"] == str(script.id)

    async def test_returns_404_for_unknown_script_id(self, client):
        with patch(
            "app.api.scripts.get_script",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = await client.get(f"/api/scripts/{uuid.uuid4()}")
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()


class TestUpdateScript:
    """Tests for PUT /api/scripts/{script_id}."""

    async def test_updates_draft_successfully(self, client):
        script = _make_script(name="Updated Script")
        with patch(
            "app.api.scripts.update_draft",
            new_callable=AsyncMock,
            return_value=script,
        ):
            response = await client.put(
                f"/api/scripts/{script.id}",
                json={
                    "raw_definition": '{"opening_response": "Hi"}',
                    "format": "json",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Updated Script"

    async def test_returns_404_for_unknown_script_id(self, client):
        with patch(
            "app.api.scripts.update_draft",
            new_callable=AsyncMock,
            side_effect=ValueError("Script not found"),
        ):
            response = await client.put(
                f"/api/scripts/{uuid.uuid4()}",
                json={
                    "raw_definition": '{"opening_response": "Hi"}',
                    "format": "json",
                },
            )
            assert response.status_code == 404

    async def test_returns_422_for_invalid_update(self, client):
        with patch(
            "app.api.scripts.update_draft",
            new_callable=AsyncMock,
            side_effect=ScriptValidationError(
                [{"loc": ("debtor_persona",), "msg": "field required"}]
            ),
        ):
            response = await client.put(
                f"/api/scripts/{uuid.uuid4()}",
                json={
                    "raw_definition": "{}",
                    "format": "json",
                },
            )
            assert response.status_code == 422


class TestDeleteScript:
    """Tests for DELETE /api/scripts/{script_id}."""

    async def test_deletes_script_successfully(self, client):
        with patch(
            "app.api.scripts.delete_script",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = await client.delete(f"/api/scripts/{uuid.uuid4()}")
            assert response.status_code == 204

    async def test_returns_404_for_unknown_script_id(self, client):
        with patch(
            "app.api.scripts.delete_script",
            new_callable=AsyncMock,
            side_effect=ValueError("Script not found"),
        ):
            response = await client.delete(f"/api/scripts/{uuid.uuid4()}")
            assert response.status_code == 404


class TestPublishScript:
    """Tests for POST /api/scripts/{script_id}/publish."""

    async def test_publishes_script_successfully(self, client):
        script = _make_script(name="Published Script", status="published")
        with patch(
            "app.api.scripts.publish",
            new_callable=AsyncMock,
            return_value=None,
        ), patch(
            "app.api.scripts.get_script",
            new_callable=AsyncMock,
            return_value=script,
        ):
            response = await client.post(f"/api/scripts/{script.id}/publish")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "published"

    async def test_returns_422_for_invalid_publish_attempt(self, client):
        with patch(
            "app.api.scripts.publish",
            new_callable=AsyncMock,
            side_effect=ScriptValidationError(
                [{"loc": ("expected_replies",), "msg": "field required"}]
            ),
        ):
            response = await client.post(f"/api/scripts/{uuid.uuid4()}/publish")
            assert response.status_code == 422

    async def test_returns_404_for_unknown_script_id(self, client):
        with patch(
            "app.api.scripts.publish",
            new_callable=AsyncMock,
            side_effect=ValueError("Script not found"),
        ):
            response = await client.post(f"/api/scripts/{uuid.uuid4()}/publish")
            assert response.status_code == 404


class TestUnpublishScript:
    """Tests for POST /api/scripts/{script_id}/unpublish."""

    async def test_unpublishes_script_successfully(self, client):
        script = _make_script(name="Unpublished Script", status="unpublished")
        with patch(
            "app.api.scripts.unpublish",
            new_callable=AsyncMock,
            return_value=script,
        ):
            response = await client.post(f"/api/scripts/{script.id}/unpublish")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unpublished"

    async def test_returns_404_for_unknown_script_id(self, client):
        with patch(
            "app.api.scripts.unpublish",
            new_callable=AsyncMock,
            side_effect=ValueError("Script not found"),
        ):
            response = await client.post(f"/api/scripts/{uuid.uuid4()}/unpublish")
            assert response.status_code == 404


class TestListScriptVersions:
    """Tests for GET /api/scripts/{script_id}/versions."""

    async def test_returns_version_list(self, client):
        script = _make_script(name="Versioned Script")
        version = _make_version(script_id=script.id, version_number=1)
        mock_db = _mock_db_with_scalars([version])
        app.dependency_overrides[get_db_session] = lambda: mock_db
        try:
            with patch(
                "app.api.scripts.get_script",
                new_callable=AsyncMock,
                return_value=script,
            ):
                response = await client.get(f"/api/scripts/{script.id}/versions")
                assert response.status_code == 200
                data = response.json()
                assert len(data) == 1
                assert data[0]["version_number"] == 1
                assert data[0]["id"] == str(version.id)
        finally:
            app.dependency_overrides.pop(get_db_session, None)

    async def test_returns_404_for_unknown_script_id(self, client):
        with patch(
            "app.api.scripts.get_script",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = await client.get(f"/api/scripts/{uuid.uuid4()}/versions")
            assert response.status_code == 404

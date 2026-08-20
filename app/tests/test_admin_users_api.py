"""Tests for filtered admin user listing."""

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.auth import require_admin


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


def _user(**overrides):
    values = {
        "id": uuid.uuid4(),
        "email": "agent@example.com",
        "full_name": "Agent Example",
        "role": "user",
        "user_type": "agent",
        "is_active": True,
        "auth_provider": "local",
        "created_at": datetime.now(UTC),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
async def test_admin_user_filters_are_forwarded_and_shape_is_preserved(client):
    admin = _user(role="admin", user_type=None)
    app.dependency_overrides[require_admin] = lambda: admin
    users = [_user()]

    with patch(
        "app.api.admin_users.UserService.list_users",
        new_callable=AsyncMock,
        return_value=users,
    ) as list_users:
        response = await client.get(
            "/api/admin/users?search=agent&role=user&user_type=agent&is_active=true"
        )

    assert response.status_code == 200
    assert response.json()[0]["email"] == "agent@example.com"
    list_users.assert_awaited_once_with(
        search="agent", role="user", user_type="agent", is_active=True
    )


@pytest.mark.asyncio
async def test_admin_user_filters_reject_invalid_values(client):
    admin = _user(role="admin", user_type=None)
    app.dependency_overrides[require_admin] = lambda: admin

    for query in ("role=manager", "user_type=manager", "is_active=maybe"):
        response = await client.get(f"/api/admin/users?{query}")
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_admin_user_listing_keeps_authentication_behavior(client):
    response = await client.get("/api/admin/users")
    assert response.status_code == 401

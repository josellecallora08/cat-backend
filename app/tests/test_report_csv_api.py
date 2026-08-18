"""Focused HTTP tests for the authorized report CSV endpoint."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_session as get_db_session
from app.main import app
from app.services.auth import require_auth


@pytest.fixture
def client() -> AsyncClient:
    """Build an HTTP client for the FastAPI application."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture(autouse=True)
def clear_overrides():
    """Clear dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


def _override_db(db):
    async def override():
        return db

    return override


@pytest.mark.asyncio
async def test_report_csv_returns_attachment_with_safe_content(client: AsyncClient) -> None:
    """An authorized report is returned as UTF-8 CSV without changing state."""
    session_id = uuid4()
    user = SimpleNamespace(id=uuid4(), role="admin", user_type=None)
    report = SimpleNamespace(session=SimpleNamespace(id=session_id))
    db = AsyncMock()
    app.dependency_overrides[require_auth] = lambda: user
    app.dependency_overrides[get_db_session] = _override_db(db)

    with (
        patch(
            "app.api.sessions.ReportService.get_report",
            new_callable=AsyncMock,
            return_value=report,
        ),
        patch("app.api.sessions.serialize_report_csv", return_value="session_id\r\n"),
    ):
        response = await client.get(f"/api/sessions/{session_id}/report.csv")

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert response.headers["content-disposition"] == (
        f'attachment; filename="report-{session_id}.csv"'
    )
    assert response.text == "session_id\r\n"


@pytest.mark.asyncio
async def test_report_csv_denies_non_admin_before_report_lookup(client: AsyncClient) -> None:
    """Non-admin CSV requests return a safe 403 without touching report data."""
    session_id = uuid4()
    user = SimpleNamespace(id=uuid4(), role="user", user_type="agent")
    app.dependency_overrides[require_auth] = lambda: user
    app.dependency_overrides[get_db_session] = _override_db(AsyncMock())

    with (
        patch("app.api.sessions.ReportService.get_report", new_callable=AsyncMock) as get_report,
        patch("app.api.sessions.serialize_report_csv") as serialize,
    ):
        response = await client.get(f"/api/sessions/{session_id}/report.csv")

    assert response.status_code == 403
    assert response.json() == {"detail": "Admin access required"}
    assert "session_id" not in response.text
    get_report.assert_not_awaited()
    serialize.assert_not_called()


@pytest.mark.asyncio
async def test_report_csv_denies_non_admin_for_missing_session(client: AsyncClient) -> None:
    """A missing session does not alter the non-admin authorization response."""
    user = SimpleNamespace(id=uuid4(), role="user", user_type="trainer")
    app.dependency_overrides[require_auth] = lambda: user
    app.dependency_overrides[get_db_session] = _override_db(AsyncMock())

    response = await client.get(f"/api/sessions/{uuid4()}/report.csv")

    assert response.status_code == 403
    assert response.json() == {"detail": "Admin access required"}
    assert "report" not in response.text.lower()


@pytest.mark.asyncio
async def test_report_csv_requires_authentication(client: AsyncClient) -> None:
    """Unauthenticated CSV requests are rejected before report generation."""
    app.dependency_overrides[get_db_session] = _override_db(AsyncMock())

    response = await client.get(f"/api/sessions/{uuid4()}/report.csv")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_report_csv_hides_generation_failure(client: AsyncClient) -> None:
    """Unexpected export failures return a generic message without internals."""
    session_id = uuid4()
    user = SimpleNamespace(id=uuid4(), role="admin", user_type=None)
    app.dependency_overrides[require_auth] = lambda: user
    app.dependency_overrides[get_db_session] = _override_db(AsyncMock())

    with patch(
        "app.api.sessions.ReportService.get_report",
        new_callable=AsyncMock,
        side_effect=RuntimeError("SELECT secret FROM users"),
    ):
        response = await client.get(f"/api/sessions/{session_id}/report.csv")

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "The report export is temporarily unavailable. Please try again later."
    )
    assert "SELECT" not in response.text

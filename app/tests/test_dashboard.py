"""Regression tests for the authenticated dashboard score-history API."""

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.database import get_session
from app.main import app
from app.services.auth import require_auth


def _session_list_db(total=0, rows=None):
    db = AsyncMock()
    count_result = MagicMock()
    count_result.scalar_one.return_value = total
    page_result = MagicMock()
    page_result.all.return_value = rows or []
    db.execute.side_effect = [count_result, page_result]
    return db


def _admin_user():
    user = MagicMock()
    user.id = uuid.uuid4()
    user.role = "admin"
    user.user_type = None
    return user


def _evaluation(score: float, created_at: datetime, category_scores=None):
    evaluation = MagicMock()
    evaluation.overall_score = score
    evaluation.created_at = created_at
    evaluation.category_scores = category_scores or []
    evaluation.is_too_short = False
    return evaluation


def _db_returning(evaluations):
    db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = evaluations
    db.execute.return_value = result
    return db


def _dashboard_db():
    db = AsyncMock()
    results = []
    for value in (0, 0, 0, 0, None, [], 0):
        result = MagicMock()
        result.scalar_one.return_value = value
        result.scalars.return_value.all.return_value = value if isinstance(value, list) else []
        results.append(result)

    recent_result = MagicMock()
    recent_result.all.return_value = []
    trend_result = MagicMock()
    trend_result.scalars.return_value.all.return_value = []
    leaderboard_result = MagicMock()
    leaderboard_result.all.return_value = []
    db.execute.side_effect = [*results, recent_result, trend_result, leaderboard_result]
    return db


@pytest.fixture
def authenticated_dashboard_dependencies():
    admin = _admin_user()
    app.dependency_overrides[require_auth] = lambda: admin
    yield admin
    app.dependency_overrides.clear()


@pytest.fixture
async def dashboard_client(authenticated_dashboard_dependencies):
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_admin_score_history_is_chronological(dashboard_client):
    older = _evaluation(
        62.0,
        datetime(2025, 1, 1, tzinfo=UTC),
        [{"category": "compliance", "score": 60}],
    )
    newer = _evaluation(84.0, datetime(2025, 1, 2, tzinfo=UTC))
    app.dependency_overrides[get_session] = lambda: _db_returning([older, newer])
    try:
        response = await dashboard_client.get("/api/dashboard/score-history")
    finally:
        app.dependency_overrides.pop(get_session, None)

    assert response.status_code == 200
    assert response.json() == [
        {
            "session_number": 1,
            "overall_score": 62.0,
            "call_opening": None,
            "compliance": 60.0,
            "empathy_communication": None,
            "negotiation_resolution": None,
            "date": "2025-01-01T00:00:00+00:00",
        },
        {
            "session_number": 2,
            "overall_score": 84.0,
            "call_opening": None,
            "compliance": None,
            "empathy_communication": None,
            "negotiation_resolution": None,
            "date": "2025-01-02T00:00:00+00:00",
        },
    ]


@pytest.mark.asyncio
async def test_score_history_handles_nullable_malformed_and_dynamic_rubric_categories(
    dashboard_client,
):
    evaluation = _evaluation(
        77.5,
        datetime(2025, 3, 1, tzinfo=UTC),
        [
            {"category": "call_opening", "score": "81"},
            {"category": "dynamic_rubric_block", "score": 93},
            {"category": "compliance", "score": "not-a-score"},
            {"category": "empathy_communication", "score": None},
            "malformed category block",
        ],
    )
    nullable_evaluation = _evaluation(68.0, datetime(2025, 3, 2, tzinfo=UTC))
    nullable_evaluation.category_scores = None
    app.dependency_overrides[get_session] = lambda: _db_returning([evaluation, nullable_evaluation])
    try:
        response = await dashboard_client.get("/api/dashboard/score-history")
    finally:
        app.dependency_overrides.pop(get_session, None)

    assert response.status_code == 200
    assert response.json()[0] == {
        "session_number": 1,
        "overall_score": 77.5,
        "call_opening": 81.0,
        "compliance": None,
        "empathy_communication": None,
        "negotiation_resolution": None,
        "date": "2025-03-01T00:00:00+00:00",
    }
    assert response.json()[1]["compliance"] is None


@pytest.mark.asyncio
async def test_session_date_filters_apply_to_count_and_items(dashboard_client):
    from app.models import Session

    session = MagicMock(spec=Session)
    session.id = uuid.uuid4()
    session.created_at = datetime(2026, 8, 7, 23, 59, 59, tzinfo=UTC)
    session.status = "completed"
    session.persona_context = {"name": "Boundary"}
    db = _session_list_db(
        total=1,
        rows=[
            (
                session,
                None,
                SimpleNamespace(name="Scenario"),
                SimpleNamespace(full_name="Agent", email="agent@example.com"),
            )
        ],
    )
    app.dependency_overrides[get_session] = lambda: db
    try:
        response = await dashboard_client.get(
            "/api/dashboard/sessions?start_date=2026-08-01&end_date=2026-08-07"
        )
    finally:
        app.dependency_overrides.pop(get_session, None)

    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert len(response.json()["items"]) == 1
    assert db.execute.await_count == 2
    count_sql = str(db.execute.await_args_list[0].args[0])
    page_sql = str(db.execute.await_args_list[1].args[0])
    assert count_sql.count("sessions.created_at") == 2
    assert page_sql.count("sessions.created_at") == 2


@pytest.mark.asyncio
async def test_session_date_filter_rejects_invalid_range_without_query(dashboard_client):
    db = _session_list_db()
    app.dependency_overrides[get_session] = lambda: db
    try:
        response = await dashboard_client.get(
            "/api/dashboard/sessions?start_date=2026-08-08&end_date=2026-08-07"
        )
    finally:
        app.dependency_overrides.pop(get_session, None)

    assert response.status_code == 422
    assert db.execute.await_count == 0


@pytest.mark.asyncio
async def test_admin_score_history_agent_filter_returns_only_selected_agent(
    dashboard_client,
):
    selected_agent = uuid.uuid4()
    selected_evaluation = _evaluation(91.0, datetime(2025, 2, 1, tzinfo=UTC))
    db = _db_returning([selected_evaluation])
    app.dependency_overrides[get_session] = lambda: db
    try:
        response = await dashboard_client.get(
            "/api/dashboard/score-history", params={"agent_id": str(selected_agent)}
        )
    finally:
        app.dependency_overrides.pop(get_session, None)

    assert response.status_code == 200
    assert [point["overall_score"] for point in response.json()] == [91.0]
    query = str(db.execute.await_args.args[0])
    assert "sessions.agent_id" in query


@pytest.mark.asyncio
async def test_admin_score_history_returns_empty_list_without_scored_evaluations(
    dashboard_client,
):
    app.dependency_overrides[get_session] = lambda: _db_returning([])
    try:
        response = await dashboard_client.get("/api/dashboard/score-history")
    finally:
        app.dependency_overrides.pop(get_session, None)

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/dashboard", "/api/dashboard/score-history"])
async def test_dashboard_endpoints_require_credentials(client, path):
    response = await client.get(path)

    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.asyncio
async def test_admin_dashboard_returns_existing_response_shape(
    dashboard_client, authenticated_dashboard_dependencies
):
    app.dependency_overrides[get_session] = _dashboard_db
    try:
        response = await dashboard_client.get("/api/dashboard")
    finally:
        app.dependency_overrides.pop(get_session, None)

    assert response.status_code == 200
    assert response.json() == {
        "total_sessions": 0,
        "completed_sessions": 0,
        "active_sessions": 0,
        "total_scenarios": 0,
        "average_overall_score": None,
        "category_averages": [],
        "recent_sessions": [],
        "total_conversations": 0,
        "improvement_trend": None,
        "campaign_name": None,
        "campaign_id": None,
        "leaderboard": [],
    }

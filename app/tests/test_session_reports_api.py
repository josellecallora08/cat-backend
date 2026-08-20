"""Tests for the session report retrieval and export API.

Feature: session-report-generation, tasks 4.1-4.3.

Uses a real in-memory async database and the real `get_authorized_session`
policy (not mocked) so authorization behavior is exercised end-to-end,
matching the existing sessions API test conventions in test_sessions_api.py.

Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 4.1, 4.2, 4.4, 4.5, 4.6
"""

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base
from app.database import get_session as get_db_session
from app.main import app
from app.models import Campaign, CampaignAgent, Scenario, Session, SessionReport, Transcript
from app.models.campaign import CampaignRole, CampaignStatus
from app.models.session_report import SessionReportReasonCode, SessionReportStatus
from app.models.user import User
from app.services.auth import require_auth
from app.services.session_report_service import generate_report


@pytest.fixture
async def async_db():
    """In-memory SQLite database with foreign keys enabled."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)

    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


def _make_scenario() -> Scenario:
    return Scenario(
        id=uuid.uuid4(),
        name="Test Scenario",
        scenario_type="FINANCIAL_HARDSHIP",
        description="A test scenario",
        debtor_profile={
            "name": "Test Debtor",
            "outstanding_balance": "5000.00",
            "days_past_due": 30,
            "personality_profile": "cooperative",
            "conversation_goal": "negotiate payment",
        },
        is_active=True,
    )


def _make_session(
    scenario_id: uuid.UUID, agent_id: uuid.UUID, status: str = "completed"
) -> Session:
    created_at = datetime.now(UTC)
    return Session(
        id=uuid.uuid4(),
        scenario_id=scenario_id,
        agent_id=agent_id,
        status=status,
        created_at=created_at,
        ended_at=created_at if status == "completed" else None,
        persona_context={
            "name": "Test Persona",
            "communication_style": "calm",
            "emotional_state": 3,
        },
    )


def _override_db(db: AsyncSession):
    async def _override():
        yield db

    return _override


async def _seed_session_with_report(
    async_db: AsyncSession, *, status: str = "completed", agent_id=None
):
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    agent_id = agent_id or uuid.uuid4()
    session = _make_session(scenario.id, agent_id, status=status)
    async_db.add(session)
    await async_db.commit()

    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    stmt = select(Session).options(selectinload(Session.campaign)).where(Session.id == session.id)
    reloaded = (await async_db.execute(stmt)).scalar_one()

    report = None
    if status == "completed":
        report = await generate_report(async_db, reloaded)

    return session, report


def _admin_user():
    return SimpleNamespace(id=uuid.uuid4(), role="admin", user_type=None)


def _agent_user(user_id):
    return SimpleNamespace(id=user_id, role="user", user_type="agent")


def _trainer_user(user_id):
    return SimpleNamespace(id=user_id, role="user", user_type="trainer")


# --- GET /report ---


@pytest.mark.asyncio
async def test_get_report_anonymous_returns_401(client, async_db):
    session, _ = await _seed_session_with_report(async_db)
    app.dependency_overrides[get_db_session] = _override_db(async_db)

    resp = await client.get(f"/api/sessions/{session.id}/report")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_get_report_owning_agent_success(client, async_db):
    agent_id = uuid.uuid4()
    session, report = await _seed_session_with_report(async_db, agent_id=agent_id)
    app.dependency_overrides[get_db_session] = _override_db(async_db)
    app.dependency_overrides[require_auth] = lambda: _agent_user(agent_id)

    resp = await client.get(f"/api/sessions/{session.id}/report")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == str(session.id)
    assert body["report_version"] == report.report_version
    assert body["payload"] is not None


@pytest.mark.asyncio
async def test_get_report_out_of_scope_agent_returns_403(client, async_db):
    owner_id = uuid.uuid4()
    session, _ = await _seed_session_with_report(async_db, agent_id=owner_id)
    app.dependency_overrides[get_db_session] = _override_db(async_db)
    app.dependency_overrides[require_auth] = lambda: _agent_user(uuid.uuid4())

    resp = await client.get(f"/api/sessions/{session.id}/report")
    assert resp.status_code == 403
    # No report content should leak in a denied response.
    assert "payload" not in resp.text


@pytest.mark.asyncio
async def test_get_report_admin_success(client, async_db):
    session, report = await _seed_session_with_report(async_db)
    app.dependency_overrides[get_db_session] = _override_db(async_db)
    app.dependency_overrides[require_auth] = lambda: _admin_user()

    resp = await client.get(f"/api/sessions/{session.id}/report")
    assert resp.status_code == 200
    assert resp.json()["report_version"] == report.report_version


@pytest.mark.asyncio
async def test_get_report_nonexistent_session_returns_404(client, async_db):
    app.dependency_overrides[get_db_session] = _override_db(async_db)
    app.dependency_overrides[require_auth] = lambda: _admin_user()

    resp = await client.get(f"/api/sessions/{uuid.uuid4()}/report")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_report_authorized_but_no_report_returns_404(client, async_db):
    """An authorized, completed session with no generated report returns 404, not 200."""
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    agent_id = uuid.uuid4()
    session = _make_session(scenario.id, agent_id, status="completed")
    async_db.add(session)
    await async_db.commit()

    app.dependency_overrides[get_db_session] = _override_db(async_db)
    app.dependency_overrides[require_auth] = lambda: _agent_user(agent_id)

    resp = await client.get(f"/api/sessions/{session.id}/report")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_report_trainer_in_campaign_success(client, async_db):
    """A trainer whose active campaign includes the session owner can view the report."""
    campaign = Campaign(id=uuid.uuid4(), name="Campaign A", status=CampaignStatus.ACTIVE.value)
    async_db.add(campaign)
    await async_db.flush()

    trainer_id = uuid.uuid4()
    agent_id = uuid.uuid4()
    async_db.add(
        User(
            id=trainer_id,
            email="trainer@test.com",
            full_name="Trainer",
            role="user",
            user_type="trainer",
        )
    )
    async_db.add(
        User(id=agent_id, email="agent@test.com", full_name="Agent", role="user", user_type="agent")
    )
    await async_db.flush()
    async_db.add(
        CampaignAgent(campaign_id=campaign.id, agent_id=trainer_id, role=CampaignRole.TRAINER.value)
    )
    async_db.add(
        CampaignAgent(
            campaign_id=campaign.id, agent_id=agent_id, role=CampaignRole.PARTICIPANT.value
        )
    )
    await async_db.flush()

    session, report = await _seed_session_with_report(async_db, agent_id=agent_id)
    app.dependency_overrides[get_db_session] = _override_db(async_db)
    app.dependency_overrides[require_auth] = lambda: _trainer_user(trainer_id)

    resp = await client.get(f"/api/sessions/{session.id}/report")
    assert resp.status_code == 200
    assert resp.json()["report_version"] == report.report_version


@pytest.mark.asyncio
async def test_get_report_out_of_scope_trainer_returns_403(client, async_db):
    """A trainer with no matching campaign assignment is denied."""
    agent_id = uuid.uuid4()
    session, _ = await _seed_session_with_report(async_db, agent_id=agent_id)
    app.dependency_overrides[get_db_session] = _override_db(async_db)
    app.dependency_overrides[require_auth] = lambda: _trainer_user(uuid.uuid4())

    resp = await client.get(f"/api/sessions/{session.id}/report")
    assert resp.status_code == 403


# --- POST /report (generation) ---


@pytest.mark.asyncio
async def test_post_report_not_completed_returns_409(client, async_db):
    agent_id = uuid.uuid4()
    session, _ = await _seed_session_with_report(async_db, status="active", agent_id=agent_id)
    app.dependency_overrides[get_db_session] = _override_db(async_db)
    app.dependency_overrides[require_auth] = lambda: _agent_user(agent_id)

    resp = await client.post(f"/api/sessions/{session.id}/report")
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_post_report_completed_session_generates_version(client, async_db):
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    agent_id = uuid.uuid4()
    session = _make_session(scenario.id, agent_id, status="completed")
    async_db.add(session)
    await async_db.commit()

    app.dependency_overrides[get_db_session] = _override_db(async_db)
    app.dependency_overrides[require_auth] = lambda: _agent_user(agent_id)

    resp = await client.post(f"/api/sessions/{session.id}/report")
    assert resp.status_code == 201
    assert resp.json()["report_version"] == 1


@pytest.mark.asyncio
async def test_post_report_anonymous_returns_401(client, async_db):
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id, uuid.uuid4(), status="completed")
    async_db.add(session)
    await async_db.commit()
    app.dependency_overrides[get_db_session] = _override_db(async_db)

    resp = await client.post(f"/api/sessions/{session.id}/report")
    assert resp.status_code == 401


# --- Export ---


@pytest.mark.asyncio
async def test_export_json_returns_payload_with_correct_content_type(client, async_db):
    agent_id = uuid.uuid4()
    session, _report = await _seed_session_with_report(async_db, agent_id=agent_id)
    app.dependency_overrides[get_db_session] = _override_db(async_db)
    app.dependency_overrides[require_auth] = lambda: _admin_user()

    resp = await client.get(f"/api/sessions/{session.id}/report/export?format=json")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    assert "Content-Disposition" in resp.headers
    assert f"session-report_{session.id}_v1.json" in resp.headers["Content-Disposition"]
    assert len(resp.content) > 0


@pytest.mark.asyncio
async def test_export_csv_returns_flattened_body_with_neutralized_formula(client, async_db):
    """A category name starting with '=' must be neutralized in the CSV output."""
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    agent_id = uuid.uuid4()
    session = _make_session(scenario.id, agent_id, status="completed")
    async_db.add(session)
    await async_db.commit()

    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from app.models import CoachingReport

    async_db.add(
        Transcript(
            id=uuid.uuid4(),
            session_id=session.id,
            speaker="agent",
            utterance_text="n/a",
            timestamp_ms=datetime.now(UTC),
            sequence_number=0,
        )
    )
    async_db.add(
        CoachingReport(
            id=uuid.uuid4(),
            session_id=session.id,
            mistakes_by_category={
                "compliance": [
                    {
                        "transcript_position": 0,
                        "transcript_excerpt": "n/a",
                        "category": "compliance",
                        "explanation": "Missed disclosure",
                        "recommended_alternative": "=SUM(A1:A9)",
                    }
                ],
            },
            total_mistakes=1,
            no_mistakes=False,
        )
    )
    await async_db.commit()

    stmt = select(Session).options(selectinload(Session.campaign)).where(Session.id == session.id)
    reloaded = (await async_db.execute(stmt)).scalar_one()
    await generate_report(async_db, reloaded)

    app.dependency_overrides[get_db_session] = _override_db(async_db)
    app.dependency_overrides[require_auth] = lambda: _admin_user()

    resp = await client.get(f"/api/sessions/{session.id}/report/export?format=csv")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    body = resp.content.decode("utf-8")
    assert "'=SUM(A1:A9)" in body
    assert "=SUM(A1:A9)\r\n" not in body.replace("'=SUM(A1:A9)", "")  # neutralized, not raw


@pytest.mark.asyncio
async def test_export_pdf_returns_paginated_pdf(client, async_db):
    agent_id = uuid.uuid4()
    session, _ = await _seed_session_with_report(async_db, agent_id=agent_id)
    app.dependency_overrides[get_db_session] = _override_db(async_db)
    app.dependency_overrides[require_auth] = lambda: _admin_user()

    resp = await client.get(f"/api/sessions/{session.id}/report/export?format=pdf")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/pdf")
    assert resp.content.startswith(b"%PDF")
    assert len(resp.content) > 500


@pytest.mark.asyncio
async def test_export_unsupported_format_returns_400_with_no_partial_body(
    client, async_db, monkeypatch
):
    agent_id = uuid.uuid4()
    session, _ = await _seed_session_with_report(async_db, agent_id=agent_id)
    app.dependency_overrides[get_db_session] = _override_db(async_db)
    app.dependency_overrides[require_auth] = lambda: _admin_user()

    from unittest.mock import Mock

    renderer = Mock(side_effect=AssertionError("unsupported format was rendered"))
    monkeypatch.setattr("app.api.session_reports.render_export", renderer)
    resp = await client.get(f"/api/sessions/{session.id}/report/export?format=xml")
    assert resp.status_code == 400
    assert resp.content == b'{"detail":"Unsupported export format"}'
    renderer.assert_not_called()


@pytest.mark.asyncio
async def test_export_csv_denies_non_admin_before_lookup(client, async_db, monkeypatch):
    """CSV compatibility export denies users before session/report work."""
    session, _ = await _seed_session_with_report(async_db)
    app.dependency_overrides[get_db_session] = _override_db(async_db)
    app.dependency_overrides[require_auth] = lambda: _agent_user(uuid.uuid4())

    lookup = Mock(side_effect=AssertionError("report lookup must not run"))
    monkeypatch.setattr("app.api.session_reports.get_current_report", lookup)

    resp = await client.get(f"/api/sessions/{session.id}/report/export?format=csv")

    assert resp.status_code == 403
    assert resp.json() == {"detail": "Admin access required"}
    assert "session_id" not in resp.text
    lookup.assert_not_called()


@pytest.mark.asyncio
async def test_export_csv_denies_non_admin_for_missing_session(client, async_db):
    """CSV compatibility export does not reveal whether a session exists."""
    app.dependency_overrides[get_db_session] = _override_db(async_db)
    app.dependency_overrides[require_auth] = lambda: _agent_user(uuid.uuid4())

    resp = await client.get(f"/api/sessions/{uuid.uuid4()}/report/export?format=csv")

    assert resp.status_code == 403
    assert resp.json() == {"detail": "Admin access required"}


@pytest.mark.asyncio
async def test_export_out_of_scope_agent_returns_403_before_export(client, async_db):
    owner_id = uuid.uuid4()
    session, _ = await _seed_session_with_report(async_db, agent_id=owner_id)
    app.dependency_overrides[get_db_session] = _override_db(async_db)
    app.dependency_overrides[require_auth] = lambda: _agent_user(uuid.uuid4())

    resp = await client.get(f"/api/sessions/{session.id}/report/export?format=json")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_export_anonymous_returns_401(client, async_db):
    session, _ = await _seed_session_with_report(async_db)
    app.dependency_overrides[get_db_session] = _override_db(async_db)

    resp = await client.get(f"/api/sessions/{session.id}/report/export?format=json")
    assert resp.status_code == 401


# --- Route table ---


def test_report_routes_do_not_collide_with_existing_session_routes():
    """Ensure /report and /report/export are distinct from /{session_id} routes."""
    paths = {
        (frozenset(getattr(route, "methods", []) or []), getattr(route, "path", None))
        for route in app.routes
    }
    assert (frozenset({"GET"}), "/api/sessions/{session_id}/report") in paths
    assert (frozenset({"POST"}), "/api/sessions/{session_id}/report") in paths
    assert (frozenset({"GET"}), "/api/sessions/{session_id}/report/export") in paths
    assert (frozenset({"GET"}), "/api/sessions/{session_id}/report/status") in paths
    assert (frozenset({"GET"}), "/api/sessions/{session_id}") in paths
    assert (
        len([path for methods, path in paths if path == "/api/sessions/{session_id}/report/status"])
        == 1
    )
    assert "/api/sessions/{session_id}/report/status" not in {
        "/api/sessions/{session_id}",
        "/api/sessions/{session_id}/report",
        "/api/sessions/{session_id}/report/export",
    }


# --- GET /report/status ---


async def _insert_report_attempt(
    async_db: AsyncSession,
    session: Session,
    *,
    status: str,
    version: int,
) -> SessionReport:
    now = datetime.now(UTC)
    attempt = SessionReport(
        id=uuid.uuid4(),
        session_id=session.id,
        agent_id=session.agent_id,
        status=status,
        report_version=version,
        payload=None,
        content_hash=None,
        reason_code=(
            SessionReportReasonCode.GENERATION_PENDING
            if status == SessionReportStatus.PENDING
            else SessionReportReasonCode.GENERATION_FAILED
        ),
        failure_reason="safe test failure" if status == SessionReportStatus.FAILED else "pending",
        created_at=now,
        updated_at=now,
    )
    async_db.add(attempt)
    await async_db.commit()
    return attempt


@pytest.mark.asyncio
async def test_status_anonymous_and_invalid_token_return_401_without_report_content(
    client, async_db
):
    session, _ = await _seed_session_with_report(async_db)
    app.dependency_overrides[get_db_session] = _override_db(async_db)

    anonymous = await client.get(f"/api/sessions/{session.id}/report/status")
    invalid = await client.get(
        f"/api/sessions/{session.id}/report/status",
        headers={"Authorization": "Bearer invalid-token"},
    )

    assert anonymous.status_code == 401
    assert invalid.status_code == 401
    assert "payload" not in anonymous.text
    assert "payload" not in invalid.text


@pytest.mark.asyncio
async def test_status_owning_agent_returns_terminal_report_envelope(client, async_db):
    agent_id = uuid.uuid4()
    session, _ = await _seed_session_with_report(async_db, agent_id=agent_id)
    app.dependency_overrides[get_db_session] = _override_db(async_db)
    app.dependency_overrides[require_auth] = lambda: _agent_user(agent_id)

    response = await client.get(f"/api/sessions/{session.id}/report/status")

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == str(session.id)
    assert body["status"] == "empty_transcript"
    assert body["report"]["payload"] is not None


@pytest.mark.asyncio
async def test_status_admin_no_report_returns_incomplete(client, async_db):
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = _make_session(scenario.id, uuid.uuid4(), status="completed")
    async_db.add(session)
    await async_db.commit()
    app.dependency_overrides[get_db_session] = _override_db(async_db)
    app.dependency_overrides[require_auth] = lambda: _admin_user()

    response = await client.get(f"/api/sessions/{session.id}/report/status")

    assert response.status_code == 200
    assert response.json()["status"] == "incomplete"
    assert response.json()["report"] is None


@pytest.mark.asyncio
async def test_status_pending_and_failed_never_include_report_payload(client, async_db):
    agent_id = uuid.uuid4()
    session, _ = await _seed_session_with_report(async_db, status="active", agent_id=agent_id)
    await _insert_report_attempt(async_db, session, status=SessionReportStatus.PENDING, version=1)
    app.dependency_overrides[get_db_session] = _override_db(async_db)
    app.dependency_overrides[require_auth] = lambda: _agent_user(agent_id)

    pending = await client.get(f"/api/sessions/{session.id}/report/status")
    assert pending.status_code == 200
    assert pending.json()["status"] == "generating"
    assert pending.json()["report"] is None
    assert "payload" not in pending.json()["latest_attempt"]

    await async_db.rollback()
    failed_session, _ = await _seed_session_with_report(
        async_db, status="active", agent_id=agent_id
    )
    await _insert_report_attempt(
        async_db, failed_session, status=SessionReportStatus.FAILED, version=1
    )
    failed = await client.get(f"/api/sessions/{failed_session.id}/report/status")
    assert failed.status_code == 200
    assert failed.json()["status"] == "failed"
    assert failed.json()["report"] is None
    assert "payload" not in failed.text


@pytest.mark.asyncio
async def test_status_older_ready_plus_failed_preserves_ready_report(client, async_db):
    agent_id = uuid.uuid4()
    session, ready = await _seed_session_with_report(async_db, agent_id=agent_id)
    await _insert_report_attempt(
        async_db, session, status=SessionReportStatus.FAILED, version=ready.report_version + 1
    )
    app.dependency_overrides[get_db_session] = _override_db(async_db)
    app.dependency_overrides[require_auth] = lambda: _agent_user(agent_id)

    response = await client.get(f"/api/sessions/{session.id}/report/status")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["report"]["report_version"] == ready.report_version
    assert body["latest_attempt"]["status"] == "failed"
    assert "payload" in body["report"]


@pytest.mark.asyncio
async def test_status_unrelated_agent_and_out_of_scope_trainer_return_403(client, async_db):
    owner_id = uuid.uuid4()
    session, _ = await _seed_session_with_report(async_db, agent_id=owner_id)
    app.dependency_overrides[get_db_session] = _override_db(async_db)

    app.dependency_overrides[require_auth] = lambda: _agent_user(uuid.uuid4())
    unrelated = await client.get(f"/api/sessions/{session.id}/report/status")
    assert unrelated.status_code == 403

    app.dependency_overrides[require_auth] = lambda: _trainer_user(uuid.uuid4())
    trainer = await client.get(f"/api/sessions/{session.id}/report/status")
    assert trainer.status_code == 403


@pytest.mark.asyncio
async def test_status_in_scope_trainer_and_admin_return_200(client, async_db):
    campaign = Campaign(id=uuid.uuid4(), name="Campaign A", status=CampaignStatus.ACTIVE.value)
    async_db.add(campaign)
    await async_db.flush()
    trainer_id = uuid.uuid4()
    agent_id = uuid.uuid4()
    async_db.add(
        User(
            id=trainer_id,
            email="status-trainer@test.com",
            full_name="Trainer",
            role="user",
            user_type="trainer",
        )
    )
    async_db.add(
        User(
            id=agent_id,
            email="status-agent@test.com",
            full_name="Agent",
            role="user",
            user_type="agent",
        )
    )
    await async_db.flush()
    async_db.add(
        CampaignAgent(campaign_id=campaign.id, agent_id=trainer_id, role=CampaignRole.TRAINER.value)
    )
    async_db.add(
        CampaignAgent(
            campaign_id=campaign.id, agent_id=agent_id, role=CampaignRole.PARTICIPANT.value
        )
    )
    await async_db.flush()
    session, _ = await _seed_session_with_report(async_db, agent_id=agent_id)
    session.campaign_id = campaign.id
    await async_db.commit()

    app.dependency_overrides[get_db_session] = _override_db(async_db)
    app.dependency_overrides[require_auth] = lambda: _trainer_user(trainer_id)
    trainer = await client.get(f"/api/sessions/{session.id}/report/status")
    assert trainer.status_code == 200

    app.dependency_overrides[require_auth] = lambda: _admin_user()
    admin = await client.get(f"/api/sessions/{session.id}/report/status")
    assert admin.status_code == 200


@pytest.mark.asyncio
async def test_status_nonexistent_session_returns_404(client, async_db):
    missing_id = uuid.uuid4()
    app.dependency_overrides[get_db_session] = _override_db(async_db)
    app.dependency_overrides[require_auth] = lambda: _admin_user()

    response = await client.get(f"/api/sessions/{missing_id}/report/status")

    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"
    assert str(missing_id) not in response.text
    assert "payload" not in response.text


@pytest.mark.asyncio
async def test_generation_conflict_returns_safe_409(client, async_db):
    agent_id = uuid.uuid4()
    session, _ = await _seed_session_with_report(async_db, status="active", agent_id=agent_id)
    await _insert_report_attempt(async_db, session, status=SessionReportStatus.PENDING, version=1)
    app.dependency_overrides[get_db_session] = _override_db(async_db)
    app.dependency_overrides[require_auth] = lambda: _agent_user(agent_id)

    response = await client.post(f"/api/sessions/{session.id}/report")

    assert response.status_code == 409
    assert "database" not in response.text.lower()
    assert "traceback" not in response.text.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "path", "spy_name"),
    [
        ("get", "/report", "get_current_report"),
        ("get", "/report/status", "get_report_status"),
        ("post", "/report", "generate_report"),
        ("get", "/report/export?format=json", "get_current_report"),
    ],
)
async def test_denied_report_operations_do_not_execute_report_queries(
    client, async_db, monkeypatch, method, path, spy_name
):
    from unittest.mock import AsyncMock

    import app.api.session_reports as session_reports_api

    owner_id = uuid.uuid4()
    session, _ = await _seed_session_with_report(async_db, agent_id=owner_id)
    spy = AsyncMock()
    monkeypatch.setattr(session_reports_api, spy_name, spy)
    app.dependency_overrides[get_db_session] = _override_db(async_db)
    app.dependency_overrides[require_auth] = lambda: _agent_user(uuid.uuid4())

    response = await getattr(client, method)(f"/api/sessions/{session.id}{path}")

    assert response.status_code == 403
    spy.assert_not_awaited()

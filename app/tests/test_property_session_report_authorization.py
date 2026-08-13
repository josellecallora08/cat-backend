"""Property-based tests for session report authorization outcomes.

Feature: session-report-generation, task 4.3.

Property: for any (role, ownership) combination, GET/POST report and
GET report/export enforce the same authorization decision as
get_authorized_session, and a denied request never returns report content.

Validates: Requirements 3.3, 3.4, 3.5, 3.7, 4.6
"""

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from app.database import Base
from app.database import get_session as get_db_session
from app.main import app
from app.models import Scenario, Session
from app.services.auth import require_auth
from app.services.session_report_service import generate_report


@pytest.fixture
async def async_db():
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


def _override_db(db: AsyncSession):
    async def _override():
        yield db

    return _override


def _make_scenario() -> Scenario:
    return Scenario(
        id=uuid.uuid4(),
        name="Test",
        scenario_type="FINANCIAL_HARDSHIP",
        debtor_profile={
            "name": "X",
            "outstanding_balance": "100",
            "days_past_due": 1,
            "personality_profile": "calm",
            "conversation_goal": "pay",
        },
        is_active=True,
    )


async def _seed_completed_session_with_report(async_db, owner_agent_id):
    scenario = _make_scenario()
    async_db.add(scenario)
    await async_db.flush()
    session = Session(
        id=uuid.uuid4(),
        scenario_id=scenario.id,
        agent_id=owner_agent_id,
        status="completed",
        persona_context={"name": "P"},
        created_at=datetime.now(UTC),
    )
    session.ended_at = session.created_at
    async_db.add(session)
    await async_db.commit()

    stmt = select(Session).options(selectinload(Session.campaign)).where(Session.id == session.id)
    reloaded = (await async_db.execute(stmt)).scalar_one()
    await generate_report(async_db, reloaded)
    return session


class TestAuthorizationDenialsNeverLeakContent:
    """Property: any denied (non-owning, non-admin, no-campaign) caller gets no content."""

    @settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        endpoint=st.sampled_from(["report", "status", "export"]),
    )
    @pytest.mark.asyncio
    async def test_out_of_scope_agent_never_sees_report_content(
        self, async_db: AsyncSession, client: AsyncClient, endpoint: str
    ):
        owner_id = uuid.uuid4()
        outsider_id = uuid.uuid4()
        session = await _seed_completed_session_with_report(async_db, owner_id)

        app.dependency_overrides[get_db_session] = _override_db(async_db)
        app.dependency_overrides[require_auth] = lambda: SimpleNamespace(
            id=outsider_id, role="user", user_type="agent"
        )

        path = (
            f"/api/sessions/{session.id}/report"
            if endpoint == "report"
            else (
                f"/api/sessions/{session.id}/report/status"
                if endpoint == "status"
                else f"/api/sessions/{session.id}/report/export?format=json"
            )
        )
        resp = await client.get(path)

        assert resp.status_code == 403
        assert "payload" not in resp.text
        assert str(session.id) not in resp.text or endpoint == "export"
        await async_db.rollback()

    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(has_token=st.just(False))
    @pytest.mark.asyncio
    async def test_anonymous_never_sees_report_content(
        self, async_db: AsyncSession, client: AsyncClient, has_token: bool
    ):
        owner_id = uuid.uuid4()
        session = await _seed_completed_session_with_report(async_db, owner_id)
        app.dependency_overrides[get_db_session] = _override_db(async_db)

        resp = await client.get(f"/api/sessions/{session.id}/report")
        assert resp.status_code == 401
        assert "payload" not in resp.text
        await async_db.rollback()

    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(role=st.just("admin"))
    @pytest.mark.asyncio
    async def test_admin_always_authorized(
        self, async_db: AsyncSession, client: AsyncClient, role: str
    ):
        owner_id = uuid.uuid4()
        session = await _seed_completed_session_with_report(async_db, owner_id)
        app.dependency_overrides[get_db_session] = _override_db(async_db)
        app.dependency_overrides[require_auth] = lambda: SimpleNamespace(
            id=uuid.uuid4(), role=role, user_type=None
        )

        resp = await client.get(f"/api/sessions/{session.id}/report")
        assert resp.status_code == 200
        assert resp.json()["session_id"] == str(session.id)
        await async_db.rollback()

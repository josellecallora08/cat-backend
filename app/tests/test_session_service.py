"""Tests for session service layer."""

import uuid
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base
from app.models import Scenario, Session
from app.services.debtor_simulator import (
    DebtorSimulatorService,
    EmotionalState,
    PersonaContext,
)
from app.services.session_service import (
    activate_session,
    create_session,
    end_session,
    get_session,
)


@pytest.fixture
async def async_db():
    """Create an in-memory SQLite database for testing."""
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


def _make_scenario(
    name: str = "Test Scenario",
    scenario_type: str = "FINANCIAL_HARDSHIP",
    is_active: bool = True,
) -> Scenario:
    """Helper to create a Scenario instance."""
    return Scenario(
        id=uuid.uuid4(),
        name=name,
        scenario_type=scenario_type,
        description="A test scenario for training",
        debtor_profile={
            "name": "Maria Santos",
            "outstanding_balance": "5000.00",
            "days_past_due": 45,
            "personality_profile": "anxious",
            "conversation_goal": "negotiate payment plan",
        },
        is_active=is_active,
    )


def _make_mock_debtor_simulator() -> DebtorSimulatorService:
    """Create a mock DebtorSimulatorService that returns a valid persona."""
    mock_llm = AsyncMock()
    simulator = DebtorSimulatorService(llm_service=mock_llm)

    # Mock generate_persona to return a valid PersonaContext
    mock_persona = PersonaContext(
        persona_id=uuid.uuid4(),
        name="Maria Santos",
        communication_style="anxious",
        financial_circumstances={
            "income_level": "low",
            "debt_amount": 5000,
            "reason_for_delinquency": "job loss",
        },
        emotional_state=EmotionalState.DEFENSIVE,
        language="EN",
    )
    simulator.generate_persona = AsyncMock(return_value=mock_persona)
    return simulator


class TestCreateSession:
    """Tests for create_session."""

    async def test_creates_session_with_pending_status(self, async_db: AsyncSession):
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.commit()

        simulator = _make_mock_debtor_simulator()
        agent_id = uuid.uuid4()

        session = await create_session(async_db, scenario.id, agent_id, simulator)

        assert session.status == "pending"
        assert session.scenario_id == scenario.id
        assert session.agent_id == agent_id

    async def test_stores_persona_context_as_json(self, async_db: AsyncSession):
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.commit()

        simulator = _make_mock_debtor_simulator()
        agent_id = uuid.uuid4()

        session = await create_session(async_db, scenario.id, agent_id, simulator)

        assert session.persona_context is not None
        assert session.persona_context["name"] == "Maria Santos"
        assert session.persona_context["communication_style"] == "anxious"
        assert "financial_circumstances" in session.persona_context
        assert session.persona_context["emotional_state"] == EmotionalState.DEFENSIVE.value

    async def test_calls_debtor_simulator_with_scenario_data(self, async_db: AsyncSession):
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.commit()

        simulator = _make_mock_debtor_simulator()
        agent_id = uuid.uuid4()

        await create_session(async_db, scenario.id, agent_id, simulator)

        simulator.generate_persona.assert_called_once()
        call_args = simulator.generate_persona.call_args[0][0]
        assert call_args["scenario_type"] == "FINANCIAL_HARDSHIP"
        assert call_args["debtor_profile"]["name"] == "Maria Santos"

    async def test_raises_error_for_nonexistent_scenario(self, async_db: AsyncSession):
        simulator = _make_mock_debtor_simulator()
        agent_id = uuid.uuid4()
        fake_id = uuid.uuid4()

        with pytest.raises(ValueError, match="not found or inactive"):
            await create_session(async_db, fake_id, agent_id, simulator)

    async def test_raises_error_for_inactive_scenario(self, async_db: AsyncSession):
        scenario = _make_scenario(is_active=False)
        async_db.add(scenario)
        await async_db.commit()

        simulator = _make_mock_debtor_simulator()
        agent_id = uuid.uuid4()

        with pytest.raises(ValueError, match="not found or inactive"):
            await create_session(async_db, scenario.id, agent_id, simulator)

    async def test_session_persisted_to_database(self, async_db: AsyncSession):
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.commit()

        simulator = _make_mock_debtor_simulator()
        agent_id = uuid.uuid4()

        session = await create_session(async_db, scenario.id, agent_id, simulator)

        # Verify we can fetch it back
        fetched = await get_session(async_db, session.id)
        assert fetched is not None
        assert fetched.id == session.id
        assert fetched.status == "pending"


class TestGetSession:
    """Tests for get_session."""

    async def test_returns_session_by_id(self, async_db: AsyncSession):
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.commit()

        session = Session(
            id=uuid.uuid4(),
            scenario_id=scenario.id,
            agent_id=uuid.uuid4(),
            status="active",
            persona_context={"name": "Test Persona"},
        )
        async_db.add(session)
        await async_db.commit()

        result = await get_session(async_db, session.id)
        assert result is not None
        assert result.id == session.id
        assert result.status == "active"

    async def test_returns_none_for_nonexistent_id(self, async_db: AsyncSession):
        result = await get_session(async_db, uuid.uuid4())
        assert result is None


class TestEndSession:
    """Tests for end_session."""

    async def test_transitions_active_session_to_completed(self, async_db: AsyncSession):
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.commit()

        session = Session(
            id=uuid.uuid4(),
            scenario_id=scenario.id,
            agent_id=uuid.uuid4(),
            status="active",
            persona_context={"name": "Test Persona"},
        )
        async_db.add(session)
        await async_db.commit()

        result = await end_session(async_db, session.id)
        assert result.status == "completed"

    async def test_sets_ended_at_timestamp(self, async_db: AsyncSession):
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.commit()

        session = Session(
            id=uuid.uuid4(),
            scenario_id=scenario.id,
            agent_id=uuid.uuid4(),
            status="active",
            persona_context={"name": "Test Persona"},
        )
        async_db.add(session)
        await async_db.commit()

        result = await end_session(async_db, session.id)
        assert result.ended_at is not None

    async def test_can_end_pending_session(self, async_db: AsyncSession):
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.commit()

        session = Session(
            id=uuid.uuid4(),
            scenario_id=scenario.id,
            agent_id=uuid.uuid4(),
            status="pending",
            persona_context={"name": "Test Persona"},
        )
        async_db.add(session)
        await async_db.commit()

        result = await end_session(async_db, session.id)
        assert result.status == "completed"

    async def test_raises_error_for_nonexistent_session(self, async_db: AsyncSession):
        with pytest.raises(ValueError, match="not found"):
            await end_session(async_db, uuid.uuid4())

    async def test_raises_error_for_already_completed_session(self, async_db: AsyncSession):
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.commit()

        session = Session(
            id=uuid.uuid4(),
            scenario_id=scenario.id,
            agent_id=uuid.uuid4(),
            status="completed",
            persona_context={"name": "Test Persona"},
        )
        async_db.add(session)
        await async_db.commit()

        with pytest.raises(ValueError, match="Cannot end session"):
            await end_session(async_db, session.id)


class TestActivateSession:
    """Tests for activate_session."""

    async def test_transitions_pending_to_active(self, async_db: AsyncSession):
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.commit()

        session = Session(
            id=uuid.uuid4(),
            scenario_id=scenario.id,
            agent_id=uuid.uuid4(),
            status="pending",
            persona_context={"name": "Test Persona"},
        )
        async_db.add(session)
        await async_db.commit()

        result = await activate_session(async_db, session.id)
        assert result.status == "active"

    async def test_raises_error_for_nonexistent_session(self, async_db: AsyncSession):
        with pytest.raises(ValueError, match="not found"):
            await activate_session(async_db, uuid.uuid4())

    async def test_raises_error_for_non_pending_session(self, async_db: AsyncSession):
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.commit()

        session = Session(
            id=uuid.uuid4(),
            scenario_id=scenario.id,
            agent_id=uuid.uuid4(),
            status="active",
            persona_context={"name": "Test Persona"},
        )
        async_db.add(session)
        await async_db.commit()

        with pytest.raises(ValueError, match="must be 'pending'"):
            await activate_session(async_db, session.id)

    async def test_raises_error_for_completed_session(self, async_db: AsyncSession):
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.commit()

        session = Session(
            id=uuid.uuid4(),
            scenario_id=scenario.id,
            agent_id=uuid.uuid4(),
            status="completed",
            persona_context={"name": "Test Persona"},
        )
        async_db.add(session)
        await async_db.commit()

        with pytest.raises(ValueError, match="must be 'pending'"):
            await activate_session(async_db, session.id)

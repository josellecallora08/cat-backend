"""Tests for session service layer."""

import json
import uuid
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base
from app.models import Scenario, Session, User
from app.models.user import UserRole
from app.services.debtor_simulator import (
    DebtorSimulatorService,
    EmotionalState,
    PersonaContext,
)
from app.services.script_registry import ScriptVersion, create_draft, publish, unpublish
from app.services.session_service import (
    activate_session,
    create_session,
    end_session,
    get_session,
)

# A minimal Script_Contract satisfying every required field/sub-field, used
# to publish a Published_Script for a scenario before create_session tests
# that expect success (create_session now requires an active Published_Script
# per Requirements 4.1-4.3).
VALID_SCRIPT_CONTRACT = {
    "debtor_persona": {
        "name": "Maria Santos",
        "communication_style": "anxious and apologetic",
        "background": "single parent, recently lost a second job",
    },
    "financial_situation": {
        "outstanding_balance": 5000.00,
        "days_past_due": 45,
        "reason_for_delinquency": "temporary job loss",
    },
    "opening_response": "Hello, I'm calling about my account.",
    "expected_replies": [
        {
            "agent_statement": "Can you make a payment today?",
            "debtor_reply": "I can try to pay something small.",
        }
    ],
    "trigger_phrases": [
        {"phrase": "legal action", "behavior": "become distressed"}
    ],
    "emotional_state_rules": [
        {"trigger": "threat", "state_change": "increase anxiety"}
    ],
    "payment_conditions": [
        {"condition": "partial payment", "term": "50 dollars now", "accepted": True}
    ],
    "escalation_conditions": [
        {"condition": "hostile language", "behavior": "end call", "ends_call": True}
    ],
    "prohibited_responses": ["I refuse to ever pay"],
    "conversation_goal": {
        "target_outcome": "payment plan agreed",
        "completion_condition": "debtor agrees to a plan",
    },
}


def _make_admin_user() -> User:
    """Create a valid Administrator `User` instance for script `created_by`/
    `published_by` FKs."""
    return User(
        id=uuid.uuid4(),
        email=f"{uuid.uuid4()}@example.com",
        hashed_password="not-a-real-hash",
        full_name="Test Admin",
        role=UserRole.ADMIN.value,
        is_active=True,
    )


async def _publish_script_for_scenario(
    db: AsyncSession, scenario_id: uuid.UUID
) -> ScriptVersion:
    """Create and publish a minimal valid Script for `scenario_id`, so
    `create_session` (which now requires an active Published_Script) can
    succeed in tests exercising the happy path.

    Returns the `ScriptVersion` created by `publish`."""
    admin = _make_admin_user()
    db.add(admin)
    await db.commit()

    script = await create_draft(
        db,
        admin_id=admin.id,
        name="Test Script",
        scenario_id=scenario_id,
        format="json",
        raw_definition=json.dumps(VALID_SCRIPT_CONTRACT),
    )
    return await publish(db, script.id, admin.id)


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

    # Disable FK enforcement before dropping tables: Script.current_version_id
    # and ScriptVersion.script_id form a circular FK relationship, and once
    # rows populate both sides (a Published_Script), SQLite's per-statement
    # FK checking can otherwise reject DROP TABLE regardless of drop order.
    async with engine.begin() as conn:
        await conn.execute(text("PRAGMA foreign_keys=OFF"))
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
        await _publish_script_for_scenario(async_db, scenario.id)

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
        await _publish_script_for_scenario(async_db, scenario.id)

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
        await _publish_script_for_scenario(async_db, scenario.id)

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
        await _publish_script_for_scenario(async_db, scenario.id)

        simulator = _make_mock_debtor_simulator()
        agent_id = uuid.uuid4()

        session = await create_session(async_db, scenario.id, agent_id, simulator)

        # Verify we can fetch it back
        fetched = await get_session(async_db, session.id)
        assert fetched is not None
        assert fetched.id == session.id
        assert fetched.status == "pending"

    async def test_pins_script_version_id_when_published_script_exists(
        self, async_db: AsyncSession
    ):
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.commit()
        script_version = await _publish_script_for_scenario(async_db, scenario.id)

        simulator = _make_mock_debtor_simulator()
        agent_id = uuid.uuid4()

        session = await create_session(async_db, scenario.id, agent_id, simulator)

        assert session.script_version_id == script_version.id

    async def test_raises_descriptive_error_when_script_is_draft_only(
        self, async_db: AsyncSession
    ):
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.commit()

        admin = _make_admin_user()
        async_db.add(admin)
        await async_db.commit()

        await create_draft(
            async_db,
            admin_id=admin.id,
            name="Draft-Only Script",
            scenario_id=scenario.id,
            format="json",
            raw_definition=json.dumps(VALID_SCRIPT_CONTRACT),
        )

        simulator = _make_mock_debtor_simulator()
        agent_id = uuid.uuid4()

        with pytest.raises(ValueError, match="Published_Script"):
            await create_session(async_db, scenario.id, agent_id, simulator)

    async def test_raises_descriptive_error_when_script_is_unpublished(
        self, async_db: AsyncSession
    ):
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.commit()

        admin = _make_admin_user()
        async_db.add(admin)
        await async_db.commit()

        script = await create_draft(
            async_db,
            admin_id=admin.id,
            name="Unpublished Script",
            scenario_id=scenario.id,
            format="json",
            raw_definition=json.dumps(VALID_SCRIPT_CONTRACT),
        )
        await publish(async_db, script.id, admin.id)
        await unpublish(async_db, script.id, admin.id)

        simulator = _make_mock_debtor_simulator()
        agent_id = uuid.uuid4()

        with pytest.raises(ValueError, match="Published_Script"):
            await create_session(async_db, scenario.id, agent_id, simulator)

    async def test_raises_descriptive_error_when_no_script_exists(
        self, async_db: AsyncSession
    ):
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.commit()

        simulator = _make_mock_debtor_simulator()
        agent_id = uuid.uuid4()

        with pytest.raises(ValueError, match="Published_Script"):
            await create_session(async_db, scenario.id, agent_id, simulator)


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

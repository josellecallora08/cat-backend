"""Tests for scenario repository layer."""

import uuid

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base
from app.models import Scenario
from app.services.scenario_repository import get_scenario_by_id, list_active_scenarios


@pytest.fixture
async def async_db():
    """Create an in-memory SQLite database for testing."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)

    # Enable foreign key support for SQLite
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
    description: str = "A test scenario",
) -> Scenario:
    """Helper to create a Scenario instance."""
    return Scenario(
        id=uuid.uuid4(),
        name=name,
        scenario_type=scenario_type,
        description=description,
        debtor_profile={
            "name": "John Doe",
            "outstanding_balance": "5000.00",
            "days_past_due": 30,
            "personality_profile": "cooperative",
            "conversation_goal": "negotiate payment plan",
        },
        is_active=is_active,
    )


class TestListActiveScenarios:
    """Tests for list_active_scenarios."""

    async def test_returns_empty_list_when_no_scenarios(self, async_db: AsyncSession):
        result = await list_active_scenarios(async_db)
        assert result == []

    async def test_returns_only_active_scenarios(self, async_db: AsyncSession):
        active = _make_scenario(name="Active Scenario", is_active=True)
        inactive = _make_scenario(name="Inactive Scenario", is_active=False)
        async_db.add_all([active, inactive])
        await async_db.commit()

        result = await list_active_scenarios(async_db)
        assert len(result) == 1
        assert result[0].name == "Active Scenario"

    async def test_returns_scenarios_ordered_by_name(self, async_db: AsyncSession):
        s1 = _make_scenario(name="Zebra Scenario")
        s2 = _make_scenario(name="Alpha Scenario")
        s3 = _make_scenario(name="Middle Scenario")
        async_db.add_all([s1, s2, s3])
        await async_db.commit()

        result = await list_active_scenarios(async_db)
        names = [s.name for s in result]
        assert names == ["Alpha Scenario", "Middle Scenario", "Zebra Scenario"]

    async def test_excludes_all_inactive_scenarios(self, async_db: AsyncSession):
        s1 = _make_scenario(name="Inactive 1", is_active=False)
        s2 = _make_scenario(name="Inactive 2", is_active=False)
        async_db.add_all([s1, s2])
        await async_db.commit()

        result = await list_active_scenarios(async_db)
        assert result == []


class TestGetScenarioById:
    """Tests for get_scenario_by_id."""

    async def test_returns_active_scenario_by_id(self, async_db: AsyncSession):
        scenario = _make_scenario(name="Findable Scenario")
        async_db.add(scenario)
        await async_db.commit()

        result = await get_scenario_by_id(async_db, scenario.id)
        assert result is not None
        assert result.id == scenario.id
        assert result.name == "Findable Scenario"

    async def test_returns_none_for_inactive_scenario(self, async_db: AsyncSession):
        scenario = _make_scenario(name="Hidden Scenario", is_active=False)
        async_db.add(scenario)
        await async_db.commit()

        result = await get_scenario_by_id(async_db, scenario.id)
        assert result is None

    async def test_returns_none_for_nonexistent_id(self, async_db: AsyncSession):
        result = await get_scenario_by_id(async_db, uuid.uuid4())
        assert result is None

    async def test_returns_scenario_with_correct_fields(self, async_db: AsyncSession):
        scenario = _make_scenario(
            name="Full Scenario",
            scenario_type="ANGRY_CUSTOMER",
            description="Detailed description",
        )
        async_db.add(scenario)
        await async_db.commit()

        result = await get_scenario_by_id(async_db, scenario.id)
        assert result is not None
        assert result.name == "Full Scenario"
        assert result.scenario_type == "ANGRY_CUSTOMER"
        assert result.description == "Detailed description"
        assert result.debtor_profile["name"] == "John Doe"

"""Tests for campaign_scenario_service, focused on the agent-visible scenario list.

Regression coverage for a bug where `get_agent_campaign_scenarios` used a
SQL-level `SELECT DISTINCT` across full `Scenario` rows. Postgres cannot
compare its `json` column type for equality, so that query raised
`UndefinedFunctionError: could not identify an equality operator for type json`
in production (surfaced to clients as a 500 on `GET /api/scenarios`). SQLite
does not enforce this restriction, so these tests instead pin the *dedup by
scenario id* behavior that replaced the SQL-level DISTINCT, to guard against
regressing back to a full-row DISTINCT/`.distinct()` call.
"""

import uuid

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base
from app.models import Scenario
from app.models.campaign import Campaign, CampaignAgent, CampaignStatus, campaign_scenarios
from app.models.user import User, UserRole, UserType
from app.services.campaign_scenario_service import get_agent_campaign_scenarios


@pytest.fixture
async def async_db():
    """Create an in-memory SQLite database for testing with foreign keys enabled."""
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


def _make_agent() -> User:
    return User(
        id=uuid.uuid4(),
        email=f"{uuid.uuid4()}@test.com",
        full_name="Test Agent",
        hashed_password="hashed",
        role=UserRole.USER.value,
        user_type=UserType.AGENT.value,
        is_active=True,
    )


def _make_campaign(status: str = CampaignStatus.ACTIVE.value) -> Campaign:
    return Campaign(id=uuid.uuid4(), name=f"Campaign {uuid.uuid4()}", status=status)


def _make_scenario(name: str = "Scenario", is_active: bool = True) -> Scenario:
    return Scenario(
        id=uuid.uuid4(),
        name=name,
        scenario_type="FINANCIAL_HARDSHIP",
        description="A test scenario",
        debtor_profile={
            "name": "John Doe",
            "outstanding_balance": "5000.00",
            "days_past_due": 30,
        },
        is_active=is_active,
    )


class TestGetAgentCampaignScenarios:
    """Tests for get_agent_campaign_scenarios."""

    async def test_returns_empty_list_when_agent_has_no_active_campaigns(
        self, async_db: AsyncSession
    ):
        agent = _make_agent()
        async_db.add(agent)
        await async_db.commit()

        result = await get_agent_campaign_scenarios(async_db, agent.id)
        assert result == []

    async def test_returns_scenarios_from_agents_active_campaign(
        self, async_db: AsyncSession
    ):
        agent = _make_agent()
        campaign = _make_campaign()
        scenario = _make_scenario(name="Visible Scenario")
        async_db.add_all([agent, campaign, scenario])
        await async_db.flush()

        async_db.add(CampaignAgent(campaign_id=campaign.id, agent_id=agent.id))
        await async_db.execute(
            campaign_scenarios.insert().values(
                campaign_id=campaign.id, scenario_id=scenario.id
            )
        )
        await async_db.commit()

        result = await get_agent_campaign_scenarios(async_db, agent.id)
        assert len(result) == 1
        assert result[0].id == scenario.id
        assert result[0].name == "Visible Scenario"

    async def test_deduplicates_scenario_shared_across_two_active_campaigns(
        self, async_db: AsyncSession
    ):
        """A scenario assigned to two of the agent's active campaigns should
        appear exactly once, without relying on a SQL-level DISTINCT over
        full rows (which breaks on the `json` debtor_profile column in
        Postgres)."""
        agent = _make_agent()
        campaign_a = _make_campaign()
        campaign_b = _make_campaign()
        scenario = _make_scenario(name="Shared Scenario")
        async_db.add_all([agent, campaign_a, campaign_b, scenario])
        await async_db.flush()

        async_db.add_all(
            [
                CampaignAgent(campaign_id=campaign_a.id, agent_id=agent.id),
                CampaignAgent(campaign_id=campaign_b.id, agent_id=agent.id),
            ]
        )
        await async_db.execute(
            campaign_scenarios.insert().values(
                [
                    {"campaign_id": campaign_a.id, "scenario_id": scenario.id},
                    {"campaign_id": campaign_b.id, "scenario_id": scenario.id},
                ]
            )
        )
        await async_db.commit()

        result = await get_agent_campaign_scenarios(async_db, agent.id)
        assert len(result) == 1
        assert result[0].id == scenario.id

    async def test_excludes_scenarios_from_non_active_campaigns(
        self, async_db: AsyncSession
    ):
        agent = _make_agent()
        draft_campaign = _make_campaign(status=CampaignStatus.DRAFT.value)
        scenario = _make_scenario(name="Draft-only Scenario")
        async_db.add_all([agent, draft_campaign, scenario])
        await async_db.flush()

        async_db.add(CampaignAgent(campaign_id=draft_campaign.id, agent_id=agent.id))
        await async_db.execute(
            campaign_scenarios.insert().values(
                campaign_id=draft_campaign.id, scenario_id=scenario.id
            )
        )
        await async_db.commit()

        result = await get_agent_campaign_scenarios(async_db, agent.id)
        assert result == []

    async def test_excludes_inactive_scenarios(self, async_db: AsyncSession):
        agent = _make_agent()
        campaign = _make_campaign()
        inactive_scenario = _make_scenario(name="Retired Scenario", is_active=False)
        async_db.add_all([agent, campaign, inactive_scenario])
        await async_db.flush()

        async_db.add(CampaignAgent(campaign_id=campaign.id, agent_id=agent.id))
        await async_db.execute(
            campaign_scenarios.insert().values(
                campaign_id=campaign.id, scenario_id=inactive_scenario.id
            )
        )
        await async_db.commit()

        result = await get_agent_campaign_scenarios(async_db, agent.id)
        assert result == []

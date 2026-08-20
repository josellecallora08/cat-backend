"""Regression tests for published negotiation-standard session pinning."""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base
from app.models import (
    Campaign,
    CampaignAgent,
    NegotiationStandard,
    NegotiationStandardVersion,
    Scenario,
    User,
)
from app.services.debtor_simulator import EmotionalState, PersonaContext
from app.services.session_service import PublishedStandardRequiredError, create_session


class FakeDebtorSimulator:
    """Deterministic persona generator for session creation tests."""

    async def generate_persona(self, _scenario: dict) -> PersonaContext:
        return PersonaContext(
            persona_id=uuid.uuid4(),
            name="Debtor",
            communication_style="cooperative",
            financial_circumstances={"debt_amount": 1000},
            emotional_state=EmotionalState.NEUTRAL,
            language="EN",
        )


@pytest.fixture
async def db_session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


async def _setup(
    db: AsyncSession, published: bool
) -> tuple[User, Campaign, Scenario, NegotiationStandardVersion | None]:
    user = User(
        id=uuid.uuid4(),
        email=f"{uuid.uuid4()}@example.test",
        full_name="Agent",
        role="user",
        user_type="agent",
    )
    campaign = Campaign(id=uuid.uuid4(), name=f"Campaign {uuid.uuid4()}")
    scenario = Scenario(
        id=uuid.uuid4(),
        name="Scenario",
        scenario_type="PAYMENT_EXTENSION",
        description="Test",
        debtor_profile={"name": "Debtor", "personality_profile": "calm"},
    )
    campaign.scenarios.append(scenario)
    campaign.agent_assignments.append(CampaignAgent(agent_id=user.id, role="participant"))
    db.add_all([user, campaign, scenario])
    await db.flush()
    version = None
    if published:
        standard = NegotiationStandard(
            campaign_id=campaign.id,
            name="Standard",
            status="published",
            draft_content={"schema_version": 1, "overall_passing_score": 70, "blocks": []},
            created_by=user.id,
            updated_by=user.id,
        )
        db.add(standard)
        await db.flush()
        version = NegotiationStandardVersion(
            standard_id=standard.id,
            version_number=1,
            snapshot=standard.draft_content,
            content_hash="a" * 64,
            created_by=user.id,
            published_by=user.id,
        )
        db.add(version)
        await db.flush()
        standard.current_version_id = version.id
    await db.commit()
    return user, campaign, scenario, version


@pytest.mark.asyncio
async def test_missing_published_standard_blocks_selected_campaign(
    db_session: AsyncSession,
) -> None:
    user, campaign, scenario, _version = await _setup(db_session, published=False)

    with pytest.raises(PublishedStandardRequiredError) as error:
        await create_session(db_session, scenario.id, user.id, FakeDebtorSimulator(), campaign.id)

    assert error.value.campaign_id == campaign.id


@pytest.mark.asyncio
async def test_session_pins_exact_published_version(db_session: AsyncSession) -> None:
    user, campaign, scenario, version = await _setup(db_session, published=True)

    session = await create_session(
        db_session, scenario.id, user.id, FakeDebtorSimulator(), campaign.id
    )

    assert session.negotiation_standard_version_id == version.id

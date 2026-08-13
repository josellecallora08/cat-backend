"""Integration coverage for rubric synchronization and version pinning."""

from __future__ import annotations

import asyncio
import json
import os
import uuid

import pytest
from scripts.seed_rubrics import seed_rubrics
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
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
from app.services.session_service import create_session


@pytest.fixture
async def db_session() -> AsyncSession:
    """Provide an isolated async database containing the complete model schema."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest.fixture
def enabled_seed(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Enable local source loading for one integration test."""
    source_path = tmp_path / "rubrics.json"
    monkeypatch.setattr(settings, "rubric_seed_enabled", True)
    monkeypatch.setattr(settings, "rubric_source", str(source_path))
    monkeypatch.setattr(settings, "rubric_source_id", None)
    return source_path


class FakeDebtorSimulator:
    """Deterministic persona generator required by session creation."""

    async def generate_persona(self, _scenario: dict) -> PersonaContext:
        return PersonaContext(
            persona_id=uuid.uuid4(),
            name="Debtor",
            communication_style="cooperative",
            financial_circumstances={"debt_amount": 1000},
            emotional_state=EmotionalState.NEUTRAL,
            language="EN",
        )


def _content(overall_passing_score: int = 70) -> dict:
    """Return valid publishable content with a controllable fingerprint."""
    return {
        "schema_version": 1,
        "overall_passing_score": overall_passing_score,
        "blocks": [
            {
                "id": "communication",
                "category": "Communication",
                "weight": 100,
                "passing_score": 70,
                "scoring_instructions": "Assess the communication quality.",
                "positive_behaviors": [],
                "violations": [],
                "penalties": [],
                "recommendation_guidance": "Recommend clear communication.",
                "display_order": 0,
            }
        ],
    }


async def _campaign(db: AsyncSession) -> tuple[User, Campaign, Scenario]:
    """Create an admin, campaign, and scenario suitable for seeded sessions."""
    admin = User(
        id=uuid.uuid4(),
        email=f"{uuid.uuid4()}@example.test",
        full_name="Rubric Administrator",
        role="admin",
        user_type="trainer",
    )
    campaign = Campaign(id=uuid.uuid4(), name=f"Campaign {uuid.uuid4()}")
    scenario = Scenario(
        id=uuid.uuid4(),
        name="Payment scenario",
        scenario_type="PAYMENT_EXTENSION",
        description="Test scenario",
        debtor_profile={"name": "Debtor", "personality_profile": "calm"},
    )
    campaign.scenarios.append(scenario)
    campaign.agent_assignments.append(CampaignAgent(agent_id=admin.id, role="participant"))
    db.add_all([admin, campaign, scenario])
    await db.commit()
    return admin, campaign, scenario


def _write_source(path, campaign_id: uuid.UUID, content: dict, *, publish: bool = False) -> None:
    """Write one strict source definition to the configured local source."""
    path.write_text(
        json.dumps(
            {
                "source_id": "approved-rubrics-v1",
                "definitions": [
                    {
                        "campaign_id": str(campaign_id),
                        "name": "Collections Quality Rubric",
                        "draft_content": content,
                        "publish": publish,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_seed_creates_then_reuses_standard_and_version(
    db_session: AsyncSession, enabled_seed
) -> None:
    """Repeated unchanged input creates no duplicate standards or versions."""
    admin, campaign, _scenario = await _campaign(db_session)
    _write_source(enabled_seed, campaign.id, _content())

    first = await seed_rubrics(db_session, initiated_by=admin.id)
    second = await seed_rubrics(db_session, initiated_by=admin.id)

    assert first.status == "success"
    assert first.created_rubrics == 1
    assert first.created_versions == 1
    assert second.reused_rubrics == 1
    assert second.reused_versions == 1
    assert second.created_rubrics == 0
    assert second.created_versions == 0
    assert await db_session.scalar(select(func.count(NegotiationStandard.id))) == 1
    assert await db_session.scalar(select(func.count(NegotiationStandardVersion.id))) == 1


@pytest.mark.asyncio
async def test_seed_changed_content_preserves_previous_version(
    db_session: AsyncSession, enabled_seed
) -> None:
    """A changed fingerprint adds one immutable historical version."""
    admin, campaign, _scenario = await _campaign(db_session)
    _write_source(enabled_seed, campaign.id, _content())
    first = await seed_rubrics(db_session, initiated_by=admin.id)
    old_version_id = first.definition_outcomes[0].version_id

    _write_source(enabled_seed, campaign.id, _content(90))
    second = await seed_rubrics(db_session, initiated_by=admin.id)

    versions = list((await db_session.scalars(select(NegotiationStandardVersion))).all())
    assert second.created_rubrics == 0
    assert second.created_versions == 1
    assert len(versions) == 2
    assert old_version_id in {version.id for version in versions}
    assert {version.version_number for version in versions} == {1, 2}


@pytest.mark.asyncio
async def test_seed_publication_and_invalid_publication_rollback(
    db_session: AsyncSession, enabled_seed
) -> None:
    """Publication switches the pointer, while invalid content leaves it unchanged."""
    admin, campaign, _scenario = await _campaign(db_session)
    _write_source(enabled_seed, campaign.id, _content(), publish=True)
    first = await seed_rubrics(db_session, initiated_by=admin.id)
    standard = await db_session.scalar(select(NegotiationStandard))
    original_current_id = standard.current_version_id

    _write_source(enabled_seed, campaign.id, _content(71), publish=True)
    published = await seed_rubrics(db_session, initiated_by=admin.id)
    assert published.published_versions == 1
    await db_session.refresh(standard)
    assert standard.current_version_id != original_current_id

    _write_source(enabled_seed, campaign.id, {**_content(), "blocks": []}, publish=True)
    rejected = await seed_rubrics(db_session, initiated_by=admin.id)
    await db_session.refresh(standard)
    assert rejected.status == "failed"
    assert rejected.rejected_definitions == 1
    assert standard.current_version_id == published.definition_outcomes[0].version_id
    assert first.definition_outcomes[0].version_id != standard.current_version_id


@pytest.mark.asyncio
async def test_session_pins_seeded_published_version(
    db_session: AsyncSession, enabled_seed
) -> None:
    """A session retains the exact published version selected by the seed flow."""
    admin, campaign, scenario = await _campaign(db_session)
    _write_source(enabled_seed, campaign.id, _content(), publish=True)
    result = await seed_rubrics(db_session, initiated_by=admin.id)
    version_id = result.definition_outcomes[0].version_id

    session = await create_session(
        db_session,
        scenario.id,
        admin.id,
        FakeDebtorSimulator(),
        campaign.id,
    )

    assert session.negotiation_standard_version_id == version_id


@pytest.fixture
async def postgres_session_factory():
    """Return an isolated PostgreSQL session factory when explicitly configured.

    PostgreSQL tests are opt-in so local and CI runs without a dedicated test database
    remain deterministic and never probe the application's configured database.
    """
    database_url = os.getenv("CAT_TEST_POSTGRES_URL")
    if not database_url:
        pytest.skip("CAT_TEST_POSTGRES_URL is not configured")
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if not database_url.startswith("postgresql+asyncpg://"):
        pytest.skip("CAT_TEST_POSTGRES_URL must use PostgreSQL")

    engine = create_async_engine(database_url, pool_pre_ping=True)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(lambda sync_connection: None)
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
    except (OSError, SQLAlchemyError) as exc:
        await engine.dispose()
        pytest.skip(f"PostgreSQL test database is unavailable: {type(exc).__name__}")

    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        await engine.dispose()


@pytest.mark.db_integration
@pytest.mark.asyncio
async def test_concurrent_seed_runs_create_one_identity_and_version(
    postgres_session_factory, enabled_seed, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent PostgreSQL runs produce one standard and one matching version."""
    monkeypatch.setattr(settings, "rubric_seed_enabled", True)
    admin_id = uuid.uuid4()
    async with postgres_session_factory() as setup_db:
        admin = User(
            id=admin_id,
            email=f"{admin_id}@example.test",
            full_name="Concurrency Administrator",
            role="admin",
            user_type="trainer",
        )
        campaign = Campaign(id=uuid.uuid4(), name=f"Concurrent Campaign {admin_id}")
        setup_db.add_all([admin, campaign])
        await setup_db.commit()
        campaign_id = campaign.id

    _write_source(enabled_seed, campaign_id, _content())

    async def run_once():
        async with postgres_session_factory() as db:
            return await seed_rubrics(db, initiated_by=admin_id)

    first, second = await asyncio.gather(run_once(), run_once())
    assert {first.status, second.status} == {"success"}
    assert all(result.definition_outcomes[0].version_id for result in (first, second))
    assert {first.definition_outcomes[0].status, second.definition_outcomes[0].status} <= {
        "created",
        "reused",
    }

    async with postgres_session_factory() as verification_db:
        standards = await verification_db.scalar(
            select(func.count(NegotiationStandard.id)).where(
                NegotiationStandard.source_id == "approved-rubrics-v1",
                NegotiationStandard.source_rubric_key == "collections quality rubric",
            )
        )
        standard = await verification_db.scalar(
            select(NegotiationStandard).where(
                NegotiationStandard.source_id == "approved-rubrics-v1",
                NegotiationStandard.source_rubric_key == "collections quality rubric",
            )
        )
        versions = await verification_db.scalar(
            select(func.count(NegotiationStandardVersion.id)).where(
                NegotiationStandardVersion.standard_id == standard.id
            )
        )

    assert standards == 1
    assert versions == 1

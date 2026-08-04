"""Service tests for negotiation standard lifecycle operations."""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base
from app.models import Campaign, User
from app.schemas.negotiation_standard import NegotiationStandardContent
from app.services.negotiation_standard_service import (
    StandardConflictError,
    StandardValidationError,
    archive_standard,
    canonical_content_hash,
    create_standard,
    get_standard,
    list_versions,
    publish_standard,
    update_draft,
    validate_draft,
)


def _content(weight: int = 100) -> NegotiationStandardContent:
    return NegotiationStandardContent.model_validate(
        {
            "overall_passing_score": 70,
            "blocks": [
                {
                    "id": "opening",
                    "category": "Opening",
                    "weight": weight,
                    "passing_score": 70,
                    "scoring_instructions": "Score only observed behavior.",
                    "positive_behaviors": [
                        {
                            "id": "greeting",
                            "name": "Greeting",
                            "description": "Greets the customer professionally.",
                            "evidence_instructions": "Cite the greeting.",
                        }
                    ],
                    "violations": [
                        {
                            "id": "rude-tone",
                            "name": "Rude tone",
                            "description": "Uses an unnecessarily hostile tone.",
                            "evidence_instructions": "Cite the hostile wording.",
                        }
                    ],
                    "penalties": [
                        {"violation_id": "rude-tone", "deduction": 10, "max_occurrences": 1}
                    ],
                    "recommendation_guidance": "Use respectful language.",
                    "display_order": 0,
                }
            ],
        }
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


async def _seed(db: AsyncSession) -> tuple[User, Campaign]:
    admin = User(
        id=uuid.uuid4(),
        email=f"{uuid.uuid4()}@example.test",
        full_name="Admin",
        role="admin",
        hashed_password="hash",
    )
    campaign = Campaign(id=uuid.uuid4(), name=f"Campaign {uuid.uuid4()}")
    db.add_all([admin, campaign])
    await db.commit()
    return admin, campaign


@pytest.mark.asyncio
async def test_draft_validation_and_publish_are_atomic(db_session: AsyncSession) -> None:
    admin, campaign = await _seed(db_session)
    standard = await create_standard(
        db_session, campaign.id, admin.id, "Collection", "Draft", _content(90)
    )

    validation = await validate_draft(db_session, campaign.id, admin.id)
    assert validation.valid is False
    assert validation.weight_total == 90

    with pytest.raises(StandardValidationError):
        await publish_standard(db_session, campaign.id, admin.id)

    assert standard.current_version_id is None


@pytest.mark.asyncio
async def test_publish_is_idempotent_and_stores_canonical_snapshot(db_session: AsyncSession) -> None:
    admin, campaign = await _seed(db_session)
    content = _content()
    await create_standard(db_session, campaign.id, admin.id, "Collection", None, content)

    first = await publish_standard(db_session, campaign.id, admin.id, "initial")
    second = await publish_standard(db_session, campaign.id, admin.id, "ignored")
    versions, total = await list_versions(db_session, campaign.id)

    assert first.id == second.id
    assert total == 1
    assert versions[0].content_hash == canonical_content_hash(content)
    assert versions[0].snapshot == content.model_dump(mode="json")


@pytest.mark.asyncio
async def test_stale_revision_and_archived_mutation_are_conflicts(db_session: AsyncSession) -> None:
    admin, campaign = await _seed(db_session)
    await create_standard(db_session, campaign.id, admin.id, "Collection", None, _content())
    standard = await get_standard(db_session, campaign.id)

    with pytest.raises(StandardConflictError):
        await update_draft(
            db_session,
            campaign.id,
            admin.id,
            expected_revision=standard.revision - 1,
            name="Stale",
        )

    await archive_standard(db_session, campaign.id, admin.id)
    with pytest.raises(StandardConflictError):
        await update_draft(
            db_session,
            campaign.id,
            admin.id,
            expected_revision=standard.revision + 1,
            name="Archived",
        )

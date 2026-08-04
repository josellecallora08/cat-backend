"""API authorization and contract tests for negotiation standards."""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from fastapi import HTTPException

from app.database import Base, get_session
from app.main import app
from app.models import Campaign, User
from app.schemas.negotiation_standard import NegotiationStandardContent
from app.services.auth import require_admin


def _content() -> dict:
    return {
        "overall_passing_score": 70,
        "blocks": [
            {
                "id": "opening",
                "category": "Opening",
                "weight": 100,
                "passing_score": 70,
                "scoring_instructions": "Score observed behavior.",
                "positive_behaviors": [
                    {
                        "id": "greeting",
                        "name": "Greeting",
                        "description": "Greets professionally.",
                        "evidence_instructions": "Cite the greeting.",
                    }
                ],
                "violations": [
                    {
                        "id": "rude-tone",
                        "name": "Rude tone",
                        "description": "Uses hostile wording.",
                        "evidence_instructions": "Cite the wording.",
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


@pytest.fixture
async def api_database():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override_session():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_session] = override_session
    try:
        async with factory() as session:
            admin = User(
                id=uuid.uuid4(),
                email=f"{uuid.uuid4()}@example.test",
                full_name="Admin",
                role="admin",
                hashed_password="hash",
            )
            campaign = Campaign(id=uuid.uuid4(), name=f"Campaign {uuid.uuid4()}")
            session.add_all([admin, campaign])
            await session.commit()
            yield admin, campaign
    finally:
        app.dependency_overrides.pop(get_session, None)
        app.dependency_overrides.pop(require_admin, None)
        await engine.dispose()


@pytest.mark.asyncio
async def test_admin_lifecycle_routes_return_structured_contract(api_database) -> None:
    admin, campaign = api_database

    async def admin_override():
        return admin

    app.dependency_overrides[require_admin] = admin_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        base = f"/api/campaigns/{campaign.id}/negotiation-standard"
        created = await client.post(
            base,
            json={"name": "Collection", "draft_content": _content()},
        )
        assert created.status_code == 201
        assert created.json()["status"] == "draft"

        validation = await client.post(f"{base}/validate")
        assert validation.status_code == 200
        assert validation.json()["valid"] is True

        published = await client.post(f"{base}/publish", json={"publication_note": "v1"})
        assert published.status_code == 201
        assert published.json()["version_number"] == 1

        versions = await client.get(f"{base}/versions")
        assert versions.status_code == 200
        assert versions.json()["total"] == 1


@pytest.mark.asyncio
async def test_non_admin_mutation_is_forbidden(api_database) -> None:
    _admin, campaign = api_database

    async def agent_override():
        raise HTTPException(status_code=403, detail="Admin access required")

    app.dependency_overrides[require_admin] = agent_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/api/campaigns/{campaign.id}/negotiation-standard",
            json={"name": "Collection", "draft_content": _content()},
        )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_invalid_publish_returns_structured_validation_error(api_database) -> None:
    admin, campaign = api_database

    async def admin_override():
        return admin

    app.dependency_overrides[require_admin] = admin_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        base = f"/api/campaigns/{campaign.id}/negotiation-standard"
        invalid = _content()
        invalid["blocks"][0]["weight"] = 90
        created = await client.post(
            base,
            json={"name": "Collection", "draft_content": invalid},
        )
        assert created.status_code == 201
        response = await client.post(f"{base}/publish")

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "validation_failed"
    assert detail["weight_total"] == 90
    assert detail["errors"][0]["path"] == "blocks"

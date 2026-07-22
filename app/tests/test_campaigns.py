"""Tests for campaign management API endpoints."""

import uuid
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models.campaign import CampaignRole
from app.schemas.campaign import CampaignListItem, PaginatedCampaigns
from app.services.auth import require_admin


def _mock_admin_user():
    """Return a mock admin user for dependency override."""
    user = MagicMock()
    user.id = uuid.uuid4()
    user.email = "admin@test.com"
    user.full_name = "Test Admin"
    user.role = "admin"
    return user


def _make_campaign(
    name: str = "Test Campaign",
    status: str = "draft",
    description: str | None = "A test campaign",
    start_date: date | None = None,
    end_date: date | None = None,
    scenarios: list | None = None,
    agent_assignments: list | None = None,
) -> MagicMock:
    """Create a mock Campaign ORM object with required attributes."""
    campaign = MagicMock()
    campaign.id = uuid.uuid4()
    campaign.name = name
    campaign.description = description
    campaign.status = status
    campaign.start_date = start_date or date(2025, 1, 1)
    campaign.end_date = end_date or date(2025, 6, 30)
    campaign.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    campaign.updated_at = datetime(2025, 1, 1, tzinfo=timezone.utc)

    if scenarios is None:
        scenario = MagicMock()
        scenario.id = uuid.uuid4()
        scenario.name = "Scenario A"
        scenario.scenario_type = "FINANCIAL_HARDSHIP"
        scenarios = [scenario]
    campaign.scenarios = scenarios

    if agent_assignments is None:
        assignment = MagicMock()
        assignment.agent = MagicMock()
        assignment.agent.id = uuid.uuid4()
        assignment.agent.full_name = "Agent Smith"
        assignment.agent.email = "agent@test.com"
        assignment.role = CampaignRole.PARTICIPANT.value
        agent_assignments = [assignment]
    campaign.agent_assignments = agent_assignments

    return campaign


@pytest.fixture
def admin_override():
    """Override require_admin dependency for the duration of the test."""
    mock_user = _mock_admin_user()
    app.dependency_overrides[require_admin] = lambda: mock_user
    yield mock_user
    app.dependency_overrides.clear()


@pytest.fixture
async def client(admin_override):
    """Async test client with admin auth overridden."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def unauth_client():
    """Async test client without auth overrides (for auth tests)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestCreateCampaign:
    """Tests for POST /api/campaigns."""

    async def test_creates_campaign_successfully(self, client):
        campaign = _make_campaign(name="New Campaign")
        with patch(
            "app.api.campaigns.create_campaign",
            new_callable=AsyncMock,
            return_value=campaign,
        ):
            agent_id = str(uuid.uuid4())
            response = await client.post(
                "/api/campaigns",
                json={
                    "name": "New Campaign",
                    "description": "desc",
                    "start_date": "2025-01-01",
                    "end_date": "2025-06-30",
                    "scenario_ids": [str(uuid.uuid4())],
                    "agents": [{"agent_id": agent_id, "role": "participant"}],
                },
            )
            assert response.status_code == 201
            data = response.json()
            assert data["name"] == "New Campaign"
            assert data["status"] == "draft"
            assert len(data["scenarios"]) == 1
            assert len(data["agents"]) == 1
            assert data["agents"][0]["role"] == "participant"

    async def test_rejects_duplicate_name(self, client):
        with patch(
            "app.api.campaigns.create_campaign",
            new_callable=AsyncMock,
            side_effect=ValueError("Campaign name already exists"),
        ):
            response = await client.post(
                "/api/campaigns",
                json={
                    "name": "Existing Campaign",
                    "agents": [{"agent_id": str(uuid.uuid4()), "role": "participant"}],
                },
            )
            assert response.status_code == 409
            assert "already exists" in response.json()["detail"]

    async def test_rejects_invalid_dates(self, client):
        response = await client.post(
            "/api/campaigns",
            json={
                "name": "Bad Dates",
                "start_date": "2025-06-30",
                "end_date": "2025-01-01",
                "agents": [{"agent_id": str(uuid.uuid4()), "role": "participant"}],
            },
        )
        assert response.status_code == 422

    async def test_accepts_empty_scenario_ids(self, client):
        campaign = _make_campaign(name="No Scenarios", scenarios=[])
        with patch(
            "app.api.campaigns.create_campaign",
            new_callable=AsyncMock,
            return_value=campaign,
        ):
            response = await client.post(
                "/api/campaigns",
                json={
                    "name": "No Scenarios",
                    "scenario_ids": [],
                    "agents": [{"agent_id": str(uuid.uuid4()), "role": "participant"}],
                },
            )
            assert response.status_code == 201

    async def test_rejects_empty_agents(self, client):
        response = await client.post(
            "/api/campaigns",
            json={
                "name": "No Agents",
                "agents": [],
            },
        )
        assert response.status_code == 422

    async def test_rejects_non_existent_scenario_ids(self, client):
        with patch(
            "app.api.campaigns.create_campaign",
            new_callable=AsyncMock,
            side_effect=ValueError("One or more scenario_ids are invalid or inactive"),
        ):
            response = await client.post(
                "/api/campaigns",
                json={
                    "name": "Bad Scenarios",
                    "scenario_ids": [str(uuid.uuid4())],
                    "agents": [{"agent_id": str(uuid.uuid4()), "role": "participant"}],
                },
            )
            assert response.status_code == 422
            assert "scenario_ids" in response.json()["detail"]

    async def test_requires_admin_auth(self, unauth_client):
        response = await unauth_client.post(
            "/api/campaigns",
            json={
                "name": "Unauthorized",
                "agents": [{"agent_id": str(uuid.uuid4()), "role": "participant"}],
            },
        )
        assert response.status_code in (401, 403)


class TestListCampaigns:
    """Tests for GET /api/campaigns."""

    async def test_returns_paginated_list(self, client):
        paginated = PaginatedCampaigns(
            items=[
                CampaignListItem(
                    id=uuid.uuid4(),
                    name="Campaign 1",
                    status="draft",
                    scenarios_count=2,
                    agents_count=3,
                    start_date=date(2025, 1, 1),
                    end_date=date(2025, 6, 30),
                    created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                    updated_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                ),
            ],
            total=1,
            page=1,
            page_size=15,
            total_pages=1,
        )
        with patch(
            "app.api.campaigns.list_campaigns",
            new_callable=AsyncMock,
            return_value=paginated,
        ):
            response = await client.get("/api/campaigns")
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1
            assert data["page"] == 1
            assert data["page_size"] == 15
            assert len(data["items"]) == 1
            assert data["items"][0]["name"] == "Campaign 1"

    async def test_filters_by_status(self, client):
        paginated = PaginatedCampaigns(
            items=[
                CampaignListItem(
                    id=uuid.uuid4(),
                    name="Active Campaign",
                    status="active",
                    scenarios_count=1,
                    agents_count=1,
                    start_date=date(2025, 1, 1),
                    end_date=date(2025, 6, 30),
                    created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                    updated_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                ),
            ],
            total=1,
            page=1,
            page_size=15,
            total_pages=1,
        )
        with patch(
            "app.api.campaigns.list_campaigns",
            new_callable=AsyncMock,
            return_value=paginated,
        ) as mock_list:
            response = await client.get("/api/campaigns?status=active")
            assert response.status_code == 200
            # Verify the status filter was passed to the service
            mock_list.assert_called_once()
            call_args = mock_list.call_args
            assert call_args[0][2] == 15  # page_size
            assert call_args[0][3] == "active"  # status filter

    async def test_excludes_archived_by_default(self, client):
        paginated = PaginatedCampaigns(
            items=[],
            total=0,
            page=1,
            page_size=15,
            total_pages=0,
        )
        with patch(
            "app.api.campaigns.list_campaigns",
            new_callable=AsyncMock,
            return_value=paginated,
        ) as mock_list:
            response = await client.get("/api/campaigns")
            assert response.status_code == 200
            # status filter should be None when not explicitly provided
            call_args = mock_list.call_args
            assert call_args[0][3] is None


class TestGetCampaign:
    """Tests for GET /api/campaigns/{id}."""

    async def test_returns_campaign_with_scenarios_and_agents(self, client):
        campaign = _make_campaign(name="Detail Campaign")
        with patch(
            "app.api.campaigns.get_campaign_by_id",
            new_callable=AsyncMock,
            return_value=campaign,
        ):
            response = await client.get(f"/api/campaigns/{campaign.id}")
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Detail Campaign"
            assert len(data["scenarios"]) == 1
            assert data["scenarios"][0]["name"] == "Scenario A"
            assert len(data["agents"]) == 1
            assert data["agents"][0]["full_name"] == "Agent Smith"
            assert data["agents"][0]["role"] == "participant"

    async def test_returns_404_for_non_existent_id(self, client):
        with patch(
            "app.api.campaigns.get_campaign_by_id",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = await client.get(f"/api/campaigns/{uuid.uuid4()}")
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()


class TestUpdateCampaign:
    """Tests for PUT /api/campaigns/{id}."""

    async def test_updates_campaign_fields(self, client):
        campaign = _make_campaign(name="Updated Campaign", description="New desc")
        with patch(
            "app.api.campaigns.update_campaign",
            new_callable=AsyncMock,
            return_value=campaign,
        ):
            response = await client.put(
                f"/api/campaigns/{campaign.id}",
                json={
                    "name": "Updated Campaign",
                    "description": "New desc",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Updated Campaign"
            assert data["description"] == "New desc"

    async def test_replaces_scenario_and_agent_associations(self, client):
        new_scenario = MagicMock()
        new_scenario.id = uuid.uuid4()
        new_scenario.name = "Scenario B"
        new_scenario.scenario_type = "ANGRY_CUSTOMER"

        new_assignment = MagicMock()
        new_assignment.agent = MagicMock()
        new_assignment.agent.id = uuid.uuid4()
        new_assignment.agent.full_name = "Agent Neo"
        new_assignment.agent.email = "neo@test.com"
        new_assignment.role = CampaignRole.TEAM_LEAD.value

        campaign = _make_campaign(
            name="Reassigned",
            scenarios=[new_scenario],
            agent_assignments=[new_assignment],
        )
        with patch(
            "app.api.campaigns.update_campaign",
            new_callable=AsyncMock,
            return_value=campaign,
        ):
            response = await client.put(
                f"/api/campaigns/{campaign.id}",
                json={
                    "scenario_ids": [str(new_scenario.id)],
                    "agents": [
                        {
                            "agent_id": str(new_assignment.agent.id),
                            "role": "team_lead",
                        },
                    ],
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["scenarios"][0]["name"] == "Scenario B"
            assert data["agents"][0]["full_name"] == "Agent Neo"
            assert data["agents"][0]["role"] == "team_lead"

    async def test_validates_updated_dates(self, client):
        """Pydantic schema rejects end_date <= start_date when both are provided."""
        response = await client.put(
            f"/api/campaigns/{uuid.uuid4()}",
            json={
                "start_date": "2025-12-01",
                "end_date": "2025-01-01",
            },
        )
        assert response.status_code == 422

    async def test_returns_404_for_non_existent_campaign(self, client):
        with patch(
            "app.api.campaigns.update_campaign",
            new_callable=AsyncMock,
            side_effect=ValueError("Campaign not found"),
        ):
            response = await client.put(
                f"/api/campaigns/{uuid.uuid4()}",
                json={"name": "Ghost"},
            )
            assert response.status_code == 404


class TestDeleteCampaign:
    """Tests for DELETE /api/campaigns/{id}."""

    async def test_archives_campaign(self, client):
        with patch(
            "app.api.campaigns.archive_campaign",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = await client.delete(f"/api/campaigns/{uuid.uuid4()}")
            assert response.status_code == 204

    async def test_returns_404_for_non_existent_campaign(self, client):
        with patch(
            "app.api.campaigns.archive_campaign",
            new_callable=AsyncMock,
            side_effect=ValueError("Campaign not found"),
        ):
            response = await client.delete(f"/api/campaigns/{uuid.uuid4()}")
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

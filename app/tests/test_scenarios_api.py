"""Tests for scenario API endpoints."""

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models import Scenario


def _make_scenario(
    name: str = "Test Scenario",
    scenario_type: str = "FINANCIAL_HARDSHIP",
    is_active: bool = True,
    description: str = "A test scenario",
    debtor_profile: dict = None,
) -> Scenario:
    """Helper to create a Scenario instance."""
    return Scenario(
        id=uuid.uuid4(),
        name=name,
        scenario_type=scenario_type,
        description=description,
        is_active=is_active,
        debtor_profile=debtor_profile or {
            "name": "John Doe",
            "outstanding_balance": "5000.00",
            "days_past_due": 30,
            "personality_profile": "cooperative",
            "conversation_goal": "negotiate payment plan",
        },
    )


@pytest.fixture
async def client():
    """Async test client for the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestListScenarios:
    """Tests for GET /api/scenarios."""

    async def test_returns_empty_list_when_no_scenarios(self, client):
        with patch(
            "app.api.scenarios.list_active_scenarios",
            new_callable=AsyncMock,
            return_value=[],
        ):
            response = await client.get("/api/scenarios")
            assert response.status_code == 200
            assert response.json() == []

    async def test_returns_scenarios_with_name_and_type(self, client):
        scenario = _make_scenario(name="Financial Hardship Scenario", scenario_type="FINANCIAL_HARDSHIP")
        with patch(
            "app.api.scenarios.list_active_scenarios",
            new_callable=AsyncMock,
            return_value=[scenario],
        ):
            response = await client.get("/api/scenarios")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["name"] == "Financial Hardship Scenario"
            assert data[0]["scenario_type"] == "FINANCIAL_HARDSHIP"
            assert data[0]["id"] == str(scenario.id)

    async def test_returns_multiple_scenarios(self, client):
        scenarios = [
            _make_scenario(name="Scenario A", scenario_type="FINANCIAL_HARDSHIP"),
            _make_scenario(name="Scenario B", scenario_type="ANGRY_CUSTOMER"),
            _make_scenario(name="Scenario C", scenario_type="PAYMENT_EXTENSION"),
        ]
        with patch(
            "app.api.scenarios.list_active_scenarios",
            new_callable=AsyncMock,
            return_value=scenarios,
        ):
            response = await client.get("/api/scenarios")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 3
            assert data[0]["name"] == "Scenario A"
            assert data[1]["scenario_type"] == "ANGRY_CUSTOMER"
            assert data[2]["scenario_type"] == "PAYMENT_EXTENSION"


class TestGetScenario:
    """Tests for GET /api/scenarios/{scenario_id}."""

    async def test_returns_scenario_with_full_debtor_profile(self, client):
        scenario = _make_scenario(
            name="Hardship Case",
            scenario_type="FINANCIAL_HARDSHIP",
            description="Agent must handle a financially distressed debtor.",
        )
        with patch(
            "app.api.scenarios.get_scenario_by_id",
            new_callable=AsyncMock,
            return_value=scenario,
        ):
            response = await client.get(f"/api/scenarios/{scenario.id}")
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == str(scenario.id)
            assert data["name"] == "Hardship Case"
            assert data["scenario_type"] == "FINANCIAL_HARDSHIP"
            assert data["description"] == "Agent must handle a financially distressed debtor."
            profile = data["debtor_profile"]
            assert profile["name"] == "John Doe"
            assert profile["outstanding_balance"] == "5000.00"
            assert profile["days_past_due"] == 30
            assert profile["personality_profile"] == "cooperative"
            assert profile["conversation_goal"] == "negotiate payment plan"

    async def test_returns_404_for_missing_scenario(self, client):
        missing_id = uuid.uuid4()
        with patch(
            "app.api.scenarios.get_scenario_by_id",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = await client.get(f"/api/scenarios/{missing_id}")
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    async def test_returns_422_for_incomplete_debtor_profile(self, client):
        # Missing required fields in debtor_profile
        scenario = _make_scenario(
            debtor_profile={
                "name": "Incomplete",
                # missing outstanding_balance, days_past_due, etc.
            }
        )
        with patch(
            "app.api.scenarios.get_scenario_by_id",
            new_callable=AsyncMock,
            return_value=scenario,
        ):
            response = await client.get(f"/api/scenarios/{scenario.id}")
            assert response.status_code == 422
            assert "incomplete" in response.json()["detail"].lower()

    async def test_returns_422_for_empty_profile_fields(self, client):
        # Profile with empty string fields that fail validation
        scenario = _make_scenario(
            debtor_profile={
                "name": "   ",  # whitespace-only
                "outstanding_balance": "5000.00",
                "days_past_due": 30,
                "personality_profile": "cooperative",
                "conversation_goal": "negotiate",
            }
        )
        with patch(
            "app.api.scenarios.get_scenario_by_id",
            new_callable=AsyncMock,
            return_value=scenario,
        ):
            response = await client.get(f"/api/scenarios/{scenario.id}")
            assert response.status_code == 422
            assert "incomplete" in response.json()["detail"].lower()

    async def test_returns_422_for_invalid_balance(self, client):
        # Profile with zero balance (must be > 0)
        scenario = _make_scenario(
            debtor_profile={
                "name": "Jane",
                "outstanding_balance": "0",
                "days_past_due": 30,
                "personality_profile": "cooperative",
                "conversation_goal": "negotiate",
            }
        )
        with patch(
            "app.api.scenarios.get_scenario_by_id",
            new_callable=AsyncMock,
            return_value=scenario,
        ):
            response = await client.get(f"/api/scenarios/{scenario.id}")
            assert response.status_code == 422

    async def test_returns_404_for_invalid_uuid_format(self, client):
        response = await client.get("/api/scenarios/not-a-uuid")
        assert response.status_code == 422  # FastAPI returns 422 for invalid path params

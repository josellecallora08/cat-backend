"""Property-based tests for scenario response completeness.

Feature: collection-agent-trainer, Property 1: Scenario response completeness

**Validates: Requirements 1.1, 1.2**

Property 1: For any scenario stored in the database with is_active = true,
the list endpoint response item SHALL contain a non-empty name and a valid
scenario_type, and the detail endpoint response SHALL include all debtor profile
fields: name, outstanding_balance, days_past_due, personality_profile, and
conversation_goal.
"""

import asyncio
import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models import Scenario
from app.schemas import ScenarioType


# --- Strategies ---

scenario_types = st.sampled_from([t.value for t in ScenarioType])

valid_names = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=1,
    max_size=100,
).filter(lambda s: s.strip() != "")

valid_balances = st.decimals(
    min_value=Decimal("0.01"),
    max_value=Decimal("999999.99"),
    allow_nan=False,
    allow_infinity=False,
    places=2,
)

valid_days_past_due = st.integers(min_value=1, max_value=10000)

valid_personality_profiles = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=1,
    max_size=200,
).filter(lambda s: s.strip() != "")

valid_conversation_goals = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=1,
    max_size=200,
).filter(lambda s: s.strip() != "")

valid_descriptions = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z")),
    min_size=1,
    max_size=300,
)


@st.composite
def valid_debtor_profiles(draw):
    """Generate valid debtor profile dicts."""
    return {
        "name": draw(valid_names),
        "outstanding_balance": str(draw(valid_balances)),
        "days_past_due": draw(valid_days_past_due),
        "personality_profile": draw(valid_personality_profiles),
        "conversation_goal": draw(valid_conversation_goals),
    }


@st.composite
def valid_scenarios(draw):
    """Generate a valid Scenario model instance with random data."""
    scenario_id = uuid.uuid4()
    name = draw(valid_names)
    scenario_type = draw(scenario_types)
    description = draw(valid_descriptions)
    debtor_profile = draw(valid_debtor_profiles())

    return Scenario(
        id=scenario_id,
        name=name,
        scenario_type=scenario_type,
        description=description,
        debtor_profile=debtor_profile,
        is_active=True,
    )


# --- Helpers ---


async def _get_list_response(scenarios_list):
    """Call the list endpoint with mocked repository."""
    with patch(
        "app.api.scenarios.list_active_scenarios",
        new_callable=AsyncMock,
        return_value=scenarios_list,
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/api/scenarios")


async def _get_detail_response(scenario):
    """Call the detail endpoint with mocked repository."""
    with patch(
        "app.api.scenarios.get_scenario_by_id",
        new_callable=AsyncMock,
        return_value=scenario,
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(f"/api/scenarios/{scenario.id}")


# --- Property Tests ---


class TestScenarioResponseCompleteness:
    """Property 1: Scenario response completeness.

    Feature: collection-agent-trainer, Property 1: Scenario response completeness
    """

    @given(scenario=valid_scenarios())
    @settings(max_examples=100)
    def test_list_item_has_non_empty_name_and_valid_type(self, scenario: Scenario):
        """List endpoint items always have a non-empty name and valid scenario_type.

        For any valid scenario, the list endpoint response item SHALL contain
        a non-empty name and a valid scenario_type.
        """
        response = asyncio.run(_get_list_response([scenario]))

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1

        item = data[0]
        # Verify non-empty name
        assert "name" in item
        assert isinstance(item["name"], str)
        assert len(item["name"]) > 0

        # Verify valid scenario_type
        assert "scenario_type" in item
        valid_types = {t.value for t in ScenarioType}
        assert item["scenario_type"] in valid_types

    @given(scenario=valid_scenarios())
    @settings(max_examples=100)
    def test_detail_includes_all_debtor_profile_fields(self, scenario: Scenario):
        """Detail endpoint response includes all required debtor profile fields.

        For any valid scenario, the detail endpoint response SHALL include all
        debtor profile fields: name, outstanding_balance, days_past_due,
        personality_profile, and conversation_goal.
        """
        response = asyncio.run(_get_detail_response(scenario))

        assert response.status_code == 200
        data = response.json()

        # Verify top-level fields
        assert "id" in data
        assert "name" in data
        assert isinstance(data["name"], str)
        assert len(data["name"]) > 0
        assert "scenario_type" in data
        valid_types = {t.value for t in ScenarioType}
        assert data["scenario_type"] in valid_types

        # Verify debtor_profile contains all required fields
        assert "debtor_profile" in data
        profile = data["debtor_profile"]

        required_fields = [
            "name",
            "outstanding_balance",
            "days_past_due",
            "personality_profile",
            "conversation_goal",
        ]
        for field in required_fields:
            assert field in profile, f"Missing debtor profile field: {field}"

        # Verify field values are non-empty/valid
        assert isinstance(profile["name"], str)
        assert len(profile["name"]) > 0

        assert profile["outstanding_balance"] is not None
        assert Decimal(profile["outstanding_balance"]) > 0

        assert isinstance(profile["days_past_due"], int)
        assert profile["days_past_due"] >= 1

        assert isinstance(profile["personality_profile"], str)
        assert len(profile["personality_profile"]) > 0

        assert isinstance(profile["conversation_goal"], str)
        assert len(profile["conversation_goal"]) > 0

    @given(scenarios_list=st.lists(valid_scenarios(), min_size=1, max_size=5))
    @settings(max_examples=100)
    def test_list_all_items_have_required_fields(self, scenarios_list: list):
        """All items in a multi-scenario list have non-empty name and valid type.

        For any list of valid scenarios, every list item SHALL contain a
        non-empty name and a valid scenario_type.
        """
        response = asyncio.run(_get_list_response(scenarios_list))

        assert response.status_code == 200
        data = response.json()
        assert len(data) == len(scenarios_list)

        valid_types = {t.value for t in ScenarioType}
        for item in data:
            assert "name" in item
            assert isinstance(item["name"], str)
            assert len(item["name"]) > 0

            assert "scenario_type" in item
            assert item["scenario_type"] in valid_types

            assert "id" in item

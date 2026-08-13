"""Regression tests for scenario seed repair behavior."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.models import Scenario
from app.services.seed_scenarios import (
    PAYMENT_ARRANGEMENT_SCENARIO,
    PAYMENT_ARRANGEMENT_SCENARIO_NAME,
    seed_bpi_payment_arrangement_scenario,
)


@pytest.mark.asyncio
async def test_bpi_seed_repairs_existing_legacy_scenario_without_new_association() -> None:
    """An existing legacy row is repaired while its campaign association remains intact."""
    campaign_id = uuid4()
    scenario_id = uuid4()
    scenario = Scenario(
        id=scenario_id,
        name=PAYMENT_ARRANGEMENT_SCENARIO_NAME,
        scenario_type="collections",
        description="Legacy scenario",
        debtor_profile={"name": "Maria Santos"},
        is_active=False,
    )
    db = AsyncMock()
    db.execute.side_effect = [
        SimpleNamespace(scalar_one_or_none=lambda: SimpleNamespace(id=campaign_id)),
        SimpleNamespace(scalar_one_or_none=lambda: scenario),
        SimpleNamespace(scalar_one_or_none=lambda: scenario_id),
    ]

    result = await seed_bpi_payment_arrangement_scenario(db)

    assert result == (campaign_id, scenario_id)
    assert scenario.scenario_type == "PAYMENT_EXTENSION"
    assert scenario.debtor_profile == PAYMENT_ARRANGEMENT_SCENARIO
    assert scenario.is_active is False
    db.commit.assert_awaited_once()
    assert db.execute.await_count == 3

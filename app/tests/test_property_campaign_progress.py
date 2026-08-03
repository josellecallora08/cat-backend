"""Property-based tests for campaign progress calculation.

Feature: campaign-based-session-selection
Properties 6, 7, and 10: progress correctness, campaign ordering, and no ordering enforcement.
"""

from types import SimpleNamespace
from uuid import UUID, uuid4
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.services.campaign_progress_service import (
    get_agent_campaigns_with_progress,
    get_campaign_progress,
)


uuid_strategy = st.uuids(version=4)
name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu")),
    min_size=1,
    max_size=20,
)


def _query_result(*rows: SimpleNamespace) -> MagicMock:
    result = MagicMock()
    result.all.return_value = list(rows)
    return result


def _progress_database(
    campaign_id: UUID,
    scenario_rows: list[SimpleNamespace],
) -> MagicMock:
    campaign_result = MagicMock()
    campaign_result.one_or_none.return_value = SimpleNamespace(
        id=campaign_id,
        name="Campaign",
    )
    db = MagicMock()
    db.execute = AsyncMock(side_effect=[campaign_result, _query_result(*scenario_rows)])
    return db


class TestCampaignProgressProperties:
    """Universal properties for campaign progress service responses."""

    @given(
        scenarios=st.lists(
            st.tuples(uuid_strategy, st.booleans()),
            max_size=12,
            unique_by=lambda item: item[0],
        )
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_progress_counts_distinct_completed_campaign_scenarios(
        self,
        scenarios: list[tuple[UUID, bool]],
    ) -> None:
        """**Validates: Requirements 4.1, 4.2, 6.1, 6.2**"""
        campaign_id, agent_id = uuid4(), uuid4()
        scenario_rows = [
            SimpleNamespace(
                id=scenario_id,
                name=f"Scenario {index}",
                scenario_type="standard",
                accomplished=accomplished,
            )
            for index, (scenario_id, accomplished) in enumerate(scenarios)
        ]
        db = _progress_database(campaign_id, scenario_rows)

        response = await get_campaign_progress(db, campaign_id, agent_id)

        expected_count = sum(accomplished for _, accomplished in scenarios)
        assert response.total_scenarios == len(scenarios)
        assert response.accomplished_scenarios == expected_count
        assert response.is_completed is (
            bool(scenarios) and expected_count == len(scenarios)
        )
        assert [item.accomplished for item in response.scenarios] == [
            accomplished for _, accomplished in scenarios
        ]

    @given(scenario_count=st.integers(min_value=0, max_value=12))
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_zero_scenario_campaign_is_completed(
        self, scenario_count: int
    ) -> None:
        """**Validates: Requirements 4.6, 6.6**"""
        campaign_id, agent_id = uuid4(), uuid4()
        rows = [
            SimpleNamespace(
                id=uuid4(),
                name=f"Scenario {index}",
                scenario_type="standard",
                accomplished=True,
            )
            for index in range(scenario_count)
        ]
        db = _progress_database(campaign_id, rows)
        response = await get_campaign_progress(db, campaign_id, agent_id)

        if scenario_count == 0:
            assert response.total_scenarios == 0
            assert response.accomplished_scenarios == 0
            assert response.scenarios == []
            assert response.is_completed is True

    @given(
        campaigns=st.lists(
            st.tuples(name_strategy, st.booleans()),
            min_size=1,
            max_size=12,
            unique_by=lambda item: item[0].casefold(),
        )
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_campaign_list_orders_incomplete_then_alphabetically(
        self, campaigns: list[tuple[str, bool]]
    ) -> None:
        """**Validates: Requirement 6.3**"""
        campaign_rows = [
            SimpleNamespace(id=uuid4(), name=name, description=None)
            for name, _ in campaigns
        ]
        progress_rows = [
            SimpleNamespace(
                campaign_id=row.id,
                total_scenarios=1,
                accomplished_scenarios=int(is_completed),
            )
            for row, (_, is_completed) in zip(campaign_rows, campaigns)
        ]
        db = MagicMock()
        db.execute = AsyncMock(
            side_effect=[_query_result(*campaign_rows), _query_result(*progress_rows)]
        )

        result = await get_agent_campaigns_with_progress(db, uuid4())

        expected = sorted(campaigns, key=lambda item: (item[1], item[0].casefold()))
        assert [(item.name, item.is_completed) for item in result] == expected

    @given(
        completed_indices=st.sets(st.integers(min_value=0, max_value=11)),
        scenario_count=st.integers(min_value=0, max_value=12),
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_completion_does_not_depend_on_scenario_start_order(
        self, completed_indices: set[int], scenario_count: int
    ) -> None:
        """**Validates: Requirements 7.2, 7.3**"""
        scenario_ids = [uuid4() for _ in range(scenario_count)]
        rows = [
            SimpleNamespace(
                id=scenario_id,
                name=f"Scenario {index}",
                scenario_type="standard",
                accomplished=index in completed_indices,
            )
            for index, scenario_id in enumerate(scenario_ids)
        ]
        db = _progress_database(uuid4(), rows)
        response = await get_campaign_progress(db, uuid4(), uuid4())

        expected = [index in completed_indices for index in range(scenario_count)]
        assert sorted(item.scenario_id for item in response.scenarios) == sorted(
            scenario_ids
        )
        assert {item.accomplished for item in response.scenarios} == set(expected)
        assert response.is_completed is (bool(expected) and all(expected))

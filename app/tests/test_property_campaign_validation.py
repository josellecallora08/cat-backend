"""Property-based tests for campaign session validation.

Feature: campaign-based-session-selection
Properties 3, 4, and 5: validation order, admin bypass, and campaign boundaries.

**Validates: Requirements 2.3, 2.4, 2.5, 2.6, 3.1, 3.2, 3.3, 3.4, 3.5**
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException, status
from hypothesis import given, settings
from hypothesis import strategies as st

from app.models.campaign import CampaignStatus
from app.services.campaign_validation_service import validate_campaign_context


campaign_statuses = st.sampled_from(
    [
        CampaignStatus.DRAFT.value,
        CampaignStatus.ACTIVE.value,
        CampaignStatus.COMPLETED.value,
        CampaignStatus.ARCHIVED.value,
    ]
)


class QueryResult:
    """Minimal result object matching the scalar query API used by the service."""

    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


def _campaign(campaign_id: UUID, campaign_status: str) -> MagicMock:
    campaign = MagicMock()
    campaign.id = campaign_id
    campaign.status = campaign_status
    return campaign


def _db(*results: QueryResult) -> MagicMock:
    db = MagicMock()
    db.execute = AsyncMock(side_effect=results)
    return db


class TestCampaignValidationProperties:
    """Properties for ordered campaign-context validation."""

    @given(
        campaign_exists=st.booleans(),
        assigned=st.booleans(),
        campaign_status=campaign_statuses,
        scenario_in_campaign=st.booleans(),
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_first_failing_condition_determines_response(
        self,
        campaign_exists: bool,
        assigned: bool,
        campaign_status: str,
        scenario_in_campaign: bool,
    ) -> None:
        """**Validates: Requirements 3.1, 3.2, 3.3, 3.4**"""
        campaign_id, agent_id, scenario_id = uuid4(), uuid4(), uuid4()
        campaign = _campaign(campaign_id, campaign_status) if campaign_exists else None
        db = _db(
            QueryResult(campaign),
            QueryResult(campaign_id if assigned else None),
            QueryResult(scenario_id if scenario_in_campaign else None),
        )

        expected = None
        if not campaign_exists:
            expected = (status.HTTP_404_NOT_FOUND, "Campaign not found")
        elif not assigned:
            expected = (
                status.HTTP_403_FORBIDDEN,
                "Agent is not assigned to this campaign",
            )
        elif campaign_status != CampaignStatus.ACTIVE.value:
            expected = (status.HTTP_400_BAD_REQUEST, "Campaign is not active")
        elif not scenario_in_campaign:
            expected = (
                status.HTTP_400_BAD_REQUEST,
                "Scenario does not belong to this campaign",
            )

        if expected is None:
            await validate_campaign_context(db, campaign_id, agent_id, scenario_id)
            assert db.execute.await_count == 3
        else:
            with pytest.raises(HTTPException) as error:
                await validate_campaign_context(db, campaign_id, agent_id, scenario_id)
            assert (error.value.status_code, error.value.detail) == expected

            expected_queries = 1 if not campaign_exists else 2
            if (
                campaign_exists
                and assigned
                and campaign_status == CampaignStatus.ACTIVE.value
            ):
                expected_queries = 3
            assert db.execute.await_count == expected_queries

    @given(campaign_status=campaign_statuses)
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_admin_bypasses_assignment_and_status_checks(
        self, campaign_status: str
    ) -> None:
        """**Validates: Requirement 3.5**"""
        campaign_id, agent_id, scenario_id = uuid4(), uuid4(), uuid4()
        db = _db(
            QueryResult(_campaign(campaign_id, campaign_status)),
            QueryResult(scenario_id),
        )

        await validate_campaign_context(
            db,
            campaign_id,
            agent_id,
            scenario_id,
            is_admin=True,
        )

        assert db.execute.await_count == 2

    @given(campaign_status=st.just(CampaignStatus.ACTIVE.value))
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_scenario_outside_campaign_is_rejected(
        self, campaign_status: str
    ) -> None:
        """**Validates: Requirements 2.5, 2.6**"""
        campaign_id, agent_id, scenario_id = uuid4(), uuid4(), uuid4()
        db = _db(
            QueryResult(_campaign(campaign_id, campaign_status)),
            QueryResult(agent_id),
            QueryResult(None),
        )

        with pytest.raises(HTTPException) as error:
            await validate_campaign_context(db, campaign_id, agent_id, scenario_id)

        assert error.value.status_code == status.HTTP_400_BAD_REQUEST
        assert error.value.detail == "Scenario does not belong to this campaign"

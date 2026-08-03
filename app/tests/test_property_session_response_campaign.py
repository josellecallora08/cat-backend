"""Property-based tests for campaign context in session responses.

Feature: campaign-based-session-selection
Property 8: Session Response Campaign Context.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.api.sessions import _session_to_response, list_sessions
from app.models.user import UserRole


campaign_strategy = st.one_of(
    st.none(),
    st.builds(
        lambda campaign_id, name: SimpleNamespace(id=campaign_id, name=name),
        campaign_id=st.uuids(version=4),
        name=st.text(min_size=1, max_size=100),
    ),
)


def _session(campaign: SimpleNamespace | None) -> SimpleNamespace:
    """Build the minimum session shape consumed by response serializers."""
    return SimpleNamespace(
        id=uuid4(),
        scenario_id=uuid4(),
        agent_id=uuid4(),
        status="active",
        campaign_id=campaign.id if campaign else None,
        campaign=campaign,
        persona_context=None,
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ended_at=None,
    )


class TestSessionResponseCampaignContext:
    """Universal campaign context guarantees for detail and list responses."""

    @given(campaign=campaign_strategy)
    @settings(max_examples=100)
    def test_detail_response_matches_campaign_relationship(
        self, campaign: SimpleNamespace | None
    ) -> None:
        """**Validates: Requirements 8.1, 8.3, 8.4, 8.5**"""
        response = _session_to_response(_session(campaign))

        assert response.campaign_id == (campaign.id if campaign else None)
        assert response.campaign_name == (campaign.name if campaign else None)

    @given(campaign=campaign_strategy)
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_list_response_matches_campaign_relationship(
        self, campaign: SimpleNamespace | None
    ) -> None:
        """**Validates: Requirements 8.2, 8.3, 8.4, 8.5**"""
        session = _session(campaign)
        count_result = MagicMock()
        count_result.scalar_one.return_value = 1
        rows_result = MagicMock()
        rows_result.all.return_value = [(session, None)]
        db = MagicMock()
        db.execute = AsyncMock(side_effect=[count_result, rows_result])
        user = SimpleNamespace(
            id=session.agent_id,
            role=UserRole.ADMIN.value,
            user_type=None,
        )

        response = await list_sessions(db=db, current_user=user)

        item = response.items[0]
        expected_campaign_id = str(campaign.id) if campaign else None
        expected_campaign_name = campaign.name if campaign else None
        assert item.campaign_id == expected_campaign_id
        assert item.campaign_name == expected_campaign_name

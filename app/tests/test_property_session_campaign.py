"""Property-based tests for session campaign persistence and compatibility.

Feature: campaign-based-session-selection
Properties 1, 2, and 11: persistence, deletion nullification, and no-campaign creation.

**Validates: Requirements 1.2, 1.3, 1.4, 2.1, 2.2, 3.6**
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Campaign, Session
from app.services.session_service import create_session
from app.tests.test_session_service import _make_mock_debtor_simulator, _make_scenario


pytest_plugins = ("app.tests.test_session_service",)


campaign_names = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
    min_size=1,
    max_size=30,
)


def _make_campaign(name: str) -> Campaign:
    """Create an active campaign with a unique database identity."""
    return Campaign(
        id=uuid.uuid4(),
        name=f"{name}-{uuid.uuid4().hex}",
        status="active",
    )


class TestSessionCampaignProperties:
    """Database-backed properties for campaign-aware sessions."""

    @given(name=campaign_names)
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @pytest.mark.asyncio
    async def test_campaign_id_persists_on_created_session(
        self, async_db: AsyncSession, name: str
    ) -> None:
        """**Validates: Requirements 1.2, 2.1**"""
        campaign = _make_campaign(name)
        scenario = _make_scenario()
        async_db.add_all([campaign, scenario])
        await async_db.commit()

        created = await create_session(
            async_db,
            scenario.id,
            uuid.uuid4(),
            _make_mock_debtor_simulator(),
            campaign_id=campaign.id,
        )
        result = await async_db.execute(select(Session).where(Session.id == created.id))
        persisted = result.scalar_one()

        assert persisted.campaign_id == campaign.id

    @given(name=campaign_names)
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @pytest.mark.asyncio
    async def test_deleting_campaign_nullifies_linked_sessions(
        self, async_db: AsyncSession, name: str
    ) -> None:
        """**Validates: Requirement 1.4**"""
        campaign = _make_campaign(name)
        scenario = _make_scenario()
        session = Session(
            id=uuid.uuid4(),
            scenario_id=scenario.id,
            agent_id=uuid.uuid4(),
            campaign_id=campaign.id,
            status="completed",
            persona_context={"name": "Test Persona"},
        )
        async_db.add_all([campaign, scenario, session])
        await async_db.commit()

        await async_db.delete(campaign)
        await async_db.commit()
        await async_db.refresh(session)

        assert session.campaign_id is None
        assert session.status == "completed"
        assert session.scenario_id == scenario.id
        assert session.persona_context == {"name": "Test Persona"}

    @given(st.just(None))
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @pytest.mark.asyncio
    async def test_session_creation_without_campaign_keeps_null_link(
        self, async_db: AsyncSession, _unused: None
    ) -> None:
        """**Validates: Requirements 1.3, 2.2, 3.6**"""
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.commit()

        created = await create_session(
            async_db,
            scenario.id,
            uuid.uuid4(),
            _make_mock_debtor_simulator(),
            campaign_id=None,
        )
        result = await async_db.execute(select(Session).where(Session.id == created.id))
        persisted = result.scalar_one()

        assert persisted.campaign_id is None
        assert persisted.scenario_id == scenario.id


class TestCriteriaCoachingCampaignSerializerExploration:
    """Property 1 probes for campaign-present and campaign-absent serialization."""

    @given(
        campaign_name=campaign_names,
        campaign_id=st.uuids(version=4),
    )
    @settings(max_examples=25)
    def test_loaded_campaign_relationship_is_serialized_safely(
        self, campaign_name: str, campaign_id: uuid.UUID
    ) -> None:
        """**Validates: Requirements 2.1, 2.2**"""
        from app.api.sessions import _session_to_response

        campaign = SimpleNamespace(id=campaign_id, name=campaign_name)
        session = SimpleNamespace(
            id=uuid.uuid4(),
            scenario_id=uuid.uuid4(),
            campaign_id=campaign_id,
            campaign=campaign,
            persona_context=None,
            status="completed",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ended_at=None,
            negotiation_standard_version=None,
        )

        response = _session_to_response(session)

        assert response.campaign_id == campaign_id
        assert response.campaign_name == campaign_name

    def test_absent_campaign_relationship_serializes_null_name(self) -> None:
        """**Validates: Requirements 2.2**"""
        from app.api.sessions import _session_to_response

        session = SimpleNamespace(
            id=uuid.uuid4(),
            scenario_id=uuid.uuid4(),
            campaign_id=None,
            campaign=None,
            persona_context=None,
            status="completed",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ended_at=None,
            negotiation_standard_version=None,
        )

        assert _session_to_response(session).campaign_name is None

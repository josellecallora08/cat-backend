"""Service layer for campaign-scenario association operations."""

from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Scenario
from app.models.campaign import (
    Campaign,
    CampaignAgent,
    CampaignStatus,
    campaign_scenarios,
)
from app.schemas.campaign import AgentCampaignScenarioItem, CampaignScenarioItem


async def add_scenarios_to_campaign(
    db: AsyncSession,
    campaign_id: UUID,
    scenario_ids: list[UUID],
) -> list[CampaignScenarioItem]:
    """Add scenarios to a campaign, skipping duplicates.

    Validates that the campaign exists and is not archived, and that all
    scenario IDs reference existing active scenarios.

    Args:
        db: Async database session.
        campaign_id: The campaign to add scenarios to.
        scenario_ids: List of scenario UUIDs to associate.

    Returns:
        Updated list of campaign scenarios sorted alphabetically by name.

    Raises:
        ValueError: If campaign not found/archived or scenario IDs are invalid.
    """
    await _get_campaign_or_raise(db, campaign_id)
    await _validate_scenario_ids_with_details(db, scenario_ids)

    # Fetch existing associations to skip duplicates
    existing_ids = await _get_existing_scenario_ids(db, campaign_id)
    new_ids = [sid for sid in scenario_ids if sid not in existing_ids]

    # Insert only new associations
    if new_ids:
        values = [{"campaign_id": campaign_id, "scenario_id": sid} for sid in new_ids]
        await db.execute(campaign_scenarios.insert().values(values))
        await db.commit()

    return await _get_sorted_campaign_scenarios(db, campaign_id)


async def remove_scenario_from_campaign(
    db: AsyncSession,
    campaign_id: UUID,
    scenario_id: UUID,
) -> list[CampaignScenarioItem]:
    """Remove a single scenario from a campaign.

    Args:
        db: Async database session.
        campaign_id: The campaign to remove the scenario from.
        scenario_id: The scenario UUID to disassociate.

    Returns:
        Updated list of campaign scenarios sorted alphabetically by name.

    Raises:
        ValueError: If campaign not found/archived or scenario not associated.
    """
    await _get_campaign_or_raise(db, campaign_id)

    # Verify the scenario is currently associated
    existing_ids = await _get_existing_scenario_ids(db, campaign_id)
    if scenario_id not in existing_ids:
        raise ValueError("Scenario is not assigned to this campaign")

    # Delete the association row
    await db.execute(
        delete(campaign_scenarios).where(
            campaign_scenarios.c.campaign_id == campaign_id,
            campaign_scenarios.c.scenario_id == scenario_id,
        )
    )
    await db.commit()

    return await _get_sorted_campaign_scenarios(db, campaign_id)


async def get_agent_campaign_scenarios(
    db: AsyncSession,
    agent_id: UUID,
) -> list[AgentCampaignScenarioItem]:
    """Get all active scenarios from an agent's active campaigns (deduplicated).

    Queries campaigns with status "active" where the agent is assigned,
    then collects scenarios that have `is_active` set to true.

    Args:
        db: Async database session.
        agent_id: The agent's user ID.

    Returns:
        Deduplicated list of active scenarios across the agent's active campaigns.
    """
    # Find active campaigns assigned to this agent
    agent_campaign_stmt = (
        select(CampaignAgent.campaign_id)
        .join(Campaign, Campaign.id == CampaignAgent.campaign_id)
        .where(
            CampaignAgent.agent_id == agent_id,
            Campaign.status == CampaignStatus.ACTIVE.value,
        )
    )
    result = await db.execute(agent_campaign_stmt)
    active_campaign_ids = [row[0] for row in result.all()]

    if not active_campaign_ids:
        return []

    # Collect active scenarios from those campaigns, deduplicated
    scenarios_stmt = (
        select(Scenario)
        .join(
            campaign_scenarios,
            campaign_scenarios.c.scenario_id == Scenario.id,
        )
        .where(
            campaign_scenarios.c.campaign_id.in_(active_campaign_ids),
            Scenario.is_active == True,  # noqa: E712
        )
        .distinct()
    )
    result = await db.execute(scenarios_stmt)
    scenarios = list(result.scalars().all())

    return [
        AgentCampaignScenarioItem(
            id=s.id,
            name=s.name,
            scenario_type=s.scenario_type,
            description=s.description or "",
        )
        for s in scenarios
    ]


async def get_agent_scenario_ids(
    db: AsyncSession,
    agent_id: UUID,
) -> set[UUID]:
    """Return deduplicated scenario IDs from an agent's active campaigns.

    Args:
        db: Async database session.
        agent_id: The agent's user ID.

    Returns:
        Set of scenario UUIDs visible to this agent.
    """
    agent_campaign_stmt = (
        select(CampaignAgent.campaign_id)
        .join(Campaign, Campaign.id == CampaignAgent.campaign_id)
        .where(
            CampaignAgent.agent_id == agent_id,
            Campaign.status == CampaignStatus.ACTIVE.value,
        )
    )
    result = await db.execute(agent_campaign_stmt)
    active_campaign_ids = [row[0] for row in result.all()]

    if not active_campaign_ids:
        return set()

    scenarios_stmt = (
        select(Scenario.id)
        .join(
            campaign_scenarios,
            campaign_scenarios.c.scenario_id == Scenario.id,
        )
        .where(
            campaign_scenarios.c.campaign_id.in_(active_campaign_ids),
            Scenario.is_active == True,  # noqa: E712
        )
        .distinct()
    )
    result = await db.execute(scenarios_stmt)
    return {row[0] for row in result.all()}


# --- Private helpers ---


async def _get_campaign_or_raise(db: AsyncSession, campaign_id: UUID) -> Campaign:
    """Fetch campaign by ID or raise ValueError if not found or archived."""
    stmt = select(Campaign).where(
        Campaign.id == campaign_id,
        Campaign.status != CampaignStatus.ARCHIVED.value,
    )
    result = await db.execute(stmt)
    campaign = result.scalar_one_or_none()
    if campaign is None:
        raise ValueError("Campaign not found")
    return campaign


async def _validate_scenario_ids_with_details(
    db: AsyncSession,
    scenario_ids: list[UUID],
) -> None:
    """Validate all scenario IDs reference existing active scenarios.

    On failure, collects invalid IDs and raises ValueError with the list.
    """
    stmt = select(Scenario.id).where(
        Scenario.id.in_(scenario_ids),
        Scenario.is_active == True,  # noqa: E712
    )
    result = await db.execute(stmt)
    valid_ids = {row[0] for row in result.all()}

    invalid_ids = [sid for sid in scenario_ids if sid not in valid_ids]
    if invalid_ids:
        raise ValueError(f"Invalid scenario IDs: {invalid_ids}")


async def _get_existing_scenario_ids(
    db: AsyncSession,
    campaign_id: UUID,
) -> set[UUID]:
    """Return the set of scenario IDs already associated with a campaign."""
    stmt = select(campaign_scenarios.c.scenario_id).where(
        campaign_scenarios.c.campaign_id == campaign_id
    )
    result = await db.execute(stmt)
    return {row[0] for row in result.all()}


async def _get_sorted_campaign_scenarios(
    db: AsyncSession,
    campaign_id: UUID,
) -> list[CampaignScenarioItem]:
    """Fetch all scenarios for a campaign, sorted alphabetically by name."""
    stmt = (
        select(Scenario)
        .join(
            campaign_scenarios,
            campaign_scenarios.c.scenario_id == Scenario.id,
        )
        .where(campaign_scenarios.c.campaign_id == campaign_id)
        .order_by(Scenario.name.asc())
    )
    result = await db.execute(stmt)
    scenarios = list(result.scalars().all())

    return [
        CampaignScenarioItem(
            id=s.id,
            name=s.name,
            scenario_type=s.scenario_type,
        )
        for s in scenarios
    ]

"""Campaign boundary validation for session creation."""

from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.campaign import (
    Campaign,
    CampaignAgent,
    CampaignStatus,
    campaign_scenarios,
)


async def validate_campaign_context(
    db: AsyncSession,
    campaign_id: UUID,
    agent_id: UUID,
    scenario_id: UUID,
    is_admin: bool = False,
) -> None:
    """Validate a campaign, agent assignment, and scenario association.

    Validation is deliberately ordered so that a missing campaign is reported before
    authorization or membership details. Administrators bypass assignment and status
    checks, but the campaign and scenario must still exist in the requested context.

    Args:
        db: Database session used for validation queries.
        campaign_id: Campaign requested for the new session.
        agent_id: Authenticated user's ID.
        scenario_id: Scenario requested for the new session.
        is_admin: Whether the authenticated user has administrative privileges.

    Raises:
        HTTPException: If any campaign boundary condition is not satisfied.
    """
    campaign = await _get_campaign(db, campaign_id)
    if campaign is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Campaign not found",
        )

    if not is_admin:
        await _require_assignment(db, campaign_id, agent_id)
        _require_active_campaign(campaign)

    await _require_scenario_membership(db, campaign_id, scenario_id)


async def _get_campaign(db: AsyncSession, campaign_id: UUID) -> Campaign | None:
    """Return the campaign for an ID, including inactive campaigns."""
    result = await db.execute(select(Campaign).where(Campaign.id == campaign_id))
    return result.scalar_one_or_none()


async def _require_assignment(
    db: AsyncSession,
    campaign_id: UUID,
    agent_id: UUID,
) -> None:
    """Ensure the user has any assignment on the campaign."""
    result = await db.execute(
        select(CampaignAgent.campaign_id).where(
            CampaignAgent.campaign_id == campaign_id,
            CampaignAgent.agent_id == agent_id,
        )
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Agent is not assigned to this campaign",
        )


def _require_active_campaign(campaign: Campaign) -> None:
    """Ensure a non-admin may create sessions only in active campaigns."""
    if campaign.status != CampaignStatus.ACTIVE.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Campaign is not active",
        )


async def _require_scenario_membership(
    db: AsyncSession,
    campaign_id: UUID,
    scenario_id: UUID,
) -> None:
    """Ensure the scenario is assigned to the requested campaign."""
    result = await db.execute(
        select(campaign_scenarios.c.scenario_id).where(
            campaign_scenarios.c.campaign_id == campaign_id,
            campaign_scenarios.c.scenario_id == scenario_id,
        )
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Scenario does not belong to this campaign",
        )

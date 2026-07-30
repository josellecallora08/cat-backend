"""Service layer for trainer campaign resolution and access verification."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.campaign import (
    Campaign,
    CampaignAgent,
    CampaignRole,
    CampaignStatus,
)


async def get_trainer_campaign(db: AsyncSession, trainer_id: UUID) -> Campaign | None:
    """Resolve the trainer's primary active campaign assignment.

    Returns the first active campaign where the trainer has role='trainer'
    in campaign_agents. Returns None if no active campaign is assigned.
    """
    stmt = (
        select(Campaign)
        .join(CampaignAgent, CampaignAgent.campaign_id == Campaign.id)
        .where(
            CampaignAgent.agent_id == trainer_id,
            CampaignAgent.role == CampaignRole.TRAINER.value,
            Campaign.status == CampaignStatus.ACTIVE.value,
        )
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_trainer_campaign_agent_ids(
    db: AsyncSession,
    campaign_id: UUID,
) -> list[UUID]:
    """Get all agent IDs (participants) assigned to a campaign.

    Returns agent_id UUIDs for users with participant or team_lead roles.
    Trainers are excluded since they are not agents in the campaign.
    """
    stmt = select(CampaignAgent.agent_id).where(
        CampaignAgent.campaign_id == campaign_id,
        CampaignAgent.role.in_(
            [
                CampaignRole.PARTICIPANT.value,
                CampaignRole.TEAM_LEAD.value,
            ]
        ),
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def verify_trainer_campaign_access(
    db: AsyncSession,
    trainer_id: UUID,
    campaign_id: UUID,
) -> bool:
    """Check if a trainer is assigned to a specific campaign.

    Returns True if a campaign_agents record exists with the given trainer_id,
    campaign_id, and role='trainer'. Returns False otherwise.
    """
    stmt = select(CampaignAgent.agent_id).where(
        CampaignAgent.agent_id == trainer_id,
        CampaignAgent.campaign_id == campaign_id,
        CampaignAgent.role == CampaignRole.TRAINER.value,
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none() is not None

"""Services for calculating agent progress within training campaigns."""

from uuid import UUID

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Scenario, Session
from app.models.campaign import Campaign, CampaignAgent, campaign_scenarios
from app.schemas.campaign_progress import (
    AgentCampaignWithProgress,
    CampaignProgressResponse,
    ScenarioProgressItem,
)


async def get_campaign_progress(
    db: AsyncSession,
    campaign_id: UUID,
    agent_id: UUID,
) -> CampaignProgressResponse:
    """Return detailed scenario progress for an agent in a campaign."""
    campaign_result = await db.execute(
        select(Campaign.id, Campaign.name).where(Campaign.id == campaign_id)
    )
    campaign = campaign_result.one_or_none()
    if campaign is None:
        raise ValueError(f"Campaign with id {campaign_id} not found")

    completed_exists = (
        select(Session.id)
        .where(
            Session.scenario_id == campaign_scenarios.c.scenario_id,
            Session.campaign_id == campaign_id,
            Session.agent_id == agent_id,
            Session.status == "completed",
        )
        .exists()
    )
    scenario_statement = (
        select(
            Scenario.id,
            Scenario.name,
            Scenario.scenario_type,
            completed_exists.label("accomplished"),
        )
        .join(campaign_scenarios, campaign_scenarios.c.scenario_id == Scenario.id)
        .where(campaign_scenarios.c.campaign_id == campaign_id)
        .order_by(func.lower(Scenario.name), Scenario.id)
    )
    scenario_rows = (await db.execute(scenario_statement)).all()
    scenarios = [
        ScenarioProgressItem(
            scenario_id=row.id,
            scenario_name=row.name,
            scenario_type=row.scenario_type,
            accomplished=bool(row.accomplished),
        )
        for row in scenario_rows
    ]
    accomplished_count = sum(item.accomplished for item in scenarios)
    total_count = len(scenarios)
    return CampaignProgressResponse(
        campaign_id=campaign.id,
        campaign_name=campaign.name,
        total_scenarios=total_count,
        accomplished_scenarios=accomplished_count,
        is_completed=total_count > 0 and accomplished_count == total_count,
        scenarios=scenarios,
    )


async def get_agent_campaigns_with_progress(
    db: AsyncSession,
    agent_id: UUID,
) -> list[AgentCampaignWithProgress]:
    """Return active campaigns assigned to an agent with progress summaries."""
    campaign_statement = (
        select(Campaign.id, Campaign.name, Campaign.description)
        .join(CampaignAgent, CampaignAgent.campaign_id == Campaign.id)
        .where(
            CampaignAgent.agent_id == agent_id,
            Campaign.status == "active",
        )
        .order_by(Campaign.name, Campaign.id)
    )
    campaigns = (await db.execute(campaign_statement)).all()
    if not campaigns:
        return []

    completed_scenario = case((Session.id.is_not(None), campaign_scenarios.c.scenario_id))
    progress_statement = (
        select(
            campaign_scenarios.c.campaign_id,
            func.count(func.distinct(campaign_scenarios.c.scenario_id)).label("total_scenarios"),
            func.count(func.distinct(completed_scenario)).label("accomplished_scenarios"),
        )
        .outerjoin(
            Session,
            (Session.scenario_id == campaign_scenarios.c.scenario_id)
            & (Session.campaign_id == campaign_scenarios.c.campaign_id)
            & (Session.agent_id == agent_id)
            & (Session.status == "completed"),
        )
        .where(
            campaign_scenarios.c.campaign_id.in_(campaign.id for campaign in campaigns),
        )
        .group_by(campaign_scenarios.c.campaign_id)
    )
    progress_rows = (await db.execute(progress_statement)).all()
    progress_by_campaign = {row.campaign_id: row for row in progress_rows}

    summaries = []
    for campaign in campaigns:
        progress = progress_by_campaign.get(campaign.id)
        total_count = int(progress.total_scenarios) if progress else 0
        accomplished_count = int(progress.accomplished_scenarios) if progress else 0
        summaries.append(
            AgentCampaignWithProgress(
                id=campaign.id,
                name=campaign.name,
                description=campaign.description,
                total_scenarios=total_count,
                accomplished_scenarios=accomplished_count,
                is_completed=total_count == 0 or accomplished_count == total_count,
            )
        )
    return sorted(
        summaries,
        key=lambda campaign: (
            campaign.is_completed,
            campaign.name.casefold(),
            campaign.id,
        ),
    )

"""Service layer for campaign CRUD operations."""

import math
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.campaign import (
    Campaign,
    CampaignAgent,
    CampaignStatus,
    campaign_scenarios,
)
from app.models.user import User, UserRole, UserType
from app.models import Scenario
from app.schemas.campaign import (
    AgentAssignment,
    CampaignCreate,
    CampaignListItem,
    CampaignUpdate,
    PaginatedCampaigns,
)


async def list_campaigns(
    db: AsyncSession,
    page: int,
    page_size: int,
    status_filter: str | None,
) -> PaginatedCampaigns:
    """List campaigns with pagination and optional status filter.

    Excludes archived campaigns unless explicitly filtered.
    """
    base_query = select(Campaign)

    if status_filter:
        base_query = base_query.where(Campaign.status == status_filter)
    else:
        base_query = base_query.where(Campaign.status != CampaignStatus.ARCHIVED.value)

    total = await _count_campaigns(db, base_query)
    total_pages = math.ceil(total / page_size) if total > 0 else 0

    stmt = (
        base_query.order_by(Campaign.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    result = await db.execute(stmt)
    campaigns = list(result.scalars().all())

    items = [_to_list_item(c) for c in campaigns]

    return PaginatedCampaigns(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


async def create_campaign(db: AsyncSession, data: CampaignCreate) -> Campaign:
    """Create a new campaign with scenario and agent associations.

    Raises:
        ValueError: If name already exists, or scenario/agent IDs are invalid.
    """
    await _check_name_uniqueness(db, data.name)

    scenario_ids = data.scenario_ids or []
    if scenario_ids:
        await _validate_scenario_ids(db, scenario_ids)
    await _validate_agent_ids(db, data.agents)

    campaign = Campaign(
        name=data.name,
        description=data.description,
        start_date=data.start_date,
        end_date=data.end_date,
        status=CampaignStatus.DRAFT.value,
    )
    db.add(campaign)
    await db.flush()

    if scenario_ids:
        await _insert_scenario_associations(db, campaign.id, scenario_ids)
    await _insert_agent_associations(db, campaign.id, data.agents)
    await db.commit()
    await db.refresh(campaign)
    return campaign


async def get_campaign_by_id(db: AsyncSession, campaign_id: UUID) -> Campaign | None:
    """Fetch a single campaign by ID, excluding archived ones.

    Returns None if not found or archived.
    """
    stmt = select(Campaign).where(
        Campaign.id == campaign_id,
        Campaign.status != CampaignStatus.ARCHIVED.value,
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def update_campaign(
    db: AsyncSession,
    campaign_id: UUID,
    data: CampaignUpdate,
) -> Campaign:
    """Partially update a campaign and replace associations if provided.

    Raises:
        ValueError: If campaign not found, name conflicts, or dates invalid.
    """
    campaign = await _get_campaign_or_raise(db, campaign_id)
    await _apply_field_updates(db, campaign, data)

    if data.scenario_ids is not None:
        await _validate_scenario_ids(db, data.scenario_ids)
        await _replace_scenario_associations(db, campaign_id, data.scenario_ids)

    if data.agents is not None:
        await _validate_agent_ids(db, data.agents)
        await _replace_agent_associations(db, campaign_id, data.agents)

    await db.commit()
    await db.refresh(campaign)
    return campaign


async def archive_campaign(db: AsyncSession, campaign_id: UUID) -> None:
    """Archive a campaign by setting its status to archived.

    Raises:
        ValueError: If campaign not found.
    """
    campaign = await _get_campaign_or_raise(db, campaign_id)
    campaign.status = CampaignStatus.ARCHIVED.value
    await db.commit()


# --- Private helpers ---


async def _count_campaigns(db: AsyncSession, base_query) -> int:
    """Count total campaigns matching the base query."""
    count_stmt = select(func.count()).select_from(base_query.subquery())
    result = await db.execute(count_stmt)
    return result.scalar_one()


def _to_list_item(campaign: Campaign) -> CampaignListItem:
    """Convert a Campaign ORM instance to a CampaignListItem schema."""
    return CampaignListItem(
        id=campaign.id,
        name=campaign.name,
        status=campaign.status,
        scenarios_count=len(campaign.scenarios),
        agents_count=len(campaign.agent_assignments),
        start_date=campaign.start_date,
        end_date=campaign.end_date,
        created_at=campaign.created_at,
        updated_at=campaign.updated_at,
    )


async def _check_name_uniqueness(
    db: AsyncSession,
    name: str,
    exclude_id: UUID | None = None,
) -> None:
    """Raise ValueError if a campaign with the given name already exists."""
    stmt = select(Campaign.id).where(Campaign.name == name)
    if exclude_id:
        stmt = stmt.where(Campaign.id != exclude_id)
    result = await db.execute(stmt)
    if result.scalar_one_or_none() is not None:
        raise ValueError("Campaign name already exists")


async def _validate_scenario_ids(db: AsyncSession, scenario_ids: list[UUID]) -> None:
    """Raise ValueError if any scenario ID is invalid or inactive."""
    stmt = select(func.count()).where(
        Scenario.id.in_(scenario_ids),
        Scenario.is_active == True,  # noqa: E712
    )
    result = await db.execute(stmt)
    count = result.scalar_one()
    if count != len(scenario_ids):
        raise ValueError("One or more scenario_ids are invalid or inactive")


async def _validate_agent_ids(
    db: AsyncSession,
    agents: list[AgentAssignment],
) -> None:
    """Raise ValueError if any agent ID is invalid or inactive."""
    agent_ids = [a.agent_id for a in agents]
    stmt = select(func.count()).where(
        User.id.in_(agent_ids),
        User.is_active == True,  # noqa: E712
        User.role == UserRole.USER.value,
        User.user_type == UserType.AGENT.value,
    )
    result = await db.execute(stmt)
    count = result.scalar_one()
    if count != len(agent_ids):
        raise ValueError("One or more agent_ids are invalid or inactive")


async def _insert_scenario_associations(
    db: AsyncSession,
    campaign_id: UUID,
    scenario_ids: list[UUID],
) -> None:
    """Insert campaign-scenario association rows."""
    values = [{"campaign_id": campaign_id, "scenario_id": sid} for sid in scenario_ids]
    await db.execute(campaign_scenarios.insert().values(values))


async def _insert_agent_associations(
    db: AsyncSession,
    campaign_id: UUID,
    agents: list[AgentAssignment],
) -> None:
    """Insert campaign-agent association rows with role."""
    for agent in agents:
        assignment = CampaignAgent(
            campaign_id=campaign_id,
            agent_id=agent.agent_id,
            role=agent.role.value,
        )
        db.add(assignment)


async def _replace_scenario_associations(
    db: AsyncSession,
    campaign_id: UUID,
    scenario_ids: list[UUID],
) -> None:
    """Delete existing scenario associations and insert new ones."""
    await db.execute(
        delete(campaign_scenarios).where(
            campaign_scenarios.c.campaign_id == campaign_id
        )
    )
    await _insert_scenario_associations(db, campaign_id, scenario_ids)


async def _replace_agent_associations(
    db: AsyncSession,
    campaign_id: UUID,
    agents: list[AgentAssignment],
) -> None:
    """Delete existing agent associations and insert new ones."""
    await db.execute(
        delete(CampaignAgent).where(CampaignAgent.campaign_id == campaign_id)
    )
    await _insert_agent_associations(db, campaign_id, agents)


async def _get_campaign_or_raise(db: AsyncSession, campaign_id: UUID) -> Campaign:
    """Fetch campaign by ID or raise ValueError if not found."""
    stmt = select(Campaign).where(
        Campaign.id == campaign_id,
        Campaign.status != CampaignStatus.ARCHIVED.value,
    )
    result = await db.execute(stmt)
    campaign = result.scalar_one_or_none()
    if campaign is None:
        raise ValueError("Campaign not found")
    return campaign


async def _apply_field_updates(
    db: AsyncSession,
    campaign: Campaign,
    data: CampaignUpdate,
) -> None:
    """Apply scalar field updates to the campaign instance."""
    if data.name is not None and data.name != campaign.name:
        await _check_name_uniqueness(db, data.name, exclude_id=campaign.id)
        campaign.name = data.name

    if data.description is not None:
        campaign.description = data.description

    if data.status is not None:
        campaign.status = data.status.value

    _apply_date_updates(campaign, data)


def _apply_date_updates(campaign: Campaign, data: CampaignUpdate) -> None:
    """Apply and validate date field changes."""
    start = data.start_date if data.start_date is not None else campaign.start_date
    end = data.end_date if data.end_date is not None else campaign.end_date

    if start is not None and end is not None and end <= start:
        raise ValueError("end_date must be after start_date")

    if data.start_date is not None:
        campaign.start_date = data.start_date
    if data.end_date is not None:
        campaign.end_date = data.end_date

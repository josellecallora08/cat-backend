"""Campaign management API endpoints (admin only)."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models.campaign import Campaign, CampaignAgent
from app.models.user import User
from app.schemas.campaign import (
    CampaignAgentItem,
    CampaignCreate,
    CampaignDetail,
    CampaignScenarioItem,
    CampaignUpdate,
    PaginatedCampaigns,
)
from app.schemas.campaign_progress import (
    AgentCampaignWithProgress,
    CampaignProgressResponse,
)
from app.services.auth import require_admin, require_agent
from app.services.campaign_progress_service import (
    get_agent_campaigns_with_progress,
    get_campaign_progress,
)
from app.services.campaign_service import (
    archive_campaign,
    create_campaign,
    get_campaign_by_id,
    list_campaigns,
    update_campaign,
)


router = APIRouter()


def _campaign_to_detail(campaign) -> CampaignDetail:
    """Convert a Campaign ORM instance to a CampaignDetail response schema."""
    return CampaignDetail(
        id=campaign.id,
        name=campaign.name,
        description=campaign.description,
        status=campaign.status,
        start_date=campaign.start_date,
        end_date=campaign.end_date,
        scenarios=[
            CampaignScenarioItem(
                id=s.id,
                name=s.name,
                scenario_type=s.scenario_type,
            )
            for s in campaign.scenarios
        ],
        agents=[
            CampaignAgentItem(
                id=assignment.agent.id,
                full_name=assignment.agent.full_name,
                email=assignment.agent.email,
                role=assignment.role,
            )
            for assignment in campaign.agent_assignments
        ],
        created_at=campaign.created_at,
        updated_at=campaign.updated_at,
    )


@router.get("", response_model=PaginatedCampaigns)
async def list_campaigns_endpoint(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=15, ge=1, le=100),
    status: str | None = Query(default=None),
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> PaginatedCampaigns:
    """List campaigns with pagination and optional status filter."""
    return await list_campaigns(db, page, page_size, status)


@router.post("", response_model=CampaignDetail, status_code=201)
async def create_campaign_endpoint(
    body: CampaignCreate,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> CampaignDetail:
    """Create a new campaign with assigned scenarios and agents."""
    try:
        campaign = await create_campaign(db, body)
    except ValueError as e:
        msg = str(e)
        if "already exists" in msg:
            raise HTTPException(status_code=409, detail=msg)
        raise HTTPException(status_code=422, detail=msg)
    return _campaign_to_detail(campaign)


@router.get("/my", response_model=list[AgentCampaignWithProgress])
async def list_my_campaigns_endpoint(
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(require_agent),
) -> list[AgentCampaignWithProgress]:
    """Return active campaigns assigned to the authenticated agent."""
    return await get_agent_campaigns_with_progress(db, current_user.id)


@router.get("/{campaign_id}/progress", response_model=CampaignProgressResponse)
async def get_campaign_progress_endpoint(
    campaign_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(require_agent),
) -> CampaignProgressResponse:
    """Return scenario completion progress for an assigned agent."""
    campaign = await db.scalar(select(Campaign.id).where(Campaign.id == campaign_id))
    if campaign is None:
        raise HTTPException(status_code=404, detail="Campaign not found")

    assignment = await db.scalar(
        select(CampaignAgent.campaign_id).where(
            CampaignAgent.campaign_id == campaign_id,
            CampaignAgent.agent_id == current_user.id,
        )
    )
    if assignment is None:
        raise HTTPException(
            status_code=403,
            detail="Agent is not assigned to this campaign",
        )

    try:
        return await get_campaign_progress(db, campaign_id, current_user.id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="Campaign not found") from error


@router.get("/{campaign_id}", response_model=CampaignDetail)
async def get_campaign_endpoint(
    campaign_id: UUID,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> CampaignDetail:
    """Get full campaign detail including assigned scenarios and agents."""
    campaign = await get_campaign_by_id(db, campaign_id)
    if campaign is None:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return _campaign_to_detail(campaign)


@router.put("/{campaign_id}", response_model=CampaignDetail)
async def update_campaign_endpoint(
    campaign_id: UUID,
    body: CampaignUpdate,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> CampaignDetail:
    """Partially update a campaign's fields and associations."""
    try:
        campaign = await update_campaign(db, campaign_id, body)
    except ValueError as e:
        msg = str(e)
        if "not found" in msg:
            raise HTTPException(status_code=404, detail=msg)
        if "already exists" in msg:
            raise HTTPException(status_code=409, detail=msg)
        raise HTTPException(status_code=422, detail=msg)
    return _campaign_to_detail(campaign)


@router.delete("/{campaign_id}", status_code=204)
async def delete_campaign_endpoint(
    campaign_id: UUID,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> None:
    """Archive a campaign (soft delete)."""
    try:
        await archive_campaign(db, campaign_id)
    except ValueError as e:
        msg = str(e)
        if "not found" in msg:
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=422, detail=msg)

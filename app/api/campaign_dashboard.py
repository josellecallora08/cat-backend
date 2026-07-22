"""Campaign dashboard API endpoints (admin only)."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models.user import User
from app.schemas.campaign_dashboard import (
    AgentProgressResponse,
    CampaignDashboardResponse,
)
from app.services.auth import require_admin
from app.services.campaign_dashboard_service import (
    get_agent_progress,
    get_campaign_dashboard,
)


router = APIRouter()


@router.get(
    "/{campaign_id}/dashboard",
    response_model=CampaignDashboardResponse,
)
async def campaign_dashboard(
    campaign_id: UUID,
    agent_id: UUID | None = Query(default=None),
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> CampaignDashboardResponse:
    """Get aggregated dashboard metrics for a campaign."""
    try:
        return await get_campaign_dashboard(db, campaign_id, agent_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get(
    "/{campaign_id}/agents/{agent_id}/progress",
    response_model=AgentProgressResponse,
)
async def agent_progress(
    campaign_id: UUID,
    agent_id: UUID,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> AgentProgressResponse:
    """Get detailed progress for a single agent within a campaign."""
    try:
        return await get_agent_progress(db, campaign_id, agent_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

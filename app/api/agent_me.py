"""Agent-facing API endpoints for authenticated agent's own resources."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models.user import User
from app.schemas.campaign import AgentCampaignScenarioItem
from app.services.auth import require_agent
from app.services.campaign_scenario_service import get_agent_campaign_scenarios


router = APIRouter()


@router.get(
    "/campaign-scenarios",
    response_model=list[AgentCampaignScenarioItem],
)
async def agent_campaign_scenarios(
    user: User = Depends(require_agent),
    db: AsyncSession = Depends(get_session),
) -> list[AgentCampaignScenarioItem]:
    """Return all active scenarios from the authenticated agent's active campaigns.

    Requires agent-level auth (agent or admin role). Returns a deduplicated list
    of scenarios across all active campaigns the user is assigned to.
    """
    return await get_agent_campaign_scenarios(db, user.id)

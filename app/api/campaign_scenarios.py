"""Campaign scenario association API endpoints (admin only)."""

import uuid as uuid_lib
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import Scenario
from app.models.campaign import campaign_scenarios
from app.models.user import User
from app.schemas import CreateScenarioRequest
from app.schemas.campaign import (
    AddScenariosRequest,
    CampaignScenariosResponse,
)
from app.services.auth import require_admin
from app.services.campaign_scenario_service import (
    _get_campaign_or_raise,
    add_scenarios_to_campaign,
    remove_scenario_from_campaign,
)


router = APIRouter()


@router.post(
    "/{campaign_id}/scenarios",
    response_model=CampaignScenariosResponse,
)
async def add_scenarios_endpoint(
    campaign_id: UUID,
    body: AddScenariosRequest,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> CampaignScenariosResponse:
    """Add scenarios to a campaign.

    Accepts a list of scenario IDs (1–50) and associates them with the campaign,
    skipping any that are already assigned. Returns the updated scenario list.
    """
    try:
        scenarios = await add_scenarios_to_campaign(db, campaign_id, body.scenario_ids)
    except ValueError as e:
        msg = str(e)
        if "Campaign not found" in msg:
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=422, detail=msg)
    return CampaignScenariosResponse(scenarios=scenarios)


@router.delete(
    "/{campaign_id}/scenarios/{scenario_id}",
    response_model=CampaignScenariosResponse,
)
async def remove_scenario_endpoint(
    campaign_id: UUID,
    scenario_id: UUID,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> CampaignScenariosResponse:
    """Remove a single scenario from a campaign.

    Returns the updated scenario list after removal.
    """
    try:
        scenarios = await remove_scenario_from_campaign(db, campaign_id, scenario_id)
    except ValueError as e:
        msg = str(e)
        raise HTTPException(status_code=404, detail=msg)
    return CampaignScenariosResponse(scenarios=scenarios)


class CreatedScenarioResponse(BaseModel):
    """Response for a newly created scenario."""

    id: UUID
    name: str
    scenario_type: str
    description: str
    debtor_profile: dict


@router.post(
    "/{campaign_id}/scenarios/create",
    response_model=CreatedScenarioResponse,
    status_code=201,
)
async def create_scenario_for_campaign(
    campaign_id: UUID,
    body: CreateScenarioRequest,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
) -> CreatedScenarioResponse:
    """Create a custom scenario and add it to the campaign atomically.

    Creates a new scenario from the provided fields and links it to the
    specified campaign in a single transaction.
    """
    # Verify campaign exists and is not archived
    try:
        await _get_campaign_or_raise(db, campaign_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Build debtor_profile JSON
    debtor_profile = {
        "name": body.debtor_profile.name,
        "outstanding_balance": str(body.debtor_profile.outstanding_balance),
        "days_past_due": body.debtor_profile.days_past_due,
        "personality_profile": body.debtor_profile.personality_profile,
        "conversation_goal": body.debtor_profile.conversation_goal,
        "prompt_blocks": body.debtor_profile.prompt_blocks,
    }

    # Create scenario
    scenario_id = uuid_lib.uuid4()
    scenario = Scenario(
        id=scenario_id,
        name=body.name,
        scenario_type=body.scenario_type.value,
        description=body.description,
        debtor_profile=debtor_profile,
        is_active=True,
    )
    db.add(scenario)

    # Link to campaign
    await db.execute(
        campaign_scenarios.insert().values(
            campaign_id=campaign_id,
            scenario_id=scenario_id,
        )
    )

    # Atomic commit
    await db.commit()
    await db.refresh(scenario)

    return CreatedScenarioResponse(
        id=scenario.id,
        name=scenario.name,
        scenario_type=scenario.scenario_type,
        description=scenario.description or "",
        debtor_profile=scenario.debtor_profile,
    )

"""Scenario selection API endpoints."""

import json
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import Scenario
from app.models.user import User, UserRole, UserType
from app.schemas import (
    DebtorProfileSchema,
    ScenarioListItem,
    ScenarioResponse,
    ScenarioType,
)
from app.services.auth import get_current_user, require_admin
from app.services.campaign_scenario_service import get_agent_campaign_scenarios
from app.services.llm_service import LLMMessage, LLMService
from app.services.scenario_repository import get_scenario_by_id, list_active_scenarios

router = APIRouter()


@router.get("", response_model=list[ScenarioListItem])
async def list_scenarios(
    db: AsyncSession = Depends(get_session),
    user: User | None = Depends(get_current_user),
) -> list[ScenarioListItem]:
    """List active training scenarios. Agents see only their campaign scenarios."""
    if (
        user
        and user.role == UserRole.USER.value
        and user.user_type == UserType.AGENT.value
    ):
        items = await get_agent_campaign_scenarios(db, user.id)
        return [
            ScenarioListItem(
                id=item.id,
                name=item.name,
                scenario_type=ScenarioType(item.scenario_type),
                description=item.description,
            )
            for item in items
        ]

    scenarios = await list_active_scenarios(db)
    return [
        ScenarioListItem(
            id=s.id,
            name=s.name,
            scenario_type=ScenarioType(s.scenario_type),
            description=s.description or "",
        )
        for s in scenarios
    ]


@router.get("/{scenario_id}", response_model=ScenarioResponse)
async def get_scenario(scenario_id: UUID, db: AsyncSession = Depends(get_session)):
    """Get scenario details including debtor profile."""
    scenario = await get_scenario_by_id(db, scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Scenario not found")

    # Validate debtor profile data completeness (Requirement 1.6)
    try:
        debtor_profile = DebtorProfileSchema(**scenario.debtor_profile)
    except (ValidationError, TypeError, KeyError):
        raise HTTPException(
            status_code=422,
            detail="Scenario cannot be loaded: debtor profile data is incomplete",
        )

    return ScenarioResponse(
        id=scenario.id,
        name=scenario.name,
        scenario_type=ScenarioType(scenario.scenario_type),
        description=scenario.description or "",
        debtor_profile=debtor_profile,
    )


# --- AI-Powered Scenario Generation ---

logger = logging.getLogger(__name__)


class GenerateScenarioRequest(BaseModel):
    """Request body for AI-generated scenario creation."""

    prompt: str  # Free-text description of the scenario to generate
    scenario_type: str = "FINANCIAL_HARDSHIP"  # Default type, LLM may override


class GenerateScenarioResponse(BaseModel):
    """Response after generating and saving a new scenario."""

    id: UUID
    name: str
    scenario_type: str
    description: str
    debtor_profile: dict


SCENARIO_GENERATION_PROMPT = """You are a training scenario designer for a debt collection agent training platform in the Philippines.

Based on the user's description, generate a complete training scenario with a realistic Filipino debtor profile.

The scenario should be challenging and realistic. The debtor should have a believable backstory, personality, and reason for delinquency.

The user may provide:
- Debtor name and gender
- Outstanding amount and days past due
- A backstory/situation
- Special behavioral instructions (e.g., "will hang up if...", "will agree to pay if...")

Include any behavioral instructions in the personality_profile field so the debtor behaves accordingly during training.

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{
    "name": "<short scenario name, 3-5 words>",
    "scenario_type": "<one of: FINANCIAL_HARDSHIP, ANGRY_CUSTOMER, PAYMENT_EXTENSION, BALANCE_DISPUTE>",
    "description": "<2-3 sentence description of the scenario and what the agent will practice>",
    "debtor_profile": {
        "name": "<realistic Filipino name>",
        "outstanding_balance": "<amount as string, e.g. '25000.00'>",
        "days_past_due": <integer>,
        "personality_profile": "<detailed personality description in 1-2 sentences>",
        "conversation_goal": "<what the debtor is trying to achieve in the call>"
    }
}

Make the scenario feel real and grounded in Filipino culture. The debtor should have natural motivations and realistic financial circumstances."""


@router.post("/generate", response_model=GenerateScenarioResponse)
async def generate_scenario(
    body: GenerateScenarioRequest,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
):
    """Generate a new scenario from a natural language prompt using AI.

    The LLM creates a complete debtor profile and scenario details based on
    the user's description. The scenario is saved to the database and
    immediately available for training.
    """
    llm_service = LLMService()

    messages = [
        LLMMessage(role="system", content=SCENARIO_GENERATION_PROMPT),
        LLMMessage(
            role="user",
            content=f"Create a scenario based on this description: {body.prompt}",
        ),
    ]

    try:
        response = await llm_service.chat_completion(
            messages,
            temperature=0.8,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.error("Scenario generation LLM call failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to generate scenario")

    # Parse LLM response
    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content[content.index("\n") + 1 :]
        if content.endswith("```"):
            content = content[:-3]
        data = json.loads(content.strip())
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("Failed to parse scenario generation response: %s", e)
        raise HTTPException(
            status_code=500, detail="Failed to parse generated scenario"
        )

    # Validate required fields
    debtor_profile = data.get("debtor_profile", {})
    if not debtor_profile.get("name") or not debtor_profile.get("outstanding_balance"):
        raise HTTPException(
            status_code=500, detail="Generated scenario has incomplete profile"
        )

    # Save to database
    import uuid

    scenario = Scenario(
        id=uuid.uuid4(),
        name=data.get("name", "Generated Scenario"),
        scenario_type=data.get("scenario_type", body.scenario_type),
        description=data.get("description", "AI-generated training scenario"),
        debtor_profile=debtor_profile,
        is_active=True,
    )

    db.add(scenario)
    await db.commit()
    await db.refresh(scenario)

    logger.info("Generated new scenario: %s (id=%s)", scenario.name, scenario.id)

    return GenerateScenarioResponse(
        id=scenario.id,
        name=scenario.name,
        scenario_type=scenario.scenario_type,
        description=scenario.description,
        debtor_profile=scenario.debtor_profile,
    )


# --- CRUD Endpoints (Admin) ---


class UpdateScenarioRequest(BaseModel):
    """Request body for updating a scenario."""

    name: str | None = None
    scenario_type: str | None = None
    description: str | None = None
    debtor_profile: dict | None = None
    is_active: bool | None = None


class ScenarioDetailResponse(BaseModel):
    """Full scenario detail for admin views."""

    id: UUID
    name: str
    scenario_type: str
    description: str
    debtor_profile: dict
    is_active: bool


@router.put("/{scenario_id}", response_model=ScenarioDetailResponse)
async def update_scenario(
    scenario_id: UUID,
    body: UpdateScenarioRequest,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
):
    """Update a scenario's fields (admin only)."""
    from app.services.scenario_repository import get_scenario_by_id

    scenario = await get_scenario_by_id(db, scenario_id, include_inactive=True)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Scenario not found")

    if body.name is not None:
        scenario.name = body.name
    if body.scenario_type is not None:
        scenario.scenario_type = body.scenario_type
    if body.description is not None:
        scenario.description = body.description
    if body.debtor_profile is not None:
        scenario.debtor_profile = body.debtor_profile
    if body.is_active is not None:
        scenario.is_active = body.is_active

    await db.commit()
    await db.refresh(scenario)

    return ScenarioDetailResponse(
        id=scenario.id,
        name=scenario.name,
        scenario_type=scenario.scenario_type,
        description=scenario.description or "",
        debtor_profile=scenario.debtor_profile,
        is_active=scenario.is_active,
    )


@router.delete("/{scenario_id}", status_code=204)
async def delete_scenario(
    scenario_id: UUID,
    db: AsyncSession = Depends(get_session),
    admin: User = Depends(require_admin),
):
    """Delete (deactivate) a scenario (admin only)."""
    from app.services.scenario_repository import get_scenario_by_id

    scenario = await get_scenario_by_id(db, scenario_id, include_inactive=True)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Scenario not found")

    scenario.is_active = False
    await db.commit()

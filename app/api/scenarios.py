"""Scenario selection API endpoints."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.schemas import DebtorProfileSchema, ScenarioListItem, ScenarioResponse, ScenarioType
from app.services.scenario_repository import get_scenario_by_id, list_active_scenarios

router = APIRouter()


@router.get("", response_model=List[ScenarioListItem])
async def list_scenarios(db: AsyncSession = Depends(get_session)):
    """List all active training scenarios."""
    scenarios = await list_active_scenarios(db)
    return [
        ScenarioListItem(
            id=s.id,
            name=s.name,
            scenario_type=ScenarioType(s.scenario_type),
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

"""Pydantic schemas for campaign progress API responses."""

from uuid import UUID

from pydantic import BaseModel


class ScenarioProgressItem(BaseModel):
    """Progress status for a scenario assigned to a campaign."""

    model_config = {"extra": "forbid"}

    scenario_id: UUID
    scenario_name: str
    scenario_type: str
    accomplished: bool


class CampaignProgressResponse(BaseModel):
    """Detailed progress for an agent within a campaign."""

    model_config = {"extra": "forbid"}

    campaign_id: UUID
    campaign_name: str
    total_scenarios: int
    accomplished_scenarios: int
    is_completed: bool
    scenarios: list[ScenarioProgressItem]


class AgentCampaignWithProgress(BaseModel):
    """Campaign summary including the requesting agent's progress."""

    model_config = {"extra": "forbid"}

    id: UUID
    name: str
    description: str | None
    total_scenarios: int
    accomplished_scenarios: int
    is_completed: bool

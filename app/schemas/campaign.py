"""Pydantic schemas for Campaign API request/response models."""

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.models.campaign import CampaignRole, CampaignStatus


# --- Request Schemas ---


class AgentAssignment(BaseModel):
    """An agent-role pair for campaign assignment."""

    model_config = {"extra": "forbid"}

    agent_id: UUID
    role: CampaignRole


class CampaignCreate(BaseModel):
    """Request body for creating a new campaign."""

    model_config = {"extra": "forbid"}

    name: str = Field(min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    start_date: date | None = None
    end_date: date | None = None
    scenario_ids: list[UUID] | None = None
    agents: list[AgentAssignment] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dates(self) -> "CampaignCreate":
        """Ensure end_date is strictly after start_date when both are provided."""
        if self.start_date is not None and self.end_date is not None:
            if self.end_date <= self.start_date:
                raise ValueError("end_date must be after start_date")
        return self


class CampaignUpdate(BaseModel):
    """Request body for updating an existing campaign (partial update)."""

    model_config = {"extra": "forbid"}

    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    status: CampaignStatus | None = None
    start_date: date | None = None
    end_date: date | None = None
    scenario_ids: list[UUID] | None = None
    agents: list[AgentAssignment] | None = None

    @model_validator(mode="after")
    def validate_dates(self) -> "CampaignUpdate":
        """Validate date consistency when both dates are provided."""
        if self.start_date is not None and self.end_date is not None:
            if self.end_date <= self.start_date:
                raise ValueError("end_date must be after start_date")
        return self


# --- Response Schemas ---


class CampaignScenarioItem(BaseModel):
    """A scenario assigned to a campaign."""

    model_config = {"extra": "forbid"}

    id: UUID
    name: str
    scenario_type: str


class CampaignAgentItem(BaseModel):
    """An agent assigned to a campaign."""

    model_config = {"extra": "forbid"}

    id: UUID
    full_name: str
    email: str
    role: str


class CampaignListItem(BaseModel):
    """Summary schema for campaign list endpoint."""

    model_config = {"extra": "forbid"}

    id: UUID
    name: str
    status: str
    scenarios_count: int
    agents_count: int
    start_date: date | None
    end_date: date | None
    created_at: datetime
    updated_at: datetime


class CampaignDetail(BaseModel):
    """Full campaign detail response including assigned scenarios and agents."""

    model_config = {"extra": "forbid"}

    id: UUID
    name: str
    description: str | None
    status: str
    start_date: date | None
    end_date: date | None
    scenarios: list[CampaignScenarioItem]
    agents: list[CampaignAgentItem]
    created_at: datetime
    updated_at: datetime


class PaginatedCampaigns(BaseModel):
    """Paginated response for campaign list endpoint."""

    model_config = {"extra": "forbid"}

    items: list[CampaignListItem]
    total: int
    page: int
    page_size: int
    total_pages: int


# --- Campaign Scenarios Schemas ---


class AddScenariosRequest(BaseModel):
    """Request to add scenarios to a campaign."""

    model_config = {"extra": "forbid"}

    scenario_ids: list[UUID] = Field(min_length=1, max_length=50)


class CampaignScenariosResponse(BaseModel):
    """Updated scenario list after add/remove."""

    model_config = {"extra": "forbid"}

    scenarios: list[CampaignScenarioItem]


class AgentCampaignScenarioItem(BaseModel):
    """A scenario available to an agent via their campaign assignments."""

    model_config = {"extra": "forbid"}

    id: UUID
    name: str
    scenario_type: str
    description: str

"""Pydantic schemas for Campaign API request/response models."""

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.models.campaign import CampaignStatus


# --- Request Schemas ---


class CampaignCreate(BaseModel):
    """Request body for creating a new campaign."""

    model_config = {"extra": "forbid"}

    name: str = Field(min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    start_date: date
    end_date: date
    scenario_ids: list[UUID] = Field(min_length=1)
    agent_ids: list[UUID] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dates(self) -> "CampaignCreate":
        """Ensure end_date is strictly after start_date."""
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
    scenario_ids: list[UUID] | None = Field(default=None, min_length=1)
    agent_ids: list[UUID] | None = Field(default=None, min_length=1)

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


class CampaignListItem(BaseModel):
    """Summary schema for campaign list endpoint."""

    model_config = {"extra": "forbid"}

    id: UUID
    name: str
    status: str
    scenarios_count: int
    agents_count: int
    start_date: date
    end_date: date
    created_at: datetime


class CampaignDetail(BaseModel):
    """Full campaign detail response including assigned scenarios and agents."""

    model_config = {"extra": "forbid"}

    id: UUID
    name: str
    description: str | None
    status: str
    start_date: date
    end_date: date
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

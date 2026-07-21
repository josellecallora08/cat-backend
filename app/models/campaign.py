"""Campaign model for training initiative management."""

import uuid
from enum import Enum

from sqlalchemy import Column, Date, DateTime, ForeignKey, String, Table, Text, Uuid
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class CampaignStatus(str, Enum):
    """Valid campaign lifecycle statuses."""

    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


# Association tables
campaign_scenarios = Table(
    "campaign_scenarios",
    Base.metadata,
    Column(
        "campaign_id",
        Uuid,
        ForeignKey("campaigns.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "scenario_id",
        Uuid,
        ForeignKey("scenarios.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)

campaign_agents = Table(
    "campaign_agents",
    Base.metadata,
    Column(
        "campaign_id",
        Uuid,
        ForeignKey("campaigns.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "agent_id",
        Uuid,
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class Campaign(Base):
    """A training campaign grouping scenarios and agents over a date range."""

    __tablename__ = "campaigns"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default=CampaignStatus.DRAFT.value)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    scenarios = relationship("Scenario", secondary=campaign_scenarios, lazy="selectin")
    agents = relationship("User", secondary=campaign_agents, lazy="selectin")

"""SQLAlchemy ORM models for the Collection Agent Trainer database."""

import uuid

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base
from app.models.campaign import (
    Campaign as Campaign,
)
from app.models.campaign import (
    CampaignAgent as CampaignAgent,
)
from app.models.campaign import (
    CampaignRole as CampaignRole,
)
from app.models.campaign import (
    CampaignStatus as CampaignStatus,
)
from app.models.campaign import (
    campaign_scenarios as campaign_scenarios,
)
from app.models.negotiation_standard import (
    ImmutableVersionError as ImmutableVersionError,
)
from app.models.negotiation_standard import (
    NegotiationStandard as NegotiationStandard,
)
from app.models.negotiation_standard import (
    NegotiationStandardVersion as NegotiationStandardVersion,
)
from app.models.script import (
    Script as Script,
)
from app.models.script import (
    ScriptStatus as ScriptStatus,
)
from app.models.script import (
    ScriptVersion as ScriptVersion,
)
from app.models.script_upload import ScriptUpload as ScriptUpload
from app.models.script_upload import UploadStatus as UploadStatus
from app.models.session_report import (
    SessionReport as SessionReport,
)
from app.models.session_report import (
    SessionReportStatus as SessionReportStatus,
)
from app.models.user import User as User
from app.models.user import UserRole as UserRole


# Use JSONB on PostgreSQL, JSON on other backends (e.g., SQLite for tests)
JSONVariant = JSON().with_variant(JSONB, "postgresql")


class Scenario(Base):
    """Training scenario with debtor profile configuration."""

    __tablename__ = "scenarios"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    scenario_type = Column(String(50), nullable=False)
    description = Column(Text)
    debtor_profile = Column(JSONVariant, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    sessions = relationship("Session", back_populates="scenario")


class Session(Base):
    """A single training session linking a scenario to a conversation."""

    __tablename__ = "sessions"
    __table_args__ = (
        UniqueConstraint(
            "agent_id",
            "creation_key",
            name="uq_sessions_agent_creation_key",
        ),
    )

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    scenario_id = Column(Uuid, ForeignKey("scenarios.id"), nullable=False)
    agent_id = Column(Uuid, nullable=False)
    creation_key = Column(Uuid, nullable=True)
    status = Column(String(20), default="pending", nullable=False)
    persona_context = Column(JSONVariant)
    script_version_id = Column(Uuid, ForeignKey("script_versions.id"), nullable=True)
    campaign_id = Column(
        Uuid,
        ForeignKey("campaigns.id", ondelete="SET NULL"),
        nullable=True,
    )
    negotiation_standard_version_id = Column(
        Uuid,
        ForeignKey("negotiation_standard_versions.id", ondelete="RESTRICT"),
        nullable=True,
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    ended_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    scenario = relationship("Scenario", back_populates="sessions")
    campaign = relationship("Campaign", lazy="selectin")
    transcripts = relationship("Transcript", back_populates="session")
    evaluation = relationship("Evaluation", back_populates="session", uselist=False)
    negotiation_standard_version = relationship(
        "NegotiationStandardVersion",
        back_populates="sessions",
        lazy="selectin",
    )
    coaching_report = relationship("CoachingReport", back_populates="session", uselist=False)
    learning_plan = relationship("LearningPlan", back_populates="session", uselist=False)
    reports = relationship("SessionReport", back_populates="session", cascade="all, delete-orphan")


class Transcript(Base):
    """Individual utterance entry within a session transcript."""

    __tablename__ = "transcripts"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    session_id = Column(Uuid, ForeignKey("sessions.id"), nullable=False)
    speaker = Column(String(10), nullable=False)
    utterance_text = Column(Text, nullable=False)
    timestamp_ms = Column(DateTime(timezone=True), nullable=False)
    sequence_number = Column(Integer, nullable=False)

    # Relationships
    session = relationship("Session", back_populates="transcripts")


class Evaluation(Base):
    """Performance evaluation result for a completed session."""

    __tablename__ = "evaluations"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    session_id = Column(Uuid, ForeignKey("sessions.id"), nullable=False, unique=True)
    overall_score = Column(Float, nullable=False)
    category_scores = Column(JSONVariant, nullable=False)
    strengths = Column(JSONVariant, nullable=False)
    weaknesses = Column(JSONVariant, nullable=False)
    negotiation_standard_version_id = Column(
        Uuid,
        ForeignKey("negotiation_standard_versions.id", ondelete="RESTRICT"),
        nullable=True,
    )
    standard_snapshot = Column(JSONVariant, nullable=True)
    weighted_total = Column(Float, nullable=True)
    passing_score = Column(Integer, nullable=True)
    passed = Column(Boolean, nullable=True)
    rubric_result = Column(JSONVariant, nullable=True)
    is_too_short = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    session = relationship("Session", back_populates="evaluation")
    negotiation_standard_version = relationship(
        "NegotiationStandardVersion",
        back_populates="evaluations",
    )


class CoachingReport(Base):
    """Coaching report identifying mistakes and recommended alternatives."""

    __tablename__ = "coaching_reports"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    session_id = Column(Uuid, ForeignKey("sessions.id"), nullable=False, unique=True)
    mistakes_by_category = Column(JSONVariant, nullable=False)
    total_mistakes = Column(Integer, nullable=False)
    no_mistakes = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    session = relationship("Session", back_populates="coaching_report")


class LearningPlan(Base):
    """Personalized learning plan mapping weak competencies to scenarios."""

    __tablename__ = "learning_plans"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    session_id = Column(Uuid, ForeignKey("sessions.id"), nullable=False, unique=True)
    agent_id = Column(Uuid, nullable=False)
    weak_competencies = Column(JSONVariant, nullable=False)
    all_passing = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    session = relationship("Session", back_populates="learning_plan")


class SystemConfig(Base):
    """Key-value configuration store for admin-managed system settings."""

    __tablename__ = "system_config"

    key = Column(String(100), primary_key=True)
    value = Column(Text, nullable=False, server_default="")
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

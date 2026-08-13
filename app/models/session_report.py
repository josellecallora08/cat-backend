"""Persisted, versioned session report snapshots.

A SessionReport is an immutable, append-only aggregation of a completed
session's artifacts (transcript, evaluation, coaching, learning plan) plus
session/campaign/pinned-standard metadata. Regeneration inserts a new row
with an incremented `report_version` rather than mutating a prior payload.
"""

import uuid

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Index,
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


# Use JSONB on PostgreSQL, JSON on other backends (e.g., SQLite for tests).
JSONVariant = JSON().with_variant(JSONB, "postgresql")


class SessionReportStatus:
    """Valid lifecycle statuses for a SessionReport row."""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class SessionReportReasonCode:
    """String constants shared by persistence code without importing Pydantic."""

    ARTIFACT_MISSING = "artifact_missing"
    EMPTY_TRANSCRIPT = "empty_transcript"
    NOT_APPLICABLE = "not_applicable"
    SESSION_TOO_SHORT = "session_too_short"
    GENERATION_PENDING = "generation_pending"
    GENERATION_FAILED = "generation_failed"
    LEGACY_ONLY = "legacy_only"
    NO_EVIDENCE = "no_evidence"
    NO_COACHING = "no_coaching"
    NO_LEARNING_PLAN = "no_learning_plan"


class SessionReport(Base):
    """A single versioned snapshot of a session's aggregated report.

    One session may have many rows (one per `report_version`); the current
    report is the highest `report_version` with `status == "ready"`. A
    stored `payload` is never rewritten in place — see design constraint
    2.3 (immutability) in the session-report-generation spec.
    """

    __tablename__ = "session_reports"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    session_id = Column(Uuid, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    agent_id = Column(Uuid, nullable=False)
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
    status = Column(String(20), nullable=False, default=SessionReportStatus.PENDING)
    report_version = Column(Integer, nullable=False, default=1)
    payload = Column(JSONVariant, nullable=True)
    content_hash = Column(String(64), nullable=True)
    generated_by = Column(Uuid, nullable=True)
    # ``failure_reason`` remains the legacy safe display/detail field.  The
    # finite ``reason_code`` is the machine-readable row-level contract.
    failure_reason = Column(Text, nullable=True)
    reason_code = Column(String(40), nullable=True)
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
    session = relationship("Session", back_populates="reports")

    __table_args__ = (
        UniqueConstraint("session_id", "report_version", name="uq_session_reports_session_version"),
        Index(
            "ix_session_reports_session_id_version_desc",
            "session_id",
            report_version.desc(),
        ),
    )

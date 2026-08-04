"""Campaign-linked negotiation standard and immutable published versions."""

import uuid

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    Uuid,
    UniqueConstraint,
    event,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base

JSONVariant = JSON().with_variant(JSONB, "postgresql")


class ImmutableVersionError(RuntimeError):
    """Raised when a published negotiation standard version is changed."""


class NegotiationStandard(Base):
    """Editable campaign-owned negotiation standard aggregate."""

    __tablename__ = "negotiation_standards"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    campaign_id = Column(Uuid, ForeignKey("campaigns.id"), nullable=False, unique=True)
    name = Column(String(120), nullable=False)
    description = Column(String(1000), nullable=True)
    status = Column(String(20), nullable=False, default="draft")
    overall_passing_score = Column(Integer, nullable=False, default=70)
    draft_content = Column(JSONVariant, nullable=True)
    current_version_id = Column(
        Uuid,
        ForeignKey(
            "negotiation_standard_versions.id",
            use_alter=True,
            name="fk_negotiation_standards_current_version",
        ),
        nullable=True,
    )
    revision = Column(Integer, nullable=False, default=1)
    created_by = Column(Uuid, ForeignKey("users.id"), nullable=False)
    updated_by = Column(Uuid, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    campaign = relationship("Campaign", back_populates="negotiation_standard")
    versions = relationship(
        "NegotiationStandardVersion",
        back_populates="standard",
        foreign_keys="NegotiationStandardVersion.standard_id",
        order_by="NegotiationStandardVersion.version_number.desc()",
    )
    current_version = relationship(
        "NegotiationStandardVersion",
        foreign_keys=[current_version_id],
        uselist=False,
        post_update=True,
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_negotiation_standards_campaign_id", "campaign_id"),
        Index("ix_negotiation_standards_status", "status"),
    )


class NegotiationStandardVersion(Base):
    """Immutable published snapshot of a negotiation standard."""

    __tablename__ = "negotiation_standard_versions"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    standard_id = Column(Uuid, ForeignKey("negotiation_standards.id"), nullable=False)
    version_number = Column(Integer, nullable=False)
    schema_version = Column(Integer, nullable=False, default=1)
    snapshot = Column(JSONVariant, nullable=False)
    content_hash = Column(String(64), nullable=False)
    created_by = Column(Uuid, ForeignKey("users.id"), nullable=False)
    published_by = Column(Uuid, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    published_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    publication_note = Column(String(500), nullable=True)

    standard = relationship(
        "NegotiationStandard",
        back_populates="versions",
        foreign_keys=[standard_id],
        lazy="selectin",
    )
    sessions = relationship(
        "Session",
        back_populates="negotiation_standard_version",
        lazy="selectin",
    )
    evaluations = relationship("Evaluation", back_populates="negotiation_standard_version")

    __table_args__ = (
        UniqueConstraint("standard_id", "version_number"),
        Index("ix_negotiation_standard_versions_standard_id", "standard_id"),
        Index("ix_negotiation_standard_versions_published_at", "published_at"),
    )


def _reject_version_update(
    _mapper: object,
    _connection: object,
    _target: NegotiationStandardVersion,
) -> None:
    """Reject SQLAlchemy updates to immutable snapshots."""
    raise ImmutableVersionError("Published negotiation standard versions are immutable")


def _reject_version_delete(
    _mapper: object,
    _connection: object,
    _target: NegotiationStandardVersion,
) -> None:
    """Reject SQLAlchemy deletes of immutable snapshots."""
    raise ImmutableVersionError("Published negotiation standard versions cannot be deleted")


event.listen(NegotiationStandardVersion, "before_update", _reject_version_update)
event.listen(NegotiationStandardVersion, "before_delete", _reject_version_delete)

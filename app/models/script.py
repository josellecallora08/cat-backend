"""Script and ScriptVersion models for the Script_Registry subsystem."""

import uuid
from enum import Enum

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Uuid,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base

# Use JSONB on PostgreSQL, JSON on other backends (e.g., SQLite for tests)
JSONVariant = JSON().with_variant(JSONB, "postgresql")


class ScriptStatus(str, Enum):
    """Valid script lifecycle statuses."""

    DRAFT = "draft"
    PUBLISHED = "published"
    UNPUBLISHED = "unpublished"


class Script(Base):
    """An admin-managed script definition tied to a scenario."""

    __tablename__ = "scripts"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    scenario_id = Column(
        Uuid, ForeignKey("scenarios.id"), nullable=False, unique=True
    )
    name = Column(String(255), nullable=False)
    status = Column(String(20), nullable=False, default=ScriptStatus.DRAFT.value)
    format = Column(String(10), nullable=False)
    draft_content = Column(JSONVariant, nullable=True)
    current_version_id = Column(
        Uuid,
        ForeignKey(
            "script_versions.id",
            use_alter=True,
            name="fk_scripts_current_version_id_script_versions",
        ),
        nullable=True,
    )
    is_deleted = Column(Boolean, nullable=False, default=False)
    created_by = Column(Uuid, ForeignKey("users.id"), nullable=False)
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
    versions = relationship(
        "ScriptVersion",
        back_populates="script",
        foreign_keys="ScriptVersion.script_id",
    )


class ScriptVersion(Base):
    """An immutable published snapshot of a Script's validated content."""

    __tablename__ = "script_versions"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    script_id = Column(Uuid, ForeignKey("scripts.id"), nullable=False)
    version_number = Column(Integer, nullable=False)
    content = Column(JSONVariant, nullable=False)
    published_by = Column(Uuid, ForeignKey("users.id"), nullable=False)
    published_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    script = relationship(
        "Script", back_populates="versions", foreign_keys=[script_id]
    )

    __table_args__ = (UniqueConstraint("script_id", "version_number"),)

"""ScriptUpload model: persistent metadata for uploaded training documents."""

import uuid
from enum import Enum

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.sql import func

from app.database import Base


class UploadStatus(str, Enum):
    """Upload processing status."""
    PENDING = "pending"
    SCANNING = "scanning"
    EXTRACTING = "extracting"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


class ScriptUpload(Base):
    """Persistent metadata for an uploaded training document.

    Tracks the lifecycle of an uploaded file from quarantine through
    scanning, extraction, and eventual deletion of the original.
    """

    __tablename__ = "script_uploads"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)

    # File identity
    filename_original = Column(String(255), nullable=False)
    mime_type = Column(String(100), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    content_hash = Column(String(64), nullable=False)  # SHA-256 hex digest

    # Storage
    storage_key = Column(String(255), nullable=False)  # UUID filename in quarantine

    # Uploader
    uploaded_by = Column(Uuid, ForeignKey("users.id"), nullable=False)

    # Processing status
    scan_status = Column(String(20), nullable=False, default="pending")  # pending/clean/infected/error
    scan_signature = Column(String(255), nullable=True)  # malware signature if detected
    extraction_status = Column(String(20), nullable=False, default="pending")  # pending/completed/failed
    extraction_error = Column(Text, nullable=True)  # error message if extraction failed

    # Extracted content (stored as sanitized pending text, NOT yet a ScriptContract)
    extracted_content = Column(Text, nullable=True)

    # Scenario link (validated before use)
    scenario_id = Column(Uuid, ForeignKey("scenarios.id"), nullable=True)

    # Overall status
    status = Column(String(20), nullable=False, default=UploadStatus.PENDING.value)

    # Optional link to script (set when extracted content is used to create/update a script)
    script_id = Column(Uuid, ForeignKey("scripts.id"), nullable=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    quarantine_expires_at = Column(DateTime(timezone=True), nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)  # When original file was deleted

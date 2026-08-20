"""Strict schemas for the normalized post-session report contract."""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ReportSectionName(StrEnum):
    """Stable names for independently loaded report sections."""

    METADATA = "metadata"
    TRANSCRIPT = "transcript"
    EVALUATION = "evaluation"
    COACHING = "coaching"
    LEARNING_PLAN = "learning_plan"
    SUMMARY = "summary"


class SectionState(StrEnum):
    """Lifecycle state of a report section."""

    LOADING = "loading"
    LOADED = "loaded"
    EMPTY = "empty"
    FAILED = "failed"


class ReportCompletion(StrEnum):
    """Aggregate completion state of a report."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"
    FAILED = "failed"


class FailureClass(StrEnum):
    """Stable classification shared by report failures and release gates."""

    BACKEND = "backend"
    FRONTEND = "frontend"
    HTTP_E2E = "http_e2e"
    DATA_CONTRACT = "data_contract"
    ACCESSIBILITY = "accessibility"
    RESPONSIVE_LAYOUT = "responsive_layout"
    EXPORT = "export"
    PRINT = "print"
    INFRASTRUCTURE = "infrastructure"


class EvaluationKind(StrEnum):
    """Origin of the persisted evaluation result."""

    CURRENT = "current"
    LEGACY = "legacy"


class ScoreStatus(StrEnum):
    """Availability and semantic status of the report score."""

    EVALUATED = "evaluated"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class ReportFailure(BaseModel):
    """Safe, user-facing details for a failed report section."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    class_: FailureClass = Field(alias="class")
    code: str = Field(min_length=1, max_length=100)
    safe_message: str = Field(min_length=1, max_length=500)
    correlation_id: str | None = Field(default=None, max_length=100)


class SectionEnvelope(BaseModel):
    """One independently loaded report section."""

    model_config = ConfigDict(extra="forbid")

    name: ReportSectionName
    state: SectionState
    data: Any | None = None
    unavailable_reason: str | None = Field(default=None, max_length=500)
    failure: ReportFailure | None = None
    updated_at: datetime | None = None


class EvaluationVersionMetadata(BaseModel):
    """Immutable evaluation-version metadata attached to a report."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: EvaluationKind
    id: UUID | None = None
    number: int | None = Field(default=None, ge=1)
    name: str | None = Field(default=None, min_length=1, max_length=200)


class ReportSessionMetadata(BaseModel):
    """Safe session identity and lifecycle metadata displayed by a report."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    status: str = Field(min_length=1, max_length=50)
    created_at: datetime | None = None
    ended_at: datetime | None = None
    scenario_id: UUID | None = None
    scenario_name: str | None = Field(default=None, max_length=255)
    campaign_id: UUID | None = None
    campaign_name: str | None = Field(default=None, max_length=255)
    participant_name: str | None = Field(default=None, max_length=255)


class ReportResponse(BaseModel):
    """Normalized aggregate report response."""

    model_config = ConfigDict(extra="forbid")

    session: ReportSessionMetadata
    report_status: ReportCompletion
    score_status: ScoreStatus
    evaluation_version: EvaluationVersionMetadata
    sections: list[SectionEnvelope] = Field(min_length=1)
    correlation_id: str | None = Field(default=None, max_length=100)


class CsvReportStatus(BaseModel):
    """Stable status fields emitted for each CSV report row."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    session_id: UUID
    report_status: ReportCompletion
    section_name: ReportSectionName
    section_state: SectionState
    evaluation_kind: EvaluationKind
    evaluation_version_id: UUID | None = None
    evaluation_version_number: int | None = Field(default=None, ge=1)
    score_status: ScoreStatus
    unavailable_reason: str | None = Field(default=None, max_length=500)
    failure_class: FailureClass | None = None

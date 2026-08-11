"""Strict, typed schemas for immutable session-report snapshots.

The existing ``reason`` display fields remain for GET compatibility.  New
``reason_code`` fields provide a finite, machine-readable reason contract and
are validated against the section in which they occur.
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas import (
    CompetencyScore,
    LearningPlanItem,
    MistakeItem,
    PersonaSummary,
    SessionStatus,
    StrengthItem,
    TranscriptEntry,
    WeaknessItem,
)
from app.schemas.rubric_evaluation import (
    CanonicalEvaluationResult,
    RubricCoachingBlock,
)


class ReportReasonCode(str, Enum):
    """Finite reason vocabulary used by report sections and status envelopes.

    This intentionally remains a string-compatible enum so JSON output is
    unchanged for callers that already consume string values.
    """

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


# The enum is used directly in annotations so both JSON strings and
# ``ReportReasonCode`` members validate consistently.
ReportReasonCodeValue = ReportReasonCode


class ReportReason(BaseModel):
    """A typed reason plus optional safe, display-only text."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    code: ReportReasonCodeValue
    message: Optional[str] = Field(default=None, min_length=1, max_length=500)


# Section matrix.  Generation reasons are row/status concerns and cannot be
# attached to report content sections.
SECTION_REASON_MATRIX: dict[str, frozenset[str]] = {
    "transcript": frozenset({"artifact_missing", "empty_transcript"}),
    "evaluation": frozenset(
        {
            "artifact_missing",
            "not_applicable",
            "session_too_short",
            "legacy_only",
            "no_evidence",
        }
    ),
    "coaching": frozenset({"artifact_missing", "no_coaching"}),
    "learning_plan": frozenset({"artifact_missing", "no_learning_plan"}),
}


def _reason_value(value: object) -> str | None:
    """Return a reason code string from enum/literal-compatible input."""
    if value is None:
        return None
    if isinstance(value, ReportReason):
        value = value.code
    if isinstance(value, Enum):
        return value.value
    return str(value)


def _validate_section_reason(
    section: str,
    available: bool,
    reason_code: str | None,
    *,
    content_empty: bool,
) -> None:
    allowed = SECTION_REASON_MATRIX[section]
    if reason_code is not None and reason_code not in allowed:
        raise ValueError(f"{reason_code!r} is not valid for the {section} section")
    if not available and reason_code is None:
        raise ValueError(f"unavailable {section} sections require a typed reason_code")
    if not available and not content_empty:
        raise ValueError(f"unavailable {section} sections must not contain content")


class SessionReportSummary(BaseModel):
    """Session, scenario, campaign, and pinned-standard identity."""

    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    scenario_id: UUID
    agent_id: UUID
    campaign_id: Optional[UUID] = None
    campaign_name: Optional[str] = None
    persona: Optional[PersonaSummary] = None
    status: SessionStatus
    created_at: datetime
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[float] = Field(default=None, ge=0)
    standard_id: Optional[UUID] = None
    standard_version_id: Optional[UUID] = None
    standard_version_number: Optional[int] = Field(default=None, ge=1)
    standard_name: Optional[str] = None

    @model_validator(mode="after")
    def validate_timestamps(self) -> "SessionReportSummary":
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        if self.ended_at is not None:
            if self.ended_at.tzinfo is None or self.ended_at.utcoffset() is None:
                raise ValueError("ended_at must be timezone-aware")
            if self.ended_at < self.created_at:
                raise ValueError("ended_at must not precede created_at")
        if self.status == SessionStatus.COMPLETED and self.ended_at is None:
            raise ValueError("completed summaries require ended_at")
        return self


class TranscriptSection(BaseModel):
    """Ordered transcript entries with an optional typed terminal reason."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    available: bool = True
    reason: Optional[str] = Field(default=None, min_length=1, max_length=500)
    reason_code: ReportReasonCodeValue | None = None
    entries: list[TranscriptEntry] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def infer_legacy_reason_code(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if data.get("reason_code") is None and not data.get("entries"):
            if data.get("available", True):
                data["reason_code"] = "empty_transcript"
            elif data.get("reason") == "No transcript recorded for this session":
                data["reason_code"] = "artifact_missing"
        return data

    @model_validator(mode="after")
    def validate_transcript_section(self) -> "TranscriptSection":
        _validate_section_reason(
            "transcript",
            self.available,
            _reason_value(self.reason_code),
            content_empty=not self.entries,
        )
        if self.reason_code == "empty_transcript" and self.entries:
            raise ValueError("empty_transcript requires an empty transcript")
        if not self.entries and self.reason_code != "empty_transcript":
            raise ValueError("empty transcripts require the empty_transcript reason_code")
        if self.entries and self.reason_code == "empty_transcript":
            raise ValueError("empty_transcript requires an empty transcript")
        for entry in self.entries:
            if entry.timestamp.tzinfo is None or entry.timestamp.utcoffset() is None:
                raise ValueError("transcript timestamps must be timezone-aware")
        sequences = [entry.sequence_number for entry in self.entries]
        if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
            raise ValueError("transcript sequence numbers must be strictly increasing and unique")
        identities = {
            (entry.speaker, entry.text, entry.timestamp, entry.sequence_number)
            for entry in self.entries
        }
        if len(identities) != len(self.entries):
            raise ValueError("transcript entries must have unique identities")
        return self


class LegacyEvaluationResult(BaseModel):
    """Legacy competency-based evaluation result for pre-rubric sessions."""

    model_config = ConfigDict(extra="forbid")

    category_scores: list[CompetencyScore] = Field(default_factory=list)
    overall_score: float = Field(ge=0, le=100)
    strengths: list[StrengthItem] = Field(default_factory=list)
    weaknesses: list[WeaknessItem] = Field(default_factory=list)


EvaluationMode = Literal["canonical", "legacy", "not_applicable", "too_short"]


class EvaluationSection(BaseModel):
    """Evaluation branch with terminal scored-field exclusivity."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    available: bool
    reason: Optional[str] = Field(default=None, min_length=1, max_length=500)
    reason_code: ReportReasonCodeValue | None = None
    mode: Optional[EvaluationMode] = None
    canonical: Optional[CanonicalEvaluationResult] = None
    legacy: Optional[LegacyEvaluationResult] = None
    weighted_total: Optional[float] = Field(default=None, ge=0, le=100)
    passing_score: Optional[int] = Field(default=None, ge=0, le=100)
    passed: Optional[bool] = None
    standard_version_number: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="before")
    @classmethod
    def infer_legacy_reason_code(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if data.get("reason_code") is None:
            mode = data.get("mode")
            reason = data.get("reason")
            if mode == "too_short":
                data["reason_code"] = "session_too_short"
            elif mode == "not_applicable":
                data["reason_code"] = "not_applicable"
            elif not data.get("available", True) and reason == "No evaluation recorded for this session":
                data["reason_code"] = "artifact_missing"
        return data

    @model_validator(mode="after")
    def validate_evaluation_section(self) -> "EvaluationSection":
        _validate_section_reason(
            "evaluation",
            self.available,
            _reason_value(self.reason_code),
            content_empty=self.canonical is None and self.legacy is None,
        )
        if self.available and self.mode is None:
            raise ValueError("available evaluation sections require a mode")
        if self.mode == "canonical" and (self.canonical is None or self.legacy is not None):
            raise ValueError("canonical evaluation mode requires only canonical data")
        if self.mode == "legacy" and (self.legacy is None or self.canonical is not None):
            raise ValueError("legacy evaluation mode requires only legacy data")
        if self.reason_code == "legacy_only" and self.mode != "legacy":
            raise ValueError("legacy_only is valid only for a legacy evaluation branch")
        if self.reason_code == "no_evidence" and self.mode != "canonical":
            raise ValueError("no_evidence is valid only for a canonical evaluation branch")
        if self.mode in {"not_applicable", "too_short"}:
            if self.reason_code not in {"not_applicable", "session_too_short"}:
                raise ValueError("terminal evaluation modes require their matching reason_code")
            if any(value is not None for value in (self.weighted_total, self.passing_score, self.passed)):
                raise ValueError("terminal evaluations must not contain scored outcome fields")
            if self.mode == "too_short" and (self.canonical is not None or self.legacy is not None):
                raise ValueError("too_short evaluations must not contain a scored branch")
        if self.reason_code in {"not_applicable", "session_too_short"} and self.mode not in {
            "not_applicable",
            "too_short",
        }:
            raise ValueError("terminal evaluation reasons require a terminal evaluation mode")
        return self


CoachingMode = Literal["canonical", "legacy"]


class CoachingSection(BaseModel):
    """Coaching artifact section, grouped by rubric block when canonical."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    available: bool
    reason: Optional[str] = Field(default=None, min_length=1, max_length=500)
    reason_code: ReportReasonCodeValue | None = None
    mode: Optional[CoachingMode] = None
    blocks: list[RubricCoachingBlock] = Field(default_factory=list)
    legacy_mistakes_by_category: dict[str, list[MistakeItem]] = Field(default_factory=dict)
    total_mistakes: Optional[int] = Field(default=None, ge=0)
    no_mistakes: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def infer_legacy_reason_code(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if data.get("reason_code") is None and not data.get("available", True):
            if data.get("reason") == "No coaching report recorded for this session":
                data["reason_code"] = "artifact_missing"
        return data

    @model_validator(mode="after")
    def validate_coaching_section(self) -> "CoachingSection":
        _validate_section_reason(
            "coaching",
            self.available,
            _reason_value(self.reason_code),
            content_empty=not self.blocks and not self.legacy_mistakes_by_category,
        )
        if self.available and self.mode is None:
            raise ValueError("available coaching sections require a mode")
        if self.mode == "canonical" and self.legacy_mistakes_by_category:
            raise ValueError("canonical coaching must not contain legacy coaching")
        if self.mode == "legacy" and self.blocks:
            raise ValueError("legacy coaching must not contain canonical blocks")
        block_ids: set[str] = set()
        criterion_ids: set[tuple[str, str]] = set()
        evidence_ids: set[tuple[str, str, int]] = set()
        for block in self.blocks:
            if block.rubric_block_id in block_ids:
                raise ValueError("coaching rubric block identities must be unique")
            block_ids.add(block.rubric_block_id)
            for recommendation in block.recommendations:
                if recommendation.rubric_block_id != block.rubric_block_id:
                    raise ValueError("coaching recommendation references the wrong rubric block")
                criterion = (block.rubric_block_id, recommendation.criterion_id)
                if criterion in criterion_ids:
                    raise ValueError("coaching criterion identities must be unique")
                criterion_ids.add(criterion)
                evidence = (
                    block.rubric_block_id,
                    recommendation.criterion_id,
                    recommendation.evidence_sequence_number,
                )
                if evidence in evidence_ids:
                    raise ValueError("coaching evidence identities must be unique")
                evidence_ids.add(evidence)
        if self.reason_code == "no_coaching" and self.available:
            raise ValueError("no_coaching is valid only for unavailable coaching")
        return self


class LearningPlanSection(BaseModel):
    """Learning-plan artifact section."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    available: bool
    reason: Optional[str] = Field(default=None, min_length=1, max_length=500)
    reason_code: ReportReasonCodeValue | None = None
    items: list[LearningPlanItem] = Field(default_factory=list)
    all_passing: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def infer_legacy_reason_code(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if data.get("reason_code") is None and not data.get("available", True):
            if data.get("reason") == "No learning plan recorded for this session":
                data["reason_code"] = "artifact_missing"
        return data

    @model_validator(mode="after")
    def validate_learning_plan_section(self) -> "LearningPlanSection":
        _validate_section_reason(
            "learning_plan",
            self.available,
            _reason_value(self.reason_code),
            content_empty=not self.items,
        )
        identities: set[tuple[str, str]] = set()
        for item in self.items:
            if (item.rubric_block_id is None) != (item.criterion_id is None):
                raise ValueError("learning-plan rubric block and criterion must be paired")
            if item.rubric_block_id is not None and item.criterion_id is not None:
                identity = (item.rubric_block_id, item.criterion_id)
                if identity in identities:
                    raise ValueError("learning-plan criterion identities must be unique")
                identities.add(identity)
        if self.reason_code == "no_learning_plan" and self.available:
            raise ValueError("no_learning_plan is valid only for unavailable learning plans")
        return self


class SessionReportPayload(BaseModel):
    """Complete assembled, immutable session-report payload."""

    model_config = ConfigDict(extra="forbid")

    summary: SessionReportSummary
    transcript: TranscriptSection
    evaluation: EvaluationSection
    coaching: CoachingSection
    learning_plan: LearningPlanSection

    @model_validator(mode="after")
    def validate_payload_identity(self) -> "SessionReportPayload":
        if self.transcript.reason_code == "empty_transcript" and self.transcript.entries:
            raise ValueError("empty transcript payload cannot contain entries")
        if self.evaluation.mode == "legacy" and self.evaluation.canonical is not None:
            raise ValueError("canonical and legacy evaluation branches are mutually exclusive")

        transcript_sequences = {entry.sequence_number for entry in self.transcript.entries}
        canonical = self.evaluation.canonical
        if canonical is not None:
            block_ids: set[str] = set()
            for category in canonical.categories:
                if category.rubric_block_id in block_ids:
                    raise ValueError("canonical rubric block identities must be unique")
                block_ids.add(category.rubric_block_id)
                criterion_ids = list(category.failed_criteria)
                if len(criterion_ids) != len(set(criterion_ids)):
                    raise ValueError("canonical criterion identities must be unique")
                evidence_ids: set[tuple[int, str, str]] = set()
                for evidence in category.evidence:
                    if evidence.sequence_number not in transcript_sequences:
                        raise ValueError("canonical evidence references an unknown transcript entry")
                    identity = (evidence.sequence_number, evidence.speaker, evidence.excerpt)
                    if identity in evidence_ids:
                        raise ValueError("canonical evidence identities must be unique")
                    evidence_ids.add(identity)
                for strength in category.strengths:
                    if not set(strength.evidence_sequence_numbers) <= transcript_sequences:
                        raise ValueError("canonical strength references an unknown transcript entry")
                for violation in category.violations:
                    if not set(violation.evidence_sequence_numbers) <= transcript_sequences:
                        raise ValueError("canonical violation references an unknown transcript entry")
                for recommendation_input in category.recommendation_inputs:
                    if recommendation_input.transcript_sequence_number not in transcript_sequences:
                        raise ValueError("canonical recommendation input references an unknown transcript entry")
            recommendation_ids: set[tuple[str, str, int]] = set()
            for recommendation in canonical.recommendations:
                if recommendation.rubric_block_id not in block_ids:
                    raise ValueError("canonical recommendation references an unknown rubric block")
                if recommendation.evidence_sequence_number not in transcript_sequences:
                    raise ValueError("canonical recommendation references an unknown transcript entry")
                identity = (
                    recommendation.rubric_block_id,
                    recommendation.criterion_id,
                    recommendation.evidence_sequence_number,
                )
                if identity in recommendation_ids:
                    raise ValueError("canonical recommendation identities must be unique")
                recommendation_ids.add(identity)

        if self.learning_plan.available:
            for item in self.learning_plan.items:
                if item.scenario_id is not None and item.scenario_id != self.summary.scenario_id:
                    raise ValueError("learning-plan scenario reference does not match the session scenario")
        return self


class ReportAttemptMetadata(BaseModel):
    """Safe metadata for a pending or failed generation attempt."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["pending", "failed"]
    report_version: int = Field(ge=1)
    reason: ReportReason
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def validate_attempt_reason(self) -> "ReportAttemptMetadata":
        expected = "generation_pending" if self.status == "pending" else "generation_failed"
        if self.reason.code != expected:
            raise ValueError(f"{self.status} attempts require reason code {expected}")
        if self.updated_at < self.created_at:
            raise ValueError("attempt updated_at must not precede created_at")
        return self


class ReadyReport(BaseModel):
    """The unchanged successful GET report envelope, strongly typed."""

    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    report_version: int = Field(ge=1)
    status: Literal["ready"]
    content_hash: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    created_at: datetime
    payload: SessionReportPayload

    @model_validator(mode="after")
    def validate_payload_identity(self) -> "ReadyReport":
        if self.payload.summary.session_id != self.session_id:
            raise ValueError("ready report session_id must match payload summary")
        return self


class ReportMissingStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    status: Literal["missing"]
    reason: ReportReason
    latest_attempt: None = None
    report: None = None

    @model_validator(mode="after")
    def validate_reason(self) -> "ReportMissingStatus":
        if self.reason.code != "artifact_missing":
            raise ValueError("missing status requires artifact_missing")
        return self


class ReportIncompleteStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    status: Literal["incomplete"]
    reason: ReportReason
    missing_sections: list[ReportReasonCodeValue] = Field(min_length=1)
    latest_attempt: ReportAttemptMetadata | None = None
    report: None = None

    @model_validator(mode="after")
    def validate_reason(self) -> "ReportIncompleteStatus":
        if self.reason.code != "artifact_missing":
            raise ValueError("incomplete status requires artifact_missing")
        if any(code in {"generation_pending", "generation_failed"} for code in self.missing_sections):
            raise ValueError("incomplete missing_sections must contain section reasons")
        return self


class ReportGeneratingStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    status: Literal["generating"]
    reason: ReportReason
    latest_attempt: ReportAttemptMetadata
    report: None = None

    @model_validator(mode="after")
    def validate_generation(self) -> "ReportGeneratingStatus":
        if self.reason.code != "generation_pending" or self.latest_attempt.status != "pending":
            raise ValueError("generating status requires a pending generation attempt")
        return self


class ReportFailedStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    status: Literal["failed"]
    reason: ReportReason
    latest_attempt: ReportAttemptMetadata
    report: None = None

    @model_validator(mode="after")
    def validate_failure(self) -> "ReportFailedStatus":
        if self.reason.code != "generation_failed" or self.latest_attempt.status != "failed":
            raise ValueError("failed status requires a failed generation attempt")
        return self


class ReportReadyStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    status: Literal["ready"]
    report: ReadyReport
    latest_attempt: ReportAttemptMetadata | None = None

    @model_validator(mode="after")
    def validate_ready(self) -> "ReportReadyStatus":
        if self.report.session_id != self.session_id:
            raise ValueError("ready status session_id must match report")
        if self.latest_attempt is not None and self.latest_attempt.status != "failed":
            raise ValueError("ready status latest_attempt may only describe a failed regeneration")
        return self


class _TerminalReportStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    report: ReadyReport
    latest_attempt: ReportAttemptMetadata | None = None

    @model_validator(mode="after")
    def validate_terminal_report(self) -> "_TerminalReportStatus":
        if self.report.session_id != self.session_id:
            raise ValueError("terminal status session_id must match report")
        if self.latest_attempt is not None:
            raise ValueError("terminal status must not include a generation attempt")
        return self


class ReportNotApplicableStatus(_TerminalReportStatus):
    status: Literal["not_applicable"]
    reason: ReportReason

    @model_validator(mode="after")
    def validate_mode(self) -> "ReportNotApplicableStatus":
        if self.reason.code != "not_applicable" or self.report.payload.evaluation.mode != "not_applicable":
            raise ValueError("not_applicable status requires a not_applicable evaluation")
        return self


class ReportTooShortStatus(_TerminalReportStatus):
    status: Literal["too_short"]
    reason: ReportReason

    @model_validator(mode="after")
    def validate_mode(self) -> "ReportTooShortStatus":
        if self.reason.code != "session_too_short" or self.report.payload.evaluation.mode != "too_short":
            raise ValueError("too_short status requires a too_short evaluation")
        return self


class ReportLegacyOnlyStatus(_TerminalReportStatus):
    status: Literal["legacy_only"]
    reason: ReportReason

    @model_validator(mode="after")
    def validate_mode(self) -> "ReportLegacyOnlyStatus":
        evaluation = self.report.payload.evaluation
        if self.reason.code != "legacy_only" or evaluation.mode != "legacy" or evaluation.canonical is not None:
            raise ValueError("legacy_only status requires a legacy-only evaluation")
        return self


class ReportEmptyTranscriptStatus(_TerminalReportStatus):
    status: Literal["empty_transcript"]
    reason: ReportReason

    @model_validator(mode="after")
    def validate_mode(self) -> "ReportEmptyTranscriptStatus":
        transcript = self.report.payload.transcript
        if self.reason.code != "empty_transcript" or transcript.entries:
            raise ValueError("empty_transcript status requires an empty transcript")
        return self


class ReportNoEvidenceStatus(_TerminalReportStatus):
    status: Literal["no_evidence"]
    reason: ReportReason

    @model_validator(mode="after")
    def validate_mode(self) -> "ReportNoEvidenceStatus":
        evaluation = self.report.payload.evaluation
        if self.reason.code != "no_evidence" or evaluation.mode != "canonical":
            raise ValueError("no_evidence status requires a canonical evaluation")
        return self


ReportStatusEnvelope = Annotated[
    Union[
        ReportMissingStatus,
        ReportIncompleteStatus,
        ReportGeneratingStatus,
        ReportFailedStatus,
        ReportReadyStatus,
        ReportNotApplicableStatus,
        ReportTooShortStatus,
        ReportLegacyOnlyStatus,
        ReportEmptyTranscriptStatus,
        ReportNoEvidenceStatus,
    ],
    Field(discriminator="status"),
]

# Compatibility aliases make the variant names discoverable to callers that
# use either the concise or explicit naming convention.
MissingReportStatus = ReportMissingStatus
IncompleteReportStatus = ReportIncompleteStatus
GeneratingReportStatus = ReportGeneratingStatus
FailedReportStatus = ReportFailedStatus
ReadyReportStatus = ReportReadyStatus
NotApplicableReportStatus = ReportNotApplicableStatus
TooShortReportStatus = ReportTooShortStatus
LegacyOnlyReportStatus = ReportLegacyOnlyStatus
EmptyTranscriptReportStatus = ReportEmptyTranscriptStatus
NoEvidenceReportStatus = ReportNoEvidenceStatus

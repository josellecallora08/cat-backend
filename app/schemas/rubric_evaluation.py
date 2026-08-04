"""Strict schemas for AI-extracted rubric observations."""

from typing import Annotated, Literal
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator


Slug = Annotated[str, Field(min_length=1, max_length=120, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]
NonEmptyText = Annotated[str, Field(min_length=1, max_length=10_000)]
SequenceNumber = Annotated[int, Field(ge=0)]


class RubricEvidence(BaseModel):
    """A verbatim, attributable excerpt from the persisted transcript."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    sequence_number: SequenceNumber
    speaker: Literal["agent", "debtor"]
    excerpt: NonEmptyText
    explanation: NonEmptyText


class RubricStrength(BaseModel):
    """Evidence-backed positive criterion observation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    criterion_id: Slug
    explanation: NonEmptyText
    evidence_sequence_numbers: list[SequenceNumber] = Field(min_length=1)


class RubricViolation(BaseModel):
    """Evidence-backed violation observation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    violation_id: Slug
    explanation: NonEmptyText
    evidence_sequence_numbers: list[SequenceNumber] = Field(min_length=1)


class RubricRecommendationInput(BaseModel):
    """Input used later to create a grounded recommendation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    criterion_id: Slug
    transcript_sequence_number: SequenceNumber
    need: NonEmptyText


class RubricRecommendation(BaseModel):
    """Backend-owned, evidence-linked coaching recommendation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    rubric_block_id: Slug
    criterion_id: Slug
    evidence_sequence_number: SequenceNumber
    explanation: NonEmptyText
    recommended_response: NonEmptyText
    coaching_advice: NonEmptyText


class RubricCategoryObservation(BaseModel):
    """One AI observation for a published rubric block."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    rubric_block_id: Slug
    raw_score: int | None = Field(ge=0, le=100)
    evidence: list[RubricEvidence]
    strengths: list[RubricStrength]
    violations: list[RubricViolation]
    failed_criteria: list[Slug]
    recommendation_inputs: list[RubricRecommendationInput]


class RubricAppliedTechnique(BaseModel):
    """A technique observed in execution, with no missed-opportunity type."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    technique_name: NonEmptyText
    execution_type: Literal["Executed Properly", "Weak Execution", "Misapplied"]
    execution_description: NonEmptyText
    evidence_sequence_numbers: list[SequenceNumber] = Field(min_length=1)


class RubricAppliedTechniques(BaseModel):
    """Compatibility list of techniques actually attempted by the agent."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    techniques_used: list[RubricAppliedTechnique]
    reason_if_empty: NonEmptyText


class RubricMissedOpportunity(BaseModel):
    """A technique that was contextually applicable but not evidenced."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    technique_name: NonEmptyText
    reason: NonEmptyText


class RubricMissedOpportunities(BaseModel):
    """Compatibility list of evidence-grounded missed opportunities."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    missed_techniques: list[RubricMissedOpportunity]
    reason_if_empty: NonEmptyText


class RubricAIObservation(BaseModel):
    """Complete strict observation returned by the evaluator model."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: Literal["evaluated", "not_applicable"]
    summary: NonEmptyText
    categories: list[RubricCategoryObservation] = Field(min_length=1)
    applied_techniques: RubricAppliedTechniques
    missed_opportunities: RubricMissedOpportunities

    @model_validator(mode="after")
    def require_not_applicable_scores(self) -> "RubricAIObservation":
        """Require scores for evaluated results and none for not-applicable results."""
        has_missing_score = any(category.raw_score is None for category in self.categories)
        if self.status == "evaluated" and has_missing_score:
            raise ValueError("evaluated observations require a raw_score for every category")
        if self.status == "not_applicable" and any(
            category.raw_score is not None for category in self.categories
        ):
            raise ValueError("not_applicable observations must not contain category scores")
        return self

    @model_validator(mode="after")
    def enforce_technique_disjointness(self) -> "RubricAIObservation":
        """Prevent a technique from appearing in both compatibility lists."""
        applied = {item.technique_name.casefold() for item in self.applied_techniques.techniques_used}
        missed = {
            item.technique_name.casefold()
            for item in self.missed_opportunities.missed_techniques
        }
        if len(applied) != len(self.applied_techniques.techniques_used):
            raise ValueError("applied techniques must be unique")
        if len(missed) != len(self.missed_opportunities.missed_techniques):
            raise ValueError("missed opportunities must be unique")
        if applied & missed:
            raise ValueError("applied and missed techniques must be disjoint")
        return self


class RubricCategoryScore(BaseModel):
    """Deterministic score breakdown for one rubric block."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @field_serializer("weighted_contribution", when_used="json")
    def serialize_weighted_contribution(self, value: Decimal) -> float:
        return float(value)

    rubric_block_id: Slug
    category: NonEmptyText
    raw_score: int | None = Field(default=None, ge=0, le=100)
    penalty_total: int = Field(ge=0)
    penalized_score: int | None = Field(default=None, ge=0, le=100)
    weight: int = Field(ge=0, le=100)
    weighted_contribution: Decimal = Field(ge=0, le=100)
    passing_score: int = Field(ge=0, le=100)
    passed: bool
    evidence: list[RubricEvidence]
    strengths: list[RubricStrength]
    violations: list[RubricViolation]
    failed_criteria: list[Slug]
    recommendation_inputs: list[RubricRecommendationInput]


class CanonicalEvaluationResult(BaseModel):
    """Backend-owned canonical rubric result."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @field_serializer("weighted_total", when_used="json")
    def serialize_weighted_total(self, value: Decimal) -> float:
        return float(value)

    status: Literal["evaluated", "not_applicable"]
    summary: NonEmptyText
    categories: list[RubricCategoryScore]
    weighted_total: Decimal = Field(ge=0, le=100)
    passing_score: int = Field(ge=0, le=100)
    passed: bool
    applied_techniques: RubricAppliedTechniques
    missed_opportunities: RubricMissedOpportunities
    recommendations: list[RubricRecommendation] = Field(default_factory=list)

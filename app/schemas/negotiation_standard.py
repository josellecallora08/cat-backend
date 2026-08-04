"""Strict Pydantic contracts for administrator-managed negotiation rubrics."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


_ID_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
_MAX_SHORT_TEXT = 120
_MAX_LONG_TEXT = 10_000

Slug = Annotated[
    str,
    Field(
        min_length=1,
        max_length=_MAX_SHORT_TEXT,
        pattern=_ID_PATTERN,
    ),
]
ShortText = Annotated[str, Field(min_length=1, max_length=_MAX_SHORT_TEXT)]
LongText = Annotated[str, Field(min_length=1, max_length=_MAX_LONG_TEXT)]
Percentage = Annotated[int, Field(ge=0, le=100)]


class RubricCriterion(BaseModel):
    """A positive behavior that can be supported by transcript evidence."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: Slug
    name: ShortText
    description: LongText
    evidence_instructions: LongText


class RubricViolation(BaseModel):
    """A behavior violation that can be detected and penalized."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: Slug
    name: ShortText
    description: LongText
    evidence_instructions: LongText


class RubricPenalty(BaseModel):
    """A deterministic deduction associated with a rubric violation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    violation_id: Slug
    deduction: Percentage
    max_occurrences: Annotated[int, Field(ge=0, le=100)]


class RubricBlock(BaseModel):
    """One weighted, scored rubric category."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: Slug
    category: ShortText
    weight: Percentage
    passing_score: Percentage
    scoring_instructions: LongText
    positive_behaviors: list[RubricCriterion]
    violations: list[RubricViolation]
    penalties: list[RubricPenalty]
    recommendation_guidance: LongText
    display_order: Annotated[int, Field(ge=0)]


class NegotiationStandardContent(BaseModel):
    """Complete draft/published content for a negotiation standard."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Annotated[int, Field(ge=1)] = 1
    overall_passing_score: Percentage
    blocks: list[RubricBlock]


class ValidationIssue(BaseModel):
    """A structured issue produced by aggregate standard validation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    code: ShortText
    path: ShortText
    message: LongText


class ValidationResult(BaseModel):
    """Aggregate validation outcome and its complete issue list."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    valid: bool
    weight_total: int
    errors: list[ValidationIssue]

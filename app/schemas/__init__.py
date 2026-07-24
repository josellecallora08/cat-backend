"""Pydantic schemas for API request/response models."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


# --- Enums ---


class ScenarioType(str, Enum):
    """Available training scenario types."""

    FINANCIAL_HARDSHIP = "FINANCIAL_HARDSHIP"
    ANGRY_CUSTOMER = "ANGRY_CUSTOMER"
    PAYMENT_EXTENSION = "PAYMENT_EXTENSION"
    BALANCE_DISPUTE = "BALANCE_DISPUTE"


class SessionStatus(str, Enum):
    """Session lifecycle states."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"


class EvaluationCategory(str, Enum):
    """Competency evaluation categories with defined weights."""

    CALL_OPENING = "call_opening"
    COMPLIANCE = "compliance"
    EMPATHY_COMMUNICATION = "empathy_communication"
    NEGOTIATION_RESOLUTION = "negotiation_resolution"


# --- Debtor Profile ---


class DebtorProfileSchema(BaseModel):
    """Debtor profile with validation for scenario completeness."""

    name: str = Field(min_length=1)
    outstanding_balance: Decimal = Field(gt=0)
    days_past_due: int = Field(ge=1)
    personality_profile: str = Field(min_length=1)
    conversation_goal: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_profile_completeness(self) -> "DebtorProfileSchema":
        """Ensures all required string fields are non-empty (not just whitespace)."""
        fields = [self.name, self.personality_profile, self.conversation_goal]
        if any(f.strip() == "" for f in fields):
            raise ValueError("Debtor profile has incomplete data")
        return self


# --- Scenario Schemas ---


class ScenarioListItem(BaseModel):
    """Summary schema for scenario list endpoint."""

    id: UUID
    name: str
    scenario_type: ScenarioType
    description: str = ""


class ScenarioResponse(BaseModel):
    """Full scenario detail response including debtor profile."""

    id: UUID
    name: str
    scenario_type: ScenarioType
    description: str
    debtor_profile: DebtorProfileSchema


class CreateDebtorProfilePayload(BaseModel):
    """Debtor profile payload for manual scenario creation."""

    name: str = Field(min_length=1, max_length=255)
    outstanding_balance: Decimal = Field(gt=0)
    days_past_due: int = Field(ge=1)
    personality_profile: str = Field(default="", max_length=2000)
    conversation_goal: str = Field(default="", max_length=2000)
    prompt_blocks: list[str] = Field(min_length=1)

    @field_validator("prompt_blocks")
    @classmethod
    def validate_prompt_blocks_non_empty(cls, v: list[str]) -> list[str]:
        """Filter empty blocks and ensure at least one non-empty remains."""
        filtered = [block for block in v if block.strip()]
        if not filtered:
            raise ValueError("At least one non-empty prompt block is required")
        return filtered


class CreateScenarioRequest(BaseModel):
    """Request body for manual scenario creation."""

    name: str = Field(min_length=1, max_length=255)
    scenario_type: ScenarioType
    description: str = Field(default="", max_length=2000)
    debtor_profile: CreateDebtorProfilePayload


# --- Session Schemas ---


class PersonaSummary(BaseModel):
    """Summary of the generated debtor persona for a session."""

    name: str
    communication_style: str
    emotional_state: str


class SessionCreate(BaseModel):
    """Request body for creating a new training session."""

    scenario_id: UUID


class SessionResponse(BaseModel):
    """Response schema for session details."""

    id: UUID
    scenario_id: UUID
    persona: Optional[PersonaSummary] = None
    status: SessionStatus
    created_at: datetime
    ended_at: Optional[datetime] = None


# --- Transcript Schemas ---


class TranscriptEntry(BaseModel):
    """A single utterance entry in a session transcript."""

    speaker: Literal["agent", "debtor"]
    text: str = Field(min_length=1)
    timestamp: datetime
    sequence_number: int = Field(ge=0)


# --- Evaluation Schemas ---


class StrengthItem(BaseModel):
    """An identified strength from the evaluation."""

    description: str = Field(min_length=1)
    category: EvaluationCategory
    transcript_excerpt: str = Field(min_length=1)


class WeaknessItem(BaseModel):
    """An identified weakness from the evaluation."""

    description: str = Field(min_length=1)
    category: EvaluationCategory
    transcript_excerpt: str = Field(min_length=1)


class CompetencyScore(BaseModel):
    """Score for a single evaluation category."""

    category: EvaluationCategory
    score: int = Field(ge=0, le=100)
    strengths: List[StrengthItem] = Field(default_factory=list)
    weaknesses: List[WeaknessItem] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """Complete evaluation result for a session."""

    session_id: UUID
    category_scores: List[CompetencyScore]
    overall_score: float = Field(ge=0, le=100)
    strengths: List[StrengthItem] = Field(min_length=1, max_length=5)
    weaknesses: List[WeaknessItem] = Field(min_length=1, max_length=5)
    is_too_short: bool = False


# --- Coaching Schemas ---


class MistakeItem(BaseModel):
    """A specific mistake identified in the transcript."""

    transcript_position: int = Field(ge=0)
    transcript_excerpt: str = Field(min_length=1)
    category: EvaluationCategory
    explanation: str = Field(min_length=1)
    recommended_alternative: str = Field(min_length=1)


class CoachingReportSchema(BaseModel):
    """Coaching report with mistakes grouped by evaluation category."""

    session_id: UUID
    mistakes_by_category: Dict[EvaluationCategory, List[MistakeItem]]
    total_mistakes: int = Field(ge=0)
    no_mistakes: bool = False


# --- Learning Plan Schemas ---


class LearningPlanItem(BaseModel):
    """A single weak competency with recommended scenario."""

    category: EvaluationCategory
    score: int = Field(ge=0, le=100)
    recommended_scenario: str = Field(min_length=1)


class LearningPlanSchema(BaseModel):
    """Personalized learning plan based on evaluation results."""

    session_id: UUID
    weak_competencies: List[LearningPlanItem] = Field(default_factory=list)
    all_passing: bool = False

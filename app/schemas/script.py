"""Pydantic schemas for the Script_Contract (AI debtor script definitions).

Defines the structural contract every Script must satisfy: required
top-level fields, required substructure for singleton and list-entry
fields, and the Prohibited_Responses / Expected_Replies conflict rule.

Configurable size/entry-count/free-text-length *limits* (Requirement 2.10)
are enforced separately in `app/services/script_validator.py`, not here —
this module only enforces structural correctness plus a fixed
`max_length=2000` ceiling on free-text fields (matching the default limit).
"""

from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

# Free-text fields share a common ceiling; kept as a local constant here
# (structural default) while the configurable version lives in app/config.py
# and is enforced by app/services/script_validator.py.
_FREE_TEXT_MAX_LENGTH = 2000

FreeText = Annotated[str, Field(min_length=1, max_length=_FREE_TEXT_MAX_LENGTH)]


class TriggerPhraseEntry(BaseModel):
    """One Trigger_Phrase entry: a phrase and its corresponding debtor behavior."""

    model_config = {"extra": "forbid"}

    phrase: FreeText
    behavior: FreeText


class ExpectedReplyEntry(BaseModel):
    """One Expected_Replies entry: an anticipated agent statement and the debtor's reply."""

    model_config = {"extra": "forbid"}

    agent_statement: FreeText
    debtor_reply: FreeText


class EmotionalStateRule(BaseModel):
    """One Emotional_State_Rules entry: an agent tone/event and the resulting state change."""

    model_config = {"extra": "forbid"}

    trigger: FreeText
    state_change: FreeText


class EscalationConditionEntry(BaseModel):
    """One Escalation_Conditions entry: a condition, its behavior, and whether it ends the call."""

    model_config = {"extra": "forbid"}

    condition: FreeText
    behavior: FreeText
    ends_call: bool


class PaymentConditionEntry(BaseModel):
    """One Payment_Conditions entry: a condition and the accepted/rejected payment term."""

    model_config = {"extra": "forbid"}

    condition: FreeText
    term: FreeText
    accepted: bool


class DebtorPersona(BaseModel):
    """The Debtor_Persona field: identity, communication style, and background."""

    model_config = {"extra": "forbid"}

    name: FreeText
    communication_style: FreeText
    background: FreeText


class FinancialSituation(BaseModel):
    """The Financial_Situation field: balance, days past due, and delinquency reason."""

    model_config = {"extra": "forbid"}

    outstanding_balance: Decimal = Field(gt=0)
    days_past_due: int = Field(ge=0)
    reason_for_delinquency: FreeText


class ConversationGoal(BaseModel):
    """The Conversation_Goal field: target outcome and call-completion condition."""

    model_config = {"extra": "forbid"}

    target_outcome: FreeText
    completion_condition: FreeText


class ScriptContract(BaseModel):
    """The full Script_Contract: all fields required by Requirement 1.1.

    Every field below is required (no defaults) so that omitting any one of
    them from a submission is reported as a missing field (Requirement 1.2)
    rather than silently defaulted. Entry-count limits on the list fields
    are intentionally NOT enforced here (Requirement 2.10 requires those
    limits to be runtime-configurable); they are checked by
    `app/services/script_validator.py` against `app/config.py` settings.
    """

    model_config = {"extra": "forbid"}

    debtor_persona: DebtorPersona
    financial_situation: FinancialSituation
    opening_response: FreeText
    expected_replies: list[ExpectedReplyEntry]
    trigger_phrases: list[TriggerPhraseEntry]
    emotional_state_rules: list[EmotionalStateRule]
    payment_conditions: list[PaymentConditionEntry]
    escalation_conditions: list[EscalationConditionEntry]
    prohibited_responses: list[FreeText]
    conversation_goal: ConversationGoal

    @model_validator(mode="after")
    def validate_no_prohibited_expected_conflict(self) -> "ScriptContract":
        """Reject Prohibited_Responses entries that duplicate an Expected_Reply.

        Comparison is trimmed and case-insensitive, per Requirement 1.9.
        """
        expected_replies_normalized = {
            entry.debtor_reply.strip().casefold(): entry.debtor_reply
            for entry in self.expected_replies
        }

        conflicts = [
            prohibited
            for prohibited in self.prohibited_responses
            if prohibited.strip().casefold() in expected_replies_normalized
        ]

        if conflicts:
            raise ValueError(
                "prohibited_responses entries conflict with expected_replies "
                f"debtor_reply entries (case-insensitive, trimmed match): {conflicts}"
            )

        return self


# --- API Request/Response Schemas (Script Registry) ---
#
# These schemas cover the Script_Registry HTTP surface (create/update/list/
# detail/versions), mirroring the naming and structural conventions used by
# `app/schemas/campaign.py` (`CampaignCreate`/`CampaignDetail`/etc.), while
# `ScriptContract` above remains the pure structural contract used by
# `app/services/script_validator.py`.

from datetime import datetime
from typing import Literal
from uuid import UUID

ScriptFormat = Literal["json", "yaml"]


class ScriptCreateRequest(BaseModel):
    """Request body for creating a new Draft_Script."""

    model_config = {"extra": "forbid"}

    name: str = Field(min_length=1, max_length=255)
    scenario_id: UUID
    format: ScriptFormat
    raw_definition: str = Field(min_length=1)


class ScriptUpdateRequest(BaseModel):
    """Request body for replacing a script's draft content."""

    model_config = {"extra": "forbid"}

    raw_definition: str = Field(min_length=1)
    format: ScriptFormat


class ScriptVersionItem(BaseModel):
    """A single immutable published snapshot of a Script's content."""

    model_config = {"extra": "forbid"}

    id: UUID
    version_number: int
    content: dict
    published_by: UUID
    published_at: datetime


class ScriptListItem(BaseModel):
    """Summary schema for the script list endpoint."""

    model_config = {"extra": "forbid"}

    id: UUID
    name: str
    scenario_id: UUID
    status: str
    format: str
    created_at: datetime
    updated_at: datetime


class ScriptDetail(BaseModel):
    """Full script detail response.

    Omits a nested `versions` list — those are served by the separate
    `GET /{script_id}/versions` endpoint per design.md's API table.
    """

    model_config = {"extra": "forbid"}

    id: UUID
    name: str
    scenario_id: UUID
    status: str
    format: str
    draft_content: dict | None
    current_version_id: UUID | None
    created_at: datetime
    updated_at: datetime


class PaginatedScripts(BaseModel):
    """Paginated response for the script list endpoint."""

    model_config = {"extra": "forbid"}

    items: list[ScriptListItem]
    total: int
    page: int
    page_size: int
    total_pages: int

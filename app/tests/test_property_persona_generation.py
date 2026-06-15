"""Property-based tests for persona generation completeness.

Feature: collection-agent-trainer, Property 3: Persona generation completeness

**Validates: Requirements 2.1**

Property 3: For any valid scenario, the Debtor Simulator's persona generation
SHALL produce a persona containing a non-empty name, a communication_style from
the defined set, populated financial_circumstances, and an emotional_state from
the defined scale.
"""

import json
import uuid
from typing import Any
from unittest.mock import AsyncMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.services.debtor_simulator import (
    VALID_COMMUNICATION_STYLES,
    DebtorSimulatorService,
    EmotionalState,
    PersonaContext,
)
from app.services.llm_service import LLMResponse, LLMServiceProtocol


# --- Strategies ---

# Valid communication styles for generated LLM responses
communication_styles = st.sampled_from(sorted(VALID_COMMUNICATION_STYLES))

# Valid emotional states (1-5)
emotional_states = st.integers(min_value=1, max_value=5)

# Valid languages
languages = st.sampled_from(["EN", "TL", "TAGLISH"])

# Non-empty names for persona generation
persona_names = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
    min_size=1,
    max_size=80,
).filter(lambda s: s.strip() != "")

# Financial circumstances - non-empty dicts with various structures
income_levels = st.sampled_from(["low", "medium", "high"])

debt_amounts = st.integers(min_value=100, max_value=1_000_000)

reasons_for_delinquency = st.sampled_from([
    "Job loss",
    "Medical emergency",
    "Business failure",
    "Divorce",
    "Unexpected expenses",
    "Reduced income",
    "Natural disaster",
    "Family emergency",
])

# Build financial circumstances dictionaries with at least one key
financial_circumstances_strategy = st.fixed_dictionaries({
    "income_level": income_levels,
    "debt_amount": debt_amounts,
    "reason_for_delinquency": reasons_for_delinquency,
})

# Scenario types
scenario_types = st.sampled_from([
    "FINANCIAL_HARDSHIP",
    "ANGRY_CUSTOMER",
    "PAYMENT_EXTENSION",
    "BALANCE_DISPUTE",
])

# Build valid scenario dicts
valid_scenarios = st.fixed_dictionaries({
    "scenario_type": scenario_types,
    "description": st.text(min_size=1, max_size=100),
    "debtor_profile": st.fixed_dictionaries({
        "name": persona_names,
        "outstanding_balance": st.integers(min_value=100, max_value=500000),
        "days_past_due": st.integers(min_value=1, max_value=365),
        "personality_profile": st.sampled_from(["cooperative", "anxious", "hostile", "evasive"]),
        "conversation_goal": st.text(min_size=1, max_size=100).filter(lambda s: s.strip() != ""),
    }),
})


# Composite strategy to generate a valid LLM JSON response
@st.composite
def valid_llm_persona_responses(draw):
    """Generate a valid LLM JSON response for persona generation."""
    name = draw(persona_names)
    style = draw(communication_styles)
    financial = draw(financial_circumstances_strategy)
    state = draw(emotional_states)
    language = draw(languages)

    response_dict = {
        "name": name,
        "communication_style": style,
        "financial_circumstances": financial,
        "emotional_state": state,
        "language": language,
    }
    return json.dumps(response_dict)


class TestPersonaGenerationCompleteness:
    """Property 3: Persona generation completeness.

    Feature: collection-agent-trainer, Property 3: Persona generation completeness

    For any valid scenario, the Debtor Simulator's persona generation SHALL produce
    a persona containing a non-empty name, a communication_style from the defined set,
    populated financial_circumstances, and an emotional_state from the defined scale.
    """

    @given(
        llm_response=valid_llm_persona_responses(),
        scenario=valid_scenarios,
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_persona_has_all_required_fields(
        self,
        llm_response: str,
        scenario: dict[str, Any],
    ):
        """Generated persona always has all required fields populated.

        Given a valid LLM JSON response with various communication styles,
        emotional states, and financial circumstances, calling generate_persona
        SHALL produce a PersonaContext with:
        - Non-empty name
        - communication_style from VALID_COMMUNICATION_STYLES
        - Non-empty financial_circumstances dict
        - Valid EmotionalState value (1-5)
        - Valid language (EN, TL, or TAGLISH)
        - Valid UUID persona_id
        """
        # Arrange: mock LLM to return the generated response
        mock_llm = AsyncMock(spec=LLMServiceProtocol)
        mock_llm.chat_completion.return_value = LLMResponse(
            content=llm_response,
            model="test-model",
            usage={"prompt_tokens": 50, "completion_tokens": 50, "total_tokens": 100},
        )

        service = DebtorSimulatorService(llm_service=mock_llm)

        # Act
        persona = await service.generate_persona(scenario)

        # Assert: all required fields are populated and valid
        assert isinstance(persona, PersonaContext)

        # Non-empty name
        assert persona.name is not None
        assert isinstance(persona.name, str)
        assert len(persona.name.strip()) > 0

        # communication_style from defined set
        assert persona.communication_style in VALID_COMMUNICATION_STYLES

        # Non-empty financial_circumstances dict
        assert isinstance(persona.financial_circumstances, dict)
        assert len(persona.financial_circumstances) > 0

        # Valid EmotionalState value
        assert isinstance(persona.emotional_state, EmotionalState)
        assert persona.emotional_state.value in range(1, 6)

        # Valid language
        assert persona.language in ("EN", "TL", "TAGLISH")

        # Valid UUID persona_id
        assert isinstance(persona.persona_id, uuid.UUID)

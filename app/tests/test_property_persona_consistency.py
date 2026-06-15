"""Property-based tests for persona consistency across conversation turns.

Feature: collection-agent-trainer, Property 4: Persona consistency across conversation turns

**Validates: Requirements 2.2**

Property 4: For any active session with multiple conversation turns, the persona's
personality_trait, financial_circumstances, and name SHALL remain identical across
all turns in the session.
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.services.debtor_simulator import (
    DebtorSimulatorService,
    EmotionalState,
    PersonaContext,
)
from app.services.llm_service import LLMResponse


# --- Strategies ---

# Generate random persona names
persona_names = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
    min_size=1,
    max_size=50,
).filter(lambda s: s.strip() != "")

# Valid communication styles
communication_styles = st.sampled_from(["cooperative", "evasive", "hostile", "anxious"])

# Generate random financial circumstances
financial_circumstances_strategy = st.fixed_dictionaries({
    "income_level": st.sampled_from(["low", "medium", "high"]),
    "debt_amount": st.integers(min_value=1000, max_value=500000),
    "reason_for_delinquency": st.text(
        alphabet=st.characters(whitelist_categories=("L", "Zs")),
        min_size=5,
        max_size=100,
    ).filter(lambda s: s.strip() != ""),
})

# Generate random emotional states
emotional_states = st.sampled_from(list(EmotionalState))

# Generate random agent messages (various lengths/content)
agent_messages = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Zs")),
    min_size=5,
    max_size=200,
).filter(lambda s: s.strip() != "")

# Number of conversation turns to simulate (2-5)
num_turns = st.integers(min_value=2, max_value=5)


class TestPersonaConsistencyAcrossTurns:
    """Property 4: Persona consistency across conversation turns.

    Feature: collection-agent-trainer, Property 4: Persona consistency across conversation turns
    """

    @given(
        name=persona_names,
        style=communication_styles,
        finances=financial_circumstances_strategy,
        initial_state=emotional_states,
        messages=st.lists(agent_messages, min_size=2, max_size=5),
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_name_remains_constant_across_turns(
        self,
        name: str,
        style: str,
        finances: dict,
        initial_state: EmotionalState,
        messages: list[str],
    ):
        """Persona name remains identical after multiple generate_response calls."""
        persona = PersonaContext(
            persona_id=uuid.uuid4(),
            name=name,
            communication_style=style,
            financial_circumstances=finances,
            emotional_state=initial_state,
        )

        mock_llm = AsyncMock()
        mock_llm.chat_completion = AsyncMock(
            return_value=LLMResponse(content="I understand.", model="test-model")
        )
        service = DebtorSimulatorService(llm_service=mock_llm)

        for msg in messages:
            await service.generate_response(persona, msg)
            assert persona.name == name

    @given(
        name=persona_names,
        style=communication_styles,
        finances=financial_circumstances_strategy,
        initial_state=emotional_states,
        messages=st.lists(agent_messages, min_size=2, max_size=5),
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_communication_style_remains_constant_across_turns(
        self,
        name: str,
        style: str,
        finances: dict,
        initial_state: EmotionalState,
        messages: list[str],
    ):
        """Persona communication_style remains identical after multiple generate_response calls."""
        persona = PersonaContext(
            persona_id=uuid.uuid4(),
            name=name,
            communication_style=style,
            financial_circumstances=finances,
            emotional_state=initial_state,
        )

        mock_llm = AsyncMock()
        mock_llm.chat_completion = AsyncMock(
            return_value=LLMResponse(content="Okay, let me think.", model="test-model")
        )
        service = DebtorSimulatorService(llm_service=mock_llm)

        for msg in messages:
            await service.generate_response(persona, msg)
            assert persona.communication_style == style

    @given(
        name=persona_names,
        style=communication_styles,
        finances=financial_circumstances_strategy,
        initial_state=emotional_states,
        messages=st.lists(agent_messages, min_size=2, max_size=5),
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_financial_circumstances_remain_constant_across_turns(
        self,
        name: str,
        style: str,
        finances: dict,
        initial_state: EmotionalState,
        messages: list[str],
    ):
        """Persona financial_circumstances remains identical after multiple generate_response calls."""
        persona = PersonaContext(
            persona_id=uuid.uuid4(),
            name=name,
            communication_style=style,
            financial_circumstances=finances,
            emotional_state=initial_state,
        )

        # Take a snapshot of financial circumstances before
        original_finances = dict(finances)

        mock_llm = AsyncMock()
        mock_llm.chat_completion = AsyncMock(
            return_value=LLMResponse(content="I'll try my best.", model="test-model")
        )
        service = DebtorSimulatorService(llm_service=mock_llm)

        for msg in messages:
            await service.generate_response(persona, msg)
            assert persona.financial_circumstances == original_finances

    @given(
        name=persona_names,
        style=communication_styles,
        finances=financial_circumstances_strategy,
        initial_state=emotional_states,
        messages=st.lists(agent_messages, min_size=2, max_size=5),
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_all_persona_identity_fields_consistent_across_turns(
        self,
        name: str,
        style: str,
        finances: dict,
        initial_state: EmotionalState,
        messages: list[str],
    ):
        """All persona identity fields (name, communication_style, financial_circumstances)
        remain identical across all conversation turns simultaneously."""
        persona = PersonaContext(
            persona_id=uuid.uuid4(),
            name=name,
            communication_style=style,
            financial_circumstances=finances,
            emotional_state=initial_state,
        )

        original_finances = dict(finances)

        mock_llm = AsyncMock()
        mock_llm.chat_completion = AsyncMock(
            return_value=LLMResponse(content="That sounds reasonable.", model="test-model")
        )
        service = DebtorSimulatorService(llm_service=mock_llm)

        for msg in messages:
            await service.generate_response(persona, msg)
            # Verify all identity fields remain unchanged
            assert persona.name == name, (
                f"Name changed from '{name}' to '{persona.name}'"
            )
            assert persona.communication_style == style, (
                f"Communication style changed from '{style}' to '{persona.communication_style}'"
            )
            assert persona.financial_circumstances == original_finances, (
                f"Financial circumstances changed from {original_finances} "
                f"to {persona.financial_circumstances}"
            )

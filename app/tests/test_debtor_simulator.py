"""Tests for the Debtor Simulator service - persona generation (Task 5.1)."""

import json
import uuid
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.services.debtor_simulator import (
    VALID_COMMUNICATION_STYLES,
    DebtorSimulatorService,
    EmotionalState,
    PersonaContext,
    _build_persona_generation_prompt,
    _parse_persona_response,
)
from app.services.llm_service import LLMMessage, LLMResponse, LLMServiceProtocol


# --- Fixtures ---


@pytest.fixture
def sample_scenario() -> dict[str, Any]:
    """A valid scenario for testing persona generation."""
    return {
        "id": str(uuid.uuid4()),
        "name": "Financial Hardship Scenario",
        "scenario_type": "FINANCIAL_HARDSHIP",
        "description": "Debtor lost their job and cannot make payments",
        "debtor_profile": {
            "name": "Maria Santos",
            "outstanding_balance": 15000,
            "days_past_due": 45,
            "personality_profile": "anxious",
            "conversation_goal": "Request payment plan",
        },
    }


@pytest.fixture
def valid_llm_persona_response() -> str:
    """A valid JSON response from the LLM for persona generation."""
    return json.dumps(
        {
            "name": "Maria Santos",
            "communication_style": "anxious",
            "financial_circumstances": {
                "income_level": "low",
                "debt_amount": 15000,
                "reason_for_delinquency": "Job loss due to company downsizing",
            },
            "emotional_state": 2,
            "language": "EN",
        }
    )


@pytest.fixture
def mock_llm_service(valid_llm_persona_response: str) -> AsyncMock:
    """Mock LLM service returning a valid persona response."""
    mock = AsyncMock(spec=LLMServiceProtocol)
    mock.chat_completion.return_value = LLMResponse(
        content=valid_llm_persona_response,
        model="qwen3:32b",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    )
    return mock


# --- Unit Tests: Prompt Building ---


class TestBuildPersonaGenerationPrompt:
    """Tests for the system prompt builder."""

    def test_includes_scenario_type(self, sample_scenario: dict[str, Any]):
        prompt = _build_persona_generation_prompt(sample_scenario)
        assert "FINANCIAL_HARDSHIP" in prompt

    def test_includes_debtor_name(self, sample_scenario: dict[str, Any]):
        prompt = _build_persona_generation_prompt(sample_scenario)
        assert "Maria Santos" in prompt

    def test_includes_personality_profile(self, sample_scenario: dict[str, Any]):
        prompt = _build_persona_generation_prompt(sample_scenario)
        assert "anxious" in prompt

    def test_includes_balance_and_days(self, sample_scenario: dict[str, Any]):
        prompt = _build_persona_generation_prompt(sample_scenario)
        assert "15000" in prompt
        assert "45" in prompt

    def test_includes_conversation_goal(self, sample_scenario: dict[str, Any]):
        prompt = _build_persona_generation_prompt(sample_scenario)
        assert "Request payment plan" in prompt

    def test_includes_json_format_instructions(self, sample_scenario: dict[str, Any]):
        prompt = _build_persona_generation_prompt(sample_scenario)
        assert "communication_style" in prompt
        assert "financial_circumstances" in prompt
        assert "emotional_state" in prompt

    def test_handles_missing_fields_gracefully(self):
        minimal_scenario = {"debtor_profile": {}, "scenario_type": "ANGRY_CUSTOMER"}
        prompt = _build_persona_generation_prompt(minimal_scenario)
        assert "ANGRY_CUSTOMER" in prompt
        assert "Unknown" in prompt


# --- Unit Tests: Response Parsing ---


class TestParsePersonaResponse:
    """Tests for parsing LLM responses into PersonaContext."""

    def test_parses_valid_json(self, sample_scenario: dict[str, Any]):
        response = json.dumps(
            {
                "name": "Juan Dela Cruz",
                "communication_style": "evasive",
                "financial_circumstances": {
                    "income_level": "medium",
                    "debt_amount": 25000,
                    "reason_for_delinquency": "Medical emergency",
                },
                "emotional_state": 3,
                "language": "TL",
            }
        )
        persona = _parse_persona_response(response, sample_scenario)

        assert persona.name == "Juan Dela Cruz"
        assert persona.communication_style == "evasive"
        assert persona.financial_circumstances["income_level"] == "medium"
        assert persona.emotional_state == EmotionalState.NEUTRAL
        assert persona.language == "TL"
        assert isinstance(persona.persona_id, uuid.UUID)
        assert persona.conversation_history == []

    def test_parses_json_with_markdown_fences(self, sample_scenario: dict[str, Any]):
        response = '```json\n{"name": "Ana Reyes", "communication_style": "hostile", "financial_circumstances": {"income_level": "low", "debt_amount": 5000, "reason_for_delinquency": "Dispute"}, "emotional_state": 1, "language": "EN"}\n```'
        persona = _parse_persona_response(response, sample_scenario)

        assert persona.name == "Ana Reyes"
        assert persona.communication_style == "hostile"
        assert persona.emotional_state == EmotionalState.HOSTILE

    def test_raises_on_invalid_json(self, sample_scenario: dict[str, Any]):
        with pytest.raises(ValueError, match="not valid JSON"):
            _parse_persona_response("not json at all", sample_scenario)

    def test_raises_on_empty_name(self, sample_scenario: dict[str, Any]):
        response = json.dumps(
            {
                "name": "",
                "communication_style": "cooperative",
                "financial_circumstances": {"income_level": "high", "debt_amount": 1000, "reason_for_delinquency": "Forgot"},
                "emotional_state": 5,
                "language": "EN",
            }
        )
        with pytest.raises(ValueError, match="name is empty"):
            _parse_persona_response(response, sample_scenario)

    def test_raises_on_invalid_communication_style(self, sample_scenario: dict[str, Any]):
        response = json.dumps(
            {
                "name": "Test Person",
                "communication_style": "aggressive",
                "financial_circumstances": {"income_level": "low", "debt_amount": 1000, "reason_for_delinquency": "Unknown"},
                "emotional_state": 3,
                "language": "EN",
            }
        )
        with pytest.raises(ValueError, match="Invalid communication_style"):
            _parse_persona_response(response, sample_scenario)

    def test_raises_on_empty_financial_circumstances(self, sample_scenario: dict[str, Any]):
        response = json.dumps(
            {
                "name": "Test Person",
                "communication_style": "cooperative",
                "financial_circumstances": {},
                "emotional_state": 3,
                "language": "EN",
            }
        )
        with pytest.raises(ValueError, match="financial_circumstances cannot be empty"):
            _parse_persona_response(response, sample_scenario)

    def test_defaults_invalid_emotional_state_to_neutral(self, sample_scenario: dict[str, Any]):
        response = json.dumps(
            {
                "name": "Test Person",
                "communication_style": "cooperative",
                "financial_circumstances": {"income_level": "high", "debt_amount": 500, "reason_for_delinquency": "Oversight"},
                "emotional_state": 99,
                "language": "EN",
            }
        )
        persona = _parse_persona_response(response, sample_scenario)
        assert persona.emotional_state == EmotionalState.NEUTRAL

    def test_defaults_invalid_language_to_en(self, sample_scenario: dict[str, Any]):
        response = json.dumps(
            {
                "name": "Test Person",
                "communication_style": "anxious",
                "financial_circumstances": {"income_level": "low", "debt_amount": 2000, "reason_for_delinquency": "Lost job"},
                "emotional_state": 2,
                "language": "FR",
            }
        )
        persona = _parse_persona_response(response, sample_scenario)
        assert persona.language == "EN"

    def test_all_valid_communication_styles_accepted(self, sample_scenario: dict[str, Any]):
        for style in VALID_COMMUNICATION_STYLES:
            response = json.dumps(
                {
                    "name": "Test Person",
                    "communication_style": style,
                    "financial_circumstances": {"income_level": "medium", "debt_amount": 3000, "reason_for_delinquency": "Test"},
                    "emotional_state": 3,
                    "language": "EN",
                }
            )
            persona = _parse_persona_response(response, sample_scenario)
            assert persona.communication_style == style

    def test_all_valid_emotional_states_accepted(self, sample_scenario: dict[str, Any]):
        for state in EmotionalState:
            response = json.dumps(
                {
                    "name": "Test Person",
                    "communication_style": "cooperative",
                    "financial_circumstances": {"income_level": "low", "debt_amount": 1000, "reason_for_delinquency": "Test"},
                    "emotional_state": state.value,
                    "language": "EN",
                }
            )
            persona = _parse_persona_response(response, sample_scenario)
            assert persona.emotional_state == state


# --- Integration Tests: DebtorSimulatorService.generate_persona ---


class TestDebtorSimulatorServiceGeneratePersona:
    """Tests for the full generate_persona flow with mocked LLM."""

    @pytest.mark.asyncio
    async def test_generate_persona_returns_complete_context(
        self, mock_llm_service: AsyncMock, sample_scenario: dict[str, Any]
    ):
        service = DebtorSimulatorService(llm_service=mock_llm_service)
        persona = await service.generate_persona(sample_scenario)

        assert isinstance(persona, PersonaContext)
        assert persona.name == "Maria Santos"
        assert persona.communication_style == "anxious"
        assert persona.financial_circumstances["income_level"] == "low"
        assert persona.financial_circumstances["debt_amount"] == 15000
        assert persona.emotional_state == EmotionalState.DEFENSIVE
        assert persona.language == "EN"
        assert isinstance(persona.persona_id, uuid.UUID)
        assert persona.conversation_history == []

    @pytest.mark.asyncio
    async def test_generate_persona_calls_llm_with_correct_messages(
        self, mock_llm_service: AsyncMock, sample_scenario: dict[str, Any]
    ):
        service = DebtorSimulatorService(llm_service=mock_llm_service)
        await service.generate_persona(sample_scenario)

        mock_llm_service.chat_completion.assert_called_once()
        call_args = mock_llm_service.chat_completion.call_args

        messages = call_args[0][0]
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert "FINANCIAL_HARDSHIP" in messages[0].content
        assert "Maria Santos" in messages[0].content

    @pytest.mark.asyncio
    async def test_generate_persona_passes_json_response_format(
        self, mock_llm_service: AsyncMock, sample_scenario: dict[str, Any]
    ):
        service = DebtorSimulatorService(llm_service=mock_llm_service)
        await service.generate_persona(sample_scenario)

        call_kwargs = mock_llm_service.chat_completion.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_generate_persona_raises_on_invalid_llm_response(
        self, sample_scenario: dict[str, Any]
    ):
        mock_llm = AsyncMock(spec=LLMServiceProtocol)
        mock_llm.chat_completion.return_value = LLMResponse(
            content="This is not JSON",
            model="test-model",
        )

        service = DebtorSimulatorService(llm_service=mock_llm)
        with pytest.raises(ValueError, match="not valid JSON"):
            await service.generate_persona(sample_scenario)

    @pytest.mark.asyncio
    async def test_generate_persona_with_different_scenario_types(
        self, mock_llm_service: AsyncMock
    ):
        """Verify different scenario types produce valid calls."""
        scenarios = [
            {
                "scenario_type": "ANGRY_CUSTOMER",
                "description": "Irate customer disputing charges",
                "debtor_profile": {
                    "name": "Pedro Reyes",
                    "outstanding_balance": 5000,
                    "days_past_due": 90,
                    "personality_profile": "hostile",
                    "conversation_goal": "Dispute resolution",
                },
            },
            {
                "scenario_type": "BALANCE_DISPUTE",
                "description": "Customer claims balance is wrong",
                "debtor_profile": {
                    "name": "Elena Cruz",
                    "outstanding_balance": 8500,
                    "days_past_due": 30,
                    "personality_profile": "evasive",
                    "conversation_goal": "Balance verification",
                },
            },
        ]

        service = DebtorSimulatorService(llm_service=mock_llm_service)
        for scenario in scenarios:
            persona = await service.generate_persona(scenario)
            assert isinstance(persona, PersonaContext)
            assert persona.name  # non-empty name
            assert persona.communication_style in VALID_COMMUNICATION_STYLES

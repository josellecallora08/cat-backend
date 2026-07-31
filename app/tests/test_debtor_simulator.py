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
    contains_prohibited_response,
    select_opening_response,
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


# --- Unit Tests: Script_Version Pinning Isolation (Task 13.17) ---


class TestScriptVersionPinningIsolation:
    """Verify a running call keeps using its pinned Script_Version content
    even after a newer Script_Version is published mid-call (Req 4.4, 4.11).

    At this file's level (no DB), the "pin" is simply: the caller always
    passes the SAME `script_content` dict on every turn. These tests show
    that `select_opening_response`/`generate_response` have no mechanism to
    reach for a script_content dict that was never passed to them -- a
    newly published version's content only matters if a caller starts
    passing it, which a properly isolated running call never does.
    """

    @pytest.fixture
    def script_content_v1(self) -> dict[str, Any]:
        """The Script_Version content pinned at session/call start."""
        return {
            "opening_response": "Hello, this is Maria.",
            "prohibited_responses": ["I refuse to pay this debt"],
            "trigger_phrases": [],
            "escalation_conditions": [],
            "payment_conditions": [],
            "expected_replies": [],
        }

    @pytest.fixture
    def script_content_v2(self) -> dict[str, Any]:
        """A DIFFERENT Script_Version content 'published' mid-call by an
        admin. It is never assigned back into the running call's state --
        it only exists here to prove it has no influence on the pinned call.
        """
        return {
            "opening_response": "Good day, this is Ana.",
            "prohibited_responses": ["I will never talk to you"],
            "trigger_phrases": [],
            "escalation_conditions": [],
            "payment_conditions": [],
            "expected_replies": [],
        }

    def test_opening_response_uses_pinned_content_at_call_start(
        self, script_content_v1: dict[str, Any]
    ):
        """Req 4.4: the opening utterance comes verbatim from the pinned
        Script_Version's content."""
        assert (
            select_opening_response(script_content_v1)
            == script_content_v1["opening_response"]
        )

    @pytest.mark.asyncio
    async def test_generate_response_stays_pinned_after_mid_call_publish(
        self,
        script_content_v1: dict[str, Any],
        script_content_v2: dict[str, Any],
    ):
        """Req 4.11: a second Script_Version published mid-call must not
        affect the running session's effective content.

        The LLM is mocked to return text that matches `script_content_v2`'s
        `prohibited_responses` entry but NOT `script_content_v1`'s entry.
        Since the running call keeps passing the SAME pinned
        `script_content_v1` object on every turn, `generate_response` only
        ever checks the response against v1's prohibited list -- v2's newly
        published prohibited entry has zero effect, and the raw LLM text is
        returned unchanged (no retry, no SAFE_DEFAULT fallback).
        """
        llm_text_matching_v2_only = "I will never talk to you about this."

        mock_llm = AsyncMock(spec=LLMServiceProtocol)
        mock_llm.chat_completion.return_value = LLMResponse(
            content=llm_text_matching_v2_only,
            model="qwen3:32b",
        )

        service = DebtorSimulatorService(llm_service=mock_llm)
        persona = PersonaContext(
            persona_id=uuid.uuid4(),
            name="Maria Santos",
            communication_style="anxious",
            financial_circumstances={"income_level": "low", "debt_amount": 15000},
            emotional_state=EmotionalState.NEUTRAL,
        )

        # Sanity check: the mocked text would have been flagged as
        # prohibited had it been checked against v2's list.
        assert contains_prohibited_response(
            llm_text_matching_v2_only, script_content_v2["prohibited_responses"]
        )
        # ...but it is NOT prohibited under the pinned v1 list.
        assert not contains_prohibited_response(
            llm_text_matching_v2_only, script_content_v1["prohibited_responses"]
        )

        # The running call passes the pinned script_content_v1 on this turn,
        # exactly as it would have on every prior turn -- script_content_v2
        # (the "newly published" version) is never passed at all.
        response = await service.generate_response(
            persona,
            "Bakit ka tumatawag?",
            script_content=script_content_v1,
        )

        # Response reflects script_content_v1's data: no match against v1's
        # prohibited list means the raw LLM text passes through unchanged,
        # not the SAFE_DEFAULT fallback that would occur under v2's rules.
        assert response.text == llm_text_matching_v2_only
        # Only one LLM call was made: no prohibited-response retry was
        # triggered, confirming v2's prohibited entry was never consulted.
        mock_llm.chat_completion.assert_called_once()

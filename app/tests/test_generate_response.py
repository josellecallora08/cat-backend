"""Tests for DebtorSimulatorService.generate_response (Task 5.3)."""

import uuid
from unittest.mock import AsyncMock

import pytest

from app.services.debtor_simulator import (
    AgentTone,
    DebtorSimulatorService,
    EmotionalState,
    Message,
    PersonaContext,
    SimulatorResponse,
    detect_language,
)
from app.services.llm_service import LLMMessage, LLMResponse, LLMServiceProtocol


# --- Fixtures ---


@pytest.fixture
def sample_persona() -> PersonaContext:
    """A persona context for testing response generation."""
    return PersonaContext(
        persona_id=uuid.uuid4(),
        name="Maria Santos",
        communication_style="anxious",
        financial_circumstances={
            "income_level": "low",
            "debt_amount": 15000,
            "reason_for_delinquency": "Job loss due to company downsizing",
        },
        emotional_state=EmotionalState.DEFENSIVE,
        conversation_history=[],
        language="EN",
    )


@pytest.fixture
def persona_with_history() -> PersonaContext:
    """A persona with existing conversation history."""
    return PersonaContext(
        persona_id=uuid.uuid4(),
        name="Juan Dela Cruz",
        communication_style="evasive",
        financial_circumstances={
            "income_level": "medium",
            "debt_amount": 25000,
            "reason_for_delinquency": "Medical emergency",
        },
        emotional_state=EmotionalState.NEUTRAL,
        conversation_history=[
            Message(role="agent", content="Good morning, this is calling about your account."),
            Message(role="debtor", content="What account? I don't know what you're talking about."),
        ],
        language="EN",
    )


@pytest.fixture
def mock_llm_service() -> AsyncMock:
    """Mock LLM service that returns a simple debtor response."""
    mock = AsyncMock(spec=LLMServiceProtocol)
    mock.chat_completion.return_value = LLMResponse(
        content="I... I know I owe money. I lost my job last month. Can we work something out?",
        model="qwen3:32b",
        usage={"prompt_tokens": 200, "completion_tokens": 30, "total_tokens": 230},
    )
    return mock


# --- Unit Tests: Language Detection ---


class TestDetectLanguage:
    """Tests for the detect_language helper function."""

    def test_detects_english(self):
        assert detect_language("Hello, I'm calling about your account balance.") == "EN"

    def test_detects_tagalog(self):
        assert detect_language("Hindi ko po kaya magbayad ng utang ko ngayon.") == "TL"

    def test_detects_taglish(self):
        assert detect_language("I can't pay right now kasi wala akong trabaho.") == "TAGLISH"

    def test_empty_string_returns_en(self):
        assert detect_language("") == "EN"

    def test_single_english_word(self):
        assert detect_language("Hello") == "EN"

    def test_single_tagalog_word(self):
        assert detect_language("Salamat") == "TL"

    def test_mostly_tagalog_with_some_english(self):
        assert detect_language("Wala po akong pera, sorry na lang.") == "TL"

    def test_mostly_english_with_few_tagalog(self):
        assert detect_language("I need more time kasi mahirap ang situation ko.") == "TAGLISH"


# --- Unit Tests: SimulatorResponse Dataclass ---


class TestSimulatorResponse:
    """Tests for the SimulatorResponse dataclass."""

    def test_creates_valid_response(self):
        resp = SimulatorResponse(
            text="I can't pay right now.",
            emotional_state=EmotionalState.DEFENSIVE,
            language="EN",
        )
        assert resp.text == "I can't pay right now."
        assert resp.emotional_state == EmotionalState.DEFENSIVE
        assert resp.language == "EN"


# --- Integration Tests: generate_response ---


class TestGenerateResponse:
    """Tests for DebtorSimulatorService.generate_response."""

    @pytest.mark.asyncio
    async def test_returns_simulator_response(
        self, mock_llm_service: AsyncMock, sample_persona: PersonaContext
    ):
        service = DebtorSimulatorService(llm_service=mock_llm_service)
        result = await service.generate_response(sample_persona, "Hi, I'm calling about your balance.")

        assert isinstance(result, SimulatorResponse)
        assert result.text != ""
        assert isinstance(result.emotional_state, EmotionalState)
        assert result.language in ("EN", "TL", "TAGLISH")

    @pytest.mark.asyncio
    async def test_empathetic_message_increases_emotional_state(
        self, mock_llm_service: AsyncMock, sample_persona: PersonaContext
    ):
        """Empathetic agent tone should move state toward cooperative."""
        sample_persona.emotional_state = EmotionalState.DEFENSIVE
        service = DebtorSimulatorService(llm_service=mock_llm_service)

        result = await service.generate_response(
            sample_persona,
            "I understand this is difficult. I want to help you find a solution that works for you.",
        )

        # DEFENSIVE(2) + empathetic → NEUTRAL(3)
        assert result.emotional_state == EmotionalState.NEUTRAL
        assert sample_persona.emotional_state == EmotionalState.NEUTRAL

    @pytest.mark.asyncio
    async def test_aggressive_message_decreases_emotional_state(
        self, mock_llm_service: AsyncMock, sample_persona: PersonaContext
    ):
        """Aggressive agent tone should move state toward hostile."""
        sample_persona.emotional_state = EmotionalState.NEUTRAL
        service = DebtorSimulatorService(llm_service=mock_llm_service)

        result = await service.generate_response(
            sample_persona,
            "You must pay immediately or we will take legal action. This is your final notice.",
        )

        # NEUTRAL(3) + aggressive → DEFENSIVE(2)
        assert result.emotional_state == EmotionalState.DEFENSIVE
        assert sample_persona.emotional_state == EmotionalState.DEFENSIVE

    @pytest.mark.asyncio
    async def test_neutral_message_no_state_change(
        self, mock_llm_service: AsyncMock, sample_persona: PersonaContext
    ):
        """Neutral agent tone should not change emotional state."""
        sample_persona.emotional_state = EmotionalState.NEUTRAL
        service = DebtorSimulatorService(llm_service=mock_llm_service)

        result = await service.generate_response(
            sample_persona,
            "Your current balance is 15,000 pesos.",
        )

        assert result.emotional_state == EmotionalState.NEUTRAL

    @pytest.mark.asyncio
    async def test_updates_conversation_history(
        self, mock_llm_service: AsyncMock, sample_persona: PersonaContext
    ):
        """Both agent message and debtor response should be added to history."""
        service = DebtorSimulatorService(llm_service=mock_llm_service)
        agent_msg = "Can we discuss your payment options?"
        await service.generate_response(sample_persona, agent_msg)

        assert len(sample_persona.conversation_history) == 2
        assert sample_persona.conversation_history[0].role == "agent"
        assert sample_persona.conversation_history[0].content == agent_msg
        assert sample_persona.conversation_history[1].role == "debtor"
        assert sample_persona.conversation_history[1].content != ""

    @pytest.mark.asyncio
    async def test_conversation_history_accumulates(
        self, mock_llm_service: AsyncMock, sample_persona: PersonaContext
    ):
        """Multiple calls should accumulate history."""
        service = DebtorSimulatorService(llm_service=mock_llm_service)

        await service.generate_response(sample_persona, "First message")
        await service.generate_response(sample_persona, "Second message")

        assert len(sample_persona.conversation_history) == 4
        assert sample_persona.conversation_history[0].role == "agent"
        assert sample_persona.conversation_history[1].role == "debtor"
        assert sample_persona.conversation_history[2].role == "agent"
        assert sample_persona.conversation_history[3].role == "debtor"

    @pytest.mark.asyncio
    async def test_includes_history_in_llm_messages(
        self, mock_llm_service: AsyncMock, persona_with_history: PersonaContext
    ):
        """Existing conversation history should be included in LLM messages."""
        service = DebtorSimulatorService(llm_service=mock_llm_service)
        await service.generate_response(persona_with_history, "Let me check your account details.")

        call_args = mock_llm_service.chat_completion.call_args
        messages = call_args[0][0]

        # system + 2 history messages + 1 new message = 4
        assert len(messages) == 4
        assert messages[0].role == "system"
        assert messages[1].role == "user"  # agent history
        assert messages[1].content == "Good morning, this is calling about your account."
        assert messages[2].role == "assistant"  # debtor history
        assert messages[2].content == "What account? I don't know what you're talking about."
        assert messages[3].role == "user"  # new agent message
        assert messages[3].content == "Let me check your account details."

    @pytest.mark.asyncio
    async def test_detects_english_language(
        self, mock_llm_service: AsyncMock, sample_persona: PersonaContext
    ):
        service = DebtorSimulatorService(llm_service=mock_llm_service)
        result = await service.generate_response(
            sample_persona, "Good morning, I'm calling about your outstanding balance."
        )
        assert result.language == "EN"

    @pytest.mark.asyncio
    async def test_detects_tagalog_language(
        self, mock_llm_service: AsyncMock, sample_persona: PersonaContext
    ):
        mock_llm_service.chat_completion.return_value = LLMResponse(
            content="Hindi ko po kaya magbayad ngayon.",
            model="qwen3:32b",
        )
        service = DebtorSimulatorService(llm_service=mock_llm_service)
        result = await service.generate_response(
            sample_persona, "Magandang umaga po, ang tawag ko po ay tungkol sa utang niyo."
        )
        assert result.language == "TL"

    @pytest.mark.asyncio
    async def test_detects_taglish_language(
        self, mock_llm_service: AsyncMock, sample_persona: PersonaContext
    ):
        service = DebtorSimulatorService(llm_service=mock_llm_service)
        result = await service.generate_response(
            sample_persona, "I need to discuss your account kasi may balance pa po kayo."
        )
        assert result.language == "TAGLISH"

    @pytest.mark.asyncio
    async def test_system_prompt_contains_persona_details(
        self, mock_llm_service: AsyncMock, sample_persona: PersonaContext
    ):
        """System prompt should include persona name, style, state, and circumstances."""
        service = DebtorSimulatorService(llm_service=mock_llm_service)
        await service.generate_response(sample_persona, "Hello there")

        call_args = mock_llm_service.chat_completion.call_args
        messages = call_args[0][0]
        system_content = messages[0].content

        assert "Maria Santos" in system_content
        assert "anxious" in system_content
        assert "Job loss" in system_content or "income_level" in system_content

    @pytest.mark.asyncio
    async def test_system_prompt_contains_language_instruction(
        self, mock_llm_service: AsyncMock, sample_persona: PersonaContext
    ):
        """System prompt should instruct LLM to respond in detected language."""
        service = DebtorSimulatorService(llm_service=mock_llm_service)
        await service.generate_response(sample_persona, "Hello, how are you today?")

        call_args = mock_llm_service.chat_completion.call_args
        messages = call_args[0][0]
        system_content = messages[0].content

        assert "English" in system_content

    @pytest.mark.asyncio
    async def test_system_prompt_contains_emotional_state(
        self, mock_llm_service: AsyncMock, sample_persona: PersonaContext
    ):
        """System prompt should reflect the current emotional state."""
        sample_persona.emotional_state = EmotionalState.HOSTILE
        service = DebtorSimulatorService(llm_service=mock_llm_service)
        await service.generate_response(sample_persona, "Pay your bill now!")

        call_args = mock_llm_service.chat_completion.call_args
        messages = call_args[0][0]
        system_content = messages[0].content

        assert "hostile" in system_content

    @pytest.mark.asyncio
    async def test_calls_llm_with_temperature(
        self, mock_llm_service: AsyncMock, sample_persona: PersonaContext
    ):
        """LLM should be called with temperature for conversational variety."""
        service = DebtorSimulatorService(llm_service=mock_llm_service)
        await service.generate_response(sample_persona, "Hello")

        call_kwargs = mock_llm_service.chat_completion.call_args[1]
        assert "temperature" in call_kwargs
        assert call_kwargs["temperature"] == 0.8

    @pytest.mark.asyncio
    async def test_boundary_state_cooperative_with_empathetic_stays(
        self, mock_llm_service: AsyncMock, sample_persona: PersonaContext
    ):
        """At COOPERATIVE boundary, empathetic tone should not exceed max."""
        sample_persona.emotional_state = EmotionalState.COOPERATIVE
        service = DebtorSimulatorService(llm_service=mock_llm_service)
        result = await service.generate_response(
            sample_persona, "I really appreciate your time and want to help you."
        )
        assert result.emotional_state == EmotionalState.COOPERATIVE

    @pytest.mark.asyncio
    async def test_boundary_state_hostile_with_aggressive_stays(
        self, mock_llm_service: AsyncMock, sample_persona: PersonaContext
    ):
        """At HOSTILE boundary, aggressive tone should not go below min."""
        sample_persona.emotional_state = EmotionalState.HOSTILE
        service = DebtorSimulatorService(llm_service=mock_llm_service)
        result = await service.generate_response(
            sample_persona, "You must pay immediately or face legal action!"
        )
        assert result.emotional_state == EmotionalState.HOSTILE

    @pytest.mark.asyncio
    async def test_response_text_is_stripped(
        self, mock_llm_service: AsyncMock, sample_persona: PersonaContext
    ):
        """Response text from LLM should be stripped of whitespace."""
        mock_llm_service.chat_completion.return_value = LLMResponse(
            content="  I need more time to pay.  \n",
            model="qwen3:32b",
        )
        service = DebtorSimulatorService(llm_service=mock_llm_service)
        result = await service.generate_response(sample_persona, "When can you pay?")
        assert result.text == "I need more time to pay."

    @pytest.mark.asyncio
    async def test_persona_language_updated(
        self, mock_llm_service: AsyncMock, sample_persona: PersonaContext
    ):
        """Persona language field should be updated to match detected language."""
        assert sample_persona.language == "EN"
        service = DebtorSimulatorService(llm_service=mock_llm_service)
        mock_llm_service.chat_completion.return_value = LLMResponse(
            content="Hindi ko po kaya.",
            model="qwen3:32b",
        )
        await service.generate_response(
            sample_persona, "Kumusta po, tawag ko po ito tungkol sa balance niyo."
        )
        assert sample_persona.language == "TL"

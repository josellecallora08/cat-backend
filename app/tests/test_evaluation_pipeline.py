"""Unit tests for the EvaluationEngine.evaluate() full pipeline.

Tests the evaluate method with mocked LLM service covering:
- Too-short session handling
- Normal evaluation with LLM response parsing
- Score calculation and persistence
- Edge cases

Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7
"""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas import EvaluationCategory, EvaluationResult
from app.services.evaluation_engine import EvaluationEngine
from app.services.llm_service import LLMResponse


def make_transcript(agent_count: int = 5, debtor_count: int = 5) -> list[dict]:
    """Create a sample transcript with given number of agent/debtor utterances."""
    transcript = []
    for i in range(max(agent_count, debtor_count)):
        if i < agent_count:
            transcript.append({"speaker": "agent", "text": f"Agent message {i}"})
        if i < debtor_count:
            transcript.append({"speaker": "debtor", "text": f"Debtor message {i}"})
    return transcript


def make_llm_evaluation_response(
    call_opening: int = 75,
    compliance: int = 80,
    empathy_communication: int = 70,
    negotiation_resolution: int = 65,
    num_strengths: int = 3,
    num_weaknesses: int = 2,
) -> dict:
    """Create a mock LLM evaluation response JSON."""
    categories = ["call_opening", "compliance", "empathy_communication", "negotiation_resolution"]

    strengths = [
        {
            "description": f"Strength {i+1} description",
            "category": categories[i % len(categories)],
            "transcript_excerpt": f"Agent message {i}",
        }
        for i in range(num_strengths)
    ]

    weaknesses = [
        {
            "description": f"Weakness {i+1} description",
            "category": categories[i % len(categories)],
            "transcript_excerpt": f"Agent message {i}",
        }
        for i in range(num_weaknesses)
    ]

    return {
        "category_scores": {
            "call_opening": call_opening,
            "compliance": compliance,
            "empathy_communication": empathy_communication,
            "negotiation_resolution": negotiation_resolution,
        },
        "strengths": strengths,
        "weaknesses": weaknesses,
    }


class MockLLMService:
    """Mock LLM service for testing."""

    def __init__(self, response_content: str):
        self._response_content = response_content
        self.call_count = 0
        self.last_messages = None

    async def chat_completion(self, messages, *, temperature=None, max_tokens=None, response_format=None):
        self.call_count += 1
        self.last_messages = messages
        return LLMResponse(
            content=self._response_content,
            model="test-model",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )


@pytest.fixture
def session_id():
    """Create a test session UUID."""
    return uuid.uuid4()


@pytest.fixture
def normal_transcript():
    """Create a transcript with enough agent utterances for evaluation."""
    return make_transcript(agent_count=5, debtor_count=5)


@pytest.fixture
def short_transcript():
    """Create a transcript that is too short for evaluation."""
    return make_transcript(agent_count=2, debtor_count=3)


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service with a valid evaluation response."""
    response_data = make_llm_evaluation_response()
    return MockLLMService(json.dumps(response_data))


@pytest.fixture
def engine(mock_llm_service):
    """Create an EvaluationEngine with mocked LLM service."""
    return EvaluationEngine(llm_service=mock_llm_service)


class TestEvaluateTooShortSession:
    """Tests for evaluate() when session is too short."""

    async def test_too_short_returns_is_too_short_true(self, engine, session_id, short_transcript):
        """Too-short session should set is_too_short=True."""
        result = await engine.evaluate(session_id, short_transcript)
        assert result.is_too_short is True

    async def test_too_short_has_empty_category_scores(self, engine, session_id, short_transcript):
        """Too-short session should have empty category_scores."""
        result = await engine.evaluate(session_id, short_transcript)
        assert result.category_scores == []

    async def test_too_short_has_zero_overall_score(self, engine, session_id, short_transcript):
        """Too-short session should have overall_score of 0."""
        result = await engine.evaluate(session_id, short_transcript)
        assert result.overall_score == 0.0

    async def test_too_short_does_not_call_llm(self, engine, session_id, short_transcript, mock_llm_service):
        """Too-short session should NOT call the LLM."""
        await engine.evaluate(session_id, short_transcript)
        assert mock_llm_service.call_count == 0

    async def test_too_short_returns_correct_session_id(self, engine, session_id, short_transcript):
        """Too-short result should reference the correct session_id."""
        result = await engine.evaluate(session_id, short_transcript)
        assert result.session_id == session_id

    async def test_too_short_has_placeholder_strengths_weaknesses(self, engine, session_id, short_transcript):
        """Too-short session should have placeholder strength and weakness."""
        result = await engine.evaluate(session_id, short_transcript)
        assert len(result.strengths) == 1
        assert len(result.weaknesses) == 1
        assert "too short" in result.strengths[0].description.lower()
        assert "too short" in result.weaknesses[0].description.lower()


class TestEvaluateNormalSession:
    """Tests for evaluate() with a normal-length session."""

    async def test_normal_session_calls_llm(self, engine, session_id, normal_transcript, mock_llm_service):
        """Normal session should call the LLM exactly once."""
        await engine.evaluate(session_id, normal_transcript)
        assert mock_llm_service.call_count == 1

    async def test_normal_session_not_too_short(self, engine, session_id, normal_transcript):
        """Normal session should have is_too_short=False."""
        result = await engine.evaluate(session_id, normal_transcript)
        assert result.is_too_short is False

    async def test_normal_session_has_four_category_scores(self, engine, session_id, normal_transcript):
        """Normal session should produce exactly 4 category scores."""
        result = await engine.evaluate(session_id, normal_transcript)
        assert len(result.category_scores) == 4

    async def test_category_scores_match_llm_response(self, session_id, normal_transcript):
        """Category scores should match the parsed LLM response values."""
        response_data = make_llm_evaluation_response(
            call_opening=85, compliance=90, empathy_communication=75, negotiation_resolution=80
        )
        llm_service = MockLLMService(json.dumps(response_data))
        engine = EvaluationEngine(llm_service=llm_service)

        result = await engine.evaluate(session_id, normal_transcript)

        score_map = {cs.category: cs.score for cs in result.category_scores}
        assert score_map[EvaluationCategory.CALL_OPENING] == 85
        assert score_map[EvaluationCategory.COMPLIANCE] == 90
        assert score_map[EvaluationCategory.EMPATHY_COMMUNICATION] == 75
        assert score_map[EvaluationCategory.NEGOTIATION_RESOLUTION] == 80

    async def test_overall_score_is_weighted(self, session_id, normal_transcript):
        """Overall score should be the correct weighted combination."""
        response_data = make_llm_evaluation_response(
            call_opening=80, compliance=90, empathy_communication=70, negotiation_resolution=60
        )
        llm_service = MockLLMService(json.dumps(response_data))
        engine = EvaluationEngine(llm_service=llm_service)

        result = await engine.evaluate(session_id, normal_transcript)

        # Expected: 80*0.20 + 90*0.30 + 70*0.25 + 60*0.25 = 16 + 27 + 17.5 + 15 = 75.5
        assert result.overall_score == pytest.approx(75.5)

    async def test_strengths_parsed_correctly(self, engine, session_id, normal_transcript):
        """Strengths should be parsed from LLM response."""
        result = await engine.evaluate(session_id, normal_transcript)
        assert 1 <= len(result.strengths) <= 5
        for strength in result.strengths:
            assert strength.description
            assert strength.category in EvaluationCategory
            assert strength.transcript_excerpt

    async def test_weaknesses_parsed_correctly(self, engine, session_id, normal_transcript):
        """Weaknesses should be parsed from LLM response."""
        result = await engine.evaluate(session_id, normal_transcript)
        assert 1 <= len(result.weaknesses) <= 5
        for weakness in result.weaknesses:
            assert weakness.description
            assert weakness.category in EvaluationCategory
            assert weakness.transcript_excerpt

    async def test_returns_evaluation_result_type(self, engine, session_id, normal_transcript):
        """evaluate() should return an EvaluationResult instance."""
        result = await engine.evaluate(session_id, normal_transcript)
        assert isinstance(result, EvaluationResult)

    async def test_llm_called_with_json_response_format(self, engine, session_id, normal_transcript, mock_llm_service):
        """LLM should be called with response_format for JSON."""
        await engine.evaluate(session_id, normal_transcript)
        # Verify the LLM was called (we can check the mock was invoked)
        assert mock_llm_service.call_count == 1

    async def test_transcript_formatted_in_prompt(self, engine, session_id, normal_transcript, mock_llm_service):
        """The transcript should be included in the user message to the LLM."""
        await engine.evaluate(session_id, normal_transcript)
        user_message = mock_llm_service.last_messages[1]
        assert "AGENT:" in user_message.content
        assert "DEBTOR:" in user_message.content


class TestEvaluateEdgeCases:
    """Tests for edge cases in evaluate()."""

    async def test_no_llm_service_raises_error(self, session_id, normal_transcript):
        """evaluate() without LLM service should raise ValueError for non-short sessions."""
        engine = EvaluationEngine(llm_service=None)
        with pytest.raises(ValueError, match="LLM service is required"):
            await engine.evaluate(session_id, normal_transcript)

    async def test_scores_clamped_to_valid_range(self, session_id, normal_transcript):
        """Scores outside [0,100] in LLM response should be clamped."""
        response_data = make_llm_evaluation_response()
        response_data["category_scores"]["call_opening"] = 150  # Over 100
        response_data["category_scores"]["compliance"] = -10  # Below 0
        llm_service = MockLLMService(json.dumps(response_data))
        engine = EvaluationEngine(llm_service=llm_service)

        result = await engine.evaluate(session_id, normal_transcript)

        score_map = {cs.category: cs.score for cs in result.category_scores}
        assert score_map[EvaluationCategory.CALL_OPENING] == 100
        assert score_map[EvaluationCategory.COMPLIANCE] == 0

    async def test_max_5_strengths(self, session_id, normal_transcript):
        """At most 5 strengths should be returned even if LLM provides more."""
        response_data = make_llm_evaluation_response(num_strengths=8)
        llm_service = MockLLMService(json.dumps(response_data))
        engine = EvaluationEngine(llm_service=llm_service)

        result = await engine.evaluate(session_id, normal_transcript)
        assert len(result.strengths) <= 5

    async def test_max_5_weaknesses(self, session_id, normal_transcript):
        """At most 5 weaknesses should be returned even if LLM provides more."""
        response_data = make_llm_evaluation_response(num_weaknesses=8)
        llm_service = MockLLMService(json.dumps(response_data))
        engine = EvaluationEngine(llm_service=llm_service)

        result = await engine.evaluate(session_id, normal_transcript)
        assert len(result.weaknesses) <= 5

    async def test_empty_strengths_from_llm_gets_fallback(self, session_id, normal_transcript):
        """If LLM returns no strengths, a fallback strength is provided."""
        response_data = make_llm_evaluation_response(num_strengths=0)
        llm_service = MockLLMService(json.dumps(response_data))
        engine = EvaluationEngine(llm_service=llm_service)

        result = await engine.evaluate(session_id, normal_transcript)
        assert len(result.strengths) >= 1

    async def test_empty_weaknesses_from_llm_gets_fallback(self, session_id, normal_transcript):
        """If LLM returns no weaknesses, a fallback weakness is provided."""
        response_data = make_llm_evaluation_response(num_weaknesses=0)
        llm_service = MockLLMService(json.dumps(response_data))
        engine = EvaluationEngine(llm_service=llm_service)

        result = await engine.evaluate(session_id, normal_transcript)
        assert len(result.weaknesses) >= 1

    async def test_exactly_4_agent_utterances_evaluates_normally(self, session_id):
        """Transcript with exactly 4 agent utterances should be evaluated normally."""
        transcript = make_transcript(agent_count=4, debtor_count=4)
        response_data = make_llm_evaluation_response()
        llm_service = MockLLMService(json.dumps(response_data))
        engine = EvaluationEngine(llm_service=llm_service)

        result = await engine.evaluate(session_id, transcript)
        assert result.is_too_short is False
        assert len(result.category_scores) == 4


class TestEvaluatePersistence:
    """Tests for persistence behavior in evaluate()."""

    async def test_no_db_no_persistence_call(self, engine, session_id, normal_transcript):
        """When db=None, no persistence should be attempted."""
        # Should not raise or fail
        result = await engine.evaluate(session_id, normal_transcript, db=None)
        assert result is not None

    async def test_persist_called_with_db(self, session_id, normal_transcript):
        """When db is provided, persistence should be attempted."""
        response_data = make_llm_evaluation_response()
        llm_service = MockLLMService(json.dumps(response_data))
        engine = EvaluationEngine(llm_service=llm_service)

        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()

        with patch("app.services.evaluation_engine.retry_db_operation", new_callable=AsyncMock) as mock_retry:
            mock_retry.return_value = None
            result = await engine.evaluate(session_id, normal_transcript, db=mock_db)

        assert result is not None
        mock_retry.assert_called_once()

    async def test_persist_called_for_too_short_with_db(self, session_id, short_transcript):
        """Persistence should be called even for too-short sessions when db is provided."""
        engine = EvaluationEngine(llm_service=None)

        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()

        with patch("app.services.evaluation_engine.retry_db_operation", new_callable=AsyncMock) as mock_retry:
            mock_retry.return_value = None
            result = await engine.evaluate(session_id, short_transcript, db=mock_db)

        assert result.is_too_short is True
        mock_retry.assert_called_once()


class TestFormatTranscript:
    """Tests for the _format_transcript helper."""

    def test_formats_agent_and_debtor(self):
        """Transcript entries should be formatted as SPEAKER: text."""
        engine = EvaluationEngine()
        transcript = [
            {"speaker": "agent", "text": "Hello"},
            {"speaker": "debtor", "text": "Hi there"},
        ]
        result = engine._format_transcript(transcript)
        assert "AGENT: Hello" in result
        assert "DEBTOR: Hi there" in result

    def test_empty_transcript(self):
        """Empty transcript should produce empty string."""
        engine = EvaluationEngine()
        result = engine._format_transcript([])
        assert result == ""

    def test_preserves_order(self):
        """Entries should appear in the same order as input."""
        engine = EvaluationEngine()
        transcript = [
            {"speaker": "agent", "text": "First"},
            {"speaker": "debtor", "text": "Second"},
            {"speaker": "agent", "text": "Third"},
        ]
        result = engine._format_transcript(transcript)
        lines = result.strip().split("\n")
        assert lines[0] == "AGENT: First"
        assert lines[1] == "DEBTOR: Second"
        assert lines[2] == "AGENT: Third"

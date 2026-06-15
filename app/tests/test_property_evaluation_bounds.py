"""Property-based tests for evaluation output cardinality bounds.

Feature: collection-agent-trainer, Property 9: Evaluation output cardinality bounds

**Validates: Requirements 5.4, 5.5**

Property 9: For any completed evaluation where is_too_short is false, the strengths
list SHALL contain between 1 and 5 items (inclusive), and the weaknesses list SHALL
contain between 1 and 5 items (inclusive), each referencing a valid evaluation category
and containing a non-empty transcript excerpt.
"""

import json
import uuid

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.schemas import EvaluationCategory
from app.services.evaluation_engine import EvaluationEngine
from app.services.llm_service import LLMResponse


# --- Constants ---

VALID_CATEGORIES = [cat.value for cat in EvaluationCategory]


# --- Mock LLM Service ---


class MockLLMService:
    """Mock LLM service that returns a pre-configured evaluation response."""

    def __init__(self, response_content: str):
        self._response_content = response_content

    async def chat_completion(self, messages, *, temperature=None, max_tokens=None, response_format=None):
        return LLMResponse(
            content=self._response_content,
            model="test-model",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )


# --- Strategies ---


@st.composite
def llm_evaluation_response(draw):
    """Generate random valid LLM evaluation responses with various numbers of strengths/weaknesses (0-10).

    The LLM might return any number of items; the evaluation engine must enforce
    the 1-5 bounds.
    """
    # Generate random scores for all categories
    call_opening = draw(st.integers(min_value=0, max_value=100))
    compliance = draw(st.integers(min_value=0, max_value=100))
    empathy_communication = draw(st.integers(min_value=0, max_value=100))
    negotiation_resolution = draw(st.integers(min_value=0, max_value=100))

    # Generate random number of strengths (0-10)
    num_strengths = draw(st.integers(min_value=0, max_value=10))
    strengths = []
    for i in range(num_strengths):
        category = draw(st.sampled_from(VALID_CATEGORIES))
        description = draw(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
                min_size=1,
                max_size=100,
            ).filter(lambda s: s.strip() != "")
        )
        excerpt = draw(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
                min_size=1,
                max_size=100,
            ).filter(lambda s: s.strip() != "")
        )
        strengths.append({
            "description": description,
            "category": category,
            "transcript_excerpt": excerpt,
        })

    # Generate random number of weaknesses (0-10)
    num_weaknesses = draw(st.integers(min_value=0, max_value=10))
    weaknesses = []
    for i in range(num_weaknesses):
        category = draw(st.sampled_from(VALID_CATEGORIES))
        description = draw(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
                min_size=1,
                max_size=100,
            ).filter(lambda s: s.strip() != "")
        )
        excerpt = draw(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
                min_size=1,
                max_size=100,
            ).filter(lambda s: s.strip() != "")
        )
        weaknesses.append({
            "description": description,
            "category": category,
            "transcript_excerpt": excerpt,
        })

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


def make_normal_transcript() -> list[dict]:
    """Create a normal-length transcript with enough agent utterances (>= 4)."""
    return [
        {"speaker": "agent", "text": "Hello, this is a collection call."},
        {"speaker": "debtor", "text": "Who is this?"},
        {"speaker": "agent", "text": "I am calling about your outstanding balance."},
        {"speaker": "debtor", "text": "I don't have any money right now."},
        {"speaker": "agent", "text": "I understand. Let me see what options we have."},
        {"speaker": "debtor", "text": "Okay, what can you do?"},
        {"speaker": "agent", "text": "We can set up a payment plan."},
        {"speaker": "debtor", "text": "That sounds reasonable."},
        {"speaker": "agent", "text": "Great, let me get the details set up."},
        {"speaker": "debtor", "text": "Thank you."},
    ]


# --- Property Tests ---


class TestEvaluationOutputCardinalityBounds:
    """Property 9: Evaluation output cardinality bounds.

    Feature: collection-agent-trainer, Property 9: Evaluation output cardinality bounds
    """

    @given(llm_response_data=llm_evaluation_response())
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_strengths_count_between_1_and_5(self, llm_response_data: dict):
        """**Validates: Requirements 5.4, 5.5**

        For any completed evaluation where is_too_short is False, the strengths list
        SHALL contain between 1 and 5 items (inclusive).
        """
        mock_llm = MockLLMService(json.dumps(llm_response_data))
        engine = EvaluationEngine(llm_service=mock_llm)
        session_id = uuid.uuid4()
        transcript = make_normal_transcript()

        result = await engine.evaluate(session_id, transcript)

        assert result.is_too_short is False
        assert 1 <= len(result.strengths) <= 5, (
            f"Expected 1-5 strengths, got {len(result.strengths)} "
            f"(LLM returned {len(llm_response_data['strengths'])} strengths)"
        )

    @given(llm_response_data=llm_evaluation_response())
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_weaknesses_count_between_1_and_5(self, llm_response_data: dict):
        """**Validates: Requirements 5.4, 5.5**

        For any completed evaluation where is_too_short is False, the weaknesses list
        SHALL contain between 1 and 5 items (inclusive).
        """
        mock_llm = MockLLMService(json.dumps(llm_response_data))
        engine = EvaluationEngine(llm_service=mock_llm)
        session_id = uuid.uuid4()
        transcript = make_normal_transcript()

        result = await engine.evaluate(session_id, transcript)

        assert result.is_too_short is False
        assert 1 <= len(result.weaknesses) <= 5, (
            f"Expected 1-5 weaknesses, got {len(result.weaknesses)} "
            f"(LLM returned {len(llm_response_data['weaknesses'])} weaknesses)"
        )

    @given(llm_response_data=llm_evaluation_response())
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_each_strength_has_valid_category(self, llm_response_data: dict):
        """**Validates: Requirements 5.4, 5.5**

        Each strength SHALL reference a valid evaluation category.
        """
        mock_llm = MockLLMService(json.dumps(llm_response_data))
        engine = EvaluationEngine(llm_service=mock_llm)
        session_id = uuid.uuid4()
        transcript = make_normal_transcript()

        result = await engine.evaluate(session_id, transcript)

        assert result.is_too_short is False
        for strength in result.strengths:
            assert strength.category in EvaluationCategory, (
                f"Strength category '{strength.category}' is not a valid EvaluationCategory"
            )

    @given(llm_response_data=llm_evaluation_response())
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_each_weakness_has_valid_category(self, llm_response_data: dict):
        """**Validates: Requirements 5.4, 5.5**

        Each weakness SHALL reference a valid evaluation category.
        """
        mock_llm = MockLLMService(json.dumps(llm_response_data))
        engine = EvaluationEngine(llm_service=mock_llm)
        session_id = uuid.uuid4()
        transcript = make_normal_transcript()

        result = await engine.evaluate(session_id, transcript)

        assert result.is_too_short is False
        for weakness in result.weaknesses:
            assert weakness.category in EvaluationCategory, (
                f"Weakness category '{weakness.category}' is not a valid EvaluationCategory"
            )

    @given(llm_response_data=llm_evaluation_response())
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_each_strength_has_non_empty_description_and_excerpt(self, llm_response_data: dict):
        """**Validates: Requirements 5.4, 5.5**

        Each strength SHALL contain a non-empty description and non-empty transcript excerpt.
        """
        mock_llm = MockLLMService(json.dumps(llm_response_data))
        engine = EvaluationEngine(llm_service=mock_llm)
        session_id = uuid.uuid4()
        transcript = make_normal_transcript()

        result = await engine.evaluate(session_id, transcript)

        assert result.is_too_short is False
        for strength in result.strengths:
            assert len(strength.description) > 0, "Strength description must be non-empty"
            assert len(strength.transcript_excerpt) > 0, "Strength transcript_excerpt must be non-empty"

    @given(llm_response_data=llm_evaluation_response())
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_each_weakness_has_non_empty_description_and_excerpt(self, llm_response_data: dict):
        """**Validates: Requirements 5.4, 5.5**

        Each weakness SHALL contain a non-empty description and non-empty transcript excerpt.
        """
        mock_llm = MockLLMService(json.dumps(llm_response_data))
        engine = EvaluationEngine(llm_service=mock_llm)
        session_id = uuid.uuid4()
        transcript = make_normal_transcript()

        result = await engine.evaluate(session_id, transcript)

        assert result.is_too_short is False
        for weakness in result.weaknesses:
            assert len(weakness.description) > 0, "Weakness description must be non-empty"
            assert len(weakness.transcript_excerpt) > 0, "Weakness transcript_excerpt must be non-empty"

"""Property-based tests for coaching mistake structural completeness.

Feature: collection-agent-trainer, Property 11: Coaching mistake structural completeness

**Validates: Requirements 6.2, 6.3**

Property 11: For any identified mistake in a coaching report, it SHALL contain a
non-empty explanation describing why the behavior was ineffective, a non-empty
recommended_alternative response, a reference to the transcript_position, and an
associated category.
"""

import json
import uuid

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.schemas import EvaluationCategory, EvaluationResult, CompetencyScore, StrengthItem, WeaknessItem
from app.services.coaching_engine import CoachingEngine
from app.services.llm_service import LLMResponse


# --- Constants ---

VALID_CATEGORIES = [cat.value for cat in EvaluationCategory]


# --- Mock LLM Service ---


class MockLLMService:
    """Mock LLM service that returns a pre-configured coaching response."""

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
def llm_coaching_response(draw):
    """Generate random valid LLM coaching responses with 1-5 mistakes.

    Each mistake has a random valid category, position, excerpt, explanation,
    and recommended alternative.
    """
    num_mistakes = draw(st.integers(min_value=1, max_value=5))
    mistakes = []

    for _ in range(num_mistakes):
        category = draw(st.sampled_from(VALID_CATEGORIES))
        position = draw(st.integers(min_value=0, max_value=20))
        excerpt = draw(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
                min_size=1,
                max_size=100,
            ).filter(lambda s: s.strip() != "")
        )
        explanation = draw(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
                min_size=1,
                max_size=200,
            ).filter(lambda s: s.strip() != "")
        )
        alternative = draw(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
                min_size=1,
                max_size=200,
            ).filter(lambda s: s.strip() != "")
        )

        mistakes.append({
            "transcript_position": position,
            "transcript_excerpt": excerpt,
            "category": category,
            "explanation": explanation,
            "recommended_alternative": alternative,
        })

    return {"mistakes": mistakes}


def make_sample_transcript() -> list[dict]:
    """Create a sample transcript with enough entries for coaching analysis."""
    return [
        {"speaker": "agent", "text": "Hello, this is a collection call."},
        {"speaker": "debtor", "text": "Who is this?"},
        {"speaker": "agent", "text": "I am calling about your outstanding balance."},
        {"speaker": "debtor", "text": "I don't have any money right now."},
        {"speaker": "agent", "text": "You need to pay immediately."},
        {"speaker": "debtor", "text": "That's not possible."},
        {"speaker": "agent", "text": "Let me explain your options."},
        {"speaker": "debtor", "text": "Okay, go ahead."},
        {"speaker": "agent", "text": "We can set up a payment plan."},
        {"speaker": "debtor", "text": "Thank you."},
    ]


def make_sample_evaluation() -> EvaluationResult:
    """Create a sample evaluation result for coaching."""
    session_id = uuid.uuid4()
    return EvaluationResult(
        session_id=session_id,
        category_scores=[
            CompetencyScore(category=EvaluationCategory.CALL_OPENING, score=40),
            CompetencyScore(category=EvaluationCategory.COMPLIANCE, score=50),
            CompetencyScore(category=EvaluationCategory.EMPATHY_COMMUNICATION, score=35),
            CompetencyScore(category=EvaluationCategory.NEGOTIATION_RESOLUTION, score=55),
        ],
        overall_score=45.0,
        strengths=[
            StrengthItem(
                description="Agent offered a payment plan",
                category=EvaluationCategory.NEGOTIATION_RESOLUTION,
                transcript_excerpt="We can set up a payment plan.",
            ),
        ],
        weaknesses=[
            WeaknessItem(
                description="Aggressive demand for immediate payment",
                category=EvaluationCategory.EMPATHY_COMMUNICATION,
                transcript_excerpt="You need to pay immediately.",
            ),
            WeaknessItem(
                description="Lack of proper call opening",
                category=EvaluationCategory.CALL_OPENING,
                transcript_excerpt="Hello, this is a collection call.",
            ),
        ],
        is_too_short=False,
    )


# --- Property Tests ---


class TestCoachingMistakeStructuralCompleteness:
    """Property 11: Coaching mistake structural completeness.

    Feature: collection-agent-trainer, Property 11: Coaching mistake structural completeness
    """

    @given(llm_response_data=llm_coaching_response())
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_each_mistake_has_non_empty_explanation(self, llm_response_data: dict):
        """**Validates: Requirements 6.2, 6.3**

        For any identified mistake in a coaching report, it SHALL contain a non-empty
        explanation describing why the behavior was ineffective.
        """
        mock_llm = MockLLMService(json.dumps(llm_response_data))
        engine = CoachingEngine(llm_service=mock_llm)
        session_id = uuid.uuid4()
        transcript = make_sample_transcript()
        evaluation = make_sample_evaluation()

        report = await engine.generate_report(session_id, transcript, evaluation)

        for category, items in report.mistakes_by_category.items():
            for item in items:
                assert item.explanation is not None and len(item.explanation) > 0, (
                    f"Mistake at position {item.transcript_position} has empty explanation"
                )

    @given(llm_response_data=llm_coaching_response())
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_each_mistake_has_non_empty_recommended_alternative(self, llm_response_data: dict):
        """**Validates: Requirements 6.2, 6.3**

        For any identified mistake in a coaching report, it SHALL contain a non-empty
        recommended_alternative response.
        """
        mock_llm = MockLLMService(json.dumps(llm_response_data))
        engine = CoachingEngine(llm_service=mock_llm)
        session_id = uuid.uuid4()
        transcript = make_sample_transcript()
        evaluation = make_sample_evaluation()

        report = await engine.generate_report(session_id, transcript, evaluation)

        for category, items in report.mistakes_by_category.items():
            for item in items:
                assert item.recommended_alternative is not None and len(item.recommended_alternative) > 0, (
                    f"Mistake at position {item.transcript_position} has empty recommended_alternative"
                )

    @given(llm_response_data=llm_coaching_response())
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_each_mistake_has_valid_transcript_position(self, llm_response_data: dict):
        """**Validates: Requirements 6.2, 6.3**

        For any identified mistake in a coaching report, it SHALL contain a reference
        to the transcript_position (>= 0).
        """
        mock_llm = MockLLMService(json.dumps(llm_response_data))
        engine = CoachingEngine(llm_service=mock_llm)
        session_id = uuid.uuid4()
        transcript = make_sample_transcript()
        evaluation = make_sample_evaluation()

        report = await engine.generate_report(session_id, transcript, evaluation)

        for category, items in report.mistakes_by_category.items():
            for item in items:
                assert item.transcript_position >= 0, (
                    f"Mistake has invalid transcript_position: {item.transcript_position}"
                )

    @given(llm_response_data=llm_coaching_response())
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_each_mistake_has_valid_category(self, llm_response_data: dict):
        """**Validates: Requirements 6.2, 6.3**

        For any identified mistake in a coaching report, it SHALL have an associated
        valid evaluation category.
        """
        mock_llm = MockLLMService(json.dumps(llm_response_data))
        engine = CoachingEngine(llm_service=mock_llm)
        session_id = uuid.uuid4()
        transcript = make_sample_transcript()
        evaluation = make_sample_evaluation()

        report = await engine.generate_report(session_id, transcript, evaluation)

        for category, items in report.mistakes_by_category.items():
            for item in items:
                assert item.category in EvaluationCategory, (
                    f"Mistake has invalid category: {item.category}"
                )

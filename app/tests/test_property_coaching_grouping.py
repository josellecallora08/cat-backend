"""Property-based tests for coaching report category grouping.

Feature: collection-agent-trainer, Property 12: Coaching report category grouping

**Validates: Requirements 6.4**

Property 12: For any coaching report containing one or more mistakes, all mistakes
SHALL be grouped under valid EvaluationCategory keys, and every mistake's category
field SHALL match the key it is grouped under.
"""

import json
import uuid

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.schemas import (
    CompetencyScore,
    EvaluationCategory,
    EvaluationResult,
    StrengthItem,
    WeaknessItem,
)
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
def llm_coaching_response_with_mistakes(draw):
    """Generate random valid LLM coaching responses with 1-10 mistakes from various categories.

    Each mistake has a valid category, non-empty fields, and a transcript position.
    """
    num_mistakes = draw(st.integers(min_value=1, max_value=10))
    mistakes = []
    for _ in range(num_mistakes):
        category = draw(st.sampled_from(VALID_CATEGORIES))
        position = draw(st.integers(min_value=0, max_value=9))
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
                max_size=100,
            ).filter(lambda s: s.strip() != "")
        )
        alternative = draw(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
                min_size=1,
                max_size=100,
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
    """Create a sample transcript for coaching report generation."""
    return [
        {"speaker": "agent", "text": "Hey, you owe money. Pay up now."},
        {"speaker": "debtor", "text": "Who is this?"},
        {"speaker": "agent", "text": "This is collections. Pay $5000 today."},
        {"speaker": "debtor", "text": "I lost my job, I can't pay."},
        {"speaker": "agent", "text": "That's not my problem."},
        {"speaker": "debtor", "text": "Please, can we work something out?"},
        {"speaker": "agent", "text": "Fine, what can you pay?"},
        {"speaker": "debtor", "text": "Maybe $200 a month?"},
        {"speaker": "agent", "text": "That's too low. We need at least $500."},
        {"speaker": "debtor", "text": "I'll try my best."},
    ]


def make_sample_evaluation() -> EvaluationResult:
    """Create a sample evaluation result for coaching engine input."""
    session_id = uuid.uuid4()
    return EvaluationResult(
        session_id=session_id,
        category_scores=[
            CompetencyScore(category=EvaluationCategory.CALL_OPENING, score=30),
            CompetencyScore(category=EvaluationCategory.COMPLIANCE, score=40),
            CompetencyScore(category=EvaluationCategory.EMPATHY_COMMUNICATION, score=25),
            CompetencyScore(category=EvaluationCategory.NEGOTIATION_RESOLUTION, score=50),
        ],
        overall_score=37.25,
        strengths=[
            StrengthItem(
                description="Agent eventually offered to negotiate",
                category=EvaluationCategory.NEGOTIATION_RESOLUTION,
                transcript_excerpt="Fine, what can you pay?",
            ),
        ],
        weaknesses=[
            WeaknessItem(
                description="Unprofessional call opening",
                category=EvaluationCategory.CALL_OPENING,
                transcript_excerpt="Hey, you owe money. Pay up now.",
            ),
            WeaknessItem(
                description="Failed to show empathy",
                category=EvaluationCategory.EMPATHY_COMMUNICATION,
                transcript_excerpt="That's not my problem.",
            ),
            WeaknessItem(
                description="Threatening language",
                category=EvaluationCategory.COMPLIANCE,
                transcript_excerpt="Pay up now.",
            ),
        ],
        is_too_short=False,
    )


# --- Property Tests ---


class TestCoachingReportCategoryGrouping:
    """Property 12: Coaching report category grouping.

    Feature: collection-agent-trainer, Property 12: Coaching report category grouping
    """

    @given(llm_response_data=llm_coaching_response_with_mistakes())
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_all_keys_are_valid_evaluation_categories(self, llm_response_data: dict):
        """**Validates: Requirements 6.4**

        For any coaching report with mistakes, all keys in mistakes_by_category
        SHALL be valid EvaluationCategory values.
        """
        mock_llm = MockLLMService(json.dumps(llm_response_data))
        engine = CoachingEngine(llm_service=mock_llm)
        session_id = uuid.uuid4()
        transcript = make_sample_transcript()
        evaluation = make_sample_evaluation()

        report = await engine.generate_report(session_id, transcript, evaluation)

        for category_key in report.mistakes_by_category.keys():
            assert category_key in EvaluationCategory, (
                f"Key '{category_key}' is not a valid EvaluationCategory. "
                f"Valid categories: {[c.value for c in EvaluationCategory]}"
            )

    @given(llm_response_data=llm_coaching_response_with_mistakes())
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_mistake_category_matches_grouping_key(self, llm_response_data: dict):
        """**Validates: Requirements 6.4**

        For each category key in mistakes_by_category, every mistake item's
        category field SHALL match the key it is grouped under.
        """
        mock_llm = MockLLMService(json.dumps(llm_response_data))
        engine = CoachingEngine(llm_service=mock_llm)
        session_id = uuid.uuid4()
        transcript = make_sample_transcript()
        evaluation = make_sample_evaluation()

        report = await engine.generate_report(session_id, transcript, evaluation)

        for category_key, mistakes in report.mistakes_by_category.items():
            for mistake in mistakes:
                assert mistake.category == category_key, (
                    f"Mistake grouped under '{category_key.value}' has category "
                    f"'{mistake.category.value}' — these must match"
                )

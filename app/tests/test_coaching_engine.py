"""Unit tests for the CoachingEngine service.

Tests coaching report generation with mocked LLM responses,
mistake parsing, category grouping, and no-mistakes handling.

Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5
"""

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.schemas import (
    CoachingReportSchema,
    CompetencyScore,
    EvaluationCategory,
    EvaluationResult,
    MistakeItem,
    StrengthItem,
    WeaknessItem,
)
from app.services.coaching_engine import CoachingEngine
from app.services.llm_service import LLMResponse


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    service = AsyncMock()
    return service


@pytest.fixture
def engine(mock_llm_service):
    """Create a CoachingEngine instance with mocked LLM."""
    return CoachingEngine(llm_service=mock_llm_service)


@pytest.fixture
def sample_transcript():
    """Sample transcript for testing."""
    return [
        {"speaker": "agent", "text": "Hey, you owe money. Pay up now."},
        {"speaker": "debtor", "text": "Who is this? What are you talking about?"},
        {"speaker": "agent", "text": "This is collections. You need to pay $5000 today."},
        {"speaker": "debtor", "text": "I can't pay that right now, I lost my job."},
        {"speaker": "agent", "text": "That's not my problem. Pay or we'll take action."},
        {"speaker": "debtor", "text": "Please, can we work something out?"},
        {"speaker": "agent", "text": "Fine, what can you pay?"},
        {"speaker": "debtor", "text": "Maybe $200 a month?"},
    ]


@pytest.fixture
def sample_evaluation():
    """Sample evaluation result with weaknesses."""
    session_id = uuid4()
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
                description="Unprofessional call opening without proper identification",
                category=EvaluationCategory.CALL_OPENING,
                transcript_excerpt="Hey, you owe money. Pay up now.",
            ),
            WeaknessItem(
                description="Failed to show empathy for debtor's situation",
                category=EvaluationCategory.EMPATHY_COMMUNICATION,
                transcript_excerpt="That's not my problem. Pay or we'll take action.",
            ),
            WeaknessItem(
                description="Threatening language violates compliance standards",
                category=EvaluationCategory.COMPLIANCE,
                transcript_excerpt="Pay or we'll take action.",
            ),
        ],
        is_too_short=False,
    )


def _make_llm_response(mistakes: list[dict]) -> LLMResponse:
    """Helper to create an LLM response with given mistakes."""
    return LLMResponse(
        content=json.dumps({"mistakes": mistakes}),
        model="test-model",
    )


class TestGenerateReport:
    """Tests for CoachingEngine.generate_report."""

    @pytest.mark.asyncio
    async def test_generates_report_with_mistakes(
        self, engine: CoachingEngine, mock_llm_service, sample_transcript, sample_evaluation
    ):
        """Report should contain parsed mistakes from LLM response."""
        mistakes_data = [
            {
                "transcript_position": 0,
                "transcript_excerpt": "Hey, you owe money. Pay up now.",
                "category": "call_opening",
                "explanation": "Agent failed to identify themselves professionally.",
                "recommended_alternative": "Good morning, my name is [Agent] calling from [Company]. Am I speaking with [Debtor Name]?",
            },
            {
                "transcript_position": 4,
                "transcript_excerpt": "That's not my problem. Pay or we'll take action.",
                "category": "empathy_communication",
                "explanation": "Dismissing the debtor's hardship reduces cooperation.",
                "recommended_alternative": "I understand you're going through a difficult time. Let's see what options we can find together.",
            },
        ]
        mock_llm_service.chat_completion.return_value = _make_llm_response(mistakes_data)

        session_id = sample_evaluation.session_id
        report = await engine.generate_report(
            session_id, sample_transcript, sample_evaluation
        )

        assert isinstance(report, CoachingReportSchema)
        assert report.session_id == session_id
        assert report.total_mistakes == 2
        assert report.no_mistakes is False

    @pytest.mark.asyncio
    async def test_mistakes_grouped_by_category(
        self, engine: CoachingEngine, mock_llm_service, sample_transcript, sample_evaluation
    ):
        """Mistakes should be grouped under their respective category keys."""
        mistakes_data = [
            {
                "transcript_position": 0,
                "transcript_excerpt": "Hey, you owe money.",
                "category": "call_opening",
                "explanation": "No proper greeting.",
                "recommended_alternative": "Good morning, this is [Agent] from [Company].",
            },
            {
                "transcript_position": 2,
                "transcript_excerpt": "You need to pay $5000 today.",
                "category": "compliance",
                "explanation": "Demanding immediate full payment without options.",
                "recommended_alternative": "I'd like to discuss your account balance and explore options.",
            },
            {
                "transcript_position": 4,
                "transcript_excerpt": "That's not my problem.",
                "category": "empathy_communication",
                "explanation": "Lack of empathy.",
                "recommended_alternative": "I understand this is challenging.",
            },
        ]
        mock_llm_service.chat_completion.return_value = _make_llm_response(mistakes_data)

        report = await engine.generate_report(
            sample_evaluation.session_id, sample_transcript, sample_evaluation
        )

        assert EvaluationCategory.CALL_OPENING in report.mistakes_by_category
        assert EvaluationCategory.COMPLIANCE in report.mistakes_by_category
        assert EvaluationCategory.EMPATHY_COMMUNICATION in report.mistakes_by_category
        assert len(report.mistakes_by_category[EvaluationCategory.CALL_OPENING]) == 1
        assert len(report.mistakes_by_category[EvaluationCategory.COMPLIANCE]) == 1
        assert len(report.mistakes_by_category[EvaluationCategory.EMPATHY_COMMUNICATION]) == 1

    @pytest.mark.asyncio
    async def test_mistake_category_matches_grouping_key(
        self, engine: CoachingEngine, mock_llm_service, sample_transcript, sample_evaluation
    ):
        """Each mistake's category field should match the key it is grouped under."""
        mistakes_data = [
            {
                "transcript_position": 0,
                "transcript_excerpt": "Hey, you owe money.",
                "category": "call_opening",
                "explanation": "Bad opening.",
                "recommended_alternative": "Professional greeting.",
            },
            {
                "transcript_position": 4,
                "transcript_excerpt": "That's not my problem.",
                "category": "empathy_communication",
                "explanation": "No empathy.",
                "recommended_alternative": "I understand.",
            },
        ]
        mock_llm_service.chat_completion.return_value = _make_llm_response(mistakes_data)

        report = await engine.generate_report(
            sample_evaluation.session_id, sample_transcript, sample_evaluation
        )

        for category, items in report.mistakes_by_category.items():
            for item in items:
                assert item.category == category

    @pytest.mark.asyncio
    async def test_no_mistakes_case(
        self, engine: CoachingEngine, mock_llm_service, sample_transcript, sample_evaluation
    ):
        """When LLM returns no mistakes, report should indicate no_mistakes=True."""
        mock_llm_service.chat_completion.return_value = _make_llm_response([])

        report = await engine.generate_report(
            sample_evaluation.session_id, sample_transcript, sample_evaluation
        )

        assert report.no_mistakes is True
        assert report.total_mistakes == 0
        assert report.mistakes_by_category == {}

    @pytest.mark.asyncio
    async def test_mistake_item_has_required_fields(
        self, engine: CoachingEngine, mock_llm_service, sample_transcript, sample_evaluation
    ):
        """Each mistake should have explanation, recommended_alternative, position, and category."""
        mistakes_data = [
            {
                "transcript_position": 2,
                "transcript_excerpt": "You need to pay $5000 today.",
                "category": "compliance",
                "explanation": "Agent demanded immediate full payment.",
                "recommended_alternative": "Let's discuss your options for resolving this balance.",
            },
        ]
        mock_llm_service.chat_completion.return_value = _make_llm_response(mistakes_data)

        report = await engine.generate_report(
            sample_evaluation.session_id, sample_transcript, sample_evaluation
        )

        items = report.mistakes_by_category[EvaluationCategory.COMPLIANCE]
        assert len(items) == 1
        item = items[0]
        assert item.transcript_position == 2
        assert item.transcript_excerpt == "You need to pay $5000 today."
        assert item.category == EvaluationCategory.COMPLIANCE
        assert item.explanation == "Agent demanded immediate full payment."
        assert item.recommended_alternative == "Let's discuss your options for resolving this balance."

    @pytest.mark.asyncio
    async def test_malformed_mistake_items_are_skipped(
        self, engine: CoachingEngine, mock_llm_service, sample_transcript, sample_evaluation
    ):
        """Malformed items from LLM should be skipped without crashing."""
        mistakes_data = [
            {
                "transcript_position": 0,
                "transcript_excerpt": "Hey, you owe money.",
                "category": "call_opening",
                "explanation": "Bad opening.",
                "recommended_alternative": "Good morning.",
            },
            {
                # Missing required fields
                "transcript_position": 1,
                "category": "invalid_category",
            },
            {
                # Missing transcript_excerpt
                "transcript_position": 2,
                "category": "compliance",
                "explanation": "test",
                "recommended_alternative": "test",
            },
        ]
        mock_llm_service.chat_completion.return_value = _make_llm_response(mistakes_data)

        report = await engine.generate_report(
            sample_evaluation.session_id, sample_transcript, sample_evaluation
        )

        # Only the valid first item should be present
        assert report.total_mistakes == 1
        assert EvaluationCategory.CALL_OPENING in report.mistakes_by_category

    @pytest.mark.asyncio
    async def test_llm_called_with_transcript_and_evaluation(
        self, engine: CoachingEngine, mock_llm_service, sample_transcript, sample_evaluation
    ):
        """LLM should be called with a prompt containing transcript and evaluation info."""
        mock_llm_service.chat_completion.return_value = _make_llm_response([])

        await engine.generate_report(
            sample_evaluation.session_id, sample_transcript, sample_evaluation
        )

        mock_llm_service.chat_completion.assert_called_once()
        call_args = mock_llm_service.chat_completion.call_args
        messages = call_args[0][0]

        # Verify system prompt is set
        assert messages[0].role == "system"
        assert "coach" in messages[0].content.lower()

        # Verify user prompt contains transcript content
        user_content = messages[1].content
        assert "Hey, you owe money" in user_content
        assert "Unprofessional call opening" in user_content

    @pytest.mark.asyncio
    async def test_llm_called_with_json_response_format(
        self, engine: CoachingEngine, mock_llm_service, sample_transcript, sample_evaluation
    ):
        """LLM should be called requesting JSON response format."""
        mock_llm_service.chat_completion.return_value = _make_llm_response([])

        await engine.generate_report(
            sample_evaluation.session_id, sample_transcript, sample_evaluation
        )

        call_kwargs = mock_llm_service.chat_completion.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_persist_called_when_db_provided(
        self, engine: CoachingEngine, mock_llm_service, sample_transcript, sample_evaluation
    ):
        """When db is provided, the report should be persisted."""
        mock_llm_service.chat_completion.return_value = _make_llm_response([])
        mock_db = MagicMock()
        mock_db.commit = AsyncMock()

        report = await engine.generate_report(
            sample_evaluation.session_id, sample_transcript, sample_evaluation, db=mock_db
        )

        # Verify db.add and db.commit were called
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_persist_when_db_not_provided(
        self, engine: CoachingEngine, mock_llm_service, sample_transcript, sample_evaluation
    ):
        """When db is None, no persistence should be attempted."""
        mock_llm_service.chat_completion.return_value = _make_llm_response([])

        # Should not raise any error
        report = await engine.generate_report(
            sample_evaluation.session_id, sample_transcript, sample_evaluation, db=None
        )

        assert isinstance(report, CoachingReportSchema)

    @pytest.mark.asyncio
    async def test_multiple_mistakes_same_category(
        self, engine: CoachingEngine, mock_llm_service, sample_transcript, sample_evaluation
    ):
        """Multiple mistakes in the same category should all be grouped together."""
        mistakes_data = [
            {
                "transcript_position": 0,
                "transcript_excerpt": "Hey, you owe money.",
                "category": "compliance",
                "explanation": "No Mini-Miranda disclosure.",
                "recommended_alternative": "This is an attempt to collect a debt.",
            },
            {
                "transcript_position": 4,
                "transcript_excerpt": "Pay or we'll take action.",
                "category": "compliance",
                "explanation": "Threatening language.",
                "recommended_alternative": "There are consequences, let me explain your options.",
            },
        ]
        mock_llm_service.chat_completion.return_value = _make_llm_response(mistakes_data)

        report = await engine.generate_report(
            sample_evaluation.session_id, sample_transcript, sample_evaluation
        )

        assert report.total_mistakes == 2
        assert len(report.mistakes_by_category[EvaluationCategory.COMPLIANCE]) == 2

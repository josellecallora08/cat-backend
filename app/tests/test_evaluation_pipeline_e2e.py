"""End-to-end tests for the EvaluationPipeline orchestrator.

Tests the full post-session flow with mocked LLM:
- Transcript retrieval → evaluation → coaching → learning plan
- All artifacts persisted and associated to session_id
- Pipeline completes within 30s requirement

Validates: Requirements 5.1, 6.1, 7.8, 8.2, 8.3
"""

import json
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas import (
    CoachingReportSchema,
    EvaluationCategory,
    EvaluationResult,
    LearningPlanSchema,
)
from app.services.evaluation_pipeline import EvaluationPipeline, PipelineResult
from app.services.llm_service import LLMResponse


# --- Test Helpers ---


def make_evaluation_llm_response(
    call_opening: int = 75,
    compliance: int = 80,
    empathy_communication: int = 60,
    negotiation_resolution: int = 65,
) -> str:
    """Create a mock LLM evaluation response JSON string."""
    return json.dumps({
        "category_scores": {
            "call_opening": call_opening,
            "compliance": compliance,
            "empathy_communication": empathy_communication,
            "negotiation_resolution": negotiation_resolution,
        },
        "strengths": [
            {
                "description": "Good call greeting",
                "category": "call_opening",
                "transcript_excerpt": "Hello, this is agent speaking",
            },
            {
                "description": "Proper disclosure",
                "category": "compliance",
                "transcript_excerpt": "This call may be recorded",
            },
        ],
        "weaknesses": [
            {
                "description": "Lacked empathy when debtor expressed hardship",
                "category": "empathy_communication",
                "transcript_excerpt": "You need to pay now",
            },
            {
                "description": "Did not offer payment plan options",
                "category": "negotiation_resolution",
                "transcript_excerpt": "The full amount is due",
            },
        ],
    })


def make_coaching_llm_response() -> str:
    """Create a mock LLM coaching response JSON string."""
    return json.dumps({
        "mistakes": [
            {
                "transcript_position": 3,
                "transcript_excerpt": "You need to pay now",
                "category": "empathy_communication",
                "explanation": "This response lacked acknowledgment of the debtor's financial situation",
                "recommended_alternative": "I understand this is difficult. Let's explore options that work for your situation.",
            },
            {
                "transcript_position": 5,
                "transcript_excerpt": "The full amount is due",
                "category": "negotiation_resolution",
                "explanation": "Failed to offer flexible payment arrangements",
                "recommended_alternative": "We have several payment plan options available. Would you like to discuss what works best for you?",
            },
        ]
    })


class MockLLMService:
    """Mock LLM that returns different responses for evaluation vs coaching prompts."""

    def __init__(self):
        self._call_count = 0
        self._evaluation_response = make_evaluation_llm_response()
        self._coaching_response = make_coaching_llm_response()

    async def chat_completion(
        self,
        messages,
        *,
        temperature=None,
        max_tokens=None,
        response_format=None,
    ) -> LLMResponse:
        self._call_count += 1
        # First call is evaluation, second is coaching
        if self._call_count == 1:
            content = self._evaluation_response
        else:
            content = self._coaching_response

        return LLMResponse(
            content=content,
            model="test-model",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )


class MockTranscriptRow:
    """Mock transcript ORM row."""

    def __init__(self, speaker: str, text: str, sequence: int):
        self.speaker = speaker
        self.utterance_text = text
        self.sequence_number = sequence


# --- Fixtures ---


@pytest.fixture
def session_id():
    return uuid.uuid4()


@pytest.fixture
def agent_id():
    return uuid.uuid4()


@pytest.fixture
def mock_llm():
    return MockLLMService()


@pytest.fixture
def pipeline(mock_llm):
    return EvaluationPipeline(llm_service=mock_llm)


@pytest.fixture
def sample_transcript_rows():
    """Simulated transcript rows as returned from DB query."""
    return [
        MockTranscriptRow("agent", "Hello, this is agent speaking from collections", 0),
        MockTranscriptRow("debtor", "Hi, what is this about?", 1),
        MockTranscriptRow("agent", "This call may be recorded. I'm calling about your account", 2),
        MockTranscriptRow("debtor", "I'm going through financial hardship right now", 3),
        MockTranscriptRow("agent", "You need to pay now", 4),
        MockTranscriptRow("debtor", "I can't afford it all at once", 5),
        MockTranscriptRow("agent", "The full amount is due", 6),
        MockTranscriptRow("debtor", "Can we work something out?", 7),
        MockTranscriptRow("agent", "I'll check what options are available", 8),
        MockTranscriptRow("debtor", "Thank you", 9),
    ]


@pytest.fixture
def sample_transcript_dicts():
    """Transcript as list of dicts (what pipeline methods consume)."""
    return [
        {"speaker": "agent", "text": "Hello, this is agent speaking from collections"},
        {"speaker": "debtor", "text": "Hi, what is this about?"},
        {"speaker": "agent", "text": "This call may be recorded. I'm calling about your account"},
        {"speaker": "debtor", "text": "I'm going through financial hardship right now"},
        {"speaker": "agent", "text": "You need to pay now"},
        {"speaker": "debtor", "text": "I can't afford it all at once"},
        {"speaker": "agent", "text": "The full amount is due"},
        {"speaker": "debtor", "text": "Can we work something out?"},
        {"speaker": "agent", "text": "I'll check what options are available"},
        {"speaker": "debtor", "text": "Thank you"},
    ]


# --- Tests ---


class TestGetTranscript:
    """Tests for transcript retrieval step."""

    async def test_retrieves_transcript_as_dicts(
        self, pipeline, session_id, sample_transcript_rows
    ):
        """get_transcript should return list of dicts with speaker and text."""
        mock_db = AsyncMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = sample_transcript_rows
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute = AsyncMock(return_value=mock_result)

        transcript = await pipeline.get_transcript(session_id, mock_db)

        assert len(transcript) == 10
        assert transcript[0] == {"speaker": "agent", "text": "Hello, this is agent speaking from collections"}
        assert transcript[1] == {"speaker": "debtor", "text": "Hi, what is this about?"}

    async def test_empty_transcript_returns_empty_list(self, pipeline, session_id):
        """get_transcript with no entries returns empty list."""
        mock_db = AsyncMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute = AsyncMock(return_value=mock_result)

        transcript = await pipeline.get_transcript(session_id, mock_db)

        assert transcript == []


class TestRunEvaluation:
    """Tests for the evaluation step."""

    async def test_produces_evaluation_result(
        self, pipeline, session_id, sample_transcript_dicts
    ):
        """run_evaluation should produce a valid EvaluationResult."""
        mock_db = AsyncMock()

        with patch(
            "app.services.evaluation_engine.retry_db_operation",
            new_callable=AsyncMock,
        ):
            result = await pipeline.run_evaluation(
                session_id, sample_transcript_dicts, mock_db
            )

        assert isinstance(result, EvaluationResult)
        assert result.session_id == session_id
        assert result.is_too_short is False
        assert len(result.category_scores) == 4

    async def test_evaluation_calculates_weighted_score(
        self, pipeline, session_id, sample_transcript_dicts
    ):
        """Overall score should be the weighted combination of category scores."""
        mock_db = AsyncMock()

        with patch(
            "app.services.evaluation_engine.retry_db_operation",
            new_callable=AsyncMock,
        ):
            result = await pipeline.run_evaluation(
                session_id, sample_transcript_dicts, mock_db
            )

        # Expected: 75*0.20 + 80*0.30 + 60*0.25 + 65*0.25 = 15 + 24 + 15 + 16.25 = 70.25
        assert result.overall_score == pytest.approx(70.25)


class TestRunCoaching:
    """Tests for the coaching step."""

    async def test_produces_coaching_report(
        self, pipeline, session_id, sample_transcript_dicts
    ):
        """run_coaching should produce a valid CoachingReportSchema."""
        mock_db = AsyncMock()

        # First run evaluation to get an EvaluationResult
        with patch(
            "app.services.evaluation_engine.retry_db_operation",
            new_callable=AsyncMock,
        ):
            evaluation = await pipeline.run_evaluation(
                session_id, sample_transcript_dicts, mock_db
            )

        with patch(
            "app.services.coaching_engine.retry_db_operation",
            new_callable=AsyncMock,
        ):
            report = await pipeline.run_coaching(
                session_id, sample_transcript_dicts, evaluation, mock_db
            )

        assert isinstance(report, CoachingReportSchema)
        assert report.session_id == session_id
        assert report.total_mistakes == 2
        assert report.no_mistakes is False


class TestRunLearningPlan:
    """Tests for the learning plan step."""

    async def test_produces_learning_plan(
        self, pipeline, session_id, agent_id, sample_transcript_dicts
    ):
        """run_learning_plan should produce a valid LearningPlanSchema."""
        mock_db = AsyncMock()

        # First get evaluation
        with patch(
            "app.services.evaluation_engine.retry_db_operation",
            new_callable=AsyncMock,
        ):
            evaluation = await pipeline.run_evaluation(
                session_id, sample_transcript_dicts, mock_db
            )

        with patch(
            "app.services.learning_plan_generator.retry_db_operation",
            new_callable=AsyncMock,
        ):
            plan = await pipeline.run_learning_plan(
                session_id, agent_id, evaluation, mock_db
            )

        assert isinstance(plan, LearningPlanSchema)
        assert plan.session_id == session_id
        # empathy_communication=60 and negotiation_resolution=65 are below 70
        assert plan.all_passing is False
        assert len(plan.weak_competencies) == 2

    async def test_all_passing_when_scores_above_threshold(
        self, session_id, agent_id
    ):
        """Learning plan should have all_passing=True when all scores >= 70."""
        # Create a mock LLM that returns high scores
        high_score_response = json.dumps({
            "category_scores": {
                "call_opening": 85,
                "compliance": 90,
                "empathy_communication": 75,
                "negotiation_resolution": 80,
            },
            "strengths": [
                {"description": "Good", "category": "call_opening", "transcript_excerpt": "Hi"}
            ],
            "weaknesses": [
                {"description": "Minor", "category": "compliance", "transcript_excerpt": "Hmm"}
            ],
        })

        class HighScoreLLM:
            async def chat_completion(self, messages, **kwargs):
                return LLMResponse(content=high_score_response, model="test", usage=None)

        pipeline = EvaluationPipeline(llm_service=HighScoreLLM())
        mock_db = AsyncMock()

        transcript = [
            {"speaker": "agent", "text": f"Message {i}"} for i in range(5)
        ] + [
            {"speaker": "debtor", "text": f"Reply {i}"} for i in range(5)
        ]

        with patch(
            "app.services.evaluation_engine.retry_db_operation",
            new_callable=AsyncMock,
        ):
            evaluation = await pipeline.run_evaluation(
                session_id, transcript, mock_db
            )

        with patch(
            "app.services.learning_plan_generator.retry_db_operation",
            new_callable=AsyncMock,
        ):
            plan = await pipeline.run_learning_plan(
                session_id, agent_id, evaluation, mock_db
            )

        assert plan.all_passing is True
        assert len(plan.weak_competencies) == 0


class TestFullPipelineRun:
    """Tests for the complete run() method."""

    async def test_full_pipeline_produces_all_artifacts(
        self, pipeline, session_id, agent_id, sample_transcript_rows
    ):
        """run() should produce evaluation, coaching, and learning plan."""
        mock_db = AsyncMock()

        # Mock DB query for transcript retrieval
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = sample_transcript_rows
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch(
            "app.services.evaluation_engine.retry_db_operation",
            new_callable=AsyncMock,
        ), patch(
            "app.services.coaching_engine.retry_db_operation",
            new_callable=AsyncMock,
        ), patch(
            "app.services.learning_plan_generator.retry_db_operation",
            new_callable=AsyncMock,
        ):
            result = await pipeline.run(session_id, agent_id, mock_db)

        assert isinstance(result, PipelineResult)
        assert result.session_id == session_id
        assert len(result.transcript) == 10
        assert isinstance(result.evaluation, EvaluationResult)
        assert isinstance(result.coaching_report, CoachingReportSchema)
        assert isinstance(result.learning_plan, LearningPlanSchema)

    async def test_all_artifacts_reference_same_session_id(
        self, pipeline, session_id, agent_id, sample_transcript_rows
    ):
        """All pipeline artifacts should be associated with the same session_id."""
        mock_db = AsyncMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = sample_transcript_rows
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch(
            "app.services.evaluation_engine.retry_db_operation",
            new_callable=AsyncMock,
        ), patch(
            "app.services.coaching_engine.retry_db_operation",
            new_callable=AsyncMock,
        ), patch(
            "app.services.learning_plan_generator.retry_db_operation",
            new_callable=AsyncMock,
        ):
            result = await pipeline.run(session_id, agent_id, mock_db)

        # Requirement 8.2: all artifacts associated with session_id
        assert result.evaluation.session_id == session_id
        assert result.coaching_report.session_id == session_id
        assert result.learning_plan.session_id == session_id

    async def test_pipeline_completes_within_30_seconds(
        self, pipeline, session_id, agent_id, sample_transcript_rows
    ):
        """Pipeline should complete well within the 30s requirement (Req 5.1)."""
        mock_db = AsyncMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = sample_transcript_rows
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute = AsyncMock(return_value=mock_result)

        start = time.monotonic()

        with patch(
            "app.services.evaluation_engine.retry_db_operation",
            new_callable=AsyncMock,
        ), patch(
            "app.services.coaching_engine.retry_db_operation",
            new_callable=AsyncMock,
        ), patch(
            "app.services.learning_plan_generator.retry_db_operation",
            new_callable=AsyncMock,
        ):
            await pipeline.run(session_id, agent_id, mock_db)

        elapsed = time.monotonic() - start
        # With mocked LLM, should complete in well under 1 second
        # In production with real LLM, must be under 30s
        assert elapsed < 30.0

    async def test_pipeline_handles_empty_transcript(
        self, pipeline, session_id, agent_id
    ):
        """Pipeline should handle empty transcript gracefully (too-short detection)."""
        mock_db = AsyncMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch(
            "app.services.evaluation_engine.retry_db_operation",
            new_callable=AsyncMock,
        ), patch(
            "app.services.coaching_engine.retry_db_operation",
            new_callable=AsyncMock,
        ), patch(
            "app.services.learning_plan_generator.retry_db_operation",
            new_callable=AsyncMock,
        ):
            result = await pipeline.run(session_id, agent_id, mock_db)

        # Empty transcript → too short
        assert result.evaluation.is_too_short is True
        assert result.transcript == []

    async def test_pipeline_coaching_uses_evaluation_result(
        self, session_id, agent_id, sample_transcript_rows
    ):
        """Coaching engine should receive the evaluation result from step 2."""
        mock_db = AsyncMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = sample_transcript_rows
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute = AsyncMock(return_value=mock_result)

        mock_llm = MockLLMService()
        pipeline = EvaluationPipeline(llm_service=mock_llm)

        with patch(
            "app.services.evaluation_engine.retry_db_operation",
            new_callable=AsyncMock,
        ), patch(
            "app.services.coaching_engine.retry_db_operation",
            new_callable=AsyncMock,
        ), patch(
            "app.services.learning_plan_generator.retry_db_operation",
            new_callable=AsyncMock,
        ):
            result = await pipeline.run(session_id, agent_id, mock_db)

        # LLM called twice: once for evaluation, once for coaching
        assert mock_llm._call_count == 2
        # Coaching report has mistakes derived from the coaching LLM call
        assert result.coaching_report.total_mistakes == 2

    async def test_pipeline_learning_plan_maps_weak_categories(
        self, pipeline, session_id, agent_id, sample_transcript_rows
    ):
        """Learning plan should map weak categories to correct scenarios."""
        mock_db = AsyncMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = sample_transcript_rows
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch(
            "app.services.evaluation_engine.retry_db_operation",
            new_callable=AsyncMock,
        ), patch(
            "app.services.coaching_engine.retry_db_operation",
            new_callable=AsyncMock,
        ), patch(
            "app.services.learning_plan_generator.retry_db_operation",
            new_callable=AsyncMock,
        ):
            result = await pipeline.run(session_id, agent_id, mock_db)

        # empathy_communication=60 → Financial Hardship
        # negotiation_resolution=65 → Payment Arrangement
        weak_categories = {
            item.category for item in result.learning_plan.weak_competencies
        }
        assert EvaluationCategory.EMPATHY_COMMUNICATION in weak_categories
        assert EvaluationCategory.NEGOTIATION_RESOLUTION in weak_categories

        scenario_map = {
            item.category: item.recommended_scenario
            for item in result.learning_plan.weak_competencies
        }
        assert scenario_map[EvaluationCategory.EMPATHY_COMMUNICATION] == "Financial Hardship"
        assert scenario_map[EvaluationCategory.NEGOTIATION_RESOLUTION] == "Payment Arrangement"

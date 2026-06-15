"""Unit tests for the EvaluationEngine service.

Tests weighted score calculation and short session detection.
Validates: Requirements 5.2, 5.3, 5.7
"""

import pytest

from app.schemas import EvaluationCategory
from app.services.evaluation_engine import EvaluationEngine, CATEGORY_WEIGHTS


@pytest.fixture
def engine():
    """Create an EvaluationEngine instance."""
    return EvaluationEngine()


class TestCalculateOverallScore:
    """Tests for EvaluationEngine.calculate_overall_score."""

    def test_all_perfect_scores(self, engine: EvaluationEngine):
        """All 100s should produce an overall score of 100."""
        scores = {
            EvaluationCategory.CALL_OPENING: 100,
            EvaluationCategory.COMPLIANCE: 100,
            EvaluationCategory.EMPATHY_COMMUNICATION: 100,
            EvaluationCategory.NEGOTIATION_RESOLUTION: 100,
        }
        result = engine.calculate_overall_score(scores)
        assert result == 100.0

    def test_all_zero_scores(self, engine: EvaluationEngine):
        """All 0s should produce an overall score of 0."""
        scores = {
            EvaluationCategory.CALL_OPENING: 0,
            EvaluationCategory.COMPLIANCE: 0,
            EvaluationCategory.EMPATHY_COMMUNICATION: 0,
            EvaluationCategory.NEGOTIATION_RESOLUTION: 0,
        }
        result = engine.calculate_overall_score(scores)
        assert result == 0.0

    def test_weighted_calculation(self, engine: EvaluationEngine):
        """Verify the weighted formula: 80*0.20 + 90*0.30 + 70*0.25 + 60*0.25."""
        scores = {
            EvaluationCategory.CALL_OPENING: 80,
            EvaluationCategory.COMPLIANCE: 90,
            EvaluationCategory.EMPATHY_COMMUNICATION: 70,
            EvaluationCategory.NEGOTIATION_RESOLUTION: 60,
        }
        # Expected: 80*0.20 + 90*0.30 + 70*0.25 + 60*0.25
        # = 16 + 27 + 17.5 + 15 = 75.5
        result = engine.calculate_overall_score(scores)
        assert result == pytest.approx(75.5)

    def test_compliance_has_highest_weight(self, engine: EvaluationEngine):
        """Compliance (0.30) should have the most impact on the overall score."""
        # Only compliance is 100, rest are 0
        scores_compliance_high = {
            EvaluationCategory.CALL_OPENING: 0,
            EvaluationCategory.COMPLIANCE: 100,
            EvaluationCategory.EMPATHY_COMMUNICATION: 0,
            EvaluationCategory.NEGOTIATION_RESOLUTION: 0,
        }
        # Only call_opening is 100, rest are 0
        scores_opening_high = {
            EvaluationCategory.CALL_OPENING: 100,
            EvaluationCategory.COMPLIANCE: 0,
            EvaluationCategory.EMPATHY_COMMUNICATION: 0,
            EvaluationCategory.NEGOTIATION_RESOLUTION: 0,
        }
        compliance_result = engine.calculate_overall_score(scores_compliance_high)
        opening_result = engine.calculate_overall_score(scores_opening_high)
        assert compliance_result > opening_result

    def test_weights_sum_to_one(self):
        """Category weights must sum to 1.0 for valid percentage calculation."""
        total = sum(CATEGORY_WEIGHTS.values())
        assert total == pytest.approx(1.0)

    def test_result_in_valid_range(self, engine: EvaluationEngine):
        """Result should always be between 0 and 100."""
        scores = {
            EvaluationCategory.CALL_OPENING: 50,
            EvaluationCategory.COMPLIANCE: 75,
            EvaluationCategory.EMPATHY_COMMUNICATION: 25,
            EvaluationCategory.NEGOTIATION_RESOLUTION: 100,
        }
        result = engine.calculate_overall_score(scores)
        assert 0 <= result <= 100

    def test_missing_category_raises_error(self, engine: EvaluationEngine):
        """Missing a category should raise ValueError."""
        scores = {
            EvaluationCategory.CALL_OPENING: 80,
            EvaluationCategory.COMPLIANCE: 90,
            # Missing EMPATHY_COMMUNICATION and NEGOTIATION_RESOLUTION
        }
        with pytest.raises(ValueError, match="Missing score for category"):
            engine.calculate_overall_score(scores)

    def test_score_below_zero_raises_error(self, engine: EvaluationEngine):
        """Score below 0 should raise ValueError."""
        scores = {
            EvaluationCategory.CALL_OPENING: -1,
            EvaluationCategory.COMPLIANCE: 90,
            EvaluationCategory.EMPATHY_COMMUNICATION: 70,
            EvaluationCategory.NEGOTIATION_RESOLUTION: 60,
        }
        with pytest.raises(ValueError, match="must be between 0 and 100"):
            engine.calculate_overall_score(scores)

    def test_score_above_100_raises_error(self, engine: EvaluationEngine):
        """Score above 100 should raise ValueError."""
        scores = {
            EvaluationCategory.CALL_OPENING: 80,
            EvaluationCategory.COMPLIANCE: 101,
            EvaluationCategory.EMPATHY_COMMUNICATION: 70,
            EvaluationCategory.NEGOTIATION_RESOLUTION: 60,
        }
        with pytest.raises(ValueError, match="must be between 0 and 100"):
            engine.calculate_overall_score(scores)


class TestIsSessionTooShort:
    """Tests for EvaluationEngine.is_session_too_short."""

    def test_empty_transcript_is_too_short(self, engine: EvaluationEngine):
        """An empty transcript should be too short."""
        assert engine.is_session_too_short([]) is True

    def test_zero_agent_utterances_is_too_short(self, engine: EvaluationEngine):
        """Transcript with only debtor utterances is too short."""
        transcript = [
            {"speaker": "debtor", "text": "Hello?"},
            {"speaker": "debtor", "text": "Who is this?"},
            {"speaker": "debtor", "text": "I don't understand."},
            {"speaker": "debtor", "text": "Fine."},
        ]
        assert engine.is_session_too_short(transcript) is True

    def test_three_agent_utterances_is_too_short(self, engine: EvaluationEngine):
        """Transcript with exactly 3 agent utterances is too short."""
        transcript = [
            {"speaker": "agent", "text": "Hi, this is collections."},
            {"speaker": "debtor", "text": "Hello?"},
            {"speaker": "agent", "text": "I'm calling about your balance."},
            {"speaker": "debtor", "text": "What balance?"},
            {"speaker": "agent", "text": "Your outstanding balance of 5000."},
            {"speaker": "debtor", "text": "Oh."},
        ]
        assert engine.is_session_too_short(transcript) is True

    def test_four_agent_utterances_is_not_too_short(self, engine: EvaluationEngine):
        """Transcript with exactly 4 agent utterances is NOT too short."""
        transcript = [
            {"speaker": "agent", "text": "Hi, this is collections."},
            {"speaker": "debtor", "text": "Hello?"},
            {"speaker": "agent", "text": "I'm calling about your balance."},
            {"speaker": "debtor", "text": "What balance?"},
            {"speaker": "agent", "text": "Your outstanding balance of 5000."},
            {"speaker": "debtor", "text": "Oh."},
            {"speaker": "agent", "text": "Can we discuss a payment plan?"},
            {"speaker": "debtor", "text": "Maybe."},
        ]
        assert engine.is_session_too_short(transcript) is False

    def test_many_agent_utterances_is_not_too_short(self, engine: EvaluationEngine):
        """Transcript with many agent utterances is NOT too short."""
        transcript = [
            {"speaker": "agent", "text": f"Message {i}"}
            for i in range(10)
        ]
        assert engine.is_session_too_short(transcript) is False

    def test_mixed_speakers_counts_only_agent(self, engine: EvaluationEngine):
        """Only agent utterances count toward the threshold."""
        transcript = [
            {"speaker": "debtor", "text": "msg1"},
            {"speaker": "debtor", "text": "msg2"},
            {"speaker": "debtor", "text": "msg3"},
            {"speaker": "debtor", "text": "msg4"},
            {"speaker": "debtor", "text": "msg5"},
            {"speaker": "agent", "text": "msg6"},
            {"speaker": "agent", "text": "msg7"},
            {"speaker": "agent", "text": "msg8"},
        ]
        # Only 3 agent utterances despite many total entries
        assert engine.is_session_too_short(transcript) is True

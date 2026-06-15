"""Property test for weighted score calculation correctness.

Feature: collection-agent-trainer, Property 8: Weighted score calculation correctness

Validates: Requirements 5.2, 5.3

For any set of four category scores where each score is an integer in [0, 100],
the overall weighted score SHALL equal:
  call_opening * 0.20 + compliance * 0.30 + empathy_communication * 0.25 + negotiation_resolution * 0.25
and the result SHALL be in the range [0, 100].
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from app.schemas import EvaluationCategory
from app.services.evaluation_engine import EvaluationEngine

# Strategy: generate a score integer in [0, 100] for each category
scores_strategy = st.fixed_dictionaries({
    EvaluationCategory.CALL_OPENING: st.integers(min_value=0, max_value=100),
    EvaluationCategory.COMPLIANCE: st.integers(min_value=0, max_value=100),
    EvaluationCategory.EMPATHY_COMMUNICATION: st.integers(min_value=0, max_value=100),
    EvaluationCategory.NEGOTIATION_RESOLUTION: st.integers(min_value=0, max_value=100),
})


class TestWeightedScoreCalculation:
    """Property 8: Weighted score calculation correctness."""

    @given(scores=scores_strategy)
    @settings(max_examples=100)
    def test_weighted_sum_formula(self, scores: dict[EvaluationCategory, int]) -> None:
        """Verify the overall score matches the weighted sum formula.

        **Validates: Requirements 5.2, 5.3**
        """
        engine = EvaluationEngine()
        result = engine.calculate_overall_score(scores)

        expected = (
            scores[EvaluationCategory.CALL_OPENING] * 0.20
            + scores[EvaluationCategory.COMPLIANCE] * 0.30
            + scores[EvaluationCategory.EMPATHY_COMMUNICATION] * 0.25
            + scores[EvaluationCategory.NEGOTIATION_RESOLUTION] * 0.25
        )

        assert result == expected, (
            f"Expected weighted score {expected}, got {result}"
        )

    @given(scores=scores_strategy)
    @settings(max_examples=100)
    def test_result_in_valid_range(self, scores: dict[EvaluationCategory, int]) -> None:
        """Verify the overall score is always in [0, 100].

        **Validates: Requirements 5.2, 5.3**
        """
        engine = EvaluationEngine()
        result = engine.calculate_overall_score(scores)

        assert 0 <= result <= 100, (
            f"Overall score {result} is out of valid range [0, 100]"
        )

"""Property-based tests for short session detection.

Feature: collection-agent-trainer, Property 10: Short session detection threshold

**Validates: Requirements 5.7**

Property 10: For any transcript containing fewer than 4 utterances from the agent,
the Evaluation Engine SHALL set is_too_short to True and SHALL NOT produce category
scores or overall score.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from app.services.evaluation_engine import EvaluationEngine


# --- Strategies ---

speakers = st.sampled_from(["agent", "debtor"])

utterance_texts = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=200,
).filter(lambda s: s.strip() != "")


@st.composite
def short_session_transcript(draw):
    """Generate a transcript with 0-3 agent utterances and any number of debtor utterances.

    This ensures the transcript is always "too short" for evaluation.
    """
    num_agent = draw(st.integers(min_value=0, max_value=3))
    num_debtor = draw(st.integers(min_value=0, max_value=10))

    entries = []
    for _ in range(num_agent):
        entries.append({"speaker": "agent", "text": draw(utterance_texts)})
    for _ in range(num_debtor):
        entries.append({"speaker": "debtor", "text": draw(utterance_texts)})

    # Shuffle to randomize ordering
    shuffled = draw(st.permutations(entries))
    return list(shuffled)


@st.composite
def sufficient_session_transcript(draw):
    """Generate a transcript with 4+ agent utterances and any number of debtor utterances.

    This ensures the transcript is long enough for evaluation.
    """
    num_agent = draw(st.integers(min_value=4, max_value=15))
    num_debtor = draw(st.integers(min_value=0, max_value=10))

    entries = []
    for _ in range(num_agent):
        entries.append({"speaker": "agent", "text": draw(utterance_texts)})
    for _ in range(num_debtor):
        entries.append({"speaker": "debtor", "text": draw(utterance_texts)})

    # Shuffle to randomize ordering
    shuffled = draw(st.permutations(entries))
    return list(shuffled)


# --- Property Tests ---


class TestShortSessionDetection:
    """Property 10: Short session detection threshold.

    Feature: collection-agent-trainer, Property 10: Short session detection threshold
    """

    @given(transcript=short_session_transcript())
    @settings(max_examples=100)
    def test_short_session_detected_as_too_short(self, transcript: list):
        """**Validates: Requirements 5.7**

        For any transcript with 0-3 agent utterances (and any number of debtor
        utterances), is_session_too_short SHALL return True.
        """
        engine = EvaluationEngine()
        assert engine.is_session_too_short(transcript) is True

    @given(transcript=sufficient_session_transcript())
    @settings(max_examples=100)
    def test_sufficient_session_not_detected_as_too_short(self, transcript: list):
        """**Validates: Requirements 5.7**

        For any transcript with 4 or more agent utterances, is_session_too_short
        SHALL return False, indicating the session is eligible for scoring.
        """
        engine = EvaluationEngine()
        assert engine.is_session_too_short(transcript) is False

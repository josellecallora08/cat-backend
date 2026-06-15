"""Unit tests for emotional state transition logic (Task 5.2).

Tests classify_agent_tone() and transition_emotional_state() functions.
Validates Requirements 2.3 and 2.4.
"""

import pytest

from app.services.debtor_simulator import (
    AgentTone,
    EmotionalState,
    classify_agent_tone,
    transition_emotional_state,
)


# --- Tests for classify_agent_tone ---


class TestClassifyAgentTone:
    """Tests for keyword-based agent tone classification."""

    def test_empathetic_message(self):
        message = "I understand your situation and I want to help you find a solution."
        assert classify_agent_tone(message) == AgentTone.EMPATHETIC

    def test_aggressive_message(self):
        message = "You must pay now or face legal action and consequences."
        assert classify_agent_tone(message) == AgentTone.AGGRESSIVE

    def test_neutral_message(self):
        message = "Your account balance is five thousand pesos."
        assert classify_agent_tone(message) == AgentTone.NEUTRAL

    def test_empty_message_is_neutral(self):
        assert classify_agent_tone("") == AgentTone.NEUTRAL

    def test_empathetic_keywords_case_insensitive(self):
        message = "I UNDERSTAND and I'm SORRY about your DIFFICULT situation."
        assert classify_agent_tone(message) == AgentTone.EMPATHETIC

    def test_aggressive_keywords_case_insensitive(self):
        message = "You MUST pay IMMEDIATELY or face LEGAL ACTION."
        assert classify_agent_tone(message) == AgentTone.AGGRESSIVE

    def test_mixed_signals_more_empathetic(self):
        # More empathetic keywords than aggressive
        message = "I understand this is difficult and I want to help, but you must address this."
        assert classify_agent_tone(message) == AgentTone.EMPATHETIC

    def test_mixed_signals_more_aggressive(self):
        # More aggressive keywords than empathetic
        message = "I understand, but you must pay immediately or face consequences and legal action."
        assert classify_agent_tone(message) == AgentTone.AGGRESSIVE

    def test_equal_signals_returns_neutral(self):
        # Equal counts -> neutral (one empathetic "help", one aggressive "must")
        message = "Can you help me? You must respond."
        assert classify_agent_tone(message) == AgentTone.NEUTRAL

    def test_empathetic_phrases(self):
        message = "I hear you. Let's work with you to find options."
        assert classify_agent_tone(message) == AgentTone.EMPATHETIC

    def test_aggressive_phrases(self):
        message = "This is your final notice. Failure to pay will result in penalties."
        assert classify_agent_tone(message) == AgentTone.AGGRESSIVE

    def test_supportive_phrasing(self):
        message = "Take your time, there's no pressure. I'm here to support you and listen."
        assert classify_agent_tone(message) == AgentTone.EMPATHETIC

    def test_threatening_phrasing(self):
        message = "We will report this to the collection agency and garnish your wages."
        assert classify_agent_tone(message) == AgentTone.AGGRESSIVE


# --- Tests for transition_emotional_state ---


class TestTransitionEmotionalState:
    """Tests for emotional state transitions based on agent tone."""

    # --- Empathetic tone transitions ---

    def test_empathetic_moves_hostile_to_defensive(self):
        result = transition_emotional_state(EmotionalState.HOSTILE, AgentTone.EMPATHETIC)
        assert result == EmotionalState.DEFENSIVE

    def test_empathetic_moves_defensive_to_neutral(self):
        result = transition_emotional_state(EmotionalState.DEFENSIVE, AgentTone.EMPATHETIC)
        assert result == EmotionalState.NEUTRAL

    def test_empathetic_moves_neutral_to_receptive(self):
        result = transition_emotional_state(EmotionalState.NEUTRAL, AgentTone.EMPATHETIC)
        assert result == EmotionalState.RECEPTIVE

    def test_empathetic_moves_receptive_to_cooperative(self):
        result = transition_emotional_state(EmotionalState.RECEPTIVE, AgentTone.EMPATHETIC)
        assert result == EmotionalState.COOPERATIVE

    def test_empathetic_clamps_at_cooperative(self):
        result = transition_emotional_state(EmotionalState.COOPERATIVE, AgentTone.EMPATHETIC)
        assert result == EmotionalState.COOPERATIVE

    # --- Aggressive tone transitions ---

    def test_aggressive_moves_cooperative_to_receptive(self):
        result = transition_emotional_state(EmotionalState.COOPERATIVE, AgentTone.AGGRESSIVE)
        assert result == EmotionalState.RECEPTIVE

    def test_aggressive_moves_receptive_to_neutral(self):
        result = transition_emotional_state(EmotionalState.RECEPTIVE, AgentTone.AGGRESSIVE)
        assert result == EmotionalState.NEUTRAL

    def test_aggressive_moves_neutral_to_defensive(self):
        result = transition_emotional_state(EmotionalState.NEUTRAL, AgentTone.AGGRESSIVE)
        assert result == EmotionalState.DEFENSIVE

    def test_aggressive_moves_defensive_to_hostile(self):
        result = transition_emotional_state(EmotionalState.DEFENSIVE, AgentTone.AGGRESSIVE)
        assert result == EmotionalState.HOSTILE

    def test_aggressive_clamps_at_hostile(self):
        result = transition_emotional_state(EmotionalState.HOSTILE, AgentTone.AGGRESSIVE)
        assert result == EmotionalState.HOSTILE

    # --- Neutral tone transitions ---

    def test_neutral_no_change_from_hostile(self):
        result = transition_emotional_state(EmotionalState.HOSTILE, AgentTone.NEUTRAL)
        assert result == EmotionalState.HOSTILE

    def test_neutral_no_change_from_defensive(self):
        result = transition_emotional_state(EmotionalState.DEFENSIVE, AgentTone.NEUTRAL)
        assert result == EmotionalState.DEFENSIVE

    def test_neutral_no_change_from_neutral(self):
        result = transition_emotional_state(EmotionalState.NEUTRAL, AgentTone.NEUTRAL)
        assert result == EmotionalState.NEUTRAL

    def test_neutral_no_change_from_receptive(self):
        result = transition_emotional_state(EmotionalState.RECEPTIVE, AgentTone.NEUTRAL)
        assert result == EmotionalState.RECEPTIVE

    def test_neutral_no_change_from_cooperative(self):
        result = transition_emotional_state(EmotionalState.COOPERATIVE, AgentTone.NEUTRAL)
        assert result == EmotionalState.COOPERATIVE

    # --- Boundary verification ---

    def test_all_states_with_empathetic_stay_within_bounds(self):
        for state in EmotionalState:
            result = transition_emotional_state(state, AgentTone.EMPATHETIC)
            assert EmotionalState.HOSTILE.value <= result.value <= EmotionalState.COOPERATIVE.value

    def test_all_states_with_aggressive_stay_within_bounds(self):
        for state in EmotionalState:
            result = transition_emotional_state(state, AgentTone.AGGRESSIVE)
            assert EmotionalState.HOSTILE.value <= result.value <= EmotionalState.COOPERATIVE.value

    def test_empathetic_always_increases_or_stays_at_max(self):
        for state in EmotionalState:
            result = transition_emotional_state(state, AgentTone.EMPATHETIC)
            assert result.value >= state.value

    def test_aggressive_always_decreases_or_stays_at_min(self):
        for state in EmotionalState:
            result = transition_emotional_state(state, AgentTone.AGGRESSIVE)
            assert result.value <= state.value

    def test_non_boundary_empathetic_strictly_increases(self):
        """Property 5: Non-boundary states move at least one level toward cooperative."""
        non_boundary_states = [s for s in EmotionalState if s != EmotionalState.COOPERATIVE]
        for state in non_boundary_states:
            result = transition_emotional_state(state, AgentTone.EMPATHETIC)
            assert result.value > state.value

    def test_non_boundary_aggressive_strictly_decreases(self):
        """Property 5: Non-boundary states move at least one level toward hostile."""
        non_boundary_states = [s for s in EmotionalState if s != EmotionalState.HOSTILE]
        for state in non_boundary_states:
            result = transition_emotional_state(state, AgentTone.AGGRESSIVE)
            assert result.value < state.value

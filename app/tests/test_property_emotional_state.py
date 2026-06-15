"""Property-based tests for emotional state transitions.

Feature: collection-agent-trainer, Property 5: Emotional state transition monotonicity

**Validates: Requirements 2.3, 2.4**

Property 5: For any emotional state that is not already at the boundary, applying an
empathetic agent tone SHALL produce a new emotional state at least one level toward
cooperative (higher value), and applying an aggressive agent tone SHALL produce a new
emotional state at least one level toward hostile (lower value).
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from app.services.debtor_simulator import AgentTone, EmotionalState, transition_emotional_state


# --- Strategies ---

# All valid emotional states
all_emotional_states = st.sampled_from(list(EmotionalState))

# Non-boundary states for empathetic tone (not already at max COOPERATIVE=5)
non_max_states = st.sampled_from([
    EmotionalState.HOSTILE,
    EmotionalState.DEFENSIVE,
    EmotionalState.NEUTRAL,
    EmotionalState.RECEPTIVE,
])

# Non-boundary states for aggressive tone (not already at min HOSTILE=1)
non_min_states = st.sampled_from([
    EmotionalState.DEFENSIVE,
    EmotionalState.NEUTRAL,
    EmotionalState.RECEPTIVE,
    EmotionalState.COOPERATIVE,
])


class TestEmotionalStateTransitionMonotonicity:
    """Property 5: Emotional state transition monotonicity.

    Feature: collection-agent-trainer, Property 5: Emotional state transition monotonicity
    """

    @given(state=non_max_states)
    @settings(max_examples=100)
    def test_empathetic_tone_increases_state(self, state: EmotionalState):
        """For non-boundary states with EMPATHETIC tone: new state > current state.

        **Validates: Requirements 2.3**
        """
        new_state = transition_emotional_state(state, AgentTone.EMPATHETIC)
        assert new_state.value > state.value, (
            f"EMPATHETIC tone on {state.name}({state.value}) should increase state, "
            f"but got {new_state.name}({new_state.value})"
        )

    @given(state=non_min_states)
    @settings(max_examples=100)
    def test_aggressive_tone_decreases_state(self, state: EmotionalState):
        """For non-boundary states with AGGRESSIVE tone: new state < current state.

        **Validates: Requirements 2.4**
        """
        new_state = transition_emotional_state(state, AgentTone.AGGRESSIVE)
        assert new_state.value < state.value, (
            f"AGGRESSIVE tone on {state.name}({state.value}) should decrease state, "
            f"but got {new_state.name}({new_state.value})"
        )

    @given(state=all_emotional_states)
    @settings(max_examples=100)
    def test_neutral_tone_preserves_state(self, state: EmotionalState):
        """For any state with NEUTRAL tone: state unchanged.

        **Validates: Requirements 2.3, 2.4**
        """
        new_state = transition_emotional_state(state, AgentTone.NEUTRAL)
        assert new_state == state, (
            f"NEUTRAL tone on {state.name}({state.value}) should preserve state, "
            f"but got {new_state.name}({new_state.value})"
        )

    @given(
        state=all_emotional_states,
        tone=st.sampled_from(list(AgentTone)),
    )
    @settings(max_examples=100)
    def test_transitions_produce_valid_emotional_state(
        self, state: EmotionalState, tone: AgentTone
    ):
        """All transitions produce valid EmotionalState values (1-5).

        **Validates: Requirements 2.3, 2.4**
        """
        new_state = transition_emotional_state(state, tone)
        assert isinstance(new_state, EmotionalState), (
            f"Expected EmotionalState instance, got {type(new_state)}"
        )
        assert EmotionalState.HOSTILE.value <= new_state.value <= EmotionalState.COOPERATIVE.value, (
            f"State value {new_state.value} is outside valid range [1, 5]"
        )

"""Property-based tests for script-driven Debtor_Simulator behavior.

Feature: ai-debtor-script-contract

This module is shared/appended to across several sub-tasks under task 13
(13.2, 13.4, 13.6, 13.8, 13.10, 13.12, 13.14, 13.16), each contributing one
test class for one of the correctness properties defined in `design.md`:

    - Property 17 (this task, 13.2): Opening utterance matches the loaded
      Script_Version
    - Property 18 (task 13.4): Emotional state changes follow the loaded
      Script_Version's rules
    - Property 19 (task 13.6): Trigger phrase matches always apply their
      configured behavior
    - Property 20 (task 13.8): Escalation conditions apply their behavior
      and end the call exactly when specified
    - Property 21 (task 13.10): Prohibited responses never reach the
      Agent
    - Property 22 (task 13.12): Payment condition matches respond
      according to their configured term
    - Property 23 (task 13.14): Conversation goal completion ends the
      call according to its rule
    - Property 24 (task 13.16): An active Training_Call is isolated from
      later publishes of the same Script

The `safe_text` free-text strategy and `valid_script_contract_dicts`
composite strategy below are adapted from
`test_property_script_lifecycle.py` / `test_property_script_consumption.py`
so that generated `Script_Version.content` dicts are structurally
consistent with the real `ScriptContract` shape used throughout this spec.
Later test classes appended to this file should reuse these strategies
rather than redefining their own copies.

Property 24 (`TestPublishIsolationDuringActiveCall`, task 13.16) is
fundamentally a DB-level isolation property rather than a pure-function
property like 17-23 above, so it additionally self-contains its own copy
of the async in-memory SQLite `async_db` fixture and the
`_make_admin_user`/`_make_scenario` FK prerequisite-row helpers, following
the same pattern established in `test_property_script_consumption.py` /
`test_property_script_lifecycle.py`, since this file otherwise only
imports from the pure `app.services.debtor_simulator`/`llm_service`
modules above.
"""

import json
import uuid
from unittest.mock import AsyncMock

import pytest
import yaml
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base
from app.models import Scenario, Script, ScriptVersion, User
from app.models.user import UserRole
from app.services.debtor_simulator import (
    SAFE_DEFAULT_DEBTOR_RESPONSE,
    AgentTone,
    DebtorSimulatorService,
    EmotionalState,
    PersonaContext,
    _apply_directional_step,
    _NEGATIVE_EMOTION_KEYWORDS,
    _POSITIVE_EMOTION_KEYWORDS,
    _resolve_state_change_direction,
    contains_prohibited_response,
    evaluate_conversation_goal_completion,
    evaluate_escalation_conditions,
    match_payment_condition,
    match_trigger_phrase,
    select_opening_response,
    transition_emotional_state,
)
from app.services.llm_service import LLMResponse, LLMServiceProtocol
from app.services.script_registry import (
    create_draft,
    get_active_published_version,
    publish,
    update_draft,
)
from app.services.session_service import create_session

# --- Shared strategies (adapted from test_property_script_lifecycle.py /
# test_property_script_consumption.py) ---

# Free-text strategy: printable letters/numbers plus a few safe punctuation
# characters, with leading/trailing whitespace stripped so the value is
# guaranteed non-empty and well-formed after stripping. ASCII-restricted
# (rather than full Unicode "L"/"N" categories) to match the alphabet used
# by this spec's other property tests.
safe_text = (
    st.text(
        alphabet=st.characters(
            whitelist_categories=(),
            whitelist_characters=(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-"
            ),
        ),
        min_size=1,
        max_size=50,
    )
    .map(lambda s: s.strip())
    .filter(lambda s: s != "")
)

trigger_phrase_entries = st.fixed_dictionaries({"phrase": safe_text, "behavior": safe_text})

expected_reply_entries = st.fixed_dictionaries(
    {"agent_statement": safe_text, "debtor_reply": safe_text}
)

emotional_state_rule_entries = st.fixed_dictionaries({"trigger": safe_text, "state_change": safe_text})

escalation_condition_entries = st.fixed_dictionaries(
    {"condition": safe_text, "behavior": safe_text, "ends_call": st.booleans()}
)

payment_condition_entries = st.fixed_dictionaries(
    {"condition": safe_text, "term": safe_text, "accepted": st.booleans()}
)

debtor_persona_dicts = st.fixed_dictionaries(
    {"name": safe_text, "communication_style": safe_text, "background": safe_text}
)

financial_situation_dicts = st.fixed_dictionaries(
    {
        "outstanding_balance": st.floats(
            min_value=0.01,
            max_value=999_999.0,
            allow_nan=False,
            allow_infinity=False,
        ).map(lambda x: round(x, 2)),
        "days_past_due": st.integers(min_value=0, max_value=10_000),
        "reason_for_delinquency": safe_text,
    }
)

conversation_goal_dicts = st.fixed_dictionaries(
    {"target_outcome": safe_text, "completion_condition": safe_text}
)


@st.composite
def valid_script_contract_dicts(draw, opening_response=None):
    """Generate a structurally valid, varied Script_Contract-shaped dict.

    `prohibited_responses` is kept empty so no generated example
    accidentally triggers the (unrelated) Prohibited/Expected conflict
    rule (Property 4), which is out of scope for this module's simulator
    properties.

    Args:
        opening_response: Optional Hypothesis strategy to use for the
            `opening_response` field. If omitted, a fresh `safe_text`
            value is drawn. Callers that need to assert an exact
            round-tripped value (e.g. Property 17) should pass their own
            strategy/value here rather than re-drawing separately, so the
            generated dict and the asserted value are guaranteed to
            match.
    """
    return {
        "debtor_persona": draw(debtor_persona_dicts),
        "financial_situation": draw(financial_situation_dicts),
        "opening_response": draw(opening_response) if opening_response is not None else draw(safe_text),
        "expected_replies": draw(st.lists(expected_reply_entries, min_size=0, max_size=5)),
        "trigger_phrases": draw(st.lists(trigger_phrase_entries, min_size=0, max_size=5)),
        "emotional_state_rules": draw(
            st.lists(emotional_state_rule_entries, min_size=0, max_size=5)
        ),
        "payment_conditions": draw(st.lists(payment_condition_entries, min_size=0, max_size=5)),
        "escalation_conditions": draw(
            st.lists(escalation_condition_entries, min_size=0, max_size=5)
        ),
        "prohibited_responses": [],
        "conversation_goal": draw(conversation_goal_dicts),
    }


# --- Property Tests ---


class TestOpeningUtteranceMatchesScriptVersion:
    """Property 17: Opening utterance matches the loaded Script_Version.

    Feature: ai-debtor-script-contract, Property 17: Opening utterance
    matches the loaded Script_Version

    For any loaded Script_Version, the first debtor utterance generated
    at Training_Call start SHALL equal the opening_response defined in
    that Script_Version.

    **Validates: Requirements 4.4**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(opening_response=safe_text, script_content=valid_script_contract_dicts())
    def test_select_opening_response_returns_exact_opening_response(
        self, opening_response, script_content
    ):
        """For any varied Script_Version content dict with a given
        `opening_response` value, `select_opening_response` SHALL return
        exactly that value, byte-for-byte, with no transformation --
        regardless of the other varied content in the dict."""
        script_content["opening_response"] = opening_response

        result = select_opening_response(script_content)

        assert result == opening_response, (
            "Expected select_opening_response to return the loaded "
            f"Script_Version's opening_response verbatim, got {result!r} "
            f"!= {opening_response!r}"
        )
        assert result is opening_response or result == opening_response

    def test_select_opening_response_returns_none_when_no_script_loaded(self):
        """When no script is loaded for the session (`script_content` is
        `None`), `select_opening_response` SHALL return `None` -- the
        fallback signal for callers to use the existing LLM-based
        opening-line flow instead."""
        result = select_opening_response(None)

        assert result is None, (
            f"Expected select_opening_response(None) to return None, got "
            f"{result!r}"
        )


# --- Strategies for Property 18 (script-driven emotional state changes) ---

# All emotion keywords recognized by `_resolve_state_change_direction`'s
# heuristic, drawn directly from the module so this test stays in sync with
# the implementation's documented keyword vocabulary.
_ALL_EMOTION_KEYWORDS = sorted(_NEGATIVE_EMOTION_KEYWORDS | _POSITIVE_EMOTION_KEYWORDS)

# Modifier prefixes recognized by `_resolve_state_change_direction`: some
# invert polarity ("decrease_"/"less_"/"reduce_"/"lower_"), some keep it as
# a no-op ("increase_"/"more_"), and "" (no modifier) also keeps polarity
# as-is per the documented heuristic.
_DIRECTION_MODIFIERS = ["", "increase_", "decrease_", "more_", "less_", "reduce_", "lower_"]

# `state_change` text built from the SAME keyword vocabulary
# `_resolve_state_change_direction` recognizes, so its resulting direction
# is deterministically predictable by calling the function directly.
matching_state_change_texts = st.builds(
    lambda modifier, keyword: f"{modifier}{keyword}",
    modifier=st.sampled_from(_DIRECTION_MODIFIERS),
    keyword=st.sampled_from(_ALL_EMOTION_KEYWORDS),
)

_AGENT_TONE_VALUES = {tone.value for tone in AgentTone}

# Decoy `emotional_state_rules` entries whose `trigger` deliberately does
# NOT match any AgentTone value, so they can never accidentally satisfy the
# matching branch of `transition_emotional_state`.
non_matching_emotional_state_rule_entries = st.builds(
    lambda trigger, state_change: {"trigger": trigger, "state_change": state_change},
    trigger=safe_text.filter(lambda s: s.strip().lower() not in _AGENT_TONE_VALUES),
    state_change=safe_text,
)


class TestScriptDrivenEmotionalStateChanges:
    """Property 18: Emotional state changes follow the loaded Script_Version's rules.

    Feature: ai-debtor-script-contract, Property 18: Emotional state
    changes follow the loaded Script_Version's rules

    For any loaded Script_Version defining an Emotional_State_Rule for a
    given agent tone or event, and any conversation turn classified as
    that tone/event, the resulting emotional state change SHALL match the
    rule defined in the Script_Version rather than the simulator's
    default table.

    **Validates: Requirements 4.5**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        tone=st.sampled_from(list(AgentTone)),
        state_change_text=matching_state_change_texts,
        current=st.sampled_from(list(EmotionalState)),
        decoy_rules=st.lists(non_matching_emotional_state_rule_entries, min_size=0, max_size=3),
    )
    def test_matching_rule_direction_overrides_hardcoded_table(
        self, tone, state_change_text, current, decoy_rules
    ):
        """When a script-defined rule's `trigger` matches the classified
        tone, the resulting emotional state SHALL be computed by applying
        that rule's `state_change` direction (per
        `_resolve_state_change_direction`) rather than the hardcoded
        empathetic/neutral/aggressive table."""
        matching_rule = {"trigger": tone.value, "state_change": state_change_text}
        # Interleave decoys around the matching rule so the matching rule
        # is not always found at a fixed position in the list.
        rules = [*decoy_rules, matching_rule]

        expected_direction = _resolve_state_change_direction(state_change_text)
        expected_result = _apply_directional_step(current, expected_direction)

        result = transition_emotional_state(current, tone, emotional_state_rules=rules)

        assert result == expected_result, (
            f"Expected transition_emotional_state to follow the script's "
            f"rule (trigger={tone.value!r}, state_change={state_change_text!r} "
            f"-> direction={expected_direction}), got {result!r} instead of "
            f"{expected_result!r} for current={current!r}"
        )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        tone=st.sampled_from(list(AgentTone)),
        current=st.sampled_from(list(EmotionalState)),
        decoy_rules=st.lists(non_matching_emotional_state_rule_entries, min_size=0, max_size=5),
    )
    def test_no_matching_rule_falls_back_to_hardcoded_table(self, tone, current, decoy_rules):
        """When `emotional_state_rules` are present but none of their
        `trigger` values match the classified tone (or any event), the
        result SHALL equal the simulator's default hardcoded-table
        behavior -- i.e. calling with no rules at all."""
        result_with_rules = transition_emotional_state(
            current, tone, emotional_state_rules=decoy_rules
        )
        result_without_rules = transition_emotional_state(current, tone)

        assert result_with_rules == result_without_rules, (
            f"Expected no-matching-rule case to fall back to the hardcoded "
            f"table result {result_without_rules!r}, got {result_with_rules!r} "
            f"for tone={tone!r}, current={current!r}, decoy_rules={decoy_rules!r}"
        )

    @given(current=st.sampled_from(list(EmotionalState)))
    def test_script_rule_diverges_from_hardcoded_table_for_contrived_case(self, current):
        """Deliberately contrived case: a script rule for EMPATHETIC tone
        whose `state_change` indicates INCREASED hostility SHALL move the
        emotional state DOWN (toward HOSTILE) -- the opposite of the
        hardcoded table's default +1-toward-COOPERATIVE behavior for
        EMPATHETIC tone. This demonstrates the script's rule is actually
        being followed instead of the default table, not merely
        coinciding with it."""
        rule = {"trigger": "empathetic", "state_change": "increase_hostility"}

        expected_direction = _resolve_state_change_direction("increase_hostility")
        assert expected_direction == -1, (
            "Sanity check on the heuristic: 'increase_hostility' should "
            f"resolve to direction -1 (toward HOSTILE), got {expected_direction}"
        )

        scripted_result = transition_emotional_state(
            current, AgentTone.EMPATHETIC, emotional_state_rules=[rule]
        )
        hardcoded_result = transition_emotional_state(current, AgentTone.EMPATHETIC)

        expected_scripted = _apply_directional_step(current, expected_direction)
        assert scripted_result == expected_scripted

        assert scripted_result != hardcoded_result, (
            "Expected the script's rule (move toward HOSTILE) to diverge "
            "from the hardcoded table's default (+1 toward COOPERATIVE) "
            f"for EMPATHETIC tone, but both produced {scripted_result!r} "
            f"for current={current!r}"
        )


# --- Strategies for Property 19 (trigger-phrase matching) ---

# Longer, alphanumeric-only phrase text (no spaces/punctuation), reducing the
# chance of accidental substring collisions between independently-generated
# trigger phrases -- and between a phrase and random `safe_text` surrounding
# context -- when constructing the embedding tests below. This keeps the
# `assume()` filters in this section from over-rejecting.
distinctive_phrase_text = (
    st.text(
        alphabet=st.characters(
            whitelist_categories=(),
            whitelist_characters=(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            ),
        ),
        min_size=4,
        max_size=20,
    )
    .map(lambda s: s.strip())
    .filter(lambda s: s != "")
)

distinctive_trigger_phrase_entries = st.fixed_dictionaries(
    {"phrase": distinctive_phrase_text, "behavior": safe_text}
)

# Case-variation transforms applied to an embedded phrase, to confirm
# `match_trigger_phrase`'s case-insensitive substring matching.
_CASE_VARIATIONS = {
    "lower": str.lower,
    "upper": str.upper,
    "swapcase": str.swapcase,
    "title": str.title,
    "unchanged": lambda s: s,
}

case_variation_names = st.sampled_from(sorted(_CASE_VARIATIONS.keys()))


class TestTriggerPhraseMatching:
    """Property 19: Trigger phrase matches always apply their configured behavior.

    Feature: ai-debtor-script-contract, Property 19: Trigger phrase
    matches always apply their configured behavior

    For any loaded Script_Version and any agent input containing one of
    its Trigger_Phrases, the simulator SHALL apply the behavior
    configured for that specific phrase; for any agent input matching
    none of the Script_Version's Trigger_Phrases, no trigger-phrase
    behavior SHALL be applied.

    **Validates: Requirements 4.6**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.filter_too_much])
    @given(
        trigger_phrases=st.lists(distinctive_trigger_phrase_entries, min_size=1, max_size=5),
        prefix=safe_text,
        suffix=safe_text,
        case_variation_name=case_variation_names,
    )
    def test_message_containing_first_entrys_phrase_returns_its_behavior(
        self, trigger_phrases, prefix, suffix, case_variation_name
    ):
        """When `agent_message` deliberately contains `trigger_phrases[0]`'s
        `phrase` (embedded within surrounding text, in an arbitrary case
        variation), and none of the other entries' phrases are also
        present as substrings, `match_trigger_phrase` SHALL return exactly
        `trigger_phrases[0]`'s `behavior`."""
        first_entry = trigger_phrases[0]
        case_fn = _CASE_VARIATIONS[case_variation_name]
        varied_phrase = case_fn(first_entry["phrase"])
        agent_message = f"{prefix} {varied_phrase} {suffix}"

        lower_message = agent_message.lower()
        other_phrase_also_present = any(
            entry["phrase"].lower() in lower_message for entry in trigger_phrases[1:]
        )
        assume(not other_phrase_also_present)

        result = match_trigger_phrase(agent_message, trigger_phrases)

        assert result == first_entry["behavior"], (
            "Expected match_trigger_phrase to return trigger_phrases[0]'s "
            f"behavior {first_entry['behavior']!r} for agent_message "
            f"{agent_message!r} (embedding phrase {first_entry['phrase']!r} "
            f"as {varied_phrase!r}), got {result!r} instead"
        )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.filter_too_much])
    @given(
        trigger_phrases=st.lists(distinctive_trigger_phrase_entries, min_size=1, max_size=5),
        agent_message=safe_text,
    )
    def test_message_matching_no_phrase_returns_none(self, trigger_phrases, agent_message):
        """For any `agent_message` that is guaranteed to contain none of
        `trigger_phrases`' `phrase` values as a substring (case-insensitive),
        `match_trigger_phrase` SHALL return `None`."""
        lower_message = agent_message.lower()
        assume(
            not any(entry["phrase"].lower() in lower_message for entry in trigger_phrases)
        )

        result = match_trigger_phrase(agent_message, trigger_phrases)

        assert result is None, (
            f"Expected match_trigger_phrase to return None for agent_message "
            f"{agent_message!r} matching none of {trigger_phrases!r}, got "
            f"{result!r} instead"
        )

    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        agent_message=safe_text,
        empty_trigger_phrases=st.sampled_from([None, []]),
    )
    def test_empty_or_none_trigger_phrases_returns_none(
        self, agent_message, empty_trigger_phrases
    ):
        """For any `agent_message`, when `trigger_phrases` is `None` or an
        empty list, `match_trigger_phrase` SHALL return `None` regardless
        of the message content."""
        result = match_trigger_phrase(agent_message, empty_trigger_phrases)

        assert result is None, (
            f"Expected match_trigger_phrase(agent_message, "
            f"{empty_trigger_phrases!r}) to return None, got {result!r} "
            f"for agent_message={agent_message!r}"
        )


# --- Strategies for Property 20 (escalation conditions) ---

# Distinctive, alphanumeric-only condition text (reusing the same rationale
# as `distinctive_phrase_text` above): reduces the chance of accidental
# substring collisions between independently-generated escalation
# conditions, and between a condition and random `safe_text` surrounding
# context, when constructing the embedding tests below.
distinctive_condition_text = distinctive_phrase_text

distinctive_escalation_condition_entries = st.fixed_dictionaries(
    {
        "condition": distinctive_condition_text,
        "behavior": safe_text,
        "ends_call": st.booleans(),
    }
)


class TestEscalationConditions:
    """Property 20: Escalation conditions apply their behavior and end the
    call exactly when specified.

    Feature: ai-debtor-script-contract, Property 20: Escalation conditions
    apply their behavior and end the call exactly when specified

    For any loaded Script_Version and any conversation state satisfying
    one of its Escalation_Conditions, the simulator SHALL apply that
    condition's configured behavior, and the Training_Call SHALL end if
    and only if that condition's ends_call flag is true.

    **Validates: Requirements 4.7**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.filter_too_much])
    @given(
        escalation_conditions=st.lists(
            distinctive_escalation_condition_entries, min_size=1, max_size=5
        ),
        prefix=safe_text,
        suffix=safe_text,
        case_variation_name=case_variation_names,
    )
    def test_message_containing_first_entrys_condition_returns_its_behavior_and_ends_call_flag(
        self, escalation_conditions, prefix, suffix, case_variation_name
    ):
        """When `agent_message` deliberately contains
        `escalation_conditions[0]`'s `condition` (embedded within
        surrounding text, in an arbitrary case variation), and none of the
        other entries' conditions are also present as substrings,
        `evaluate_escalation_conditions` SHALL return exactly
        `(escalation_conditions[0]["behavior"], escalation_conditions[0]["ends_call"])`
        -- covering both the ends_call=True and ends_call=False cases, so
        the call ends if and only if that specific entry's ends_call flag
        is True, never unconditionally always/never."""
        first_entry = escalation_conditions[0]
        case_fn = _CASE_VARIATIONS[case_variation_name]
        varied_condition = case_fn(first_entry["condition"])
        agent_message = f"{prefix} {varied_condition} {suffix}"

        lower_message = agent_message.lower()
        other_condition_also_present = any(
            entry["condition"].lower() in lower_message for entry in escalation_conditions[1:]
        )
        assume(not other_condition_also_present)

        result = evaluate_escalation_conditions(agent_message, escalation_conditions)

        assert result == (first_entry["behavior"], first_entry["ends_call"]), (
            "Expected evaluate_escalation_conditions to return "
            f"escalation_conditions[0]'s (behavior, ends_call) = "
            f"({first_entry['behavior']!r}, {first_entry['ends_call']!r}) for "
            f"agent_message {agent_message!r} (embedding condition "
            f"{first_entry['condition']!r} as {varied_condition!r}), got "
            f"{result!r} instead"
        )
        assert result is not None
        _, ends_call = result
        assert ends_call is first_entry["ends_call"] or ends_call == first_entry["ends_call"], (
            "Expected the returned ends_call flag to match exactly what was "
            f"configured ({first_entry['ends_call']!r}), got {ends_call!r} "
            "-- the call must end if and only if the matched entry's "
            "ends_call is True, not unconditionally always/never."
        )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.filter_too_much])
    @given(
        escalation_conditions=st.lists(
            distinctive_escalation_condition_entries, min_size=1, max_size=5
        ),
        agent_message=safe_text,
    )
    def test_message_matching_no_condition_returns_none(
        self, escalation_conditions, agent_message
    ):
        """For any `agent_message` that is guaranteed to contain none of
        `escalation_conditions`' `condition` values as a substring
        (case-insensitive), `evaluate_escalation_conditions` SHALL return
        `None` -- no behavior is applied and no call-ending signal is
        produced."""
        lower_message = agent_message.lower()
        assume(
            not any(
                entry["condition"].lower() in lower_message
                for entry in escalation_conditions
            )
        )

        result = evaluate_escalation_conditions(agent_message, escalation_conditions)

        assert result is None, (
            f"Expected evaluate_escalation_conditions to return None for "
            f"agent_message {agent_message!r} matching none of "
            f"{escalation_conditions!r}, got {result!r} instead"
        )

    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        agent_message=safe_text,
        empty_escalation_conditions=st.sampled_from([None, []]),
    )
    def test_empty_or_none_escalation_conditions_returns_none(
        self, agent_message, empty_escalation_conditions
    ):
        """For any `agent_message`, when `escalation_conditions` is `None`
        or an empty list, `evaluate_escalation_conditions` SHALL return
        `None` regardless of the message content."""
        result = evaluate_escalation_conditions(agent_message, empty_escalation_conditions)

        assert result is None, (
            f"Expected evaluate_escalation_conditions(agent_message, "
            f"{empty_escalation_conditions!r}) to return None, got {result!r} "
            f"for agent_message={agent_message!r}"
        )


# --- Strategies for Property 21 (prohibited-response filtering) ---

# Distinctive, alphanumeric-only prohibited-response text (reusing the same
# rationale as `distinctive_phrase_text` above): reduces the chance of
# accidental substring collisions between independently-generated
# prohibited responses and surrounding "clean" raw output text used below.
distinctive_prohibited_text = distinctive_phrase_text


def _make_service_with_llm_outputs(raw_outputs: list[str]) -> "DebtorSimulatorService":
    """Build a DebtorSimulatorService whose mocked LLM returns `raw_outputs`
    in sequence, one per `chat_completion` call.

    Args:
        raw_outputs: The raw `.content` text to return, in call order. Each
            entry corresponds to exactly one `chat_completion` invocation.

    Returns:
        A DebtorSimulatorService wired to an AsyncMock LLM service.
    """
    llm_service = AsyncMock(spec=LLMServiceProtocol)
    llm_service.chat_completion = AsyncMock(
        side_effect=[LLMResponse(content=text, model="test-model") for text in raw_outputs]
    )
    return DebtorSimulatorService(llm_service=llm_service)


def _make_persona() -> "PersonaContext":
    """Build a minimal PersonaContext for generate_response tests."""
    return PersonaContext(
        persona_id=uuid.uuid4(),
        name="Test Debtor",
        communication_style="anxious",
        financial_circumstances={
            "income_level": "low",
            "debt_amount": 10000,
            "reason_for_delinquency": "job loss",
        },
        emotional_state=EmotionalState.NEUTRAL,
        conversation_history=[],
        language="EN",
    )


class TestProhibitedResponsesNeverReachAgent:
    """Property 21: Prohibited responses never reach the Agent.

    Feature: ai-debtor-script-contract, Property 21: Prohibited responses
    never reach the Agent

    For any loaded Script_Version and any sequence of raw generated debtor
    outputs (including ones that match a Prohibited_Response), the debtor
    output ultimately returned to the Agent SHALL NOT match any entry in
    that Script_Version's Prohibited_Responses.

    **Validates: Requirements 4.8**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        prohibited_responses=st.lists(distinctive_prohibited_text, min_size=1, max_size=5),
        data=st.data(),
    )
    async def test_all_attempts_prohibited_falls_back_to_safe_default(
        self, prohibited_responses, data
    ):
        """When every LLM attempt (the initial call plus all
        `_MAX_PROHIBITED_RESPONSE_RETRIES` retries) returns raw output that
        matches a prohibited response, `generate_response` SHALL exhaust
        its retry budget and return exactly `SAFE_DEFAULT_DEBTOR_RESPONSE`
        -- which itself SHALL NOT match any prohibited response."""
        total_attempts = DebtorSimulatorService._MAX_PROHIBITED_RESPONSE_RETRIES + 1

        # Every raw output literally IS one of the prohibited_responses
        # entries (drawn per-attempt so different attempts can use
        # different entries), guaranteeing each attempt matches.
        raw_outputs = data.draw(
            st.lists(
                st.sampled_from(prohibited_responses),
                min_size=total_attempts,
                max_size=total_attempts,
            )
        )

        service = _make_service_with_llm_outputs(raw_outputs)
        persona = _make_persona()

        result = await service.generate_response(
            persona,
            "Hello, this is about your account.",
            script_content={"prohibited_responses": prohibited_responses},
        )

        assert result.text == SAFE_DEFAULT_DEBTOR_RESPONSE, (
            "Expected generate_response to fall back to the safe default "
            f"line after exhausting all {total_attempts} attempts, all of "
            f"which matched a prohibited response, got {result.text!r} "
            f"instead of {SAFE_DEFAULT_DEBTOR_RESPONSE!r}"
        )
        assert not contains_prohibited_response(result.text, prohibited_responses), (
            "Expected the final returned text to never match a prohibited "
            f"response, but {result.text!r} matched one of "
            f"{prohibited_responses!r}"
        )
        assert service.llm_service.chat_completion.await_count == total_attempts

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        prohibited_responses=st.lists(distinctive_prohibited_text, min_size=1, max_size=5),
        clean_text=distinctive_prohibited_text,
        data=st.data(),
    )
    async def test_later_clean_regeneration_within_budget_is_returned(
        self, prohibited_responses, clean_text, data
    ):
        """When an earlier attempt's raw output matches a prohibited
        response but a later attempt (within the retry budget) does not,
        `generate_response` SHALL return that later clean text verbatim --
        not the safe default -- since a clean regeneration succeeded
        within budget."""
        assume(not contains_prohibited_response(clean_text, prohibited_responses))

        total_attempts = DebtorSimulatorService._MAX_PROHIBITED_RESPONSE_RETRIES + 1
        # Pick how many leading attempts are prohibited (at least 1, but
        # leaving room for the final clean attempt within budget).
        num_prohibited_leading = data.draw(st.integers(min_value=1, max_value=total_attempts - 1))

        leading_prohibited = data.draw(
            st.lists(
                st.sampled_from(prohibited_responses),
                min_size=num_prohibited_leading,
                max_size=num_prohibited_leading,
            )
        )
        # The clean attempt occupies the next slot; any remaining attempts
        # after it are never reached because the loop breaks on success, so
        # we only need `num_prohibited_leading + 1` outputs.
        raw_outputs = [*leading_prohibited, clean_text]

        service = _make_service_with_llm_outputs(raw_outputs)
        persona = _make_persona()

        result = await service.generate_response(
            persona,
            "Hello, this is about your account.",
            script_content={"prohibited_responses": prohibited_responses},
        )

        assert result.text == clean_text, (
            "Expected generate_response to return the later clean "
            f"regeneration {clean_text!r} verbatim, got {result.text!r} "
            f"instead (leading_prohibited={leading_prohibited!r})"
        )
        assert not contains_prohibited_response(result.text, prohibited_responses)
        assert service.llm_service.chat_completion.await_count == len(raw_outputs)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        prohibited_responses=st.lists(distinctive_prohibited_text, min_size=0, max_size=5),
        raw_output=distinctive_prohibited_text,
    )
    async def test_never_prohibited_output_returned_unmodified_with_single_call(
        self, prohibited_responses, raw_output
    ):
        """For any raw LLM output that never matches any prohibited
        response, `generate_response` SHALL return that raw output exactly
        (unmodified), and the LLM SHALL be called exactly once -- no
        unnecessary retries."""
        assume(not contains_prohibited_response(raw_output, prohibited_responses))

        service = _make_service_with_llm_outputs([raw_output])
        persona = _make_persona()

        result = await service.generate_response(
            persona,
            "Hello, this is about your account.",
            script_content={"prohibited_responses": prohibited_responses},
        )

        assert result.text == raw_output, (
            f"Expected generate_response to return the raw output "
            f"{raw_output!r} unmodified when it never matches a prohibited "
            f"response, got {result.text!r} instead"
        )
        assert service.llm_service.chat_completion.await_count == 1, (
            "Expected exactly one LLM call when the first attempt's output "
            f"is already clean, got {service.llm_service.chat_completion.await_count} "
            "calls instead"
        )


# --- Strategies for Property 22 (payment condition matching) ---

# Distinctive, alphanumeric-only condition text (reusing the same rationale
# as `distinctive_phrase_text` above): reduces the chance of accidental
# substring collisions between independently-generated payment conditions,
# and between a condition and random `safe_text` surrounding context, when
# constructing the embedding tests below.
distinctive_payment_condition_text = distinctive_phrase_text

distinctive_payment_condition_entries = st.fixed_dictionaries(
    {
        "condition": distinctive_payment_condition_text,
        "term": safe_text,
        "accepted": st.booleans(),
    }
)


class TestPaymentConditionMatching:
    """Property 22: Payment condition matches respond according to their
    configured term.

    Feature: ai-debtor-script-contract, Property 22: Payment condition
    matches respond according to their configured term

    For any loaded Script_Version and any agent input matching one of its
    Payment_Conditions entries, the simulator's response SHALL reflect that
    entry's configured accepted/rejected term.

    **Validates: Requirements 4.9**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.filter_too_much])
    @given(
        payment_conditions=st.lists(
            distinctive_payment_condition_entries, min_size=1, max_size=5
        ),
        prefix=safe_text,
        suffix=safe_text,
        case_variation_name=case_variation_names,
    )
    def test_message_containing_first_entrys_condition_returns_its_term_and_accepted(
        self, payment_conditions, prefix, suffix, case_variation_name
    ):
        """When `agent_message` deliberately contains
        `payment_conditions[0]`'s `condition` (embedded within surrounding
        text, in an arbitrary case variation), and none of the other
        entries' conditions are also present as substrings,
        `match_payment_condition` SHALL return exactly
        `(payment_conditions[0]["term"], payment_conditions[0]["accepted"])`
        -- covering both the accepted=True and accepted=False cases, so
        the returned term/accepted always reflects that specific entry's
        configuration, never an unconditional default."""
        first_entry = payment_conditions[0]
        case_fn = _CASE_VARIATIONS[case_variation_name]
        varied_condition = case_fn(first_entry["condition"])
        agent_message = f"{prefix} {varied_condition} {suffix}"

        lower_message = agent_message.lower()
        other_condition_also_present = any(
            entry["condition"].lower() in lower_message for entry in payment_conditions[1:]
        )
        assume(not other_condition_also_present)

        result = match_payment_condition(agent_message, payment_conditions)

        assert result == (first_entry["term"], first_entry["accepted"]), (
            "Expected match_payment_condition to return "
            f"payment_conditions[0]'s (term, accepted) = "
            f"({first_entry['term']!r}, {first_entry['accepted']!r}) for "
            f"agent_message {agent_message!r} (embedding condition "
            f"{first_entry['condition']!r} as {varied_condition!r}), got "
            f"{result!r} instead"
        )
        assert result is not None
        _, accepted = result
        assert accepted is first_entry["accepted"] or accepted == first_entry["accepted"], (
            "Expected the returned accepted flag to match exactly what was "
            f"configured ({first_entry['accepted']!r}), got {accepted!r} -- "
            "the response must reflect that specific entry's accepted term, "
            "not an unconditional always-True/always-False default."
        )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.filter_too_much])
    @given(
        payment_conditions=st.lists(
            distinctive_payment_condition_entries, min_size=1, max_size=5
        ),
        agent_message=safe_text,
    )
    def test_message_matching_no_condition_returns_none(
        self, payment_conditions, agent_message
    ):
        """For any `agent_message` that is guaranteed to contain none of
        `payment_conditions`' `condition` values as a substring
        (case-insensitive), `match_payment_condition` SHALL return `None`."""
        lower_message = agent_message.lower()
        assume(
            not any(
                entry["condition"].lower() in lower_message
                for entry in payment_conditions
            )
        )

        result = match_payment_condition(agent_message, payment_conditions)

        assert result is None, (
            f"Expected match_payment_condition to return None for "
            f"agent_message {agent_message!r} matching none of "
            f"{payment_conditions!r}, got {result!r} instead"
        )

    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        agent_message=safe_text,
        empty_payment_conditions=st.sampled_from([None, []]),
    )
    def test_empty_or_none_payment_conditions_returns_none(
        self, agent_message, empty_payment_conditions
    ):
        """For any `agent_message`, when `payment_conditions` is `None` or
        an empty list, `match_payment_condition` SHALL return `None`
        regardless of the message content."""
        result = match_payment_condition(agent_message, empty_payment_conditions)

        assert result is None, (
            f"Expected match_payment_condition(agent_message, "
            f"{empty_payment_conditions!r}) to return None, got {result!r} "
            f"for agent_message={agent_message!r}"
        )


# --- Strategies for Property 23 (conversation goal completion) ---

# Distinctive, alphanumeric-only completion_condition text (reusing the same
# rationale as `distinctive_phrase_text` above): reduces the chance of
# accidental substring collisions between an independently-generated
# `completion_condition` and random `safe_text` surrounding context when
# constructing the embedding tests below. `target_outcome` uses `safe_text`
# so it varies freely and is asserted to be returned verbatim.
distinctive_completion_condition_text = distinctive_phrase_text

distinctive_conversation_goal_dicts = st.fixed_dictionaries(
    {
        "target_outcome": safe_text,
        "completion_condition": distinctive_completion_condition_text,
    }
)

# `conversation_goal` dicts with a missing or empty `completion_condition`,
# covering both ways the field can be absent-of-content: the key omitted
# entirely, and the key present with an empty string value.
missing_or_empty_completion_condition_conversation_goals = st.one_of(
    st.builds(lambda target_outcome: {"target_outcome": target_outcome}, target_outcome=safe_text),
    st.builds(
        lambda target_outcome: {"target_outcome": target_outcome, "completion_condition": ""},
        target_outcome=safe_text,
    ),
)


class TestConversationGoalCompletion:
    """Property 23: Conversation goal completion ends the call according to
    its rule.

    Feature: ai-debtor-script-contract, Property 23: Conversation goal
    completion ends the call according to its rule

    For any loaded Script_Version and any conversation state satisfying its
    Conversation_Goal's completion_condition, the simulator SHALL end the
    Training_Call according to the ending rule specified in the
    Conversation_Goal (the target_outcome).

    **Validates: Requirements 4.10**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.filter_too_much])
    @given(
        conversation_goal=distinctive_conversation_goal_dicts,
        prefix=safe_text,
        suffix=safe_text,
        case_variation_name=case_variation_names,
    )
    def test_message_containing_completion_condition_returns_target_outcome(
        self, conversation_goal, prefix, suffix, case_variation_name
    ):
        """When `agent_message` deliberately contains the
        `conversation_goal`'s `completion_condition` (embedded within
        surrounding text, in an arbitrary case variation),
        `evaluate_conversation_goal_completion` SHALL return exactly the
        `conversation_goal`'s `target_outcome`."""
        case_fn = _CASE_VARIATIONS[case_variation_name]
        varied_condition = case_fn(conversation_goal["completion_condition"])
        agent_message = f"{prefix} {varied_condition} {suffix}"

        result = evaluate_conversation_goal_completion(agent_message, conversation_goal)

        assert result == conversation_goal["target_outcome"], (
            "Expected evaluate_conversation_goal_completion to return the "
            f"conversation_goal's target_outcome {conversation_goal['target_outcome']!r} "
            f"for agent_message {agent_message!r} (embedding completion_condition "
            f"{conversation_goal['completion_condition']!r} as {varied_condition!r}), "
            f"got {result!r} instead"
        )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.filter_too_much])
    @given(
        conversation_goal=distinctive_conversation_goal_dicts,
        agent_message=safe_text,
    )
    def test_message_not_containing_completion_condition_returns_none(
        self, conversation_goal, agent_message
    ):
        """For any `agent_message` that is guaranteed to not contain the
        `conversation_goal`'s `completion_condition` as a substring
        (case-insensitive), `evaluate_conversation_goal_completion` SHALL
        return `None`."""
        assume(
            conversation_goal["completion_condition"].lower() not in agent_message.lower()
        )

        result = evaluate_conversation_goal_completion(agent_message, conversation_goal)

        assert result is None, (
            f"Expected evaluate_conversation_goal_completion to return None "
            f"for agent_message {agent_message!r} not containing "
            f"completion_condition {conversation_goal['completion_condition']!r}, "
            f"got {result!r} instead"
        )

    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(agent_message=safe_text)
    def test_none_conversation_goal_returns_none(self, agent_message):
        """For any `agent_message`, when `conversation_goal` is `None`,
        `evaluate_conversation_goal_completion` SHALL return `None`
        regardless of the message content."""
        result = evaluate_conversation_goal_completion(agent_message, None)

        assert result is None, (
            f"Expected evaluate_conversation_goal_completion(agent_message, None) "
            f"to return None, got {result!r} for agent_message={agent_message!r}"
        )

    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        conversation_goal=missing_or_empty_completion_condition_conversation_goals,
        agent_message=safe_text,
    )
    def test_missing_or_empty_completion_condition_returns_none(
        self, conversation_goal, agent_message
    ):
        """For any `agent_message`, when the `conversation_goal`'s
        `completion_condition` is missing entirely or is an empty string,
        `evaluate_conversation_goal_completion` SHALL return `None`
        regardless of the message content."""
        result = evaluate_conversation_goal_completion(agent_message, conversation_goal)

        assert result is None, (
            f"Expected evaluate_conversation_goal_completion to return None "
            f"for conversation_goal with missing/empty completion_condition "
            f"{conversation_goal!r}, got {result!r} for agent_message={agent_message!r}"
        )


# --- Fixtures/helpers for Property 24 (DB-backed publish isolation) ---
#
# Adapted from `test_property_script_consumption.py` / from
# `test_property_script_lifecycle.py`'s conventions, self-contained here
# since this module otherwise only imports pure-function
# `debtor_simulator`/`llm_service` helpers.


@pytest.fixture
async def async_db():
    """Create an in-memory SQLite database for testing with foreign keys enabled."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)

    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as session:
        yield session

    # Disable FK enforcement before dropping tables: Script.current_version_id
    # and ScriptVersion.script_id form a circular FK relationship, and once
    # rows populate both sides (e.g. simulating a Published_Script), SQLite's
    # per-statement FK checking can otherwise reject DROP TABLE regardless of
    # drop order.
    async with engine.begin() as conn:
        await conn.execute(text("PRAGMA foreign_keys=OFF"))
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


def _make_admin_user(name: str = "Test Admin") -> User:
    """Create a valid Administrator `User` instance for `created_by` FKs."""
    return User(
        id=uuid.uuid4(),
        email=f"{uuid.uuid4()}@example.com",
        hashed_password="not-a-real-hash",
        full_name=name,
        role=UserRole.ADMIN.value,
        is_active=True,
    )


def _make_scenario(name: str = "Test Scenario") -> Scenario:
    """Create a valid `Scenario` instance for `scenario_id` FKs."""
    return Scenario(
        id=uuid.uuid4(),
        name=name,
        scenario_type="FINANCIAL_HARDSHIP",
        description="A test scenario for publish-isolation tests",
        debtor_profile={
            "name": "Maria Santos",
            "outstanding_balance": "5000.00",
            "days_past_due": 45,
            "personality_profile": "anxious",
            "conversation_goal": "negotiate payment plan",
        },
        is_active=True,
    )


def _make_mock_debtor_simulator() -> DebtorSimulatorService:
    """Create a mock DebtorSimulatorService that returns a valid persona.

    Mirrors `test_property_script_consumption.py`'s helper of the same
    name. `create_session` calls `generate_persona` as part of its
    persona-generation step; this test cares about `script_version_id`
    pinning, not persona content, so a fixed mock persona is sufficient.
    """
    mock_llm = AsyncMock()
    simulator = DebtorSimulatorService(llm_service=mock_llm)

    mock_persona = PersonaContext(
        persona_id=uuid.uuid4(),
        name="Maria Santos",
        communication_style="anxious",
        financial_circumstances={
            "income_level": "low",
            "debt_amount": 5000,
            "reason_for_delinquency": "job loss",
        },
        emotional_state=EmotionalState.DEFENSIVE,
        language="EN",
    )
    simulator.generate_persona = AsyncMock(return_value=mock_persona)
    return simulator


@st.composite
def two_distinct_script_contract_dicts(draw):
    """Generate two structurally valid Script_Contract-shaped dicts with
    distinct `opening_response` values, so version 1's and version 2's
    content can never be accidentally byte-for-byte identical -- ensuring
    an isolation-check that compares the pinned content against version 1
    (and not version 2) is meaningful for every generated example."""
    opening_response_1 = draw(safe_text)
    opening_response_2 = draw(safe_text.filter(lambda s: s != opening_response_1))

    contract_1 = draw(valid_script_contract_dicts(opening_response=st.just(opening_response_1)))
    contract_2 = draw(valid_script_contract_dicts(opening_response=st.just(opening_response_2)))

    format = draw(st.sampled_from(["json", "yaml"]))

    return contract_1, contract_2, format


def _serialize(contract_dict: dict, format: str) -> str:
    """Serialize a contract dict per `format`, matching `update_draft`'s
    expected `raw_definition` input."""
    if format == "json":
        return json.dumps(contract_dict)
    return yaml.safe_dump(contract_dict)


class TestPublishIsolationDuringActiveCall:
    """Property 24: An active Training_Call is isolated from later
    publishes of the same Script.

    Feature: ai-debtor-script-contract, Property 24: An active
    Training_Call is isolated from later publishes of the same Script

    For any loaded Script_Version and any conversation state, if an
    Administrator publishes a new Script_Version for the same Script
    while a Training_Call is active, the Debtor_Simulator SHALL continue
    using the Script_Version loaded at Training_Call start until the
    Training_Call ends, without notifying the Agent of the newer
    Script_Version during the active Training_Call.

    **Validates: Requirements 4.11**
    """

    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(case=two_distinct_script_contract_dicts())
    async def test_active_session_pinned_version_unaffected_by_later_publish(
        self, async_db: AsyncSession, case
    ):
        """For any Script published as version 1 and then consumed by a
        Training_Call (`create_session`), publishing a new version 2 for
        the SAME script while that Training_Call is still active SHALL
        NOT change the session's pinned `script_version_id`, and
        re-fetching that pinned `ScriptVersion` SHALL still return
        version 1's original content byte-for-byte -- even though the
        Script's `current_version_id` has moved on to version 2, proving
        the publish itself succeeded (isolation is about the session not
        seeing the change, not about the publish failing)."""
        contract_1, contract_2, format = case

        admin = _make_admin_user()
        scenario = _make_scenario()
        async_db.add_all([admin, scenario])
        await async_db.flush()

        script = await create_draft(
            async_db,
            admin_id=admin.id,
            name="Isolation Test Script",
            scenario_id=scenario.id,
            format=format,
            raw_definition=_serialize(contract_1, format),
        )
        version_1 = await publish(async_db, script_id=script.id, admin_id=admin.id)
        version_1_id = version_1.id
        version_1_content = version_1.content

        # Start the Training_Call: create_session pins script_version_id
        # to version 1's id at call start.
        simulator = _make_mock_debtor_simulator()
        session = await create_session(async_db, scenario.id, uuid.uuid4(), simulator)

        assert session.script_version_id == version_1_id, (
            "Expected create_session to pin script_version_id to version "
            f"1's id {version_1_id!r}, got {session.script_version_id!r}"
        )

        # While the call is "active" (session not ended), an
        # Administrator publishes a NEW version for the SAME script.
        await update_draft(
            async_db,
            script_id=script.id,
            raw_definition=_serialize(contract_2, format),
            format=format,
        )
        version_2 = await publish(async_db, script_id=script.id, admin_id=admin.id)

        assert version_2.id != version_1_id, (
            "Expected the mid-call publish to create a NEW ScriptVersion "
            f"distinct from version 1, got the same id {version_1_id!r}"
        )

        # (a) The session's script_version_id, re-fetched from the DB,
        # is STILL version 1's id -- unchanged by the later publish.
        await async_db.refresh(session)
        assert session.script_version_id == version_1_id, (
            "Expected the active session's script_version_id to remain "
            f"pinned to version 1's id {version_1_id!r} after a mid-call "
            f"publish, but it changed to {session.script_version_id!r}"
        )

        # (b) Re-fetching the ScriptVersion by the session's pinned id
        # returns version 1's original content, NOT version 2's.
        pinned_version = await async_db.get(ScriptVersion, session.script_version_id)
        assert pinned_version is not None
        assert pinned_version.content == version_1_content, (
            "Expected the pinned ScriptVersion's content to still match "
            f"version 1's original content {version_1_content!r}, got "
            f"{pinned_version.content!r}"
        )
        assert pinned_version.content != version_2.content, (
            "Expected the pinned ScriptVersion's content to differ from "
            f"version 2's content {version_2.content!r} (they were "
            "generated with distinct opening_response values), but they "
            f"matched: {pinned_version.content!r}"
        )

        # (c) Script.current_version_id now points at version 2,
        # confirming the publish DID succeed and update the Script's
        # "latest" pointer.
        refetched_script = await async_db.get(Script, script.id)
        assert refetched_script is not None
        assert refetched_script.current_version_id == version_2.id, (
            "Expected Script.current_version_id to point at version 2 "
            f"after the mid-call publish, got "
            f"{refetched_script.current_version_id!r} instead of "
            f"{version_2.id!r}"
        )

        # Optional contrast: get_active_published_version (the "for new
        # sessions" view) now returns version 2, while the EXISTING
        # session's pinned version remains version 1.
        active_version_for_new_sessions = await get_active_published_version(
            async_db, scenario.id
        )
        assert active_version_for_new_sessions is not None
        assert active_version_for_new_sessions.id == version_2.id, (
            "Expected get_active_published_version to reflect the newer "
            f"publish (version 2, id {version_2.id!r}) for any NEW "
            "session start, got "
            f"{active_version_for_new_sessions.id!r} instead"
        )
        assert session.script_version_id != active_version_for_new_sessions.id, (
            "Expected the existing active session's pinned version to "
            "diverge from what a brand-new session would now receive -- "
            f"the existing session stayed on version 1 "
            f"({session.script_version_id!r}) while new sessions would "
            f"get version 2 ({active_version_for_new_sessions.id!r})"
        )

        await async_db.rollback()

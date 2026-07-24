"""Property-based tests for Script definition format handling.

Feature: ai-debtor-script-contract, Property 5: JSON and YAML round-trip acceptance

**Validates: Requirements 2.1**

Property 5: For any structurally valid Script_Contract, serializing it to JSON
and submitting it, and separately serializing it to YAML and submitting it,
SHALL both be accepted and SHALL both parse to structurally equivalent
contract data.
"""

import json

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from app.schemas.script import ScriptContract
from app.services.script_validator import ScriptFormatError, parse_script_definition

# --- Strategies ---

# Free-text strategy: printable letters/numbers plus a few safe punctuation
# characters, with leading/trailing whitespace stripped so the value is
# guaranteed to survive both JSON and YAML round-tripping unchanged.
safe_text = (
    st.text(
        alphabet=st.characters(
            whitelist_categories=("L", "N"),
            whitelist_characters=" .,!?-",
        ),
        min_size=1,
        max_size=50,
    )
    .map(lambda s: s.strip())
    .filter(lambda s: s != "")
)

trigger_phrase_entries = st.fixed_dictionaries(
    {
        "phrase": safe_text,
        "behavior": safe_text,
    }
)

expected_reply_entries = st.fixed_dictionaries(
    {
        "agent_statement": safe_text,
        "debtor_reply": safe_text,
    }
)

emotional_state_rules = st.fixed_dictionaries(
    {
        "trigger": safe_text,
        "state_change": safe_text,
    }
)

escalation_condition_entries = st.fixed_dictionaries(
    {
        "condition": safe_text,
        "behavior": safe_text,
        "ends_call": st.booleans(),
    }
)

payment_condition_entries = st.fixed_dictionaries(
    {
        "condition": safe_text,
        "term": safe_text,
        "accepted": st.booleans(),
    }
)

debtor_personas = st.fixed_dictionaries(
    {
        "name": safe_text,
        "communication_style": safe_text,
        "background": safe_text,
    }
)

financial_situations = st.fixed_dictionaries(
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

conversation_goals = st.fixed_dictionaries(
    {
        "target_outcome": safe_text,
        "completion_condition": safe_text,
    }
)


@st.composite
def valid_script_contract_dicts(draw):
    """Generate a structurally valid Script_Contract-shaped dict.

    `prohibited_responses` is intentionally kept empty so that no generated
    example can accidentally trigger the Prohibited/Expected conflict rule
    (Property 4) — this property is only concerned with format round-trip,
    not conflict detection.
    """
    return {
        "debtor_persona": draw(debtor_personas),
        "financial_situation": draw(financial_situations),
        "opening_response": draw(safe_text),
        "expected_replies": draw(st.lists(expected_reply_entries, min_size=0, max_size=5)),
        "trigger_phrases": draw(st.lists(trigger_phrase_entries, min_size=0, max_size=5)),
        "emotional_state_rules": draw(st.lists(emotional_state_rules, min_size=0, max_size=5)),
        "payment_conditions": draw(st.lists(payment_condition_entries, min_size=0, max_size=5)),
        "escalation_conditions": draw(
            st.lists(escalation_condition_entries, min_size=0, max_size=5)
        ),
        "prohibited_responses": [],
        "conversation_goal": draw(conversation_goals),
    }


class TestScriptFormatRoundTrip:
    """Property 5: JSON and YAML round-trip acceptance.

    Feature: ai-debtor-script-contract, Property 5: JSON and YAML round-trip
    acceptance

    For any structurally valid Script_Contract, serializing it to JSON and
    submitting it, and separately serializing it to YAML and submitting it,
    SHALL both be accepted and SHALL both parse to structurally equivalent
    contract data.
    """

    @given(contract_dict=valid_script_contract_dicts())
    @settings(max_examples=100)
    def test_json_and_yaml_round_trip_to_equivalent_data(self, contract_dict):
        """JSON and YAML serializations of the same contract parse identically."""
        json_text = json.dumps(contract_dict)
        yaml_text = yaml.dump(contract_dict, default_flow_style=False, allow_unicode=True)

        # Both formats SHALL be accepted (parse without raising).
        parsed_from_json = parse_script_definition(json_text, "json")
        parsed_from_yaml = parse_script_definition(yaml_text, "yaml")

        # Both SHALL parse back to the original, structurally equivalent data.
        assert parsed_from_json == contract_dict
        assert parsed_from_yaml == contract_dict
        assert parsed_from_json == parsed_from_yaml

        # Both SHALL be accepted as structurally valid Script_Contract data.
        ScriptContract(**parsed_from_json)
        ScriptContract(**parsed_from_yaml)


# --- Property 6 strategies ---

# Arbitrary format strings that are not the exact (case-sensitive) supported
# values "json" or "yaml". Includes near-misses (different case, whitespace,
# substrings), empty strings, and arbitrary text.
unsupported_format_strings = st.text(max_size=20).filter(
    lambda s: s != "json" and s != "yaml"
)

# Arbitrary raw_text content: empty strings, garbage text, and even text that
# would be valid JSON/YAML if it were parsed under the right format — the
# format check must reject regardless of content.
arbitrary_raw_text = st.one_of(
    st.just(""),
    st.text(max_size=200),
    st.just('{"a": 1}'),
    st.just("a: 1\nb: 2\n"),
    st.just("not valid json or yaml: [[["),
)


class TestUnsupportedScriptFormatRejection:
    """Property 6: Unsupported formats are always rejected.

    Feature: ai-debtor-script-contract, Property 6: Unsupported formats are
    always rejected

    For any script submission whose declared format is not "json" and not
    "yaml", the Script_Registry SHALL reject the submission and identify the
    unsupported format, regardless of the content submitted.

    **Validates: Requirements 2.2**
    """

    @given(format=unsupported_format_strings, raw_text=arbitrary_raw_text)
    @settings(max_examples=100)
    def test_unsupported_format_always_rejected(self, format, raw_text):
        """Any format other than exactly 'json' or 'yaml' raises ScriptFormatError
        identifying the unsupported format, regardless of raw_text content."""
        with pytest.raises(ScriptFormatError) as exc_info:
            parse_script_definition(raw_text, format)

        # The unsupported format SHALL be identified in the error.
        assert repr(format) in str(exc_info.value) or format in str(exc_info.value)

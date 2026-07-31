"""Unit tests for the Script definition validator pipeline
(app/services/script_validator.py).

Covers the full `validate_script` pipeline end-to-end with concrete
examples: valid JSON, valid YAML, a missing required field, an
unsupported format, malformed JSON, and malformed YAML.
"""

import json

import pytest
import yaml

from app.schemas.script import ScriptContract
from app.services.script_validator import (
    ScriptFormatError,
    ScriptLimits,
    ScriptValidationError,
    validate_script,
)

# A generous ScriptLimits instance so limit checks never interfere with
# these tests (they only exercise parse/structure/format behavior).
GENEROUS_LIMITS = ScriptLimits(
    max_definition_size_bytes=1_000_000,
    max_trigger_phrases=100,
    max_expected_replies=100,
    max_escalation_conditions=100,
    max_field_text_length=2000,
)


def _valid_contract_dict() -> dict:
    """A fully valid Script_Contract-shaped dict, matching every required
    field in app/schemas/script.py."""
    return {
        "debtor_persona": {
            "name": "Jordan Lee",
            "communication_style": "Calm but evasive",
            "background": "Recently lost a part-time job",
        },
        "financial_situation": {
            "outstanding_balance": 1250.50,
            "days_past_due": 45,
            "reason_for_delinquency": "Reduced income after job loss",
        },
        "opening_response": "Hello, who is this calling?",
        "expected_replies": [
            {
                "agent_statement": "I'm calling about your outstanding balance.",
                "debtor_reply": "I know, I've been meaning to call you back.",
            }
        ],
        "trigger_phrases": [
            {
                "phrase": "I can't pay right now",
                "behavior": "Becomes defensive and asks for more time",
            }
        ],
        "emotional_state_rules": [
            {
                "trigger": "Agent raises voice",
                "state_change": "Debtor becomes anxious",
            }
        ],
        "payment_conditions": [
            {
                "condition": "Offered a payment plan",
                "term": "50 dollars per month for six months",
                "accepted": True,
            }
        ],
        "escalation_conditions": [
            {
                "condition": "Debtor becomes hostile",
                "behavior": "Escalate to a supervisor",
                "ends_call": False,
            }
        ],
        "prohibited_responses": ["I will never pay you anything"],
        "conversation_goal": {
            "target_outcome": "Secure a payment commitment",
            "completion_condition": "Debtor agrees to a payment plan",
        },
    }


class TestValidateScriptValidSubmissions:
    """A fully valid contract SHALL be accepted in both JSON and YAML form."""

    def test_valid_json_contract_is_accepted(self):
        raw_text = json.dumps(_valid_contract_dict())

        result = validate_script(raw_text, "json", GENEROUS_LIMITS)

        assert isinstance(result, ScriptContract)
        assert result.opening_response == "Hello, who is this calling?"

    def test_valid_yaml_contract_is_accepted(self):
        raw_text = yaml.dump(
            _valid_contract_dict(), default_flow_style=False, allow_unicode=True
        )

        result = validate_script(raw_text, "yaml", GENEROUS_LIMITS)

        assert isinstance(result, ScriptContract)
        assert result.opening_response == "Hello, who is this calling?"


class TestValidateScriptStructuralFailure:
    """A contract missing exactly one required field SHALL be rejected."""

    def test_missing_required_field_raises_validation_error(self):
        contract_dict = _valid_contract_dict()
        del contract_dict["conversation_goal"]
        raw_text = json.dumps(contract_dict)

        with pytest.raises(ScriptValidationError) as exc_info:
            validate_script(raw_text, "json", GENEROUS_LIMITS)

        error_locs = [error["loc"] for error in exc_info.value.errors]
        assert ("conversation_goal",) in error_locs


class TestValidateScriptFormatFailures:
    """Unsupported formats and unparseable content SHALL be rejected with
    ScriptFormatError."""

    def test_unsupported_format_raises_format_error(self):
        raw_text = json.dumps(_valid_contract_dict())

        with pytest.raises(ScriptFormatError):
            validate_script(raw_text, "xml", GENEROUS_LIMITS)

    def test_malformed_json_raises_format_error(self):
        # Truncated JSON: missing closing brace/bracket.
        raw_text = '{"debtor_persona": {"name": "Jordan"'

        with pytest.raises(ScriptFormatError):
            validate_script(raw_text, "json", GENEROUS_LIMITS)

    def test_unparseable_yaml_raises_format_error(self):
        # Invalid YAML syntax: unbalanced/inconsistent indentation with a
        # dangling mapping key that yaml.safe_load cannot parse.
        raw_text = "debtor_persona:\n  name: Jordan\n key: [unclosed"

        with pytest.raises(ScriptFormatError):
            validate_script(raw_text, "yaml", GENEROUS_LIMITS)

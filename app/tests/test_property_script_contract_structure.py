"""Property-based tests for Script_Contract structural validation.

Feature: ai-debtor-script-contract

This module is shared/appended to across several sub-tasks (3.5, 3.6, 3.7,
3.9), each contributing one test class for one of the correctness
properties defined in `design.md`:

    - Property 1 (this task, 3.5): Missing required fields are always
      rejected and fully reported
    - Property 2 (task 3.6): Required substructure completeness for
      singleton object fields
    - Property 3 (task 3.7): List-entry substructure completeness
    - Property 4 (task 3.9): Prohibited/Expected conflict detection

The module-level Hypothesis strategies below (particularly
`valid_script_contract_dicts`) are the shared base: a structurally valid
Script_Contract-shaped dict generator, adapted from
`test_property_script_format.py`'s strategy. Later test classes appended
to this file should reuse these strategies rather than redefining their
own copies.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.services.script_validator import (
    ScriptValidationError,
    validate_conflicts,
    validate_contract_structure,
)

# --- Shared strategies (base: adapted from test_property_script_format.py) ---

# Free-text strategy: printable letters/numbers plus a few safe punctuation
# characters, with leading/trailing whitespace stripped so the value is
# guaranteed non-empty and well-formed after stripping.
safe_text = (
    st.text(
        # Restricted to ASCII letters/digits (rather than the full Unicode
        # "L"/"N" categories) to avoid characters like the Turkish dotless
        # i ('\u0131') whose `.upper().casefold()` does not round-trip back
        # to `.casefold()` under Python's default (non-Turkish-locale) case
        # folding rules — tests in this module build case-variant strings
        # (e.g. `.upper()`) and rely on trimmed/case-insensitive comparisons
        # remaining consistent.
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

emotional_state_rule_entries = st.fixed_dictionaries(
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

debtor_persona_dicts = st.fixed_dictionaries(
    {
        "name": safe_text,
        "communication_style": safe_text,
        "background": safe_text,
    }
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
    {
        "target_outcome": safe_text,
        "completion_condition": safe_text,
    }
)

# The ten required top-level Script_Contract fields (Requirement 1.1).
TOP_LEVEL_REQUIRED_FIELDS = [
    "debtor_persona",
    "financial_situation",
    "opening_response",
    "expected_replies",
    "trigger_phrases",
    "emotional_state_rules",
    "payment_conditions",
    "escalation_conditions",
    "prohibited_responses",
    "conversation_goal",
]


@st.composite
def valid_script_contract_dicts(draw, min_expected_replies: int = 0):
    """Generate a structurally valid Script_Contract-shaped dict.

    `prohibited_responses` is kept empty by default so no generated example
    accidentally triggers the Prohibited/Expected conflict rule (Property 4)
    unless a test explicitly opts into a conflict scenario.

    Args:
        min_expected_replies: minimum number of `expected_replies` entries
            to generate. Tests that need to force a Prohibited/Expected
            conflict should pass ``min_expected_replies=1`` so there is
            always at least one `debtor_reply` to duplicate.
    """
    return {
        "debtor_persona": draw(debtor_persona_dicts),
        "financial_situation": draw(financial_situation_dicts),
        "opening_response": draw(safe_text),
        "expected_replies": draw(
            st.lists(expected_reply_entries, min_size=min_expected_replies, max_size=5)
        ),
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


def _reported_top_level_field_paths(errors):
    """Extract the set of top-level field names present in a
    ScriptValidationError's `errors` list (Pydantic `.errors()` format).

    Errors whose `loc` is empty (e.g. a model-level `model_validator`
    failure such as the Prohibited/Expected conflict check) are ignored
    here since they do not identify a single top-level field.
    """
    return {str(error["loc"][0]) for error in errors if error["loc"]}


class TestMissingRequiredFieldsReporting:
    """Property 1: Missing required fields are always rejected and fully reported.

    Feature: ai-debtor-script-contract, Property 1: Missing required fields
    are always rejected and fully reported

    For any script submission missing a non-empty subset of the ten
    required top-level fields, validation SHALL reject the submission and
    the reported errors SHALL include every missing field name — even when
    the submission simultaneously contains a Prohibited_Responses/
    Expected_Replies conflict, validation SHALL still report all missing
    fields (it does not short-circuit).

    **Validates: Requirements 1.1, 1.2**
    """

    @given(
        contract_dict=valid_script_contract_dicts(),
        fields_to_remove=st.lists(
            st.sampled_from(TOP_LEVEL_REQUIRED_FIELDS),
            min_size=1,
            max_size=len(TOP_LEVEL_REQUIRED_FIELDS),
            unique=True,
        ),
    )
    @settings(max_examples=100)
    def test_missing_fields_are_rejected_and_all_reported(self, contract_dict, fields_to_remove):
        """Removing any non-empty subset of the 10 required fields is rejected,
        and every removed field name appears in the reported errors."""
        for field in fields_to_remove:
            contract_dict.pop(field, None)

        with pytest.raises(ScriptValidationError) as exc_info:
            validate_contract_structure(contract_dict)

        reported_field_paths = _reported_top_level_field_paths(exc_info.value.errors)

        for field in fields_to_remove:
            assert field in reported_field_paths, (
                f"Expected missing field {field!r} to be reported, got: "
                f"{reported_field_paths}"
            )

    @given(
        contract_dict=valid_script_contract_dicts(min_expected_replies=1),
        fields_to_remove=st.lists(
            st.sampled_from(
                [
                    field
                    for field in TOP_LEVEL_REQUIRED_FIELDS
                    if field not in ("expected_replies", "prohibited_responses")
                ]
            ),
            min_size=1,
            max_size=len(TOP_LEVEL_REQUIRED_FIELDS) - 2,
            unique=True,
        ),
    )
    @settings(max_examples=100)
    def test_missing_fields_still_fully_reported_alongside_conflict(
        self, contract_dict, fields_to_remove
    ):
        """When the submission simultaneously has a Prohibited_Responses/
        Expected_Replies conflict AND missing fields, every missing field
        is still reported — validation does not short-circuit on the
        conflict (or vice versa)."""
        # Force a Prohibited_Responses / Expected_Replies conflict by
        # duplicating an existing expected debtor_reply.
        conflicting_reply = contract_dict["expected_replies"][0]["debtor_reply"]
        contract_dict["prohibited_responses"] = [conflicting_reply]

        for field in fields_to_remove:
            contract_dict.pop(field, None)

        with pytest.raises(ScriptValidationError) as exc_info:
            validate_contract_structure(contract_dict)

        reported_field_paths = _reported_top_level_field_paths(exc_info.value.errors)

        for field in fields_to_remove:
            assert field in reported_field_paths, (
                f"Expected missing field {field!r} to still be reported "
                f"alongside the conflict, got: {reported_field_paths}"
            )


# --- Singleton substructure completeness (task 3.6) ---

# Required sub-fields for each singleton object field, per Requirements
# 1.3 (debtor_persona), 1.4 (financial_situation), 1.8 (conversation_goal).
SINGLETON_REQUIRED_SUBFIELDS = {
    "debtor_persona": ["name", "communication_style", "background"],
    "financial_situation": ["outstanding_balance", "days_past_due", "reason_for_delinquency"],
    "conversation_goal": ["target_outcome", "completion_condition"],
}

# Required sub-fields for each emotional_state_rules list entry, per
# Requirement 1.6.
EMOTIONAL_STATE_RULE_SUBFIELDS = ["trigger", "state_change"]


@st.composite
def valid_contract_with_emotional_rule_dicts(draw):
    """Generate a valid Script_Contract dict guaranteed to have at least
    one `emotional_state_rules` entry, so a required sub-field can always
    be removed from a real entry."""
    contract_dict = draw(valid_script_contract_dicts())
    if not contract_dict["emotional_state_rules"]:
        contract_dict["emotional_state_rules"] = [draw(emotional_state_rule_entries)]
    return contract_dict


def _reported_field_paths(errors):
    """Extract the set of full dotted field paths (all `loc` parts, not
    just the top-level one) present in a ScriptValidationError's `errors`
    list (Pydantic `.errors()` format)."""
    return {".".join(str(part) for part in error["loc"]) for error in errors if error["loc"]}


class TestSingletonSubstructureCompleteness:
    """Property 2: Required substructure completeness for singleton object
    fields.

    Feature: ai-debtor-script-contract, Property 2: Required substructure
    completeness for singleton object fields

    For any script where Debtor_Persona, Financial_Situation, or
    Conversation_Goal is present but missing one of its required
    sub-fields, validation SHALL reject the submission identifying the
    missing sub-field, and SHALL accept the submission when all
    sub-fields are present and valid. Emotional_State_Rules entries
    (trigger/state_change) are tested with the same completeness
    property, treated per the schema as list entries each with a fixed
    required sub-field set.

    **Validates: Requirements 1.3, 1.4, 1.6, 1.8**
    """

    @given(
        contract_dict=valid_script_contract_dicts(),
        singleton_field=st.sampled_from(list(SINGLETON_REQUIRED_SUBFIELDS.keys())),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_missing_singleton_subfield_is_rejected_and_identified(
        self, contract_dict, singleton_field, data
    ):
        """Removing one required sub-field from debtor_persona,
        financial_situation, or conversation_goal is rejected, and the
        reported errors identify the specific missing sub-field path
        (e.g. "debtor_persona.name")."""
        subfield = data.draw(st.sampled_from(SINGLETON_REQUIRED_SUBFIELDS[singleton_field]))
        del contract_dict[singleton_field][subfield]

        with pytest.raises(ScriptValidationError) as exc_info:
            validate_contract_structure(contract_dict)

        reported_paths = _reported_field_paths(exc_info.value.errors)
        expected_path = f"{singleton_field}.{subfield}"

        assert expected_path in reported_paths, (
            f"Expected missing sub-field path {expected_path!r} to be "
            f"reported, got: {reported_paths}"
        )

    @given(
        contract_dict=valid_contract_with_emotional_rule_dicts(),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_missing_emotional_state_rule_subfield_is_rejected_and_identified(
        self, contract_dict, data
    ):
        """Removing one required sub-field (trigger or state_change) from
        an emotional_state_rules entry is rejected, and the reported
        errors identify the malformed entry (index + sub-field path)."""
        entries = contract_dict["emotional_state_rules"]
        entry_index = data.draw(st.integers(min_value=0, max_value=len(entries) - 1))
        subfield = data.draw(st.sampled_from(EMOTIONAL_STATE_RULE_SUBFIELDS))

        del entries[entry_index][subfield]

        with pytest.raises(ScriptValidationError) as exc_info:
            validate_contract_structure(contract_dict)

        reported_paths = _reported_field_paths(exc_info.value.errors)
        expected_path = f"emotional_state_rules.{entry_index}.{subfield}"

        assert expected_path in reported_paths, (
            f"Expected missing sub-field path {expected_path!r} to be "
            f"reported, got: {reported_paths}"
        )

    @given(contract_dict=valid_script_contract_dicts())
    @settings(max_examples=100)
    def test_all_subfields_present_is_accepted(self, contract_dict):
        """When all required sub-fields are present and valid across
        debtor_persona, financial_situation, conversation_goal, and every
        emotional_state_rules entry, validation succeeds (no exception)."""
        validate_contract_structure(contract_dict)


# --- List-entry substructure completeness (task 3.7) ---

# Required sub-fields for each of the four list-entry fields, per
# Requirements 1.5 (trigger_phrases), 1.7 (escalation_conditions), and the
# expected_replies/payment_conditions analogues also covered by this
# property.
LIST_ENTRY_REQUIRED_SUBFIELDS = {
    "trigger_phrases": ["phrase", "behavior"],
    "expected_replies": ["agent_statement", "debtor_reply"],
    "escalation_conditions": ["condition", "behavior", "ends_call"],
    "payment_conditions": ["condition", "term", "accepted"],
}

# Strategy used to generate a fresh entry for each list field when a
# contract needs to be topped up to guarantee non-empty lists.
_LIST_ENTRY_STRATEGIES = {
    "trigger_phrases": trigger_phrase_entries,
    "expected_replies": expected_reply_entries,
    "escalation_conditions": escalation_condition_entries,
    "payment_conditions": payment_condition_entries,
}


@st.composite
def valid_contract_with_populated_lists_dicts(draw):
    """Generate a valid Script_Contract dict guaranteed to have at least
    one entry in each of trigger_phrases, expected_replies,
    escalation_conditions, and payment_conditions, so a required
    sub-field can always be removed from a real entry in any of them."""
    contract_dict = draw(valid_script_contract_dicts())
    for field, strategy in _LIST_ENTRY_STRATEGIES.items():
        if not contract_dict[field]:
            contract_dict[field] = [draw(strategy)]
    return contract_dict


class TestListEntrySubstructureCompleteness:
    """Property 3: List-entry substructure completeness.

    Feature: ai-debtor-script-contract, Property 3: List-entry
    substructure completeness

    For any script where an entry in Trigger_Phrases, Expected_Replies,
    Escalation_Conditions, or Payment_Conditions is present but missing
    one of its required sub-fields (phrase/behavior;
    agent_statement/debtor_reply; condition/behavior/ends_call;
    condition/term/accepted), validation SHALL reject the submission
    identifying the malformed entry and missing sub-field, and SHALL
    accept the submission when every entry in these lists has all
    required sub-fields.

    **Validates: Requirements 1.5, 1.7, 1.10**
    """

    @given(
        contract_dict=valid_contract_with_populated_lists_dicts(),
        list_field=st.sampled_from(list(LIST_ENTRY_REQUIRED_SUBFIELDS.keys())),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_missing_list_entry_subfield_is_rejected_and_identified(
        self, contract_dict, list_field, data
    ):
        """Removing one required sub-field from an entry in
        trigger_phrases, expected_replies, escalation_conditions, or
        payment_conditions is rejected, and the reported errors identify
        the malformed entry's indexed path (e.g. "trigger_phrases.0.phrase")."""
        entries = contract_dict[list_field]
        entry_index = data.draw(st.integers(min_value=0, max_value=len(entries) - 1))
        subfield = data.draw(st.sampled_from(LIST_ENTRY_REQUIRED_SUBFIELDS[list_field]))

        del entries[entry_index][subfield]

        with pytest.raises(ScriptValidationError) as exc_info:
            validate_contract_structure(contract_dict)

        reported_paths = _reported_field_paths(exc_info.value.errors)
        expected_path = f"{list_field}.{entry_index}.{subfield}"

        assert expected_path in reported_paths, (
            f"Expected missing sub-field path {expected_path!r} to be "
            f"reported, got: {reported_paths}"
        )

    @given(contract_dict=valid_contract_with_populated_lists_dicts())
    @settings(max_examples=100)
    def test_all_list_entries_with_subfields_present_is_accepted(self, contract_dict):
        """When every entry in trigger_phrases, expected_replies,
        escalation_conditions, and payment_conditions has all required
        sub-fields present and valid, validation succeeds (no exception)."""
        validate_contract_structure(contract_dict)


# --- Prohibited/Expected conflict detection (task 3.9) ---


class TestProhibitedExpectedConflictDetection:
    """Property 4: Prohibited/Expected conflict detection.

    Feature: ai-debtor-script-contract, Property 4: Prohibited/Expected
    conflict detection

    For any script where a Prohibited_Responses entry textually
    duplicates (after trim/case normalization) an Expected_Replies[*].
    debtor_reply entry, validation SHALL reject the submission and
    identify the conflicting entries; for any script where no such
    duplication exists, this check alone SHALL NOT cause rejection.

    Exercised end-to-end via `validate_contract_structure` (the dict-level
    whole pipeline), which internally triggers `ScriptContract`'s own
    `model_validator` conflict check (`app/schemas/script.py`, task 2.1).

    **Validates: Requirements 1.9**
    """

    @given(
        contract_dict=valid_script_contract_dicts(min_expected_replies=1),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_conflicting_prohibited_response_is_rejected_and_identified(
        self, contract_dict, data
    ):
        """A prohibited_responses entry that textually duplicates (after
        trim/case normalization) an expected_replies[*].debtor_reply
        entry is rejected, and the reported error identifies the
        conflicting entry."""
        entry_index = data.draw(
            st.integers(min_value=0, max_value=len(contract_dict["expected_replies"]) - 1)
        )
        conflicting_reply = contract_dict["expected_replies"][entry_index]["debtor_reply"]

        # Vary whitespace/case so the duplication is only apparent after
        # trim/case normalization, not necessarily byte-for-byte.
        variant = data.draw(
            st.sampled_from(
                [
                    conflicting_reply,
                    conflicting_reply.upper(),
                    conflicting_reply.lower(),
                    f"  {conflicting_reply}  ",
                    f"\t{conflicting_reply}\n",
                ]
            )
        )
        contract_dict["prohibited_responses"] = [variant]

        with pytest.raises(ScriptValidationError) as exc_info:
            validate_contract_structure(contract_dict)

        error_texts = " ".join(
            f"{error.get('msg', '')} {error.get('ctx', '')}" for error in exc_info.value.errors
        )
        assert conflicting_reply in error_texts or variant.strip() in error_texts, (
            f"Expected the conflicting entry {conflicting_reply!r} to be "
            f"identified in the reported errors, got: {exc_info.value.errors}"
        )

    @given(
        contract_dict=valid_script_contract_dicts(min_expected_replies=1),
        candidate_prohibited=st.lists(safe_text, min_size=0, max_size=5),
    )
    @settings(max_examples=100)
    def test_non_conflicting_prohibited_responses_are_accepted(
        self, contract_dict, candidate_prohibited
    ):
        """When no prohibited_responses entry duplicates any
        expected_replies[*].debtor_reply entry (after trim/case
        normalization), the conflict check alone does not cause
        rejection."""
        expected_replies_normalized = {
            entry["debtor_reply"].strip().casefold() for entry in contract_dict["expected_replies"]
        }

        # Guarantee disjointness by construction: drop any generated
        # candidate that happens to normalize to an existing debtor_reply,
        # so this test can never accidentally exercise the conflict case.
        contract_dict["prohibited_responses"] = [
            candidate
            for candidate in candidate_prohibited
            if candidate.strip().casefold() not in expected_replies_normalized
        ]

        validate_contract_structure(contract_dict)

    @given(
        contract_dict=valid_script_contract_dicts(min_expected_replies=1),
        data=st.data(),
    )
    @settings(max_examples=50)
    def test_validate_conflicts_directly_identifies_conflicting_pair(self, contract_dict, data):
        """Sanity check: `validate_conflicts` (the standalone, non-raising
        equivalent used by the aggregator) reports the same conflicting
        pair directly on an already-constructed contract, without going
        through the raising `model_validator` path."""
        entry_index = data.draw(
            st.integers(min_value=0, max_value=len(contract_dict["expected_replies"]) - 1)
        )
        conflicting_reply = contract_dict["expected_replies"][entry_index]["debtor_reply"]
        variant = f"  {conflicting_reply.upper()}  "

        # Build a contract with no conflict first (so ScriptContract
        # construction itself doesn't raise), then attach the conflicting
        # prohibited response directly to the model instance to exercise
        # `validate_conflicts` in isolation.
        contract_dict["prohibited_responses"] = []
        contract = validate_contract_structure(contract_dict)
        contract.prohibited_responses = [variant]

        conflicts = validate_conflicts(contract)

        assert (conflicting_reply, variant) in conflicts, (
            f"Expected validate_conflicts to report ({conflicting_reply!r}, "
            f"{variant!r}), got: {conflicts}"
        )

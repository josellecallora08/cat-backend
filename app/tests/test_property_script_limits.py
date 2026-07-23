"""Property-based tests for Script definition configurable limit enforcement.

Feature: ai-debtor-script-contract

This module is shared/appended to across several sub-tasks (3.11, 3.12,
3.13), each contributing one test class for one of the correctness
properties defined in `design.md`:

    - Property 7 (this task, 3.11): Definition size limit enforcement
    - Property 8 (task 3.12): Entry-count limit enforcement
    - Property 9 (task 3.13): Free-text length limit enforcement

The module-level Hypothesis strategies below are adapted from
`test_property_script_contract_structure.py`'s shared base (particularly
`valid_script_contract_dicts`/`safe_text`), reused here so a structurally
valid `ScriptContract` can be built and checked against `validate_limits`
directly, without needing the full `validate_script` pipeline. Later test
classes appended to this file should reuse these strategies rather than
redefining their own copies.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from app.schemas.script import ScriptContract
from app.services.script_validator import ScriptLimits, validate_limits

# --- Shared strategies (adapted from test_property_script_contract_structure.py) ---

# Free-text strategy: printable letters/numbers plus a few safe punctuation
# characters, with leading/trailing whitespace stripped so the value is
# guaranteed non-empty and well-formed after stripping.
#
# Restricted to ASCII letters/digits (rather than the full Unicode "L"/"N"
# categories) to avoid characters whose casefold/encoding behavior could
# introduce unrelated edge cases — mirrors the ASCII-safe alphabet choice
# in `test_property_script_contract_structure.py`.
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


@st.composite
def valid_script_contract_dicts(draw, min_expected_replies: int = 0):
    """Generate a structurally valid Script_Contract-shaped dict.

    `prohibited_responses` is kept empty so no generated example
    accidentally triggers the Prohibited/Expected conflict rule (Property
    4), which is out of scope for the limit-enforcement properties tested
    in this module.

    Args:
        min_expected_replies: minimum number of `expected_replies` entries
            to generate.
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


@st.composite
def valid_script_contracts(draw, min_expected_replies: int = 0):
    """Generate a structurally valid, constructed `ScriptContract` instance."""
    contract_dict = draw(valid_script_contract_dicts(min_expected_replies=min_expected_replies))
    return ScriptContract(**contract_dict)


# Default limits generous enough that a contract built from the strategies
# above never trips the entry-count or free-text-length checks by
# construction (max 5 entries per list well under any generated
# `max_*_count`; max 50 chars per field well under any generated
# `max_field_text_length`). This isolates the size check under test in
# this module's Property 7 tests from the other limit types.
GENEROUS_NON_SIZE_LIMITS = {
    "max_trigger_phrases": 1000,
    "max_expected_replies": 1000,
    "max_escalation_conditions": 1000,
    "max_field_text_length": 10_000,
}


def _limits_with_size(max_definition_size_bytes: int) -> ScriptLimits:
    return ScriptLimits(
        max_definition_size_bytes=max_definition_size_bytes,
        **GENEROUS_NON_SIZE_LIMITS,
    )


def _size_violations(violations):
    """Filter a `validate_limits` violations list down to only the
    size-related ones (`type == "limit_exceeded.size"`), since other limit
    types are out of scope for Property 7."""
    return [v for v in violations if v.get("type") == "limit_exceeded.size"]


class TestDefinitionSizeLimitEnforcement:
    """Property 7: Definition size limit enforcement.

    Feature: ai-debtor-script-contract, Property 7: Definition size limit
    enforcement

    For any configured maximum definition size L and any script
    definition whose serialized size in bytes is greater than L, the
    Script_Registry SHALL reject the submission and report both L and the
    submitted size; for any script definition whose serialized size is
    less than or equal to L (and otherwise valid), size alone SHALL NOT
    cause rejection.

    **Validates: Requirements 2.3, 2.4, 2.10**
    """

    @given(
        contract=valid_script_contracts(),
        max_definition_size_bytes=st.integers(min_value=1, max_value=1_000_000),
        excess=st.integers(min_value=1, max_value=1_000_000),
    )
    @settings(max_examples=100)
    def test_oversized_definition_is_rejected_and_reports_limit_and_size(
        self, contract, max_definition_size_bytes, excess
    ):
        """A submission whose raw size in bytes is strictly greater than
        the configured maximum is rejected with a violation that reports
        both the configured limit L and the submitted size."""
        limits = _limits_with_size(max_definition_size_bytes)
        raw_size_bytes = max_definition_size_bytes + excess

        violations = validate_limits(contract, raw_size_bytes, limits)
        size_violations = _size_violations(violations)

        assert size_violations, (
            f"Expected a size-limit violation for raw_size_bytes={raw_size_bytes} "
            f"> max_definition_size_bytes={max_definition_size_bytes}, got: {violations}"
        )

        for violation in size_violations:
            ctx = violation.get("ctx", {})
            assert ctx.get("limit_bytes") == max_definition_size_bytes, (
                f"Expected violation ctx to report limit_bytes="
                f"{max_definition_size_bytes}, got: {violation}"
            )
            assert ctx.get("submitted_bytes") == raw_size_bytes, (
                f"Expected violation ctx to report submitted_bytes="
                f"{raw_size_bytes}, got: {violation}"
            )
            # The limit and submitted size must also be human-reportable
            # in the message text, not just the structured ctx.
            assert str(max_definition_size_bytes) in violation.get("msg", "")
            assert str(raw_size_bytes) in violation.get("msg", "")

    @given(
        contract=valid_script_contracts(),
        max_definition_size_bytes=st.integers(min_value=1, max_value=1_000_000),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_within_size_limit_definition_is_not_rejected_for_size_alone(
        self, contract, max_definition_size_bytes, data
    ):
        """A submission whose raw size in bytes is less than or equal to
        the configured maximum does not produce a size-limit violation,
        even though other (out-of-scope) limit types are not otherwise
        constrained here."""
        raw_size_bytes = data.draw(st.integers(min_value=0, max_value=max_definition_size_bytes))
        limits = _limits_with_size(max_definition_size_bytes)

        violations = validate_limits(contract, raw_size_bytes, limits)
        size_violations = _size_violations(violations)

        assert not size_violations, (
            f"Expected no size-limit violation for raw_size_bytes={raw_size_bytes} "
            f"<= max_definition_size_bytes={max_definition_size_bytes}, got: {size_violations}"
        )


# --- Entry-count limit strategies (task 3.12) ---

# Maps each entry-count-limited field to its per-entry Hypothesis strategy
# and the corresponding `ScriptLimits` attribute name that bounds it.
ENTRY_COUNT_FIELD_CONFIGS = {
    "trigger_phrases": (trigger_phrase_entries, "max_trigger_phrases"),
    "expected_replies": (expected_reply_entries, "max_expected_replies"),
    "escalation_conditions": (escalation_condition_entries, "max_escalation_conditions"),
}

# Generous defaults for every `ScriptLimits` field, used as a base so each
# entry-count test can override only the single limit attribute under test
# (`limit_attr`), keeping every other limit (size, other entry counts,
# free-text length) well out of reach of the small generated entries.
_GENEROUS_LIMITS_KWARGS = {
    "max_definition_size_bytes": 10_000_000,
    "max_trigger_phrases": 1000,
    "max_expected_replies": 1000,
    "max_escalation_conditions": 1000,
    "max_field_text_length": 10_000,
}


def _limits_with_count(limit_attr: str, max_count: int) -> ScriptLimits:
    kwargs = dict(_GENEROUS_LIMITS_KWARGS)
    kwargs[limit_attr] = max_count
    return ScriptLimits(**kwargs)


def _count_violations_for_field(violations, field_name: str):
    """Filter a `validate_limits` violations list down to only the
    `limit_exceeded.count` violations for the given field, since other
    limit/field types are out of scope for a single field's assertion."""
    return [
        v
        for v in violations
        if v.get("type") == "limit_exceeded.count" and v.get("loc") == (field_name,)
    ]


@st.composite
def oversized_entry_count_cases(draw):
    """Generate a (field_name, limit_attr, max_count, actual_count,
    contract_dict) case where `contract_dict[field_name]` has strictly
    more entries (`actual_count`) than a configured maximum (`max_count`)
    for that field, with every other field left at its normal
    (non-triggering) generated size."""
    field_name = draw(st.sampled_from(list(ENTRY_COUNT_FIELD_CONFIGS)))
    entry_strategy, limit_attr = ENTRY_COUNT_FIELD_CONFIGS[field_name]

    max_count = draw(st.integers(min_value=0, max_value=15))
    excess = draw(st.integers(min_value=1, max_value=15))
    actual_count = max_count + excess

    contract_dict = draw(valid_script_contract_dicts())
    contract_dict[field_name] = draw(
        st.lists(entry_strategy, min_size=actual_count, max_size=actual_count)
    )

    return field_name, limit_attr, max_count, actual_count, contract_dict


@st.composite
def within_limit_entry_count_cases(draw):
    """Generate a (field_name, limit_attr, max_count, contract_dict) case
    where `contract_dict[field_name]` has at most a configured maximum
    (`max_count`) entries for that field, with every other field left at
    its normal (non-triggering) generated size."""
    field_name = draw(st.sampled_from(list(ENTRY_COUNT_FIELD_CONFIGS)))
    entry_strategy, limit_attr = ENTRY_COUNT_FIELD_CONFIGS[field_name]

    max_count = draw(st.integers(min_value=0, max_value=15))
    actual_count = draw(st.integers(min_value=0, max_value=max_count))

    contract_dict = draw(valid_script_contract_dicts())
    contract_dict[field_name] = draw(
        st.lists(entry_strategy, min_size=actual_count, max_size=actual_count)
    )

    return field_name, limit_attr, max_count, contract_dict


class TestEntryCountLimitEnforcement:
    """Property 8: Entry-count limit enforcement.

    Feature: ai-debtor-script-contract, Property 8: Entry-count limit
    enforcement

    For any configured maximum entry count L for Trigger_Phrases,
    Expected_Replies, or Escalation_Conditions, and any script whose
    corresponding list has more than L entries, the Script_Registry SHALL
    reject the submission and identify which limit was exceeded; for any
    script whose corresponding list has at most L entries (and otherwise
    valid), the entry count alone SHALL NOT cause rejection.

    **Validates: Requirements 2.5, 2.6, 2.7, 2.9, 2.10**
    """

    @given(case=oversized_entry_count_cases())
    @settings(max_examples=100)
    def test_oversized_entry_count_is_rejected_and_identifies_limit(self, case):
        """A field whose entry count strictly exceeds its configured
        maximum is rejected with a `limit_exceeded.count` violation that
        identifies the field (via `loc`) and reports both the configured
        limit and the actual count."""
        field_name, limit_attr, max_count, actual_count, contract_dict = case
        contract = ScriptContract(**contract_dict)
        limits = _limits_with_count(limit_attr, max_count)

        violations = validate_limits(contract, raw_size_bytes=100, limits=limits)
        field_violations = _count_violations_for_field(violations, field_name)

        assert field_violations, (
            f"Expected a limit_exceeded.count violation for field={field_name!r} "
            f"with actual_count={actual_count} > max_count={max_count}, got: {violations}"
        )

        for violation in field_violations:
            ctx = violation.get("ctx", {})
            assert ctx.get("limit") == max_count, (
                f"Expected violation ctx to report limit={max_count}, got: {violation}"
            )
            assert ctx.get("actual") == actual_count, (
                f"Expected violation ctx to report actual={actual_count}, got: {violation}"
            )
            # The limit and actual count must also be human-reportable in
            # the message text, not just the structured ctx.
            assert str(max_count) in violation.get("msg", "")
            assert str(actual_count) in violation.get("msg", "")

    @given(case=within_limit_entry_count_cases())
    @settings(max_examples=100)
    def test_within_limit_entry_count_is_not_rejected_for_count_alone(self, case):
        """A field whose entry count is less than or equal to its
        configured maximum does not produce a `limit_exceeded.count`
        violation for that field, even though other (out-of-scope) limit
        types are not otherwise constrained here."""
        field_name, limit_attr, max_count, contract_dict = case
        contract = ScriptContract(**contract_dict)
        limits = _limits_with_count(limit_attr, max_count)

        violations = validate_limits(contract, raw_size_bytes=100, limits=limits)
        field_violations = _count_violations_for_field(violations, field_name)

        assert not field_violations, (
            f"Expected no limit_exceeded.count violation for field={field_name!r} "
            f"with actual_count<={max_count}, got: {field_violations}"
        )


# --- Free-text length limit strategies (task 3.13) ---

# Alphabet shared with `safe_text`/`_bounded_text` below, used directly
# (without the leading/trailing-strip-and-filter-nonempty step) when a
# generated string's *exact* length matters, since stripping could shrink
# a string below the length we deliberately constructed it to have.
_SAFE_TEXT_ALPHABET = st.characters(
    whitelist_categories=(),
    whitelist_characters=(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-"
    ),
)


def _exact_length_text(length: int):
    """A strategy for strings of exactly `length` characters, drawn from the
    same safe alphabet as `safe_text` but without stripping (which could
    change the resulting length)."""
    return st.text(alphabet=_SAFE_TEXT_ALPHABET, min_size=length, max_size=length)


def _bounded_text(max_length: int):
    """A `safe_text`-equivalent strategy whose generated strings never
    exceed `max_length` characters, for building contracts where every
    free-text field is guaranteed to respect a given configured limit."""
    return (
        st.text(
            alphabet=_SAFE_TEXT_ALPHABET,
            min_size=1,
            max_size=max_length,
        )
        .map(lambda s: s.strip())
        .filter(lambda s: s != "")
    )


# Every free-text field path `validate_limits` checks (see
# `_free_text_fields` in `app/services/script_validator.py`), described so
# `oversized_free_text_cases` below can set exactly one of them to an
# oversized value while leaving the rest of a generated contract dict
# untouched:
#   - "singleton": `path_info` is the dict-key path (1 or 2 levels deep)
#     to overwrite directly on the contract dict.
#   - "list": `path_info` is (list_field_name, subfield_name,
#     entry_strategy) — the list is regenerated with at least one entry
#     so index 0's `subfield_name` can be overwritten.
#   - "list_scalar": `prohibited_responses` is a list of plain strings
#     (no subfield), handled as its own case.
FREE_TEXT_FIELD_DESCRIPTORS = [
    ("singleton", ("debtor_persona", "name")),
    ("singleton", ("debtor_persona", "communication_style")),
    ("singleton", ("debtor_persona", "background")),
    ("singleton", ("financial_situation", "reason_for_delinquency")),
    ("singleton", ("opening_response",)),
    ("singleton", ("conversation_goal", "target_outcome")),
    ("singleton", ("conversation_goal", "completion_condition")),
    ("list", ("expected_replies", "agent_statement", expected_reply_entries)),
    ("list", ("expected_replies", "debtor_reply", expected_reply_entries)),
    ("list", ("trigger_phrases", "phrase", trigger_phrase_entries)),
    ("list", ("trigger_phrases", "behavior", trigger_phrase_entries)),
    ("list", ("emotional_state_rules", "trigger", emotional_state_rule_entries)),
    ("list", ("emotional_state_rules", "state_change", emotional_state_rule_entries)),
    ("list", ("escalation_conditions", "condition", escalation_condition_entries)),
    ("list", ("escalation_conditions", "behavior", escalation_condition_entries)),
    ("list", ("payment_conditions", "condition", payment_condition_entries)),
    ("list", ("payment_conditions", "term", payment_condition_entries)),
    ("list_scalar", ("prohibited_responses",)),
]


@st.composite
def oversized_free_text_cases(draw):
    """Generate a (field_path, max_field_text_length, actual_length,
    contract_dict) case where `contract_dict`, at `field_path`, holds text
    strictly longer (`actual_length`) than a configured maximum
    (`max_field_text_length`), while every other free-text field is left at
    its normal (non-triggering) generated size.

    `max_field_text_length` is capped below the schema's own fixed
    `FreeText` ceiling (2000) so an `actual_length` greater than it but
    still <= 2000 always exists — the realistic case of an operator
    configuring a stricter-than-2000 limit, testable via normal
    `ScriptContract` construction without bypassing schema validation.
    """
    kind, path_info = draw(st.sampled_from(FREE_TEXT_FIELD_DESCRIPTORS))

    max_field_text_length = draw(st.integers(min_value=1, max_value=1999))
    actual_length = draw(st.integers(min_value=max_field_text_length + 1, max_value=2000))
    oversized_text = draw(_exact_length_text(actual_length))

    contract_dict = draw(valid_script_contract_dicts())

    if kind == "singleton":
        if len(path_info) == 2:
            parent, leaf = path_info
            contract_dict[parent] = {**contract_dict[parent], leaf: oversized_text}
        else:
            (leaf,) = path_info
            contract_dict[leaf] = oversized_text
        field_path = path_info
    elif kind == "list":
        list_field, subfield, entry_strategy = path_info
        entries = draw(st.lists(entry_strategy, min_size=1, max_size=5))
        entries[0] = {**entries[0], subfield: oversized_text}
        contract_dict[list_field] = entries
        field_path = (list_field, 0, subfield)
    else:  # "list_scalar" -> prohibited_responses
        # Use only the single oversized entry (no additional random
        # entries) to avoid any chance of incidentally colliding with an
        # `expected_replies[*].debtor_reply` value and triggering the
        # unrelated Prohibited/Expected conflict validator (Property 4),
        # which is out of scope for this free-text-length property.
        contract_dict["prohibited_responses"] = [oversized_text]
        field_path = ("prohibited_responses", 0)

    return field_path, max_field_text_length, actual_length, contract_dict


@st.composite
def _contract_dict_with_max_field_text_length(draw, max_field_text_length: int):
    """A `valid_script_contract_dicts`-equivalent strategy where every
    free-text field is bounded by `max_field_text_length` (via
    `_bounded_text`), so the resulting dict never trips a
    `limit_exceeded.length` violation for that configured limit."""
    bounded_text = _bounded_text(max_field_text_length)

    debtor_persona = draw(
        st.fixed_dictionaries(
            {
                "name": bounded_text,
                "communication_style": bounded_text,
                "background": bounded_text,
            }
        )
    )
    financial_situation = draw(
        st.fixed_dictionaries(
            {
                "outstanding_balance": st.floats(
                    min_value=0.01,
                    max_value=999_999.0,
                    allow_nan=False,
                    allow_infinity=False,
                ).map(lambda x: round(x, 2)),
                "days_past_due": st.integers(min_value=0, max_value=10_000),
                "reason_for_delinquency": bounded_text,
            }
        )
    )
    conversation_goal = draw(
        st.fixed_dictionaries(
            {
                "target_outcome": bounded_text,
                "completion_condition": bounded_text,
            }
        )
    )
    expected_reply_entry = st.fixed_dictionaries(
        {"agent_statement": bounded_text, "debtor_reply": bounded_text}
    )
    trigger_phrase_entry = st.fixed_dictionaries({"phrase": bounded_text, "behavior": bounded_text})
    emotional_state_rule_entry = st.fixed_dictionaries(
        {"trigger": bounded_text, "state_change": bounded_text}
    )
    escalation_condition_entry = st.fixed_dictionaries(
        {"condition": bounded_text, "behavior": bounded_text, "ends_call": st.booleans()}
    )
    payment_condition_entry = st.fixed_dictionaries(
        {"condition": bounded_text, "term": bounded_text, "accepted": st.booleans()}
    )

    return {
        "debtor_persona": debtor_persona,
        "financial_situation": financial_situation,
        "opening_response": draw(bounded_text),
        "expected_replies": draw(st.lists(expected_reply_entry, min_size=0, max_size=5)),
        "trigger_phrases": draw(st.lists(trigger_phrase_entry, min_size=0, max_size=5)),
        "emotional_state_rules": draw(
            st.lists(emotional_state_rule_entry, min_size=0, max_size=5)
        ),
        "payment_conditions": draw(st.lists(payment_condition_entry, min_size=0, max_size=5)),
        "escalation_conditions": draw(
            st.lists(escalation_condition_entry, min_size=0, max_size=5)
        ),
        "prohibited_responses": [],
        "conversation_goal": conversation_goal,
    }


@st.composite
def within_limit_free_text_cases(draw):
    """Generate a (max_field_text_length, contract_dict) case where every
    free-text field in `contract_dict` is at most a configured maximum
    (`max_field_text_length`)."""
    max_field_text_length = draw(st.integers(min_value=1, max_value=2000))
    contract_dict = draw(_contract_dict_with_max_field_text_length(max_field_text_length))
    return max_field_text_length, contract_dict


def _limits_with_field_text_length(max_field_text_length: int) -> ScriptLimits:
    return ScriptLimits(
        max_definition_size_bytes=10_000_000,
        max_trigger_phrases=1000,
        max_expected_replies=1000,
        max_escalation_conditions=1000,
        max_field_text_length=max_field_text_length,
    )


class TestFreeTextLengthLimitEnforcement:
    """Property 9: Free-text length limit enforcement.

    Feature: ai-debtor-script-contract, Property 9: Free-text length limit
    enforcement

    For any configured maximum character length L and any script where a
    free-text field's length exceeds L, the Script_Registry SHALL reject
    the submission and identify which field/limit was exceeded; for any
    script where all free-text fields are at most L characters (and
    otherwise valid), text length alone SHALL NOT cause rejection.

    **Validates: Requirements 2.8, 2.9, 2.10**
    """

    @given(case=oversized_free_text_cases())
    @settings(max_examples=100)
    def test_oversized_free_text_field_is_rejected_and_identifies_field(self, case):
        """A free-text field whose length strictly exceeds the configured
        maximum is rejected with a `limit_exceeded.length` violation that
        identifies the field (via `loc`) and reports both the configured
        limit and the actual length."""
        field_path, max_field_text_length, actual_length, contract_dict = case
        contract = ScriptContract(**contract_dict)
        limits = _limits_with_field_text_length(max_field_text_length)

        violations = validate_limits(contract, raw_size_bytes=100, limits=limits)
        field_violations = [
            v
            for v in violations
            if v.get("type") == "limit_exceeded.length" and v.get("loc") == field_path
        ]

        assert field_violations, (
            f"Expected a limit_exceeded.length violation for field_path={field_path!r} "
            f"with actual_length={actual_length} > max_field_text_length="
            f"{max_field_text_length}, got: {violations}"
        )

        for violation in field_violations:
            ctx = violation.get("ctx", {})
            assert ctx.get("limit") == max_field_text_length, (
                f"Expected violation ctx to report limit={max_field_text_length}, "
                f"got: {violation}"
            )
            assert ctx.get("actual") == actual_length, (
                f"Expected violation ctx to report actual={actual_length}, got: {violation}"
            )
            # The limit and actual length must also be human-reportable in
            # the message text, not just the structured ctx.
            assert str(max_field_text_length) in violation.get("msg", "")
            assert str(actual_length) in violation.get("msg", "")

    @given(case=within_limit_free_text_cases())
    @settings(max_examples=100)
    def test_within_limit_free_text_fields_are_not_rejected_for_length_alone(self, case):
        """A script whose free-text fields are all at most the configured
        maximum length does not produce any `limit_exceeded.length`
        violation."""
        max_field_text_length, contract_dict = case
        contract = ScriptContract(**contract_dict)
        limits = _limits_with_field_text_length(max_field_text_length)

        violations = validate_limits(contract, raw_size_bytes=100, limits=limits)
        length_violations = [v for v in violations if v.get("type") == "limit_exceeded.length"]

        assert not length_violations, (
            f"Expected no limit_exceeded.length violations when every free-text field "
            f"is at most max_field_text_length={max_field_text_length}, got: "
            f"{length_violations}"
        )

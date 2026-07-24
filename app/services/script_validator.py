"""Script definition validation pipeline.

Pure, side-effect-free functions that parse a raw Script definition
(JSON or YAML text) and validate it against the Script_Contract and
configurable limits. No DB access — this module is the primary unit
under property-based testing for the Script_Registry subsystem.
"""

import json
from typing import Any, Dict, List, NamedTuple, Tuple, Union

import yaml
from pydantic import ValidationError

from app.schemas.script import ScriptContract


class ScriptFormatError(Exception):
    """Raised when a Script definition's declared format is unsupported
    or its content cannot be parsed as that format.
    """


class ScriptValidationError(Exception):
    """Raised when a Script definition fails Script_Contract structural
    validation.

    Carries the full list of validation error details (field path +
    message) discovered by Pydantic, since ``ValidationError.errors()``
    already collects every violation without short-circuiting on the
    first one. Callers must not truncate this list to a single error.
    """

    def __init__(self, errors: List[Dict[str, Any]]) -> None:
        self.errors = errors
        field_paths = ", ".join(
            ".".join(str(part) for part in error["loc"]) or "<root>"
            for error in errors
        )
        super().__init__(
            f"Script definition failed contract validation ({len(errors)} "
            f"error(s)): {field_paths}"
        )


def parse_script_definition(raw_text: str, format: str) -> Dict[str, Any]:
    """Parse a raw Script definition into a dict.

    Args:
        raw_text: The raw Script definition content (JSON or YAML text).
        format: The declared format of ``raw_text``. Must be ``"json"``
            or ``"yaml"``.

    Returns:
        The parsed Script definition as a dict.

    Raises:
        ScriptFormatError: If ``format`` is not ``"json"`` or ``"yaml"``,
            or if ``raw_text`` cannot be parsed as the declared format.
    """
    if format == "json":
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ScriptFormatError(f"Unparseable JSON content: {exc}") from exc
    elif format == "yaml":
        try:
            data = yaml.safe_load(raw_text)
        except yaml.YAMLError as exc:
            raise ScriptFormatError(f"Unparseable YAML content: {exc}") from exc
    else:
        raise ScriptFormatError(
            f"Unsupported script format: {format!r} (expected 'json' or 'yaml')"
        )

    if not isinstance(data, dict):
        raise ScriptFormatError(
            "Parsed script definition must be a mapping/object, got "
            f"{type(data).__name__}"
        )

    return data


def validate_contract_structure(data: Dict[str, Any]) -> ScriptContract:
    """Validate a parsed Script definition against the Script_Contract.

    Args:
        data: The parsed Script definition (as returned by
            ``parse_script_definition``).

    Returns:
        The validated ``ScriptContract`` instance.

    Raises:
        ScriptValidationError: If ``data`` does not satisfy the
            Script_Contract. The raised error's ``errors`` attribute
            contains every missing/invalid field path Pydantic found,
            not just the first one.
    """
    try:
        return ScriptContract(**data)
    except ValidationError as exc:
        raise ScriptValidationError(exc.errors()) from exc


def validate_conflicts(contract: ScriptContract) -> List[Tuple[str, str]]:
    """Detect Prohibited_Responses entries that conflict with Expected_Replies.

    Standalone, reusable equivalent of ``ScriptContract``'s own
    ``model_validator`` conflict check (see ``app/schemas/script.py``), for
    use by the ``validate_script`` aggregator (task 3.10). That aggregator
    needs to collect *all* violations (missing fields + conflicts + limits)
    without the schema-level validator's exception short-circuiting the
    aggregation, so this function inspects an already-constructed
    ``ScriptContract`` instance directly rather than raising.

    Args:
        contract: A validated ``ScriptContract`` instance.

    Returns:
        A list of ``(expected_reply, prohibited_response)`` pairs for every
        ``prohibited_responses`` entry that textually duplicates an
        ``expected_replies[*].debtor_reply`` entry, using the same trimmed,
        case-insensitive comparison as the schema-level validator. Empty
        if there are no conflicts.
    """
    expected_replies_normalized = {
        entry.debtor_reply.strip().casefold(): entry.debtor_reply
        for entry in contract.expected_replies
    }

    return [
        (expected_replies_normalized[prohibited.strip().casefold()], prohibited)
        for prohibited in contract.prohibited_responses
        if prohibited.strip().casefold() in expected_replies_normalized
    ]


# A limit violation uses the same shape as `pydantic.ValidationError.errors()`
# entries (``loc``/``msg``/``type``/``ctx`` keys) so it can be appended
# directly into a `ScriptValidationError`'s unified `errors` list alongside
# structural/conflict violations without any special-casing.
LimitViolation = Dict[str, Any]


class ScriptLimits(NamedTuple):
    """Configurable Script_Registry limits (Requirement 2.10).

    A plain data container so this module never needs to import
    `app.config` directly, keeping it pure and independently testable.
    Callers construct this from `app.config.settings`, e.g.::

        ScriptLimits(
            max_definition_size_bytes=settings.script_max_definition_size_bytes,
            max_trigger_phrases=settings.script_max_trigger_phrases,
            max_expected_replies=settings.script_max_expected_replies,
            max_escalation_conditions=settings.script_max_escalation_conditions,
            max_field_text_length=settings.script_max_field_text_length,
        )
    """

    max_definition_size_bytes: int
    max_trigger_phrases: int
    max_expected_replies: int
    max_escalation_conditions: int
    max_field_text_length: int


class ScriptLimitError(Exception):
    """Raised when a Script definition exceeds one or more configurable
    limits (Requirement 2.9): definition size, entry counts, or free-text
    field lengths.

    Carries the full list of limit violations found (see
    ``validate_limits``), using the same error-dict shape as
    ``ScriptValidationError.errors`` so callers can inspect/report every
    exceeded limit uniformly, without truncating to a single violation.
    """

    def __init__(self, violations: List[LimitViolation]) -> None:
        self.violations = violations
        descriptions = "; ".join(violation.get("msg", "") for violation in violations)
        super().__init__(
            f"Script definition exceeds {len(violations)} configured limit(s): "
            f"{descriptions}"
        )


def _free_text_fields(
    contract: ScriptContract,
) -> List[Tuple[Tuple[Union[str, int], ...], str]]:
    """Enumerate every free-text field on a validated Script_Contract.

    Returns a list of ``(field_path, text)`` pairs covering every
    free-text field checked against ``limits.max_field_text_length``
    (Requirement 2.8): Debtor_Persona name/communication_style/background,
    Financial_Situation.reason_for_delinquency, Opening_Response, every
    Expected_Replies entry's agent_statement/debtor_reply, every
    Trigger_Phrases entry's phrase/behavior, every Emotional_State_Rules
    entry's trigger/state_change, every Escalation_Conditions entry's
    condition/behavior, every Payment_Conditions entry's condition/term,
    every Prohibited_Responses entry, and Conversation_Goal's
    target_outcome/completion_condition.
    """
    fields: List[Tuple[Tuple[Union[str, int], ...], str]] = [
        (("debtor_persona", "name"), contract.debtor_persona.name),
        (
            ("debtor_persona", "communication_style"),
            contract.debtor_persona.communication_style,
        ),
        (("debtor_persona", "background"), contract.debtor_persona.background),
        (
            ("financial_situation", "reason_for_delinquency"),
            contract.financial_situation.reason_for_delinquency,
        ),
        (("opening_response",), contract.opening_response),
        (
            ("conversation_goal", "target_outcome"),
            contract.conversation_goal.target_outcome,
        ),
        (
            ("conversation_goal", "completion_condition"),
            contract.conversation_goal.completion_condition,
        ),
    ]

    for index, entry in enumerate(contract.expected_replies):
        fields.append((("expected_replies", index, "agent_statement"), entry.agent_statement))
        fields.append((("expected_replies", index, "debtor_reply"), entry.debtor_reply))

    for index, entry in enumerate(contract.trigger_phrases):
        fields.append((("trigger_phrases", index, "phrase"), entry.phrase))
        fields.append((("trigger_phrases", index, "behavior"), entry.behavior))

    for index, entry in enumerate(contract.emotional_state_rules):
        fields.append((("emotional_state_rules", index, "trigger"), entry.trigger))
        fields.append((("emotional_state_rules", index, "state_change"), entry.state_change))

    for index, entry in enumerate(contract.escalation_conditions):
        fields.append((("escalation_conditions", index, "condition"), entry.condition))
        fields.append((("escalation_conditions", index, "behavior"), entry.behavior))

    for index, entry in enumerate(contract.payment_conditions):
        fields.append((("payment_conditions", index, "condition"), entry.condition))
        fields.append((("payment_conditions", index, "term"), entry.term))

    for index, text in enumerate(contract.prohibited_responses):
        fields.append((("prohibited_responses", index), text))

    return fields


def validate_limits(
    contract: ScriptContract, raw_size_bytes: int, limits: ScriptLimits
) -> List[LimitViolation]:
    """Check a validated Script_Contract against configurable limits.

    Checks, without short-circuiting on the first violation found
    (Requirements 2.3-2.9):
      (a) ``raw_size_bytes`` against ``limits.max_definition_size_bytes``
      (b) ``trigger_phrases``/``expected_replies``/``escalation_conditions``
          entry counts against their respective configured maximums
      (c) every free-text field's length (see ``_free_text_fields``)
          against ``limits.max_field_text_length``

    Args:
        contract: A structurally validated ``ScriptContract`` instance.
        raw_size_bytes: The size, in bytes, of the raw (serialized) Script
            definition submission.
        limits: The configurable limits to check against.

    Returns:
        A list of violation dicts (``loc``/``msg``/``type``/``ctx`` keys,
        matching ``pydantic.ValidationError.errors()``'s shape) each
        identifying which limit was exceeded (Requirement 2.9). Empty if
        no limit is exceeded.
    """
    violations: List[LimitViolation] = []

    if raw_size_bytes > limits.max_definition_size_bytes:
        violations.append(
            {
                "loc": ("__root__",),
                "msg": (
                    f"Script definition size ({raw_size_bytes} bytes) exceeds the "
                    f"maximum allowed size ({limits.max_definition_size_bytes} bytes)"
                ),
                "type": "limit_exceeded.size",
                "ctx": {
                    "limit_bytes": limits.max_definition_size_bytes,
                    "submitted_bytes": raw_size_bytes,
                },
            }
        )

    entry_count_checks = (
        ("trigger_phrases", contract.trigger_phrases, limits.max_trigger_phrases),
        ("expected_replies", contract.expected_replies, limits.max_expected_replies),
        (
            "escalation_conditions",
            contract.escalation_conditions,
            limits.max_escalation_conditions,
        ),
    )
    for field_name, entries, max_count in entry_count_checks:
        actual_count = len(entries)
        if actual_count > max_count:
            violations.append(
                {
                    "loc": (field_name,),
                    "msg": (
                        f"{field_name} has {actual_count} entries, exceeding the "
                        f"maximum allowed count ({max_count})"
                    ),
                    "type": "limit_exceeded.count",
                    "ctx": {"limit": max_count, "actual": actual_count},
                }
            )

    for field_path, text in _free_text_fields(contract):
        actual_length = len(text)
        if actual_length > limits.max_field_text_length:
            field_path_str = ".".join(str(part) for part in field_path)
            violations.append(
                {
                    "loc": field_path,
                    "msg": (
                        f"{field_path_str} length ({actual_length} characters) exceeds "
                        f"the maximum allowed length ({limits.max_field_text_length} "
                        "characters)"
                    ),
                    "type": "limit_exceeded.length",
                    "ctx": {
                        "limit": limits.max_field_text_length,
                        "actual": actual_length,
                    },
                }
            )

    return violations


def _is_conflict_only_failure(errors: List[Dict[str, Any]]) -> bool:
    """Check whether every error in a ``ScriptValidationError.errors`` list
    originates solely from ``ScriptContract``'s prohibited/expected conflict
    ``model_validator`` (see ``app/schemas/script.py``), rather than from
    any missing/invalid field.

    That validator runs with ``mode="after"`` in Pydantic v2, which only
    fires once every individual field has *already* passed its own
    validation without error. So if every reported error has an empty
    ``loc`` and ``type == "value_error"`` (the shape of that validator's
    raised ``ValueError``), the submitted data is otherwise structurally
    complete and valid — only the conflict check itself failed.
    """
    return bool(errors) and all(
        error.get("loc") == () and error.get("type") == "value_error"
        for error in errors
    )


def validate_script(raw_text: str, format: str, limits: ScriptLimits) -> ScriptContract:
    """Run the full Script definition validation pipeline.

    Runs parse -> structure -> conflicts -> limits in sequence. Structural
    validation must succeed before conflicts/limits can be evaluated at
    all (both require an already-constructed ``ScriptContract`` instance
    to inspect), so the aggregation contract is:

      - If parsing fails, ``ScriptFormatError`` is raised immediately —
        there is no partial dict/contract state to aggregate against yet.
      - If structural validation fails for any reason *other than* the
        schema-level prohibited/expected conflict check, ``ScriptValidationError``
        is raised carrying *every* structural violation found (already
        guaranteed by ``validate_contract_structure``, tasks 3.4/3.5).
      - If structural validation fails *solely* because of the
        schema-level conflict check (see ``_is_conflict_only_failure``),
        the contract is rebuilt with ``prohibited_responses`` temporarily
        cleared (bypassing just that one validator, since every other
        field is already known-valid) so it can still be inspected. This
        lets a conflict combine with limit violations in one report
        instead of short-circuiting.
      - Once a contract instance is available (by either path above),
        ``validate_conflicts`` and ``validate_limits`` are both run
        against it, and a single ``ScriptValidationError`` aggregating
        **all** discovered conflict + limit violations is raised if
        either found any — no short-circuiting between the two.

    Args:
        raw_text: The raw Script definition content (JSON or YAML text).
        format: The declared format of ``raw_text``. Must be ``"json"``
            or ``"yaml"``.
        limits: The configurable limits to check the definition against.

    Returns:
        The validated ``ScriptContract`` instance, if the definition
        passes every stage of validation.

    Raises:
        ScriptFormatError: If ``format`` is unsupported or ``raw_text``
            cannot be parsed as the declared format.
        ScriptValidationError: If structural validation fails (other than
            a conflict-only failure), or if a contract instance is
            obtained but one or more conflict and/or limit violations
            are found.
    """
    data = parse_script_definition(raw_text, format)

    try:
        contract = validate_contract_structure(data)
    except ScriptValidationError as exc:
        if not _is_conflict_only_failure(exc.errors):
            raise
        # The only problem is the prohibited/expected conflict itself;
        # every other field is already known-valid. Rebuild bypassing
        # just that check so conflicts/limits can still be aggregated.
        contract = validate_contract_structure({**data, "prohibited_responses": []})
        contract.prohibited_responses = data.get("prohibited_responses", [])

    conflict_violations: List[LimitViolation] = [
        {
            "loc": ("prohibited_responses",),
            "msg": (
                f"prohibited_responses entry {prohibited!r} conflicts with an "
                f"expected_replies debtor_reply entry {expected_reply!r}"
            ),
            "type": "conflict.prohibited_expected",
            "ctx": {"expected_reply": expected_reply, "prohibited_response": prohibited},
        }
        for expected_reply, prohibited in validate_conflicts(contract)
    ]
    limit_violations = validate_limits(
        contract, raw_size_bytes=len(raw_text.encode("utf-8")), limits=limits
    )

    combined_violations = conflict_violations + limit_violations
    if combined_violations:
        raise ScriptValidationError(combined_violations)

    return contract

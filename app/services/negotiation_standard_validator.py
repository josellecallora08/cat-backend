"""Pure aggregate validation for negotiation standard rubric content."""

from collections import defaultdict

from app.schemas.negotiation_standard import (
    NegotiationStandardContent,
    RubricBlock,
    ValidationIssue,
    ValidationResult,
)


def _issue(code: str, path: str, message: str) -> ValidationIssue:
    """Create a structured validation issue."""
    return ValidationIssue(code=code, path=path, message=message)


def _check_unique(
    values: list[tuple[str, str]],
    errors: list[ValidationIssue],
    message: str,
) -> None:
    """Append duplicate-value issues while preserving first-seen paths."""
    seen: dict[str, str] = {}
    for value, path in values:
        key = value.casefold()
        if key in seen:
            errors.append(_issue("duplicate", path, f"{message}: '{value}' is already used."))
        else:
            seen[key] = path


def _validate_blocks(
    content: NegotiationStandardContent,
) -> list[ValidationIssue]:
    """Validate block, criterion, violation, and penalty relationships."""
    errors: list[ValidationIssue] = []
    block_ids: list[tuple[str, str]] = []
    categories: list[tuple[str, str]] = []
    display_orders: list[tuple[str, str]] = []
    all_criterion_ids: list[tuple[str, str]] = []
    all_violation_ids: list[tuple[str, str]] = []
    known_violations: dict[int, set[str]] = defaultdict(set)

    for block_index, block in enumerate(content.blocks):
        block_path = f"blocks[{block_index}]"
        block_ids.append((block.id, f"{block_path}.id"))
        categories.append((block.category, f"{block_path}.category"))
        display_orders.append((str(block.display_order), f"{block_path}.display_order"))

        for criterion_index, criterion in enumerate(block.positive_behaviors):
            all_criterion_ids.append(
                (criterion.id, f"{block_path}.positive_behaviors[{criterion_index}].id")
            )
        for violation_index, violation in enumerate(block.violations):
            violation_path = f"{block_path}.violations[{violation_index}].id"
            all_violation_ids.append((violation.id, violation_path))
            known_violations[block_index].add(violation.id.casefold())

        for penalty_index, penalty in enumerate(block.penalties):
            if penalty.violation_id.casefold() not in known_violations[block_index]:
                errors.append(
                    _issue(
                        "unknown_reference",
                        f"{block_path}.penalties[{penalty_index}].violation_id",
                        f"Penalty references unknown violation '{penalty.violation_id}'.",
                    )
                )

    _check_unique(block_ids, errors, "Duplicate block ID")
    _check_unique(categories, errors, "Duplicate category")
    _check_unique(display_orders, errors, "Duplicate display order")
    _check_unique(all_criterion_ids, errors, "Duplicate criterion ID")
    _check_unique(all_violation_ids, errors, "Duplicate violation ID")
    criterion_paths = {value.casefold(): path for value, path in all_criterion_ids}
    for violation_id, violation_path in all_violation_ids:
        if violation_id.casefold() in criterion_paths:
            errors.append(
                _issue(
                    "duplicate",
                    violation_path,
                    f"Criterion and violation IDs must be unique; '{violation_id}' is reused.",
                )
            )
    return errors


def validate_standard(
    content: NegotiationStandardContent,
    for_publish: bool,
) -> ValidationResult:
    """Validate a rubric aggregate and return all deterministic issues.

    Draft validation reports publication-relevant weight errors without mutating
    the draft. Publication validation applies the same structural checks and
    requires the exact total of 100.
    """
    errors: list[ValidationIssue] = []
    weight_total = sum(block.weight for block in content.blocks)

    if not content.blocks:
        errors.append(_issue("required", "blocks", "At least one rubric block is required."))
    errors.extend(_validate_blocks(content))

    if weight_total != 100:
        errors.append(
            _issue(
                "weights_must_total_100",
                "blocks",
                f"Rubric block weights must total 100; received {weight_total}.",
            )
        )

    # Keep the argument explicit in the public API; both draft validation and
    # publication validation report the same aggregate errors.
    _ = for_publish
    errors.sort(key=lambda item: (item.path, item.code, item.message))
    return ValidationResult(valid=not errors, weight_total=weight_total, errors=errors)

"""Pure validation of AI rubric observations against pinned transcript data."""

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.schemas.negotiation_standard import NegotiationStandardContent, ValidationIssue
from app.schemas.rubric_evaluation import RubricAIObservation


class ObservationValidationError(ValueError):
    """Raised when an AI observation cannot be trusted or grounded."""

    def __init__(self, errors: list[ValidationIssue]) -> None:
        super().__init__("Rubric observation failed validation")
        self.errors = errors


@dataclass(frozen=True)
class ValidatedObservation:
    """Observation with verified snapshot references and transcript evidence."""

    observation: RubricAIObservation
    snapshot: NegotiationStandardContent
    transcript: tuple[dict[str, Any], ...]


def _normalize(value: str) -> str:
    """Normalize only whitespace for exact evidence containment checks."""
    return " ".join(value.split())


def _error(code: str, path: str, message: str) -> ValidationIssue:
    """Create a structured observation validation error."""
    return ValidationIssue(code=code, path=path, message=message)


def _schema_errors(exc: ValidationError) -> list[ValidationIssue]:
    """Convert strict Pydantic errors into the service error contract."""
    issues: list[ValidationIssue] = []
    for error in exc.errors():
        location = (
            "".join(
                f"[{part}]" if isinstance(part, int) else (f".{part}" if index else str(part))
                for index, part in enumerate(error.get("loc", ()))
            )
            or "observation"
        )
        error_type = str(error.get("type", "invalid"))
        code = (
            "required"
            if error_type == "missing"
            else "invalid_evidence"
            if "evidence" in location
            else "invalid"
        )
        issues.append(_error(code, location, str(error.get("msg", "Invalid observation."))))
    return issues


def _transcript_index(transcript: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Index persisted transcript entries and reject duplicate sequence numbers."""
    indexed: dict[int, dict[str, Any]] = {}
    errors: list[ValidationIssue] = []
    for index, entry in enumerate(transcript):
        sequence = entry.get("sequence_number", index)
        if sequence in indexed:
            errors.append(
                _error(
                    "duplicate",
                    f"transcript[{index}].sequence_number",
                    "Sequence number is duplicated.",
                )
            )
        indexed[sequence] = entry
    if errors:
        raise ObservationValidationError(errors)
    return indexed


def validate_observation(
    observation: RubricAIObservation | dict[str, Any],
    snapshot: NegotiationStandardContent | dict[str, Any],
    transcript: list[dict[str, Any]],
) -> ValidatedObservation:
    """Validate exact rubric coverage, references, speakers, and excerpts."""
    try:
        parsed = (
            observation
            if isinstance(observation, RubricAIObservation)
            else RubricAIObservation.model_validate(observation)
        )
        rubric = (
            snapshot
            if isinstance(snapshot, NegotiationStandardContent)
            else NegotiationStandardContent.model_validate(snapshot)
        )
    except ValidationError as exc:
        raise ObservationValidationError(_schema_errors(exc)) from exc
    transcript_by_sequence = _transcript_index(transcript)
    errors: list[ValidationIssue] = []
    blocks = {block.id: block for block in rubric.blocks}
    category_ids = [category.rubric_block_id for category in parsed.categories]

    if len(category_ids) != len(set(category_ids)):
        errors.append(
            _error("duplicate", "categories", "Each rubric block must appear exactly once.")
        )
    expected = set(blocks)
    actual = set(category_ids)
    for missing in sorted(expected - actual):
        errors.append(_error("required", "categories", f"Missing category result for '{missing}'."))
    for unknown in sorted(actual - expected):
        errors.append(
            _error("unknown_reference", "categories", f"Unknown rubric block '{unknown}'.")
        )

    for category_index, category in enumerate(parsed.categories):
        block = blocks.get(category.rubric_block_id)
        if block is None:
            continue
        valid_criteria = {item.id for item in block.positive_behaviors + block.violations}
        valid_violations = {item.id for item in block.violations}
        evidence_sequences = {item.sequence_number for item in category.evidence}
        for evidence_index, evidence in enumerate(category.evidence):
            _validate_evidence(
                errors, evidence, evidence_index, transcript_by_sequence, category_index
            )
        criterion_evidence: dict[str, set[int]] = {}
        finding_ids: set[str] = set()
        for finding_index, finding in enumerate(category.strengths):
            path = f"categories[{category_index}].strengths[{finding_index}]"
            _validate_finding(
                errors,
                finding.criterion_id,
                valid_criteria,
                finding.evidence_sequence_numbers,
                evidence_sequences,
                path,
            )
            if finding.criterion_id in valid_criteria:
                finding_ids.add(finding.criterion_id)
                criterion_evidence.setdefault(finding.criterion_id, set()).update(
                    finding.evidence_sequence_numbers
                )
        for finding_index, finding in enumerate(category.violations):
            path = f"categories[{category_index}].violations[{finding_index}]"
            if finding.violation_id not in valid_violations:
                errors.append(
                    _error(
                        "unknown_reference",
                        f"{path}.violation_id",
                        "Violation is not declared in this rubric block.",
                    )
                )
            else:
                finding_ids.add(finding.violation_id)
                criterion_evidence.setdefault(finding.violation_id, set()).update(
                    finding.evidence_sequence_numbers
                )
            _validate_sequences(
                errors,
                finding.evidence_sequence_numbers,
                evidence_sequences,
                f"{path}.evidence_sequence_numbers",
            )
        failed_criteria = set(category.failed_criteria)
        for criterion_index, criterion_id in enumerate(category.failed_criteria):
            if criterion_id not in valid_criteria:
                errors.append(
                    _error(
                        "unknown_reference",
                        f"categories[{category_index}].failed_criteria[{criterion_index}]",
                        "Criterion is not declared in this rubric block.",
                    )
                )
        recommendation_criteria: set[str] = set()
        for recommendation_index, recommendation in enumerate(category.recommendation_inputs):
            path = f"categories[{category_index}].recommendation_inputs[{recommendation_index}]"
            if recommendation.criterion_id not in valid_criteria:
                errors.append(
                    _error(
                        "unknown_reference",
                        f"{path}.criterion_id",
                        "Criterion is not declared in this rubric block.",
                    )
                )
                continue
            recommendation_criteria.add(recommendation.criterion_id)
            sequence = recommendation.transcript_sequence_number
            if sequence not in transcript_by_sequence:
                errors.append(
                    _error(
                        "unknown_reference",
                        f"{path}.transcript_sequence_number",
                        "Transcript sequence does not exist.",
                    )
                )
            elif sequence not in evidence_sequences:
                errors.append(
                    _error(
                        "required",
                        f"{path}.transcript_sequence_number",
                        "Recommendation requires matching category evidence.",
                    )
                )
            elif (
                recommendation.criterion_id in criterion_evidence
                and sequence not in criterion_evidence[recommendation.criterion_id]
            ):
                errors.append(
                    _error(
                        "required",
                        f"{path}.transcript_sequence_number",
                        "Recommendation requires evidence belonging to the cited criterion.",
                    )
                )
            elif (
                recommendation.criterion_id not in criterion_evidence
                and recommendation.criterion_id not in failed_criteria
            ):
                errors.append(
                    _error(
                        "required",
                        f"{path}.criterion_id",
                        "Recommendation requires a finding or failed criterion.",
                    )
                )
        for criterion_id in sorted(failed_criteria - finding_ids - recommendation_criteria):
            criterion_index = category.failed_criteria.index(criterion_id)
            errors.append(
                _error(
                    "required",
                    f"categories[{category_index}].failed_criteria[{criterion_index}]",
                    "Failed criteria require a finding or validated recommendation input.",
                )
            )

    valid_technique_names = {
        item.name.casefold()
        for block in rubric.blocks
        for item in block.positive_behaviors + block.violations
    }
    for technique_index, technique in enumerate(parsed.applied_techniques.techniques_used):
        path = f"applied_techniques.techniques_used[{technique_index}]"
        if technique.technique_name.casefold() not in valid_technique_names:
            errors.append(
                _error(
                    "unknown_reference",
                    f"{path}.technique_name",
                    "Technique is not declared in the published rubric.",
                )
            )
        for sequence_index, sequence in enumerate(technique.evidence_sequence_numbers):
            if sequence not in transcript_by_sequence:
                errors.append(
                    _error(
                        "unknown_reference",
                        f"{path}.evidence_sequence_numbers[{sequence_index}]",
                        "Transcript sequence does not exist.",
                    )
                )
    for missed_index, missed in enumerate(parsed.missed_opportunities.missed_techniques):
        path = f"missed_opportunities.missed_techniques[{missed_index}]"
        if missed.technique_name.casefold() not in valid_technique_names:
            errors.append(
                _error(
                    "unknown_reference",
                    f"{path}.technique_name",
                    "Technique is not declared in the published rubric.",
                )
            )

    if errors:
        raise ObservationValidationError(
            sorted(errors, key=lambda item: (item.path, item.code, item.message))
        )
    return ValidatedObservation(parsed, rubric, tuple(transcript))


def _validate_sequences(
    errors: list[ValidationIssue],
    sequences: list[int],
    evidence_sequences: set[int],
    path: str,
) -> None:
    """Ensure finding evidence points at category evidence."""
    for index, sequence in enumerate(sequences):
        if sequence not in evidence_sequences:
            errors.append(
                _error(
                    "required", f"{path}[{index}]", "Finding requires matching category evidence."
                )
            )


def _validate_finding(
    errors: list[ValidationIssue],
    criterion_id: str,
    valid_criteria: set[str],
    sequences: list[int],
    evidence_sequences: set[int],
    path: str,
) -> None:
    """Validate a criterion finding and its evidence references."""
    if criterion_id not in valid_criteria:
        errors.append(
            _error(
                "unknown_reference",
                f"{path}.criterion_id",
                "Criterion is not declared in this rubric block.",
            )
        )
    _validate_sequences(errors, sequences, evidence_sequences, f"{path}.evidence_sequence_numbers")


def _validate_evidence(
    errors: list[ValidationIssue],
    evidence: Any,
    evidence_index: int,
    transcript_by_sequence: dict[int, dict[str, Any]],
    category_index: int,
) -> None:
    """Verify speaker and exact whitespace-normalized excerpt grounding."""
    path = f"categories[{category_index}].evidence[{evidence_index}]"
    entry = transcript_by_sequence.get(evidence.sequence_number)
    if entry is None:
        errors.append(
            _error(
                "unknown_reference",
                f"{path}.sequence_number",
                "Transcript sequence does not exist.",
            )
        )
        return
    if entry.get("speaker") != evidence.speaker:
        errors.append(
            _error(
                "invalid_evidence",
                f"{path}.speaker",
                "Evidence speaker does not match the persisted transcript.",
            )
        )
    if _normalize(evidence.excerpt) not in _normalize(
        str(entry.get("text", entry.get("utterance_text", "")))
    ):
        errors.append(
            _error(
                "invalid_evidence",
                f"{path}.excerpt",
                "Evidence excerpt is not contained in the persisted utterance.",
            )
        )

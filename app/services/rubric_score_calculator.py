"""Deterministic Decimal scoring for validated rubric observations."""

from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from app.schemas.negotiation_standard import NegotiationStandardContent
from app.schemas.rubric_evaluation import CanonicalEvaluationResult, RubricCategoryScore
from app.services.rubric_observation_validator import ValidatedObservation


class ScoreInvariantError(ValueError):
    """Raised when a published rubric violates a scoring invariant."""


def _money(value: Decimal) -> Decimal:
    """Round a score contribution to two decimal places using half-up semantics."""
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _penalty_total(category: Any, block: Any) -> int:
    """Calculate configured deductions with per-violation finding caps."""
    occurrences = Counter(finding.violation_id for finding in category.violations)
    return sum(
        min(occurrences.get(penalty.violation_id, 0), penalty.max_occurrences) * penalty.deduction
        for penalty in block.penalties
    )


def calculate_rubric_score(
    validated: ValidatedObservation,
    snapshot: NegotiationStandardContent | dict[str, Any] | None = None,
) -> CanonicalEvaluationResult:
    """Calculate penalties, weighted contributions, and pass/fail from backend data."""
    if snapshot is None:
        rubric = validated.snapshot
    elif isinstance(snapshot, NegotiationStandardContent):
        rubric = snapshot
    else:
        rubric = NegotiationStandardContent.model_validate(snapshot)
    if rubric.model_dump() != validated.snapshot.model_dump():
        raise ScoreInvariantError(
            "Scoring snapshot does not match the validated observation snapshot"
        )
    snapshot = rubric
    weight_total = sum(block.weight for block in snapshot.blocks)
    if weight_total != 100:
        raise ScoreInvariantError("Published rubric weights must total exactly 100")

    blocks = {block.id: block for block in snapshot.blocks}
    categories = []
    weighted_total = Decimal("0")
    ordered_observations = sorted(
        validated.observation.categories,
        key=lambda item: (blocks[item.rubric_block_id].display_order, item.rubric_block_id),
    )
    for observation in ordered_observations:
        block = blocks[observation.rubric_block_id]
        raw_score = observation.raw_score
        if raw_score is None:
            penalty_total = 0
            penalized_score = None
            contribution = Decimal("0.00")
            category_passed = False
        else:
            penalty_total = _penalty_total(observation, block)
            penalized_score = max(0, raw_score - penalty_total)
            contribution = _money(Decimal(penalized_score) * Decimal(block.weight) / Decimal(100))
            category_passed = penalized_score >= block.passing_score
        weighted_total += contribution
        categories.append(
            RubricCategoryScore(
                rubric_block_id=block.id,
                category=block.category,
                raw_score=raw_score,
                penalty_total=penalty_total,
                penalized_score=penalized_score,
                weight=block.weight,
                weighted_contribution=contribution,
                passing_score=block.passing_score,
                passed=category_passed,
                evidence=observation.evidence,
                strengths=observation.strengths,
                violations=observation.violations,
                failed_criteria=observation.failed_criteria,
                recommendation_inputs=observation.recommendation_inputs,
            )
        )

    total = _money(weighted_total)
    passed = bool(
        validated.observation.status == "evaluated"
        and total >= Decimal(snapshot.overall_passing_score)
        and all(category.passed for category in categories)
    )
    return CanonicalEvaluationResult(
        status=validated.observation.status,
        summary=validated.observation.summary,
        categories=categories,
        weighted_total=total,
        passing_score=snapshot.overall_passing_score,
        passed=passed,
        applied_techniques=validated.observation.applied_techniques,
        missed_opportunities=validated.observation.missed_opportunities,
    )

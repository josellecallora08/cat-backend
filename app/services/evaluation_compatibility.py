"""Adapters from canonical rubric results to the legacy review/Jinja contract."""

import re
from uuid import UUID

from pydantic import BaseModel, ConfigDict, model_validator

from app.schemas.rubric_evaluation import (
    CanonicalEvaluationResult,
    RubricAppliedTechniques,
    RubricMissedOpportunities,
    RubricRecommendation,
)


class LegacyReview(BaseModel):
    """Legacy review fields consumed by the supplied Jinja renderer."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    applied_techniques: RubricAppliedTechniques
    missed_opportunities: RubricMissedOpportunities

    @model_validator(mode="after")
    def lists_are_disjoint(self) -> "LegacyReview":
        applied = {
            item.technique_name.casefold() for item in self.applied_techniques.techniques_used
        }
        missed = {
            item.technique_name.casefold() for item in self.missed_opportunities.missed_techniques
        }
        if applied & missed:
            raise ValueError("legacy applied and missed techniques must be disjoint")
        return self


def to_legacy_review(
    canonical_result: CanonicalEvaluationResult | dict,
) -> LegacyReview:
    """Derive legacy fields without feeding compatibility data back into scoring."""
    canonical = (
        canonical_result
        if isinstance(canonical_result, CanonicalEvaluationResult)
        else CanonicalEvaluationResult.model_validate(canonical_result)
    )
    return LegacyReview(
        summary=canonical.summary,
        applied_techniques=canonical.applied_techniques,
        missed_opportunities=canonical.missed_opportunities,
    )


def to_jinja_context(canonical_result: CanonicalEvaluationResult | LegacyReview | dict) -> dict:
    """Return only the legacy variables expected by the Jinja template."""
    review = (
        canonical_result
        if isinstance(canonical_result, LegacyReview)
        else to_legacy_review(canonical_result)
    )
    return {
        "summary": review.summary,
        "applied_techniques": review.applied_techniques.model_dump(mode="json"),
        "missed_opportunities": review.missed_opportunities.model_dump(mode="json"),
    }


def render_legacy_review(canonical_result: CanonicalEvaluationResult | dict) -> str:
    """Render a small text fallback using the same legacy section order."""
    context = to_jinja_context(canonical_result)
    applied = context["applied_techniques"]["techniques_used"]
    missed = context["missed_opportunities"]["missed_techniques"]
    applied_text = (
        "\n".join(
            f"- {item['technique_name']}: {item['execution_description']}" for item in applied
        )
        or f"- {context['applied_techniques']['reason_if_empty']}"
    )
    missed_text = (
        "\n".join(f"- {item['technique_name']}: {item['reason']}" for item in missed)
        or f"- {context['missed_opportunities']['reason_if_empty']}"
    )
    return (
        f"Applied Technique Delivery:\n{applied_text}\n\n"
        f"Missed Opportunity:\n{missed_text}\n\n"
        f"Summary:\n{context['summary']}"
    )


# Conservative output sanitization: recommendations never echo transcript text.
def redact_recommendation_text(value: str) -> str:
    """Remove common and obfuscated PII forms before text is persisted."""
    redacted = re.sub(
        r"[\w.+-]+\s*(?:@|\[at\]|\(at\)|\s+at\s+)\s*[\w.-]+\s*(?:\.|\[dot\]|\(dot\)|\s+dot\s+)\s*[A-Za-z]{2,}",
        "[redacted email]",
        value,
        flags=re.IGNORECASE,
    )
    redacted = re.sub(
        r"(?:[$₱€£]\s*|\b(?:PHP|USD|EUR)\s*)\d[\d,]*(?:\.\d+)?",
        "[redacted amount]",
        redacted,
        flags=re.IGNORECASE,
    )
    redacted = re.sub(
        r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)",
        "[redacted phone]",
        redacted,
    )
    redacted = re.sub(r"\b\d[\d,]*(?:\.\d+)?\b", "[redacted number]", redacted)
    redacted = re.sub(
        r"\b(?:Mr|Mrs|Ms|Dr)\.?\s+[A-Z][a-z]+",
        "[redacted person]",
        redacted,
    )
    redacted = re.sub(
        r"\b[A-Z][a-z]+\s+(?=(?:from|in|at|owes|contact)\b)",
        "[redacted person] ",
        redacted,
    )
    redacted = re.sub(
        r"\b(?:in|at|from)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",
        "[redacted place]",
        redacted,
    )
    redacted = re.sub(r"\b[A-Z][A-Z0-9_]{2,}\b", "[redacted name]", redacted)
    return redacted.strip()


SAFE_EXPLANATION = "The cited rubric criterion needs improvement based on validated evidence."
SAFE_RESPONSE = "At the referenced moment, follow the cited criterion and pinned rubric guidance."
SAFE_ADVICE = "Use the pinned rubric guidance at the cited evidence moment."


def _safe_recommendation_text(value: str, fallback: str) -> str:
    """Redact generated text and guarantee a bounded non-empty fallback."""
    sanitized = redact_recommendation_text(value)[:10_000].strip()
    return sanitized or fallback


class RecommendationValidationError(ValueError):
    """Raised when a recommendation is not grounded in the pinned rubric result."""


def build_rubric_recommendations(
    canonical_result: CanonicalEvaluationResult | dict,
    snapshot: dict,
    standard_version_id: UUID | None = None,
    standard_version_number: int | None = None,
) -> list[RubricRecommendation]:
    """Build prioritized, criterion-grounded recommendations from validated data."""
    from app.schemas.negotiation_standard import NegotiationStandardContent

    canonical = (
        canonical_result
        if isinstance(canonical_result, CanonicalEvaluationResult)
        else CanonicalEvaluationResult.model_validate(canonical_result)
    )
    rubric = NegotiationStandardContent.model_validate(snapshot)
    if canonical.status == "not_applicable":
        return []
    blocks = {block.id: block for block in rubric.blocks}
    candidates: list[tuple[int, int, int, str, str, int, RubricRecommendation]] = []

    for category in canonical.categories:
        block = blocks[category.rubric_block_id]
        criteria = {item.id: item for item in block.positive_behaviors + block.violations}
        criterion_evidence: dict[str, set[int]] = {}
        findings: dict[str, str] = {}
        for finding in category.strengths:
            criterion_evidence.setdefault(finding.criterion_id, set()).update(
                finding.evidence_sequence_numbers
            )
            findings[finding.criterion_id] = finding.explanation
        for finding in category.violations:
            criterion_evidence.setdefault(finding.violation_id, set()).update(
                finding.evidence_sequence_numbers
            )
            findings[finding.violation_id] = finding.explanation

        evidence_sequences = {item.sequence_number for item in category.evidence}
        inputs = list(category.recommendation_inputs)
        if not inputs:
            inputs.extend(
                {
                    "criterion_id": violation.violation_id,
                    "transcript_sequence_number": violation.evidence_sequence_numbers[0],
                    "need": violation.explanation,
                }
                for violation in category.violations
                if violation.evidence_sequence_numbers
            )
            inputs.extend(
                {
                    "criterion_id": criterion_id,
                    "transcript_sequence_number": min(criterion_evidence[criterion_id]),
                    "need": findings[criterion_id],
                }
                for criterion_id in category.failed_criteria
                if criterion_id in criterion_evidence
            )

        gap = max(0, block.passing_score - (category.penalized_score or 0))
        for recommendation in inputs:
            item = (
                recommendation
                if hasattr(recommendation, "criterion_id")
                else type("Input", (), recommendation)()
            )
            if item.criterion_id not in criteria:
                raise RecommendationValidationError(
                    f"Unsupported criterion '{item.criterion_id}' in block '{block.id}'"
                )
            allowed_sequences = criterion_evidence.get(item.criterion_id)
            if allowed_sequences is None:
                if item.criterion_id not in category.failed_criteria:
                    raise RecommendationValidationError(
                        f"Recommendation for '{item.criterion_id}' lacks criterion evidence"
                    )
                allowed_sequences = evidence_sequences
            if item.transcript_sequence_number not in allowed_sequences:
                raise RecommendationValidationError(
                    f"Recommendation for '{item.criterion_id}' is not linked to criterion evidence"
                )
            criterion = criteria[item.criterion_id]
            source = next(
                (
                    evidence
                    for evidence in category.evidence
                    if evidence.sequence_number == item.transcript_sequence_number
                ),
                None,
            )
            explanation = _safe_recommendation_text(
                findings.get(item.criterion_id, item.need),
                SAFE_EXPLANATION,
            )
            response = _safe_recommendation_text(
                f"At the referenced moment, apply {criterion.name}: "
                f"{criterion.description} Address the identified need: {item.need}. "
                f"{block.recommendation_guidance}",
                SAFE_RESPONSE,
            )
            advice = _safe_recommendation_text(
                f"{block.recommendation_guidance} "
                f"Use this criterion's evidence instruction: {criterion.evidence_instructions}",
                SAFE_ADVICE,
            )
            generated = RubricRecommendation(
                rubric_block_id=block.id,
                block_name=block.category,
                criterion_id=item.criterion_id,
                criterion_name=criterion.name,
                display_order=block.display_order,
                evidence_sequence_number=item.transcript_sequence_number,
                source_speaker=source.speaker if source else None,
                source_excerpt=source.excerpt if source else None,
                explanation=explanation,
                recommended_response=response,
                coaching_advice=advice,
                standard_version_id=standard_version_id,
                standard_version_number=standard_version_number,
            )
            candidates.append(
                (
                    gap,
                    category.penalty_total,
                    block.display_order,
                    block.id,
                    item.criterion_id,
                    item.transcript_sequence_number,
                    generated,
                )
            )

    candidates.sort(
        key=lambda value: (
            -value[0],
            -value[1],
            value[2],
            value[3],
            value[4],
            value[5],
        )
    )
    return [item[-1] for item in candidates]

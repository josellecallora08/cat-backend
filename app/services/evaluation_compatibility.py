"""Adapters from canonical rubric results to the legacy review/Jinja contract."""

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
            item.technique_name.casefold()
            for item in self.applied_techniques.techniques_used
        }
        missed = {
            item.technique_name.casefold()
            for item in self.missed_opportunities.missed_techniques
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
    applied_text = "\n".join(
        f"- {item['technique_name']}: {item['execution_description']}"
        for item in applied
    ) or f"- {context['applied_techniques']['reason_if_empty']}"
    missed_text = "\n".join(
        f"- {item['technique_name']}: {item['reason']}"
        for item in missed
    ) or f"- {context['missed_opportunities']['reason_if_empty']}"
    return (
        f"Applied Technique Delivery:\n{applied_text}\n\n"
        f"Missed Opportunity:\n{missed_text}\n\n"
        f"Summary:\n{context['summary']}"
    )


# Conservative output sanitization: recommendations never echo transcript text.
def redact_recommendation_text(value: str) -> str:
    """Remove common PII forms before recommendation text is persisted."""
    import re

    redacted = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[redacted email]", value)
    redacted = re.sub(r"(?:[$₱€£]\s*|\b(?:PHP|USD|EUR)\s*)\d[\d,]*(?:\.\d+)?", "[redacted amount]", redacted, flags=re.IGNORECASE)
    redacted = re.sub(r"\b\d[\d,]*(?:\.\d+)?\b", "[redacted number]", redacted)
    redacted = re.sub(r"\b(?:Mr|Mrs|Ms|Dr)\.?\s+[A-Z][a-z]+", "[redacted person]", redacted)
    redacted = re.sub(r"\b(?:in|at|from)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", "[redacted place]", redacted)
    redacted = re.sub(r"\b[A-Z][a-z]{2,}\b", "[redacted name]", redacted)
    return redacted.strip()


class RecommendationValidationError(ValueError):
    """Raised when a recommendation is not grounded in the pinned rubric result."""



def build_rubric_recommendations(
    canonical_result: CanonicalEvaluationResult | dict,
    snapshot: dict,
    standard_version_id: UUID | None = None,
    standard_version_number: int | None = None,
) -> list[RubricRecommendation]:
    """Build prioritized, evidence-sequence-linked recommendations from validated data."""
    from app.schemas.negotiation_standard import NegotiationStandardContent

    canonical = (
        canonical_result
        if isinstance(canonical_result, CanonicalEvaluationResult)
        else CanonicalEvaluationResult.model_validate(canonical_result)
    )
    rubric = NegotiationStandardContent.model_validate(snapshot)
    blocks = {block.id: block for block in rubric.blocks}
    candidates: list[tuple[int, int, int, RubricRecommendation]] = []

    for category in canonical.categories:
        block = blocks[category.rubric_block_id]
        criteria = {
            item.id: item
            for item in block.positive_behaviors + block.violations
        }
        evidence_sequences = {item.sequence_number for item in category.evidence}
        findings = {
            item.criterion_id: item.explanation for item in category.strengths
        }
        findings.update({item.violation_id: item.explanation for item in category.violations})
        inputs = list(category.recommendation_inputs)
        if not inputs:
            for violation in category.violations:
                if violation.evidence_sequence_numbers:
                    inputs.append({
                        "criterion_id": violation.violation_id,
                        "transcript_sequence_number": violation.evidence_sequence_numbers[0],
                        "need": violation.explanation,
                    })
            for criterion_id in category.failed_criteria:
                if criterion_id in findings:
                    sequence = next(
                        item.evidence_sequence_numbers[0]
                        for item in category.strengths + category.violations
                        if getattr(item, "criterion_id", getattr(item, "violation_id", None)) == criterion_id
                    )
                    inputs.append({
                        "criterion_id": criterion_id,
                        "transcript_sequence_number": sequence,
                        "need": findings[criterion_id],
                    })

        gap = max(0, block.passing_score - (category.penalized_score or 0))
        valid_criteria = set(criteria)
        for recommendation in inputs:
            item = recommendation if hasattr(recommendation, "criterion_id") else type("Input", (), recommendation)()
            if item.criterion_id not in valid_criteria:
                raise RecommendationValidationError(
                    f"Unsupported criterion '{item.criterion_id}' in block '{block.id}'"
                )
            if item.transcript_sequence_number not in evidence_sequences:
                raise RecommendationValidationError(
                    f"Recommendation for '{item.criterion_id}' is not linked to category evidence"
                )
            criterion = criteria[item.criterion_id]
            explanation = findings.get(item.criterion_id, item.need)
            generated = RubricRecommendation(
                rubric_block_id=block.id,
                block_name=block.category,
                criterion_id=item.criterion_id,
                criterion_name=criterion.name,
                display_order=block.display_order,
                evidence_sequence_number=item.transcript_sequence_number,
                explanation=redact_recommendation_text(explanation),
                recommended_response="I understand your concern. Let us review the available options together.",
                coaching_advice=redact_recommendation_text(
                    f"{block.recommendation_guidance} Focus on the validated evidence at this moment."
                ),
                standard_version_id=standard_version_id,
                standard_version_number=standard_version_number,
            )
            candidates.append((gap, category.penalty_total, block.display_order, generated))

    candidates.sort(key=lambda value: (-value[0], -value[1], value[2], value[3].evidence_sequence_number))
    return [item for _, _, _, item in candidates]

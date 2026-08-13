"""Unit tests for the negotiation standard contract and aggregate validator."""

import pytest
from pydantic import ValidationError

from app.schemas.negotiation_standard import (
    NegotiationStandardContent,
    RubricBlock,
    RubricCriterion,
    RubricPenalty,
    RubricViolation,
)
from app.services.negotiation_standard_validator import validate_standard


def _criterion(identifier: str = "move-forward") -> dict[str, str]:
    return {
        "id": identifier,
        "name": "Move forward",
        "description": "Redirects the conversation toward resolution.",
        "evidence_instructions": "Cite the redirect and surrounding turn.",
    }


def _violation(identifier: str = "legal-threat") -> dict[str, str]:
    return {
        "id": identifier,
        "name": "Unsupported legal threat",
        "description": "States an unverified legal consequence as certain.",
        "evidence_instructions": "Cite the exact claim.",
    }


def _block(identifier: str, weight: int, order: int = 0) -> dict:
    return {
        "id": identifier,
        "category": identifier.replace("-", " ").title(),
        "weight": weight,
        "passing_score": 70,
        "scoring_instructions": "Score only behavior supported by transcript evidence.",
        "positive_behaviors": [_criterion(f"{identifier}-positive")],
        "violations": [_violation(f"{identifier}-violation")],
        "penalties": [
            {
                "violation_id": f"{identifier}-violation",
                "deduction": 20,
                "max_occurrences": 1,
            }
        ],
        "recommendation_guidance": "Recommend an evidence-grounded alternative.",
        "display_order": order,
    }


def _content(*blocks: dict) -> NegotiationStandardContent:
    return NegotiationStandardContent(
        overall_passing_score=70,
        blocks=list(blocks),
    )


def test_valid_contract_accepts_numeric_boundaries() -> None:
    content = NegotiationStandardContent(
        overall_passing_score=0,
        blocks=[
            RubricBlock(
                id="boundary",
                category="Boundary",
                weight=100,
                passing_score=100,
                scoring_instructions="Score the observed behavior.",
                positive_behaviors=[RubricCriterion(**_criterion())],
                violations=[RubricViolation(**_violation())],
                penalties=[
                    RubricPenalty(violation_id="legal-threat", deduction=0, max_occurrences=0)
                ],
                recommendation_guidance="Use a safer alternative.",
                display_order=0,
            )
        ],
    )

    result = validate_standard(content, for_publish=True)

    assert result.valid is True
    assert result.weight_total == 100


@pytest.mark.parametrize(
    "field,value", [("weight", -1), ("weight", 101), ("passing_score", -1), ("passing_score", 101)]
)
def test_numeric_boundaries_are_rejected(field: str, value: int) -> None:
    block = _block("opening", 100)
    block[field] = value

    with pytest.raises(ValidationError):
        _content(block)


def test_extra_fields_are_rejected() -> None:
    block = _block("opening", 100)
    block["unexpected"] = True

    with pytest.raises(ValidationError):
        _content(block)


def test_whitespace_names_and_invalid_ids_are_rejected() -> None:
    block = _block("opening", 100)
    block["category"] = "   "

    with pytest.raises(ValidationError):
        _content(block)

    block = _block("Opening", 100)
    with pytest.raises(ValidationError):
        _content(block)


def test_draft_and_publish_validation_report_weight_errors() -> None:
    draft = validate_standard(_content(_block("opening", 90)), for_publish=False)
    published = validate_standard(
        _content(_block("opening", 55), _block("resolution", 55)),
        for_publish=True,
    )

    assert draft.valid is False
    assert draft.weight_total == 90
    assert any(error.code == "weights_must_total_100" for error in draft.errors)
    assert published.valid is False
    assert published.weight_total == 110


def test_structural_validation_returns_all_errors() -> None:
    first = _block("duplicate", 40, 0)
    second = _block("duplicate", 40, 0)
    second["category"] = first["category"].lower()
    second["penalties"][0]["violation_id"] = "missing"

    result = validate_standard(_content(first, second), for_publish=False)
    codes = {error.code for error in result.errors}

    assert result.valid is False
    assert {"duplicate", "unknown_reference", "weights_must_total_100"}.issubset(codes)


def test_no_blocks_is_required() -> None:
    result = validate_standard(_content(), for_publish=True)

    assert result.valid is False
    assert result.errors[0].code == "required"

"""Property tests for legacy and too-short report compatibility semantics."""

from hypothesis import given, settings
from hypothesis import strategies as st

from app.schemas.report import EvaluationKind, ScoreStatus
from app.services.report_service import ReportService


@settings(max_examples=100)
@given(
    transcript_length=st.integers(min_value=0, max_value=3),
    legacy_score=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
    too_short_score=st.floats(
        min_value=0,
        max_value=100,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_legacy_and_too_short_results_remain_semantically_distinct(
    transcript_length: int,
    legacy_score: float,
    too_short_score: float,
) -> None:
    """Feature: report-quality-release-gates, Property 6.

    Legacy origin is represented by missing current-version metadata, while a
    too-short transcript is represented by not-applicable score semantics.
    """
    service = ReportService.__new__(ReportService)
    legacy_result = {
        "is_too_short": False,
        "rubric_result": {"status": "evaluated"},
        "overall_score": legacy_score,
        "passed": True,
        "negotiation_standard_version_id": None,
    }
    too_short_result = {
        "is_too_short": transcript_length < 4,
        "rubric_result": {"status": "not_applicable"},
        "overall_score": too_short_score,
        "passed": False,
    }

    legacy_version = service._version(None, legacy_result)  # type: ignore[arg-type]
    too_short_status = service._score_status(too_short_result)

    assert legacy_version.kind is EvaluationKind.LEGACY
    assert service._score_status(legacy_result) is ScoreStatus.EVALUATED
    assert too_short_status is ScoreStatus.NOT_APPLICABLE

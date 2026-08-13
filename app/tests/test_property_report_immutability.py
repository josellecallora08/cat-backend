"""Property tests for historical evaluation immutability."""

from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

from hypothesis import given, settings
from hypothesis import strategies as st

from app.services.report_service import ReportService


TEXT = st.text(min_size=1, max_size=40)
SCORE = st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False)
SNAPSHOT = st.fixed_dictionaries(
    {
        "standard_id": TEXT,
        "blocks": st.lists(
            st.fixed_dictionaries(
                {"id": TEXT, "label": TEXT},
                optional={"weight": SCORE},
            ),
            max_size=4,
        ),
    }
)
EVIDENCE = st.lists(TEXT, max_size=4)


@settings(max_examples=100)
@given(
    score=SCORE,
    evidence=EVIDENCE,
    historical_snapshot=SNAPSHOT,
    later_standard_name=TEXT,
    later_version_number=st.integers(min_value=1, max_value=1000),
)
def test_historical_evaluation_is_immutable_after_later_configuration_changes(
    score: float,
    evidence: list[str],
    historical_snapshot: dict[str, object],
    later_standard_name: str,
    later_version_number: int,
) -> None:
    """Feature: report-quality-release-gates, Property 5.

    A later configuration is not allowed to replace the persisted evaluation
    result, evidence, snapshot, or pinned version metadata used by a report.
    """
    version_id = uuid4()
    pinned_version = SimpleNamespace(
        version_number=7,
        standard=SimpleNamespace(name="Historical standard"),
    )
    session = SimpleNamespace(negotiation_standard_version=pinned_version)
    evaluation = {
        "overall_score": score,
        "strengths": evidence,
        "weaknesses": evidence,
        "standard_snapshot": deepcopy(historical_snapshot),
        "negotiation_standard_version_id": version_id,
        "standard_version_number": pinned_version.version_number,
        "standard_name": pinned_version.standard.name,
        "rubric_result": {"status": "evaluated", "score": score},
    }
    service = ReportService(None)  # type: ignore[arg-type]

    historical_report = {
        "evaluation": deepcopy(evaluation),
        "version": service._version(session, evaluation).model_dump(mode="json"),
    }

    later_configuration = {
        "name": later_standard_name,
        "version_number": later_version_number,
        "snapshot": {"standard_id": later_standard_name},
    }
    later_configuration["snapshot"] = deepcopy(later_configuration["snapshot"])

    assert historical_report == {
        "evaluation": evaluation,
        "version": historical_report["version"],
    }
    assert later_configuration != historical_report["evaluation"]
    assert historical_report["evaluation"]["standard_snapshot"] == historical_snapshot
    assert historical_report["evaluation"]["overall_score"] == score
    assert historical_report["evaluation"]["strengths"] == evidence
    assert historical_report["version"]["id"] == str(version_id)
    assert historical_report["version"]["number"] == pinned_version.version_number
    assert historical_report["version"]["name"] == pinned_version.standard.name

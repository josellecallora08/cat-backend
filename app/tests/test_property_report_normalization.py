"""Property tests for safe report section normalization."""

from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from app.schemas.report import ReportSectionName, SectionEnvelope, SectionState


SECTION_NAMES = st.sampled_from(list(ReportSectionName))
VALID_STATES = st.sampled_from(list(SectionState))
MALFORMED_STATES = st.text(min_size=1).filter(
    lambda value: value not in {state.value for state in SectionState}
)


@settings(max_examples=100)
@given(
    valid_name=SECTION_NAMES,
    valid_state=VALID_STATES,
    malformed_state=MALFORMED_STATES,
)
def test_malformed_section_is_rejected_without_affecting_valid_sibling(
    valid_name: ReportSectionName,
    valid_state: SectionState,
    malformed_state: str,
) -> None:
    """Feature: report-quality-release-gates, Property 3.

    A malformed section state must become a contract failure at the boundary,
    while a valid sibling remains normalizable independently.
    """
    valid_section = SectionEnvelope(name=valid_name, state=valid_state, data={"value": 1})
    assert valid_section.name == valid_name
    assert valid_section.state == valid_state

    try:
        SectionEnvelope(
            name=valid_name,
            state=malformed_state,
            data={"value": 2},
        )
    except ValidationError:
        return

    raise AssertionError("Malformed section state was accepted")

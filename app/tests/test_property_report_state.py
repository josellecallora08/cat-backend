"""Property tests for local report state transitions and retry preservation."""

import asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

from hypothesis import given, settings
from hypothesis import strategies as st

from app.schemas.report import ReportSectionName, SectionState
from app.services.report_service import ReportService


SECTION_NAMES = st.sampled_from(list(ReportSectionName))
LOADED_DATA = st.one_of(
    st.integers(), st.text(), st.dictionaries(st.text(max_size=12), st.integers())
)


@settings(max_examples=100)
@given(
    section_name=SECTION_NAMES,
    loaded_data=LOADED_DATA,
    sibling_data=LOADED_DATA,
    outcome=st.sampled_from(["loaded", "empty", "failed"]),
)
def test_section_transition_is_terminal_and_local(
    section_name: ReportSectionName,
    loaded_data: object,
    sibling_data: object,
    outcome: str,
) -> None:
    """Feature: report-quality-release-gates, Property 1:
    terminal and local section state transitions.

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 9.4, 11.1**
    """
    asyncio.run(_assert_terminal_and_local(section_name, loaded_data, sibling_data, outcome))


async def _assert_terminal_and_local(
    section_name: ReportSectionName,
    loaded_data: object,
    sibling_data: object,
    outcome: str,
) -> None:
    """Exercise one asynchronous section transition example."""
    service = ReportService(AsyncMock())
    sibling = service._loaded(ReportSectionName.METADATA, sibling_data)
    if section_name is ReportSectionName.METADATA:
        section_name = ReportSectionName.TRANSCRIPT

    loader = {
        "loaded": AsyncMock(return_value=loaded_data),
        "empty": AsyncMock(return_value=None),
        "failed": AsyncMock(side_effect=RuntimeError("temporary section failure")),
    }[outcome]
    result = await service._load_section(section_name, uuid4(), loader)

    assert result.state in {SectionState.LOADED, SectionState.EMPTY, SectionState.FAILED}
    if outcome == "loaded":
        assert result.state is SectionState.LOADED
        assert result.data == loaded_data
    elif outcome == "empty":
        assert result.state is SectionState.EMPTY
        assert result.data is None
    else:
        assert result.state is SectionState.FAILED
        assert result.failure is not None
    assert sibling.state is SectionState.LOADED
    assert sibling.data == sibling_data


@settings(max_examples=100)
@given(successful_data=LOADED_DATA, retry_data=st.one_of(LOADED_DATA, st.none()))
def test_successful_data_survives_partial_failure_and_retry(
    successful_data: object,
    retry_data: object | None,
) -> None:
    """Feature: report-quality-release-gates, Property 2:
    successful data survives partial failure and retry.

    **Validates: Requirements 1.5, 1.6, 2.4, 9.1, 9.2, 9.3**
    """
    asyncio.run(_assert_retry_preservation(successful_data, retry_data))


async def _assert_retry_preservation(successful_data: object, retry_data: object | None) -> None:
    """Exercise one asynchronous partial-failure and retry example."""
    service = ReportService(AsyncMock())
    session_id = uuid4()
    successful = service._loaded(ReportSectionName.TRANSCRIPT, successful_data)
    failed = await service._load_section(
        ReportSectionName.EVALUATION,
        session_id,
        AsyncMock(side_effect=RuntimeError("retryable failure")),
    )
    retried = await service._load_section(
        ReportSectionName.EVALUATION,
        session_id,
        AsyncMock(return_value=retry_data),
    )

    assert failed.state is SectionState.FAILED
    assert successful.state is SectionState.LOADED
    assert successful.data == successful_data
    if retry_data is None:
        assert retried.state is SectionState.EMPTY
        assert retried.data is None
    else:
        assert retried.state is SectionState.LOADED
        assert retried.data == retry_data
    assert retried.name is ReportSectionName.EVALUATION
    assert successful.name is not retried.name

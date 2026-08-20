"""HTTP contract tests for normalized report retrieval routes."""

from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_session as get_db_session
from app.main import app
from app.models.user import User
from app.schemas.report import (
    EvaluationKind,
    EvaluationVersionMetadata,
    ReportCompletion,
    ReportResponse,
    ReportSectionName,
    ReportSessionMetadata,
    ScoreStatus,
    SectionEnvelope,
    SectionState,
)
from app.services.auth import require_auth
from app.services.report_service import ReportService


@pytest.fixture
def report_response() -> ReportResponse:
    """Build a small valid aggregate response for route contract tests."""
    session_id = uuid4()
    return ReportResponse(
        session=ReportSessionMetadata(id=session_id, status="completed"),
        report_status=ReportCompletion.COMPLETE,
        score_status=ScoreStatus.UNAVAILABLE,
        evaluation_version=EvaluationVersionMetadata(kind=EvaluationKind.LEGACY),
        sections=[
            SectionEnvelope(
                name=ReportSectionName.METADATA,
                state=SectionState.LOADED,
                data={"status": "completed"},
            ),
            SectionEnvelope(
                name=ReportSectionName.TRANSCRIPT,
                state=SectionState.EMPTY,
                unavailable_reason="No transcript is available.",
            ),
        ],
    )


@pytest.mark.asyncio
async def test_get_report_and_section_routes_return_normalized_contract(
    report_response: ReportResponse,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aggregate and section routes expose the same stable section envelope."""
    user = User(id=uuid4(), email="agent@example.test", role="agent", is_active=True)

    async def fake_report(self: ReportService, session_id, current_user):
        return report_response

    async def fake_db():
        yield object()

    monkeypatch.setattr(ReportService, "get_report", fake_report)
    app.dependency_overrides[get_db_session] = fake_db
    app.dependency_overrides[require_auth] = lambda: user
    try:
        session_id = report_response.session.id
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            aggregate = await client.get(f"/api/sessions/{session_id}/report")
            section = await client.get(f"/api/sessions/{session_id}/report/sections/metadata")

        assert aggregate.status_code == 200
        assert aggregate.json()["report_status"] == "complete"
        assert section.status_code == 200
        assert section.json()["name"] == "metadata"
        assert section.json()["state"] == "loaded"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_report_section_route_rejects_unknown_section_name() -> None:
    """Unknown section identifiers fail validation instead of inferring data."""
    app.dependency_overrides[require_auth] = lambda: User(
        id=uuid4(), email="agent@example.test", role="agent", is_active=True
    )
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/api/sessions/{uuid4()}/report/sections/not-a-section")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422

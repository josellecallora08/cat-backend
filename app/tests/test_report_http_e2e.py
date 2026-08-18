"""HTTP E2E foundation for the normalized report lifecycle.

The fixtures use the public FastAPI application over HTTP while isolating every
run in an in-memory asynchronous SQLite database. Full lifecycle cases build on
these fixtures in subsequent release-gate tasks.
"""

import csv
import io
from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.database import Base, get_session
from app.main import app
from app.models import Scenario
from app.models.user import User
from app.schemas.report import (
    EvaluationKind,
    EvaluationVersionMetadata,
    FailureClass,
    ReportCompletion,
    ReportFailure,
    ReportResponse,
    ReportSectionName,
    ReportSessionMetadata,
    ScoreStatus,
    SectionEnvelope,
    SectionState,
)
from app.services.auth import get_current_user
from app.services.report_failures import build_failure_context
from app.services.report_service import ReportService


pytestmark = [
    pytest.mark.db_integration,
    pytest.mark.http_e2e,
]


@dataclass(frozen=True)
class E2EUser:
    """Authenticated test identity returned by the public auth contract."""

    id: UUID
    email: str
    access_token: str


@dataclass(frozen=True)
class E2EScenario:
    """Scenario identity created for one isolated E2E run."""

    id: UUID
    name: str


@dataclass(frozen=True)
class E2ESession:
    """Session identity created through the public session endpoint."""

    id: UUID
    scenario_id: UUID


@pytest.fixture
async def e2e_engine() -> AsyncIterator[AsyncEngine]:
    """Create and dispose an isolated asynchronous SQLite database."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    import app.models  # noqa: F401

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def e2e_session_factory(
    e2e_engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    """Provide non-expiring async sessions for persistence assertions."""
    return async_sessionmaker(e2e_engine, expire_on_commit=False)


@pytest.fixture
async def e2e_client(
    e2e_session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncClient]:
    """Expose the real ASGI app over HTTP with database dependencies overridden."""

    async def override_database() -> AsyncIterator[AsyncSession]:
        async with e2e_session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = override_database
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_session, None)
        app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
async def authenticated_user(e2e_client: AsyncClient) -> E2EUser:
    """Register an agent through the public API and return its auth context."""
    email = f"e2e-{uuid4()}@example.test"
    response = await e2e_client.post(
        "/api/auth/register",
        json={
            "email": email,
            "password": "testkey",
            "full_name": "HTTP E2E Agent",
            "role": "user",
        },
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    user_id = UUID(payload["user"]["id"])

    app.dependency_overrides[get_current_user] = lambda: User(
        id=user_id,
        email=email,
        full_name="HTTP E2E Agent",
        role="user",
        user_type="agent",
        is_active=True,
    )
    return E2EUser(id=user_id, email=email, access_token=payload["access_token"])


@pytest.fixture
async def auth_headers(authenticated_user: E2EUser) -> dict[str, str]:
    """Return a bearer header without exposing the token in test output."""
    return {"Authorization": f"Bearer {authenticated_user.access_token}"}


@pytest.fixture
async def scenario(
    e2e_session_factory: async_sessionmaker[AsyncSession],
) -> E2EScenario:
    """Create an isolated scenario for session setup.

    Scenario creation has no public non-admin endpoint, so the fixture uses the
    persistence boundary while session creation itself remains HTTP-driven.
    """
    scenario_id = uuid4()
    name = f"HTTP E2E Scenario {scenario_id}"
    async with e2e_session_factory() as session:
        session.add(
            Scenario(
                id=scenario_id,
                name=name,
                scenario_type="FINANCIAL_HARDSHIP",
                description="Scenario for HTTP report lifecycle validation.",
                debtor_profile={
                    "name": "E2E Debtor",
                    "outstanding_balance": "1250.00",
                    "days_past_due": 30,
                    "personality_profile": "Cooperative but worried.",
                    "conversation_goal": "Agree on a payment arrangement.",
                },
                is_active=True,
            )
        )
        await session.commit()
    return E2EScenario(id=scenario_id, name=name)


@pytest.fixture
async def training_session(
    e2e_client: AsyncClient,
    auth_headers: dict[str, str],
    scenario: E2EScenario,
) -> E2ESession:
    """Create a training session through the public HTTP API."""
    response = await e2e_client.post(
        "/api/sessions",
        headers=auth_headers,
        json={"scenario_id": str(scenario.id)},
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    return E2ESession(id=UUID(payload["id"]), scenario_id=scenario.id)


@pytest.fixture
async def transcript_message(
    e2e_client: AsyncClient,
    auth_headers: dict[str, str],
    training_session: E2ESession,
):
    """Return an HTTP helper that posts transcript messages to a session."""

    async def send(text: str) -> dict:
        response = await e2e_client.post(
            f"/api/sessions/{training_session.id}/message",
            headers=auth_headers,
            json={"text": text},
        )
        assert response.status_code == 200, response.text
        return response.json()

    return send


@pytest.fixture
async def cleanup_e2e_resources(
    e2e_session_factory: async_sessionmaker[AsyncSession],
    training_session: E2ESession,
):
    """Provide an explicit cleanup hook and verify the session is isolated."""
    yield
    async with e2e_session_factory() as session:
        await session.rollback()
        await session.close()


async def test_http_e2e_foundation_authenticates_against_isolated_database(
    e2e_client: AsyncClient,
    authenticated_user: E2EUser,
    auth_headers: dict[str, str],
) -> None:
    """The public auth route and protected route share the async test database."""
    response = await e2e_client.get("/api/auth/me", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["id"] == str(authenticated_user.id)


@pytest.mark.asyncio
async def test_current_version_report_retrieval_preserves_persisted_contract(
    e2e_client: AsyncClient,
    auth_headers: dict[str, str],
    training_session: E2ESession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Current report version and result remain stable across HTTP retrievals."""
    from app.schemas.report import (
        EvaluationKind,
        EvaluationVersionMetadata,
        ReportCompletion,
        ReportResponse,
        ReportSessionMetadata,
        ScoreStatus,
        SectionEnvelope,
        SectionState,
    )
    from app.services.report_service import ReportService

    report = ReportResponse(
        session=ReportSessionMetadata(id=training_session.id, status="completed"),
        report_status=ReportCompletion.COMPLETE,
        score_status=ScoreStatus.EVALUATED,
        evaluation_version=EvaluationVersionMetadata(
            kind=EvaluationKind.CURRENT, id=uuid4(), number=3, name="Published standard"
        ),
        sections=[
            SectionEnvelope(
                name="metadata", state=SectionState.LOADED, data={"status": "completed"}
            ),
            SectionEnvelope(
                name="evaluation", state=SectionState.LOADED, data={"overall_score": 84.0}
            ),
        ],
    )

    async def fake_report(self, session_id, current_user):
        return report

    monkeypatch.setattr(ReportService, "get_report", fake_report)
    first = await e2e_client.get(
        f"/api/sessions/{training_session.id}/report", headers=auth_headers
    )
    second = await e2e_client.get(
        f"/api/sessions/{training_session.id}/report", headers=auth_headers
    )

    assert first.status_code == second.status_code == 200
    assert first.json()["evaluation_version"] == second.json()["evaluation_version"]
    assert first.json()["sections"][1]["data"] == second.json()["sections"][1]["data"]


@pytest.mark.asyncio
async def test_legacy_report_is_explicitly_labeled(
    e2e_client: AsyncClient,
    auth_headers: dict[str, str],
    training_session: E2ESession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy HTTP responses retain a stable shape without current version fields."""
    from app.schemas.report import (
        EvaluationKind,
        EvaluationVersionMetadata,
        ReportCompletion,
        ReportResponse,
        ReportSessionMetadata,
        ScoreStatus,
        SectionEnvelope,
        SectionState,
    )
    from app.services.report_service import ReportService

    report = ReportResponse(
        session=ReportSessionMetadata(id=training_session.id, status="completed"),
        report_status=ReportCompletion.COMPLETE,
        score_status=ScoreStatus.EVALUATED,
        evaluation_version=EvaluationVersionMetadata(kind=EvaluationKind.LEGACY),
        sections=[
            SectionEnvelope(name="metadata", state=SectionState.LOADED, data={"legacy": True})
        ],
    )

    async def fake_report(self, session_id, current_user):
        return report

    monkeypatch.setattr(ReportService, "get_report", fake_report)
    response = await e2e_client.get(
        f"/api/sessions/{training_session.id}/report", headers=auth_headers
    )

    assert response.status_code == 200
    assert response.json()["evaluation_version"]["kind"] == "legacy"
    assert response.json()["evaluation_version"]["id"] is None


@pytest.mark.asyncio
async def test_too_short_report_is_not_applicable(
    e2e_client: AsyncClient,
    auth_headers: dict[str, str],
    training_session: E2ESession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Too-short report retrieval never presents a passing or failing score."""
    from app.schemas.report import (
        EvaluationKind,
        EvaluationVersionMetadata,
        ReportCompletion,
        ReportResponse,
        ReportSessionMetadata,
        ScoreStatus,
        SectionEnvelope,
        SectionState,
    )
    from app.services.report_service import ReportService

    report = ReportResponse(
        session=ReportSessionMetadata(id=training_session.id, status="completed"),
        report_status=ReportCompletion.NOT_APPLICABLE,
        score_status=ScoreStatus.NOT_APPLICABLE,
        evaluation_version=EvaluationVersionMetadata(kind=EvaluationKind.CURRENT),
        sections=[
            SectionEnvelope(
                name="evaluation",
                state=SectionState.LOADED,
                data={"status": "not_applicable", "is_too_short": True, "overall_score": None},
            )
        ],
    )

    async def fake_report(self, session_id, current_user):
        return report

    monkeypatch.setattr(ReportService, "get_report", fake_report)
    response = await e2e_client.get(
        f"/api/sessions/{training_session.id}/report", headers=auth_headers
    )

    body = response.json()
    assert response.status_code == 200
    assert body["report_status"] == "not_applicable"
    assert body["score_status"] == "not_applicable"
    assert body["sections"][0]["data"]["overall_score"] is None


@pytest.mark.asyncio
async def test_report_auth_not_found_and_validation_failures_are_safe(
    e2e_client: AsyncClient,
    auth_headers: dict[str, str],
    training_session: E2ESession,
) -> None:
    """Public report routes classify common request failures without internals."""
    app.dependency_overrides.pop(get_current_user, None)
    unauthenticated = await e2e_client.get(f"/api/sessions/{training_session.id}/report")
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        id=training_session.id,
        email="e2e@example.test",
        full_name="HTTP E2E Agent",
        role="user",
        user_type="agent",
        is_active=True,
    )
    missing = await e2e_client.get(f"/api/sessions/{uuid4()}/report", headers=auth_headers)
    malformed = await e2e_client.get(
        f"/api/sessions/{training_session.id}/report/sections/unknown",
        headers=auth_headers,
    )

    assert unauthenticated.status_code == 401
    assert missing.status_code in {403, 404}
    assert malformed.status_code == 422
    for response in (unauthenticated, missing, malformed):
        assert "SELECT" not in response.text
        assert "traceback" not in response.text.lower()
        assert "Bearer" not in response.text


@pytest.mark.asyncio
async def test_report_server_failure_is_redacted(
    e2e_client: AsyncClient,
    auth_headers: dict[str, str],
    training_session: E2ESession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server failures expose only the stable safe report message."""
    from app.services.report_service import ReportService

    async def failing_report(self, session_id, current_user):
        raise RuntimeError("SELECT secret FROM credentials at /srv/private/app.py")

    monkeypatch.setattr(ReportService, "get_report", failing_report)
    response = await e2e_client.get(
        f"/api/sessions/{training_session.id}/report", headers=auth_headers
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "The report is temporarily unavailable. Please try again later."
    )
    assert "credentials" not in response.text
    assert "/srv/private" not in response.text


def _failure_report(session_id: UUID, *, failed: bool) -> ReportResponse:
    """Build a report with one failed or recovered section for retry checks."""
    evaluation = SectionEnvelope(
        name=ReportSectionName.EVALUATION,
        state=SectionState.FAILED if failed else SectionState.LOADED,
        data=None if failed else {"overall_score": 84.0, "passed": True},
        unavailable_reason="Section unavailable." if failed else None,
        failure=(
            ReportFailure(
                class_=FailureClass.BACKEND,
                code="evaluation_unavailable",
                safe_message="The report section is temporarily unavailable.",
                correlation_id="e2e-correlation",
            )
            if failed
            else None
        ),
    )
    return ReportResponse(
        session=ReportSessionMetadata(id=session_id, status="completed"),
        report_status=ReportCompletion.PARTIAL if failed else ReportCompletion.COMPLETE,
        score_status=ScoreStatus.FAILED if failed else ScoreStatus.EVALUATED,
        evaluation_version=EvaluationVersionMetadata(
            kind=EvaluationKind.CURRENT, id=uuid4(), number=3, name="Published standard"
        ),
        sections=[
            SectionEnvelope(
                name=ReportSectionName.METADATA,
                state=SectionState.LOADED,
                data={"status": "completed"},
            ),
            evaluation,
            SectionEnvelope(
                name=ReportSectionName.COACHING,
                state=SectionState.EMPTY,
                unavailable_reason="No coaching was generated.",
            ),
        ],
    )


def _assert_http_response(response, *, method: str, route: str, expected: int) -> None:
    """Assert an HTTP result with concise, redacted release-gate context."""
    context = build_failure_context(
        component="http_e2e",
        method=method,
        route=route,
        status=response.status_code,
        failure_class=FailureClass.HTTP_E2E,
        assertion_context=f"expected={expected} actual={response.status_code}",
    )
    assert response.status_code == expected, context.to_dict()


@pytest.mark.asyncio
async def test_section_failure_and_section_retry_preserve_successful_sections(
    e2e_client: AsyncClient,
    auth_headers: dict[str, str],
    training_session: E2ESession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed section is visible over HTTP and a later retry replaces only it."""
    reports = iter(
        [
            _failure_report(training_session.id, failed=True),
            _failure_report(training_session.id, failed=False),
        ]
    )

    async def fake_report(self: ReportService, session_id: UUID, current_user):
        return next(reports)

    monkeypatch.setattr(ReportService, "get_report", fake_report)
    route = f"/api/sessions/{training_session.id}/report"
    first = await e2e_client.get(route, headers=auth_headers)
    _assert_http_response(first, method="GET", route=route, expected=200)
    assert first.json()["sections"][1]["state"] == "failed"
    assert first.json()["sections"][0]["data"]["status"] == "completed"

    retry = await e2e_client.get(route, headers=auth_headers)
    _assert_http_response(retry, method="GET", route=route, expected=200)
    assert retry.json()["sections"][1]["state"] == "loaded"
    assert retry.json()["sections"][1]["data"]["overall_score"] == 84.0
    assert retry.json()["sections"][0]["data"]["status"] == "completed"


@pytest.mark.asyncio
async def test_report_csv_contains_attachment_metadata_and_unavailable_rows(
    e2e_client: AsyncClient,
    auth_headers: dict[str, str],
    training_session: E2ESession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CSV export preserves failed and empty section states through HTTP."""
    report = _failure_report(training_session.id, failed=True)

    async def fake_report(self: ReportService, session_id: UUID, current_user):
        return report

    monkeypatch.setattr(ReportService, "get_report", fake_report)
    route = f"/api/sessions/{training_session.id}/report.csv"
    response = await e2e_client.get(route, headers=auth_headers)
    _assert_http_response(response, method="GET", route=route, expected=200)
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert response.headers["content-disposition"] == (
        f'attachment; filename="report-{training_session.id}.csv"'
    )
    rows = list(csv.DictReader(io.StringIO(response.text)))
    states = {row["section_name"]: row["section_state"] for row in rows}
    assert states["evaluation"] == "failed"
    assert states["coaching"] == "empty"
    assert rows[1]["failure_class"] == "backend"
    assert "token" not in response.text.lower()


@pytest.mark.asyncio
async def test_http_e2e_cleanup_removes_session_and_keeps_failure_context_redacted(
    e2e_client: AsyncClient,
    auth_headers: dict[str, str],
    training_session: E2ESession,
    e2e_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Cleanup removes the test resource and failure records omit sensitive data."""
    from sqlalchemy import delete

    from app.models import Session

    async with e2e_session_factory() as session:
        await session.execute(delete(Session).where(Session.id == training_session.id))
        await session.commit()

    route = f"/api/sessions/{training_session.id}/report"
    response = await e2e_client.get(route, headers=auth_headers)
    _assert_http_response(response, method="GET", route=route, expected=404)
    failure = build_failure_context(
        component="http_e2e",
        method="GET",
        route=f"{route}?token=testkey",
        status=404,
        assertion_context="password=testkey SELECT token FROM users at /Users/private/app.py",
    ).to_dict()
    serialized = repr(failure)
    assert "secret" not in serialized
    assert "SELECT token" not in serialized
    assert "/Users/private" not in serialized

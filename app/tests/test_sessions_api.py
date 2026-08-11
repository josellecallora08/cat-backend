"""Tests for session API endpoints.

Tests POST /api/sessions, GET /api/sessions/{id}, POST /api/sessions/{id}/end,
GET /api/sessions/{id}/transcript, GET /api/sessions/{id}/evaluation,
GET /api/sessions/{id}/coaching, GET /api/sessions/{id}/learning-plan.
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from app.database import get_session as get_db_session
from app.main import app
from app.models import (
    CoachingReport,
    Evaluation,
    LearningPlan,
    NegotiationStandard,
    NegotiationStandardVersion,
    Session,
    Transcript,
)
from app.services.auth import require_auth
from app.services.session_access import get_authorized_session


@pytest.fixture
async def client():
    """Async test client for the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture(autouse=True)
def clear_overrides():
    """Ensure dependency overrides are cleaned up after each test."""
    yield
    app.dependency_overrides.clear()


def _make_session(
    scenario_id: uuid.UUID | None = None,
    status: str = "active",
    persona_context: dict | None = None,
) -> Session:
    """Helper to create a Session model instance."""
    session = Session(
        id=uuid.uuid4(),
        scenario_id=scenario_id or uuid.uuid4(),
        agent_id=uuid.uuid4(),
        status=status,
        persona_context=persona_context
        if persona_context is not None
        else {
            "persona_id": str(uuid.uuid4()),
            "name": "Maria Santos",
            "communication_style": "anxious",
            "financial_circumstances": {
                "income_level": "low",
                "debt_amount": 5000,
                "reason_for_delinquency": "job loss",
            },
            "emotional_state": 2,
            "language": "EN",
        },
        created_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        ended_at=None,
    )
    return session


def _mock_db_returning_scalar(value):
    """Create a mock db session whose execute returns a scalar_one_or_none of `value`."""
    from unittest.mock import MagicMock

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = value
    mock_db.execute = AsyncMock(return_value=mock_result)
    return mock_db


def _mock_db_returning_scalars(values):
    """Create a mock db session whose execute returns scalars().all() of `values`."""
    from unittest.mock import MagicMock

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = values
    mock_result.scalars.return_value = mock_scalars
    mock_db.execute = AsyncMock(return_value=mock_result)
    return mock_db


@pytest.fixture(autouse=True)
def route_session_lookup_compat(monkeypatch):
    """Keep existing route tests pointed at the extracted lookup seam."""
    from app.api import sessions as sessions_api
    from app.services import session_access

    async def lookup(db, session_id):
        return await sessions_api.get_session_service(db, session_id)

    monkeypatch.setattr(session_access, "get_session", lookup)


def _override_db(mock_db):
    """Create an async generator override for get_db_session dependency."""

    async def _override():
        return mock_db

    return _override


class TestCreateSession:
    """Tests for POST /api/sessions."""

    async def test_creates_session_returns_201(self, client):
        scenario_id = uuid.uuid4()
        session = _make_session(scenario_id=scenario_id, status="pending")

        with patch(
            "app.api.sessions.create_session_service",
            new_callable=AsyncMock,
            return_value=session,
        ):
            response = await client.post(
                "/api/sessions",
                json={"scenario_id": str(scenario_id)},
            )

        assert response.status_code == 201
        data = response.json()
        assert data["id"] == str(session.id)
        assert data["scenario_id"] == str(scenario_id)
        assert data["status"] == "pending"
        assert data["persona"] is not None
        assert data["persona"]["name"] == "Maria Santos"
        assert data["persona"]["communication_style"] == "anxious"
        assert data["persona"]["emotional_state"] == "2"

    async def test_returns_404_for_invalid_scenario(self, client):
        with patch(
            "app.api.sessions.create_session_service",
            new_callable=AsyncMock,
            side_effect=ValueError("Scenario with id xyz not found or inactive"),
        ):
            response = await client.post(
                "/api/sessions",
                json={"scenario_id": str(uuid.uuid4())},
            )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_returns_422_for_missing_scenario_id(self, client):
        response = await client.post("/api/sessions", json={})
        assert response.status_code == 422


class TestGetSession:
    """Tests for GET /api/sessions/{id}."""

    @pytest.fixture(autouse=True)
    def _auth_override(self):
        """Override require_auth with an admin user for session detail tests."""
        from unittest.mock import MagicMock

        mock_user = MagicMock()
        mock_user.id = uuid.uuid4()
        mock_user.email = "admin@test.com"
        mock_user.role = "admin"
        mock_user.user_type = None
        app.dependency_overrides[require_auth] = lambda: mock_user
        yield
        app.dependency_overrides.pop(require_auth, None)

    async def test_returns_session_details(self, client):
        session = _make_session(status="active")

        with patch(
            "app.services.session_access.get_session",
            new_callable=AsyncMock,
            return_value=session,
        ):
            response = await client.get(f"/api/sessions/{session.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(session.id)
        assert data["status"] == "active"
        assert data["persona"]["name"] == "Maria Santos"
        assert data["ended_at"] is None

    async def test_returns_404_for_nonexistent_session(self, client):
        with patch(
            "app.api.sessions.get_session_service",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = await client.get(f"/api/sessions/{uuid.uuid4()}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_returns_session_without_persona_if_no_context(self, client):
        session = _make_session()
        session.persona_context = None

        with patch(
            "app.api.sessions.get_session_service",
            new_callable=AsyncMock,
            return_value=session,
        ):
            response = await client.get(f"/api/sessions/{session.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["persona"] is None

    async def test_returns_403_for_agent_accessing_other_session(self, client):
        """Agent cannot access session belonging to another user."""
        from unittest.mock import MagicMock

        agent_user = MagicMock()
        agent_user.id = uuid.uuid4()
        agent_user.email = "agent@test.com"
        agent_user.role = "user"
        agent_user.user_type = "agent"
        app.dependency_overrides[require_auth] = lambda: agent_user

        session = _make_session(status="active")
        # session.agent_id is a different UUID

        with patch(
            "app.api.sessions.get_session_service",
            new_callable=AsyncMock,
            return_value=session,
        ):
            response = await client.get(f"/api/sessions/{session.id}")

        assert response.status_code == 403
        assert "access denied" in response.json()["detail"].lower()

    async def test_agent_can_access_own_session(self, client):
        """Agent can access their own session."""
        from unittest.mock import MagicMock

        agent_user = MagicMock()
        agent_user.id = uuid.uuid4()
        agent_user.email = "agent@test.com"
        agent_user.role = "user"
        agent_user.user_type = "agent"
        app.dependency_overrides[require_auth] = lambda: agent_user

        session = _make_session(status="active")
        session.agent_id = agent_user.id

        with patch(
            "app.api.sessions.get_session_service",
            new_callable=AsyncMock,
            return_value=session,
        ):
            response = await client.get(f"/api/sessions/{session.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(session.id)


class TestSessionAccess:
    """Tests for the shared session artifact access policy."""

    async def test_session_access_allows_administrator_for_existing_session(self):
        session = _make_session()
        admin = SimpleNamespace(id=uuid.uuid4(), role="admin", user_type=None)

        with patch(
            "app.services.session_access.get_session",
            new_callable=AsyncMock,
            return_value=session,
        ):
            result = await get_authorized_session(AsyncMock(), session.id, admin)

        assert result is session

    async def test_session_access_allows_agent_who_owns_session(self):
        session = _make_session()
        agent = SimpleNamespace(id=session.agent_id, role="user", user_type="agent")

        with patch(
            "app.services.session_access.get_session",
            new_callable=AsyncMock,
            return_value=session,
        ):
            result = await get_authorized_session(AsyncMock(), session.id, agent)

        assert result is session

    async def test_session_access_denies_agent_who_does_not_own_session(self):
        session = _make_session()
        agent = SimpleNamespace(id=uuid.uuid4(), role="user", user_type="agent")

        with patch(
            "app.services.session_access.get_session",
            new_callable=AsyncMock,
            return_value=session,
        ):
            with pytest.raises(HTTPException) as error:
                await get_authorized_session(AsyncMock(), session.id, agent)

        assert error.value.status_code == 403
        assert error.value.detail == "Session access denied"

    async def test_session_access_allows_trainer_for_agent_in_active_campaign(self):
        session = _make_session()
        trainer = SimpleNamespace(id=uuid.uuid4(), role="user", user_type="trainer")
        campaign = SimpleNamespace(id=uuid.uuid4())

        with (
            patch(
                "app.services.session_access.get_session",
                new_callable=AsyncMock,
                return_value=session,
            ),
            patch(
                "app.services.session_access.get_trainer_campaign",
                new_callable=AsyncMock,
                return_value=campaign,
            ),
            patch(
                "app.services.session_access.get_trainer_campaign_agent_ids",
                new_callable=AsyncMock,
                return_value=[session.agent_id],
            ),
        ):
            result = await get_authorized_session(AsyncMock(), session.id, trainer)

        assert result is session

    @pytest.mark.parametrize("campaign", [None, SimpleNamespace(id=uuid.uuid4())])
    async def test_session_access_denies_trainer_without_access(self, campaign):
        session = _make_session()
        trainer = SimpleNamespace(id=uuid.uuid4(), role="user", user_type="trainer")
        campaign_agent_ids = [] if campaign is not None else None

        with (
            patch(
                "app.services.session_access.get_session",
                new_callable=AsyncMock,
                return_value=session,
            ),
            patch(
                "app.services.session_access.get_trainer_campaign",
                new_callable=AsyncMock,
                return_value=campaign,
            ),
            patch(
                "app.services.session_access.get_trainer_campaign_agent_ids",
                new_callable=AsyncMock,
                return_value=campaign_agent_ids or [],
            ) as get_agent_ids,
        ):
            with pytest.raises(HTTPException) as error:
                await get_authorized_session(AsyncMock(), session.id, trainer)

        assert error.value.status_code == 403
        assert error.value.detail == "Session access denied"
        if campaign is None:
            get_agent_ids.assert_not_awaited()

    async def test_session_access_returns_404_for_missing_session(self):
        session_id = uuid.uuid4()
        user = SimpleNamespace(id=uuid.uuid4(), role="admin", user_type=None)

        with patch(
            "app.services.session_access.get_session",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with pytest.raises(HTTPException) as error:
                await get_authorized_session(AsyncMock(), session_id, user)

        assert error.value.status_code == 404
        assert error.value.detail == f"Session {session_id} not found"


class TestEndSession:
    """Tests for POST /api/sessions/{id}/end."""

    async def test_ends_session_returns_completed(self, client):
        session = _make_session(status="completed")
        session.ended_at = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

        with patch(
            "app.api.sessions.end_session_service",
            new_callable=AsyncMock,
            return_value=session,
        ):
            response = await client.post(f"/api/sessions/{session.id}/end")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["ended_at"] is not None

    async def test_returns_404_for_nonexistent_session(self, client):
        with patch(
            "app.api.sessions.end_session_service",
            new_callable=AsyncMock,
            side_effect=ValueError("Session with id xyz not found"),
        ):
            response = await client.post(f"/api/sessions/{uuid.uuid4()}/end")

        assert response.status_code == 404

    async def test_returns_400_for_invalid_state_transition(self, client):
        with patch(
            "app.api.sessions.end_session_service",
            new_callable=AsyncMock,
            side_effect=ValueError(
                "Cannot end session with status 'completed'. "
                "Session must be 'pending' or 'active' to be ended."
            ),
        ):
            response = await client.post(f"/api/sessions/{uuid.uuid4()}/end")

        assert response.status_code == 400
        assert "cannot end session" in response.json()["detail"].lower()


class TestGetTranscript:
    """Tests for GET /api/sessions/{id}/transcript."""

    @pytest.fixture(autouse=True)
    def _auth_override(self):
        """Use an admin identity for existing artifact contract tests."""
        user = SimpleNamespace(id=uuid.uuid4(), role="admin", user_type=None)
        app.dependency_overrides[require_auth] = lambda: user
        yield
        app.dependency_overrides.pop(require_auth, None)

    async def test_returns_transcript_entries(self, client):
        session = _make_session()
        session_id = session.id

        transcripts = [
            Transcript(
                id=uuid.uuid4(),
                session_id=session_id,
                speaker="agent",
                utterance_text="Hello, this is regarding your account.",
                timestamp_ms=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
                sequence_number=0,
            ),
            Transcript(
                id=uuid.uuid4(),
                session_id=session_id,
                speaker="debtor",
                utterance_text="What do you want?",
                timestamp_ms=datetime(2024, 1, 15, 10, 0, 5, tzinfo=timezone.utc),
                sequence_number=1,
            ),
        ]

        mock_db = _mock_db_returning_scalars(transcripts)
        app.dependency_overrides[get_db_session] = _override_db(mock_db)

        with patch(
            "app.api.sessions.get_session_service",
            new_callable=AsyncMock,
            return_value=session,
        ):
            response = await client.get(f"/api/sessions/{session_id}/transcript")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["speaker"] == "agent"
        assert data[0]["text"] == "Hello, this is regarding your account."
        assert data[0]["sequence_number"] == 0
        assert data[1]["speaker"] == "debtor"
        assert data[1]["sequence_number"] == 1

    async def test_returns_404_for_nonexistent_session(self, client):
        with patch(
            "app.api.sessions.get_session_service",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = await client.get(f"/api/sessions/{uuid.uuid4()}/transcript")

        assert response.status_code == 404

    async def test_returns_empty_list_for_no_transcripts(self, client):
        session = _make_session()

        mock_db = _mock_db_returning_scalars([])
        app.dependency_overrides[get_db_session] = _override_db(mock_db)

        with patch(
            "app.api.sessions.get_session_service",
            new_callable=AsyncMock,
            return_value=session,
        ):
            response = await client.get(f"/api/sessions/{session.id}/transcript")

        assert response.status_code == 200
        assert response.json() == []


class TestGetEvaluation:
    """Tests for GET /api/sessions/{id}/evaluation."""

    @pytest.fixture(autouse=True)
    def _auth_override(self):
        """Use an admin identity for existing artifact contract tests."""
        user = SimpleNamespace(id=uuid.uuid4(), role="admin", user_type=None)
        app.dependency_overrides[require_auth] = lambda: user
        yield
        app.dependency_overrides.pop(require_auth, None)

    async def test_returns_evaluation_result(self, client):
        session = _make_session()
        session_id = session.id

        evaluation = Evaluation(
            id=uuid.uuid4(),
            session_id=session_id,
            overall_score=75.5,
            category_scores=[
                {
                    "category": "call_opening",
                    "score": 80,
                    "strengths": [],
                    "weaknesses": [],
                },
                {
                    "category": "compliance",
                    "score": 70,
                    "strengths": [],
                    "weaknesses": [],
                },
                {
                    "category": "empathy_communication",
                    "score": 85,
                    "strengths": [],
                    "weaknesses": [],
                },
                {
                    "category": "negotiation_resolution",
                    "score": 65,
                    "strengths": [],
                    "weaknesses": [],
                },
            ],
            strengths=[
                {
                    "description": "Good opening greeting",
                    "category": "call_opening",
                    "transcript_excerpt": "Hi, this is...",
                },
            ],
            weaknesses=[
                {
                    "description": "Missed compliance check",
                    "category": "compliance",
                    "transcript_excerpt": "I need you to pay now",
                },
            ],
            is_too_short=False,
        )

        mock_db = _mock_db_returning_scalar(evaluation)
        app.dependency_overrides[get_db_session] = _override_db(mock_db)

        with patch(
            "app.api.sessions.get_session_service",
            new_callable=AsyncMock,
            return_value=session,
        ):
            response = await client.get(f"/api/sessions/{session_id}/evaluation")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == str(session_id)
        assert data["overall_score"] == 75.5
        assert len(data["category_scores"]) == 4
        assert len(data["strengths"]) == 1
        assert len(data["weaknesses"]) == 1
        assert data["is_too_short"] is False

    async def test_returns_canonical_rubric_result_and_pinned_standard_metadata(self, client):
        session = _make_session()
        version_id = uuid.uuid4()
        standard = NegotiationStandard(
            id=uuid.uuid4(),
            campaign_id=uuid.uuid4(),
            name="Compliance Standard",
            status="published",
            revision=2,
            created_by=uuid.uuid4(),
            updated_by=uuid.uuid4(),
        )
        version = NegotiationStandardVersion(
            id=version_id,
            standard_id=standard.id,
            version_number=3,
            schema_version=1,
            snapshot={"schema_version": 1, "blocks": []},
            content_hash="a" * 64,
            created_by=uuid.uuid4(),
            published_by=uuid.uuid4(),
        )
        version.standard = standard
        canonical = {
            "status": "evaluated",
            "summary": "Evidence-grounded result.",
            "categories": [{
                "rubric_block_id": "compliance",
                "category": "Compliance",
                "raw_score": 80,
                "penalty_total": 5,
                "penalized_score": 75,
                "weight": 100,
                "weighted_contribution": 75,
                "passing_score": 70,
                "passed": True,
                "evidence": [{"sequence_number": 1, "speaker": "agent", "excerpt": "Offer", "explanation": "Clear option."}],
                "strengths": [],
                "violations": [],
                "failed_criteria": [],
                "recommendation_inputs": [],
            }],
            "weighted_total": 75,
            "passing_score": 70,
            "passed": True,
            "applied_techniques": {"techniques_used": [], "reason_if_empty": "None observed."},
            "missed_opportunities": {"missed_techniques": [], "reason_if_empty": "None missed."},
            "recommendations": [],
        }
        evaluation = Evaluation(
            id=uuid.uuid4(),
            session_id=session.id,
            overall_score=75,
            category_scores=[canonical["categories"][0]],
            strengths=[],
            weaknesses=[],
            negotiation_standard_version_id=version_id,
            standard_snapshot=version.snapshot,
            weighted_total=75,
            passing_score=70,
            passed=True,
            rubric_result=canonical,
            is_too_short=False,
        )
        evaluation.negotiation_standard_version = version
        mock_db = _mock_db_returning_scalar(evaluation)
        app.dependency_overrides[get_db_session] = _override_db(mock_db)

        with patch("app.api.sessions.get_session_service", new_callable=AsyncMock, return_value=session):
            response = await client.get(f"/api/sessions/{session.id}/evaluation")

        assert response.status_code == 200
        data = response.json()
        assert data["standard_name"] == "Compliance Standard"
        assert data["standard_version_number"] == 3
        assert data["weighted_total"] == 75
        assert data["passing_score"] == 70
        assert data["passed"] is True
        assert data["category_scores"] == []
        assert data["rubric_result"]["categories"][0]["weighted_contribution"] == 75

    async def test_returns_404_for_nonexistent_session(self, client):
        with patch(
            "app.api.sessions.get_session_service",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = await client.get(f"/api/sessions/{uuid.uuid4()}/evaluation")

        assert response.status_code == 404

    async def test_returns_404_when_no_evaluation_exists(self, client):
        session = _make_session()

        mock_db = _mock_db_returning_scalar(None)
        app.dependency_overrides[get_db_session] = _override_db(mock_db)

        with patch(
            "app.api.sessions.get_session_service",
            new_callable=AsyncMock,
            return_value=session,
        ):
            response = await client.get(f"/api/sessions/{session.id}/evaluation")

        assert response.status_code == 404
        assert "no evaluation" in response.json()["detail"].lower()


class TestGetCoaching:
    """Tests for GET /api/sessions/{id}/coaching."""

    @pytest.fixture(autouse=True)
    def _auth_override(self):
        """Use an admin identity for existing artifact contract tests."""
        user = SimpleNamespace(id=uuid.uuid4(), role="admin", user_type=None)
        app.dependency_overrides[require_auth] = lambda: user
        yield
        app.dependency_overrides.pop(require_auth, None)

    async def test_returns_coaching_report(self, client):
        session = _make_session()
        session_id = session.id

        report = CoachingReport(
            id=uuid.uuid4(),
            session_id=session_id,
            mistakes_by_category={
                "compliance": [
                    {
                        "transcript_position": 3,
                        "transcript_excerpt": "Pay now or else",
                        "category": "compliance",
                        "explanation": "Threatening language violates regulations",
                        "recommended_alternative": "I understand this is difficult. Let's discuss options.",
                    }
                ]
            },
            total_mistakes=1,
            no_mistakes=False,
        )

        mock_db = _mock_db_returning_scalar(report)
        app.dependency_overrides[get_db_session] = _override_db(mock_db)

        with patch(
            "app.api.sessions.get_session_service",
            new_callable=AsyncMock,
            return_value=session,
        ):
            response = await client.get(f"/api/sessions/{session_id}/coaching")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == str(session_id)
        assert data["total_mistakes"] == 1
        assert data["no_mistakes"] is False
        assert "compliance" in data["mistakes_by_category"]
        mistakes = data["mistakes_by_category"]["compliance"]
        assert len(mistakes) == 1
        assert mistakes[0]["transcript_position"] == 3

    async def test_returns_404_for_nonexistent_session(self, client):
        with patch(
            "app.api.sessions.get_session_service",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = await client.get(f"/api/sessions/{uuid.uuid4()}/coaching")

        assert response.status_code == 404

    async def test_returns_404_when_no_report_exists(self, client):
        session = _make_session()

        mock_db = _mock_db_returning_scalar(None)
        app.dependency_overrides[get_db_session] = _override_db(mock_db)

        with patch(
            "app.api.sessions.get_session_service",
            new_callable=AsyncMock,
            return_value=session,
        ):
            response = await client.get(f"/api/sessions/{session.id}/coaching")

        assert response.status_code == 404
        assert "no coaching report" in response.json()["detail"].lower()

    async def test_returns_canonical_rubric_coaching_without_legacy_category_duplicates(self, client):
        session = _make_session()
        version_id = uuid.uuid4()
        report = CoachingReport(
            id=uuid.uuid4(),
            session_id=session.id,
            mistakes_by_category={
                "_rubric_coaching": {
                    "standard_version_id": str(version_id),
                    "standard_version_number": 7,
                    "blocks": [{
                        "rubric_block_id": "custom-block",
                        "block_name": "Custom Block",
                        "display_order": 0,
                        "recommendations": [{
                            "rubric_block_id": "custom-block",
                            "block_name": "Custom Block",
                            "criterion_id": "custom-criterion",
                            "criterion_name": "Custom Criterion",
                            "display_order": 0,
                            "evidence_sequence_number": 3,
                            "explanation": "Needs work.",
                            "recommended_response": "Let us review this.",
                            "coaching_advice": "Use the criterion guidance.",
                            "standard_version_id": str(version_id),
                            "standard_version_number": 7,
                        }],
                    }],
                }
            },
            total_mistakes=1,
            no_mistakes=False,
        )
        app.dependency_overrides[get_db_session] = _override_db(_mock_db_returning_scalar(report))

        with patch(
            "app.api.sessions.get_session_service",
            new_callable=AsyncMock,
            return_value=session,
        ):
            response = await client.get(f"/api/sessions/{session.id}/coaching")

        assert response.status_code == 200
        data = response.json()
        assert data["mistakes_by_category"] == {}
        assert data["rubric_coaching"]["standard_version_id"] == str(version_id)
        assert data["rubric_coaching"]["blocks"][0]["block_name"] == "Custom Block"
        assert data["rubric_coaching"]["blocks"][0]["recommendations"][0]["criterion_name"] == "Custom Criterion"

    async def test_mixed_canonical_report_suppresses_legacy_mistakes(self, client):
        session = _make_session()
        version_id = uuid.uuid4()
        recommendation = {
            "rubric_block_id": "custom-block",
            "criterion_id": "custom-criterion",
            "evidence_sequence_number": 3,
            "explanation": "Needs work.",
            "recommended_response": "Try a clearer response.",
            "coaching_advice": "Use the criterion guidance.",
            "standard_version_id": str(version_id),
            "standard_version_number": 7,
        }
        report = CoachingReport(
            id=uuid.uuid4(),
            session_id=session.id,
            mistakes_by_category={
                "compliance": [{
                    "transcript_position": 1,
                    "transcript_excerpt": "Legacy duplicate",
                    "category": "compliance",
                    "explanation": "Duplicate",
                    "recommended_alternative": "Do not render this.",
                }],
                "_rubric_recommendations": [recommendation],
                "_rubric_recommendations_by_block": {"custom-block": [recommendation]},
            },
            total_mistakes=99,
            no_mistakes=False,
        )
        app.dependency_overrides[get_db_session] = _override_db(
            _mock_db_returning_scalar(report)
        )

        with patch(
            "app.api.sessions.get_session_service",
            new_callable=AsyncMock,
            return_value=session,
        ):
            response = await client.get(f"/api/sessions/{session.id}/coaching")

        assert response.status_code == 200
        data = response.json()
        assert data["mistakes_by_category"] == {}
        assert data["total_mistakes"] == 1
        assert data["no_mistakes"] is False


class TestGetLearningPlan:
    """Tests for GET /api/sessions/{id}/learning-plan."""

    @pytest.fixture(autouse=True)
    def _auth_override(self):
        """Use an admin identity for existing artifact contract tests."""
        user = SimpleNamespace(id=uuid.uuid4(), role="admin", user_type=None)
        app.dependency_overrides[require_auth] = lambda: user
        yield
        app.dependency_overrides.pop(require_auth, None)

    async def test_returns_learning_plan(self, client):
        session = _make_session()
        session_id = session.id

        plan = LearningPlan(
            id=uuid.uuid4(),
            session_id=session_id,
            agent_id=uuid.uuid4(),
            weak_competencies=[
                {
                    "category": "compliance",
                    "score": 55,
                    "recommended_scenario": "Compliance Fundamentals",
                },
                {
                    "category": "empathy_communication",
                    "score": 60,
                    "recommended_scenario": "Financial Hardship",
                },
            ],
            all_passing=False,
        )

        mock_db = _mock_db_returning_scalar(plan)
        app.dependency_overrides[get_db_session] = _override_db(mock_db)

        with patch(
            "app.api.sessions.get_session_service",
            new_callable=AsyncMock,
            return_value=session,
        ):
            response = await client.get(f"/api/sessions/{session_id}/learning-plan")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == str(session_id)
        assert data["all_passing"] is False
        assert len(data["weak_competencies"]) == 2
        assert data["weak_competencies"][0]["category"] == "compliance"
        assert data["weak_competencies"][0]["score"] == 55
        assert (
            data["weak_competencies"][0]["recommended_scenario"]
            == "Compliance Fundamentals"
        )

    async def test_returns_learning_plan_all_passing(self, client):
        session = _make_session()
        session_id = session.id

        plan = LearningPlan(
            id=uuid.uuid4(),
            session_id=session_id,
            agent_id=uuid.uuid4(),
            weak_competencies=[],
            all_passing=True,
        )

        mock_db = _mock_db_returning_scalar(plan)
        app.dependency_overrides[get_db_session] = _override_db(mock_db)

        with patch(
            "app.api.sessions.get_session_service",
            new_callable=AsyncMock,
            return_value=session,
        ):
            response = await client.get(f"/api/sessions/{session_id}/learning-plan")

        assert response.status_code == 200
        data = response.json()
        assert data["all_passing"] is True
        assert data["weak_competencies"] == []

    async def test_returns_404_for_nonexistent_session(self, client):
        with patch(
            "app.api.sessions.get_session_service",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = await client.get(f"/api/sessions/{uuid.uuid4()}/learning-plan")

        assert response.status_code == 404

    async def test_returns_404_when_no_plan_exists(self, client):
        session = _make_session()

        mock_db = _mock_db_returning_scalar(None)
        app.dependency_overrides[get_db_session] = _override_db(mock_db)

        with patch(
            "app.api.sessions.get_session_service",
            new_callable=AsyncMock,
            return_value=session,
        ):
            response = await client.get(f"/api/sessions/{session.id}/learning-plan")

        assert response.status_code == 404
        assert "no learning plan" in response.json()["detail"].lower()

    async def test_returns_canonical_rubric_item_and_scenario_id(self, client):
        session = _make_session()
        scenario_id = uuid.uuid4()
        plan = LearningPlan(
            id=uuid.uuid4(),
            session_id=session.id,
            agent_id=session.agent_id,
            weak_competencies=[
                {
                    "category": "De-escalation",
                    "score": 55,
                    "recommended_scenario": "Authorized Practice",
                    "scenario_id": scenario_id,
                    "rubric_block_id": "de-escalation",
                    "criterion_id": "calm-tone",
                    "practice_focus": "Practice calm tone (calm-tone).",
                }
            ],
            all_passing=False,
        )
        mock_db = _mock_db_returning_scalar(plan)
        app.dependency_overrides[get_db_session] = _override_db(mock_db)

        with patch(
            "app.api.sessions.get_session_service",
            new_callable=AsyncMock,
            return_value=session,
        ):
            response = await client.get(f"/api/sessions/{session.id}/learning-plan")

        assert response.status_code == 200
        item = response.json()["weak_competencies"][0]
        assert item["category"] == "De-escalation"
        assert item["rubric_block_id"] == "de-escalation"
        assert item["criterion_id"] == "calm-tone"
        assert item["practice_focus"] == "Practice calm tone (calm-tone)."
        assert item["scenario_id"] == str(scenario_id)


ARTIFACT_PATHS = (
    "transcript",
    "evaluation",
    "coaching",
    "learning-plan",
)


def _artifact_db(artifact_path: str, session_id: uuid.UUID):
    """Build an artifact query mock with the existing response contract."""
    if artifact_path == "transcript":
        return _mock_db_returning_scalars([])
    if artifact_path == "evaluation":
        return _mock_db_returning_scalar(
            Evaluation(
                id=uuid.uuid4(),
                session_id=session_id,
                overall_score=75,
                category_scores=[],
                strengths=[
                    {
                        "description": "Clear opening",
                        "category": "call_opening",
                        "transcript_excerpt": "Hello",
                    }
                ],
                weaknesses=[
                    {
                        "description": "Review the close",
                        "category": "negotiation_resolution",
                        "transcript_excerpt": "Let's discuss the next step.",
                    }
                ],
                is_too_short=False,
            )
        )
    if artifact_path == "coaching":
        return _mock_db_returning_scalar(
            CoachingReport(
                id=uuid.uuid4(),
                session_id=session_id,
                mistakes_by_category={},
                total_mistakes=0,
                no_mistakes=True,
            )
        )
    return _mock_db_returning_scalar(
        LearningPlan(
            id=uuid.uuid4(),
            session_id=session_id,
            agent_id=uuid.uuid4(),
            weak_competencies=[],
            all_passing=True,
        )
    )


class TestSessionArtifactAuthorization:
    """Authorization matrix for every protected session artifact endpoint."""

    @pytest.mark.parametrize("artifact_path", ARTIFACT_PATHS)
    async def test_session_artifacts_require_authentication(self, client, artifact_path):
        session_id = uuid.uuid4()
        mock_db = AsyncMock()
        app.dependency_overrides[get_db_session] = _override_db(mock_db)
        app.dependency_overrides.pop(require_auth, None)

        response = await client.get(f"/api/sessions/{session_id}/{artifact_path}")

        assert response.status_code == 401
        mock_db.execute.assert_not_awaited()

    @pytest.mark.parametrize("artifact_path", ARTIFACT_PATHS)
    async def test_session_artifacts_deny_before_artifact_query(
        self,
        client,
        artifact_path,
    ):
        session = _make_session()
        user = SimpleNamespace(id=uuid.uuid4(), role="user", user_type="agent")
        mock_db = AsyncMock()
        app.dependency_overrides[require_auth] = lambda: user
        app.dependency_overrides[get_db_session] = _override_db(mock_db)

        with patch(
            "app.api.sessions.get_authorized_session",
            new_callable=AsyncMock,
            side_effect=HTTPException(status_code=403, detail="Session access denied"),
        ) as authorize:
            response = await client.get(f"/api/sessions/{session.id}/{artifact_path}")

        assert response.status_code == 403
        assert response.json()["detail"] == "Session access denied"
        authorize.assert_awaited_once()
        mock_db.execute.assert_not_awaited()

    @pytest.mark.parametrize("artifact_path", ARTIFACT_PATHS)
    @pytest.mark.parametrize("role", ("owner", "trainer", "admin"))
    async def test_session_artifacts_allow_authorized_roles(
        self,
        client,
        artifact_path,
        role,
    ):
        session = _make_session()
        user = SimpleNamespace(
            id=session.agent_id if role == "owner" else uuid.uuid4(),
            role="admin" if role == "admin" else "user",
            user_type="trainer" if role == "trainer" else "agent",
        )
        mock_db = _artifact_db(artifact_path, session.id)
        app.dependency_overrides[require_auth] = lambda: user
        app.dependency_overrides[get_db_session] = _override_db(mock_db)

        with patch(
            "app.api.sessions.get_authorized_session",
            new_callable=AsyncMock,
            return_value=session,
        ):
            response = await client.get(f"/api/sessions/{session.id}/{artifact_path}")

        assert response.status_code == 200

    @pytest.mark.parametrize("artifact_path", ARTIFACT_PATHS)
    async def test_session_artifacts_return_404_for_missing_session(
        self,
        client,
        artifact_path,
    ):
        session_id = uuid.uuid4()
        user = SimpleNamespace(id=uuid.uuid4(), role="admin", user_type=None)
        mock_db = AsyncMock()
        app.dependency_overrides[require_auth] = lambda: user
        app.dependency_overrides[get_db_session] = _override_db(mock_db)

        with patch(
            "app.api.sessions.get_authorized_session",
            new_callable=AsyncMock,
            side_effect=HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found",
            ),
        ):
            response = await client.get(f"/api/sessions/{session_id}/{artifact_path}")

        assert response.status_code == 404
        mock_db.execute.assert_not_awaited()


class TestCriteriaCoachingCompletionExploration:
    """Bug-condition probes retained until the session API repair is applied."""

    def test_serializer_reads_loaded_campaign_or_returns_null(self):
        from app.api.sessions import _session_to_response

        campaign = SimpleNamespace(id=uuid.uuid4(), name="Campaign A")
        with_campaign = SimpleNamespace(
            id=uuid.uuid4(),
            scenario_id=uuid.uuid4(),
            campaign_id=campaign.id,
            campaign=campaign,
            persona_context=None,
            status="completed",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ended_at=None,
            negotiation_standard_version=None,
        )
        without_campaign = SimpleNamespace(
            id=uuid.uuid4(),
            scenario_id=uuid.uuid4(),
            campaign_id=None,
            campaign=None,
            persona_context=None,
            status="completed",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ended_at=None,
            negotiation_standard_version=None,
        )

        campaign_response = _session_to_response(with_campaign)
        empty_response = _session_to_response(without_campaign)

        assert campaign_response.campaign_id == campaign.id
        assert campaign_response.campaign_name == "Campaign A"
        assert empty_response.campaign_id is None
        assert empty_response.campaign_name is None

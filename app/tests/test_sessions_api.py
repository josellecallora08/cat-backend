"""Tests for session API endpoints.

Tests POST /api/sessions, GET /api/sessions/{id}, POST /api/sessions/{id}/end,
GET /api/sessions/{id}/transcript, GET /api/sessions/{id}/evaluation,
GET /api/sessions/{id}/coaching, GET /api/sessions/{id}/learning-plan.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_session as get_db_session
from app.main import app
from app.models import (
    CoachingReport,
    Evaluation,
    LearningPlan,
    Session,
    Transcript,
)


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

    async def test_returns_session_details(self, client):
        session = _make_session(status="active")

        with patch(
            "app.api.sessions.get_session_service",
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


class TestGetLearningPlan:
    """Tests for GET /api/sessions/{id}/learning-plan."""

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
        assert data["weak_competencies"][0]["recommended_scenario"] == "Compliance Fundamentals"

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

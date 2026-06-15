"""Unit tests for SQLAlchemy models."""

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session as DBSession

from app.database import Base
from app.models import (
    CoachingReport,
    Evaluation,
    LearningPlan,
    Scenario,
    Session,
    Transcript,
)


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with DBSession(engine) as session:
        yield session
    engine.dispose()


def test_scenario_model_creation(db_session):
    """Test that a Scenario can be created with all required fields."""
    scenario = Scenario(
        id=uuid.uuid4(),
        name="Financial Hardship",
        scenario_type="FINANCIAL_HARDSHIP",
        description="A debtor facing financial difficulties.",
        debtor_profile={
            "name": "John Doe",
            "outstanding_balance": "5000.00",
            "days_past_due": 45,
            "personality_profile": "anxious",
            "conversation_goal": "negotiate payment plan",
        },
        is_active=True,
    )
    db_session.add(scenario)
    db_session.commit()

    result = db_session.execute(select(Scenario)).scalar_one()
    assert result.name == "Financial Hardship"
    assert result.scenario_type == "FINANCIAL_HARDSHIP"
    assert result.debtor_profile["name"] == "John Doe"
    assert result.is_active is True


def test_session_model_with_foreign_key(db_session):
    """Test that a Session links to a Scenario via foreign key."""
    scenario_id = uuid.uuid4()
    scenario = Scenario(
        id=scenario_id,
        name="Angry Customer",
        scenario_type="ANGRY_CUSTOMER",
        debtor_profile={"name": "Jane Smith", "outstanding_balance": "1200.00",
                        "days_past_due": 30, "personality_profile": "hostile",
                        "conversation_goal": "resolve dispute"},
        is_active=True,
    )
    db_session.add(scenario)
    db_session.commit()

    session = Session(
        id=uuid.uuid4(),
        scenario_id=scenario_id,
        agent_id=uuid.uuid4(),
        status="pending",
        persona_context={"emotional_state": "hostile"},
    )
    db_session.add(session)
    db_session.commit()

    result = db_session.execute(select(Session)).scalar_one()
    assert result.scenario_id == scenario_id
    assert result.status == "pending"
    assert result.persona_context["emotional_state"] == "hostile"


def test_transcript_model(db_session):
    """Test Transcript model with all required fields."""
    scenario_id = uuid.uuid4()
    session_id = uuid.uuid4()

    db_session.add(Scenario(
        id=scenario_id, name="Test", scenario_type="TEST",
        debtor_profile={"name": "X", "outstanding_balance": "100",
                        "days_past_due": 1, "personality_profile": "calm",
                        "conversation_goal": "pay"},
        is_active=True,
    ))
    db_session.add(Session(
        id=session_id, scenario_id=scenario_id,
        agent_id=uuid.uuid4(), status="active",
    ))
    db_session.commit()

    transcript = Transcript(
        id=uuid.uuid4(),
        session_id=session_id,
        speaker="agent",
        utterance_text="Hello, I'm calling about your account.",
        timestamp_ms=datetime.now(timezone.utc),
        sequence_number=1,
    )
    db_session.add(transcript)
    db_session.commit()

    result = db_session.execute(select(Transcript)).scalar_one()
    assert result.speaker == "agent"
    assert result.sequence_number == 1
    assert result.utterance_text == "Hello, I'm calling about your account."


def test_evaluation_unique_session(db_session):
    """Test that Evaluation has a unique constraint on session_id."""
    scenario_id = uuid.uuid4()
    session_id = uuid.uuid4()

    db_session.add(Scenario(
        id=scenario_id, name="Test", scenario_type="TEST",
        debtor_profile={"name": "X", "outstanding_balance": "100",
                        "days_past_due": 1, "personality_profile": "calm",
                        "conversation_goal": "pay"},
        is_active=True,
    ))
    db_session.add(Session(
        id=session_id, scenario_id=scenario_id,
        agent_id=uuid.uuid4(), status="completed",
    ))
    db_session.commit()

    evaluation = Evaluation(
        id=uuid.uuid4(),
        session_id=session_id,
        overall_score=75.5,
        category_scores=[{"category": "compliance", "score": 80}],
        strengths=[{"description": "Good opening"}],
        weaknesses=[{"description": "Missed compliance"}],
        is_too_short=False,
    )
    db_session.add(evaluation)
    db_session.commit()

    result = db_session.execute(select(Evaluation)).scalar_one()
    assert result.overall_score == 75.5
    assert result.is_too_short is False


def test_coaching_report_model(db_session):
    """Test CoachingReport model creation."""
    scenario_id = uuid.uuid4()
    session_id = uuid.uuid4()

    db_session.add(Scenario(
        id=scenario_id, name="Test", scenario_type="TEST",
        debtor_profile={"name": "X", "outstanding_balance": "100",
                        "days_past_due": 1, "personality_profile": "calm",
                        "conversation_goal": "pay"},
        is_active=True,
    ))
    db_session.add(Session(
        id=session_id, scenario_id=scenario_id,
        agent_id=uuid.uuid4(), status="completed",
    ))
    db_session.commit()

    report = CoachingReport(
        id=uuid.uuid4(),
        session_id=session_id,
        mistakes_by_category={"compliance": [{"explanation": "Did not verify identity"}]},
        total_mistakes=1,
        no_mistakes=False,
    )
    db_session.add(report)
    db_session.commit()

    result = db_session.execute(select(CoachingReport)).scalar_one()
    assert result.total_mistakes == 1
    assert result.no_mistakes is False


def test_learning_plan_model(db_session):
    """Test LearningPlan model creation."""
    scenario_id = uuid.uuid4()
    session_id = uuid.uuid4()

    db_session.add(Scenario(
        id=scenario_id, name="Test", scenario_type="TEST",
        debtor_profile={"name": "X", "outstanding_balance": "100",
                        "days_past_due": 1, "personality_profile": "calm",
                        "conversation_goal": "pay"},
        is_active=True,
    ))
    db_session.add(Session(
        id=session_id, scenario_id=scenario_id,
        agent_id=uuid.uuid4(), status="completed",
    ))
    db_session.commit()

    plan = LearningPlan(
        id=uuid.uuid4(),
        session_id=session_id,
        agent_id=uuid.uuid4(),
        weak_competencies=[
            {"category": "compliance", "score": 55, "recommended_scenario": "Compliance Fundamentals"}
        ],
        all_passing=False,
    )
    db_session.add(plan)
    db_session.commit()

    result = db_session.execute(select(LearningPlan)).scalar_one()
    assert result.all_passing is False
    assert len(result.weak_competencies) == 1
    assert result.weak_competencies[0]["category"] == "compliance"

"""Complete pinned-standard evaluation flow for TASK-036."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base, get_session
from app.main import app
from app.models import Campaign, CampaignAgent, Evaluation, LearningPlan, CoachingReport, Scenario, Transcript, User
from app.services.auth import require_auth
from app.services.debtor_simulator import EmotionalState, PersonaContext
from app.services.evaluation_compatibility import render_legacy_review, to_legacy_review
from app.services.evaluation_pipeline import EvaluationPipeline
from app.services.llm_service import LLMResponse
from app.services.negotiation_standard_service import create_standard, publish_standard
from app.services.session_service import create_session
from app.services.session_service import end_session
from app.services.transcript_manager import TranscriptManager
from app.schemas.negotiation_standard import NegotiationStandardContent


class FakeDebtorSimulator:
    async def generate_persona(self, _scenario: dict) -> PersonaContext:
        return PersonaContext(
            persona_id=uuid4(),
            name="Debtor",
            communication_style="cooperative",
            financial_circumstances={"debt_amount": 1000},
            emotional_state=EmotionalState.NEUTRAL,
            language="EN",
        )


class FakeEvaluationLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    async def chat_completion(self, _messages, **_kwargs) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content=self.content, model="task-036-test")


def _content() -> NegotiationStandardContent:
    return NegotiationStandardContent.model_validate({
        "schema_version": 1,
        "overall_passing_score": 70,
        "blocks": [{
            "id": "opening",
            "category": "Opening",
            "weight": 100,
            "passing_score": 70,
            "scoring_instructions": "Score only grounded behavior.",
            "positive_behaviors": [{"id": "greeting", "name": "Greeting", "description": "Greets clearly.", "evidence_instructions": "Quote it."}],
            "violations": [{"id": "rude-tone", "name": "Rude tone", "description": "Uses hostile wording.", "evidence_instructions": "Quote it."}],
            "penalties": [{"violation_id": "rude-tone", "deduction": 10, "max_occurrences": 1}],
            "recommendation_guidance": "Use a respectful greeting and acknowledge the concern.",
            "display_order": 0,
        }],
    })


def _llm_observation() -> str:
    return '{"status":"evaluated","summary":"The agent used a grounded opening.","categories":[{"rubric_block_id":"opening","raw_score":80,"evidence":[{"sequence_number":0,"speaker":"agent","excerpt":"Hello there","explanation":"The agent greeted the debtor."}],"strengths":[{"criterion_id":"greeting","explanation":"The opening was clear.","evidence_sequence_numbers":[0]}],"violations":[{"violation_id":"rude-tone","explanation":"The tone could be warmer.","evidence_sequence_numbers":[0]}],"failed_criteria":[],"recommendation_inputs":[{"criterion_id":"rude-tone","transcript_sequence_number":0,"need":"Use a warmer tone."}]}],"applied_techniques":{"techniques_used":[{"technique_name":"Greeting","execution_type":"Executed Properly","execution_description":"The greeting was clear.","evidence_sequence_numbers":[0]}],"reason_if_empty":"None."},"missed_opportunities":{"missed_techniques":[],"reason_if_empty":"None."}}'


@pytest.fixture
async def db_session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_complete_campaign_to_pinned_evaluation_and_results_api(db_session: AsyncSession) -> None:
    admin = User(id=uuid4(), email=f"{uuid4()}@example.test", full_name="Admin", role="admin", hashed_password="hash")
    agent = User(id=uuid4(), email=f"{uuid4()}@example.test", full_name="Agent", role="user", user_type="agent", hashed_password="hash")
    campaign = Campaign(id=uuid4(), name=f"Campaign {uuid4()}")
    scenario = Scenario(id=uuid4(), name="Scenario", scenario_type="PAYMENT_EXTENSION", description="Test", debtor_profile={"name": "Debtor", "personality_profile": "calm"})
    campaign.scenarios.append(scenario)
    campaign.agent_assignments.append(CampaignAgent(agent_id=agent.id, role="participant"))
    db_session.add_all([admin, agent, campaign, scenario])
    await db_session.commit()

    await create_standard(db_session, campaign.id, admin.id, "Collection standard", None, _content())
    version = await publish_standard(db_session, campaign.id, admin.id, "release gate")
    session = await create_session(db_session, scenario.id, agent.id, FakeDebtorSimulator(), campaign.id)
    assert session.negotiation_standard_version_id == version.id

    manager = TranscriptManager(db_session)
    transcript_text = [
        "Hello there", "I need more time", "Let us review options", "I am worried",
        "We can discuss a plan", "That sounds helpful", "Let us confirm the next step", "Thank you",
    ]
    for index, text in enumerate(transcript_text):
        await manager.append_entry(session.id, "agent" if index % 2 == 0 else "debtor", text, datetime.now(timezone.utc))
    await manager.persist(session.id)
    assert await db_session.scalar(select(Transcript).where(Transcript.session_id == session.id)) is not None
    completed = await end_session(db_session, session.id)
    assert completed.status == "completed"

    llm = FakeEvaluationLLM(_llm_observation())
    pipeline = EvaluationPipeline(llm)
    result = await pipeline.run(session.id, agent.id, db_session)

    assert result.evaluation.negotiation_standard_version_id == version.id
    assert result.evaluation.standard_snapshot == version.snapshot
    assert result.evaluation.overall_score == 70.0
    assert result.evaluation.rubric_result is not None
    assert result.coaching_report.total_mistakes == 1
    assert result.learning_plan.standard_version_id == version.id
    assert result.learning_plan.weak_competencies[0].criterion_id == "rude-tone"
    assert llm.calls == 1

    stored = (await db_session.execute(select(Evaluation).where(Evaluation.session_id == session.id))).scalar_one()
    assert stored.negotiation_standard_version_id == version.id
    assert (await db_session.execute(select(CoachingReport).where(CoachingReport.session_id == session.id))).scalar_one()
    assert (await db_session.execute(select(LearningPlan).where(LearningPlan.session_id == session.id))).scalar_one()

    legacy = to_legacy_review(result.evaluation.rubric_result)
    assert "Greeting" in {item.technique_name for item in legacy.applied_techniques.techniques_used}
    assert "Applied Technique Delivery" in render_legacy_review(result.evaluation.rubric_result)

    factory = db_session.get_bind()
    async def override_session():
        yield db_session

    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[require_auth] = lambda: admin
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            evaluation_response = await client.get(f"/api/sessions/{session.id}/evaluation")
            coaching_response = await client.get(f"/api/sessions/{session.id}/coaching")
            plan_response = await client.get(f"/api/sessions/{session.id}/learning-plan")
        assert evaluation_response.status_code == 200
        assert evaluation_response.json()["standard_version_number"] == 1
        assert evaluation_response.json()["weighted_total"] == 70.0
        assert coaching_response.status_code == 200 and coaching_response.json()["rubric_recommendations"]
        assert plan_response.status_code == 200 and plan_response.json()["standard_version_id"] == str(version.id)
    finally:
        app.dependency_overrides.pop(get_session, None)
        app.dependency_overrides.pop(require_auth, None)

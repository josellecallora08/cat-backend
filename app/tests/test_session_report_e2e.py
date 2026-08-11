"""Full authenticated session-report flow over an async database.

The test database and JWT authentication are real.  The only deterministic
boundary is the external LLM transport; rubric validation/scoring, evidence
matching, coaching grouping, learning-plan scenario authorization, report
assembly, persistence, and API authorization remain production services.

Validates: Requirements 1.3, 1.4, 2.1-2.12, 3.5-3.6, 4.4, 4.6-4.10,
5.1-5.3, 5.6, 6.1-6.3, 6.6, 6.8, and PRESERVE-001 through PRESERVE-006.
"""

import copy
import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from app.database import Base, get_session
from app.main import app
from app.models import (
    Campaign,
    CampaignAgent,
    CoachingReport,
    Evaluation,
    LearningPlan,
    Scenario,
    Session,
    SessionReport,
    Transcript,
    User,
)
from app.models.campaign import CampaignRole, CampaignStatus
from app.schemas import EvaluationResult, EvaluationCategory, StrengthItem, WeaknessItem
from app.schemas.negotiation_standard import NegotiationStandardContent
from app.schemas.rubric_evaluation import CanonicalEvaluationResult
from app.services.auth import hash_password
from app.services.coaching_engine import CoachingEngine
from app.services.evaluation_compatibility import build_rubric_recommendations
from app.services.learning_plan_generator import LearningPlanGenerator
from app.services.negotiation_standard_service import create_standard, publish_standard
from app.services.rubric_observation_validator import validate_observation
from app.services.rubric_score_calculator import calculate_rubric_score
from app.services.session_report_service import generate_report
from app.services.llm_service import LLMResponse, LLMService


PASSWORD = "E2E-password-123!"


@pytest.fixture
async def async_db():
    """Create an isolated async database with SQLite foreign keys enabled."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)

    @event.listens_for(engine.sync_engine, "connect")
    def enable_foreign_keys(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session

    async with engine.begin() as connection:
        await connection.execute(text("PRAGMA foreign_keys=OFF"))
        await connection.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture(autouse=True)
def clear_app_overrides():
    """Prevent dependency overrides from leaking between E2E examples."""
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def e2e_client(async_db: AsyncSession):
    async def override_db():
        yield async_db

    app.dependency_overrides[get_session] = override_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


def _two_block_standard() -> NegotiationStandardContent:
    return NegotiationStandardContent.model_validate({
        "schema_version": 1,
        "overall_passing_score": 70,
        "blocks": [
            {
                "id": "opening",
                "category": "Opening",
                "weight": 50,
                "passing_score": 70,
                "scoring_instructions": "Use grounded opening evidence.",
                "positive_behaviors": [{
                    "id": "greeting",
                    "name": "Greeting",
                    "description": "Greets clearly.",
                    "evidence_instructions": "Quote the greeting.",
                }],
                "violations": [{
                    "id": "rushed-tone",
                    "name": "Rushed tone",
                    "description": "Uses rushed wording.",
                    "evidence_instructions": "Quote the rushed wording.",
                }],
                "penalties": [{"violation_id": "rushed-tone", "deduction": 10, "max_occurrences": 1}],
                "recommendation_guidance": "Slow down and acknowledge the debtor.",
                "display_order": 0,
            },
            {
                "id": "resolution",
                "category": "Resolution",
                "weight": 50,
                "passing_score": 70,
                "scoring_instructions": "Use grounded resolution evidence.",
                "positive_behaviors": [{
                    "id": "offer-plan",
                    "name": "Offer a plan",
                    "description": "Offers a payment plan.",
                    "evidence_instructions": "Quote the plan.",
                }],
                "violations": [{
                    "id": "no-plan",
                    "name": "No plan",
                    "description": "Does not offer a plan.",
                    "evidence_instructions": "Quote the omission.",
                }],
                "penalties": [{"violation_id": "no-plan", "deduction": 15, "max_occurrences": 1}],
                "recommendation_guidance": "Offer a realistic payment plan.",
                "display_order": 1,
            },
        ],
    })


def _transcript_values() -> list[tuple[str, str]]:
    return [
        ("agent", "Hello, this is Alex from the collections team."),
        ("debtor", "I am worried about my balance."),
        ("agent", "We need payment today."),
        ("debtor", "I cannot pay the full amount today."),
        ("agent", "I understand this is difficult."),
        ("debtor", "Thank you for listening."),
        ("agent", "The full amount is due."),
        ("debtor", "Can we discuss another option?"),
    ]


def _observation() -> dict:
    return {
        "status": "evaluated",
        "summary": "The agent opened clearly but missed a compliant resolution.",
        "categories": [
            {
                "rubric_block_id": "opening",
                "raw_score": 85,
                "evidence": [{"sequence_number": 0, "speaker": "agent", "excerpt": "Hello, this is Alex", "explanation": "A clear greeting."},
                             {"sequence_number": 2, "speaker": "agent", "excerpt": "We need payment today", "explanation": "The wording was rushed."}],
                "strengths": [{"criterion_id": "greeting", "explanation": "The call began clearly.", "evidence_sequence_numbers": [0]}],
                "violations": [{"violation_id": "rushed-tone", "explanation": "The demand was rushed.", "evidence_sequence_numbers": [2]}],
                "failed_criteria": ["rushed-tone"],
                "recommendation_inputs": [{"criterion_id": "rushed-tone", "transcript_sequence_number": 2, "need": "Use a calmer opening."}],
            },
            {
                "rubric_block_id": "resolution",
                "raw_score": 65,
                "evidence": [{"sequence_number": 6, "speaker": "agent", "excerpt": "The full amount is due", "explanation": "No flexible option was offered."}],
                "strengths": [],
                "violations": [{"violation_id": "no-plan", "explanation": "The agent did not offer a plan.", "evidence_sequence_numbers": [6]}],
                "failed_criteria": ["no-plan"],
                "recommendation_inputs": [{"criterion_id": "no-plan", "transcript_sequence_number": 6, "need": "Offer a realistic plan."}],
            },
        ],
        "applied_techniques": {"techniques_used": [{"technique_name": "Greeting", "execution_type": "Executed Properly", "execution_description": "The greeting was clear.", "evidence_sequence_numbers": [0]}], "reason_if_empty": "None."},
        "missed_opportunities": {"missed_techniques": [], "reason_if_empty": "No missed opportunities recorded."},
    }


def _llm_response(messages, **_kwargs) -> LLMResponse:
    prompt = messages[0].content.lower()
    if "persona generator" in prompt:
        content = json.dumps({
            "name": "Test Debtor",
            "communication_style": "cooperative",
            "financial_circumstances": {"income_level": "medium", "debt_amount": 5000, "reason_for_delinquency": "job loss"},
            "emotional_state": 3,
            "language": "EN",
        })
    else:
        content = json.dumps(_observation())
    return LLMResponse(content=content, model="e2e-llm", usage={})


async def _login(client: AsyncClient, user: User) -> dict[str, str]:
    response = await client.post("/api/auth/login", json={"email": user.email, "password": PASSWORD})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


async def _seed_identity(db: AsyncSession, *, campaign: bool = True) -> tuple[User, Scenario, Campaign | None, object | None]:
    agent = User(id=uuid.uuid4(), email=f"{uuid.uuid4()}@e2e.test", full_name="E2E Agent", role="user", user_type="agent", hashed_password=hash_password(PASSWORD), avatar_url=None)
    scenario = Scenario(id=uuid.uuid4(), name="Authorized Scenario", scenario_type="FINANCIAL_HARDSHIP", description="Pinned scenario", debtor_profile={"name": "Test Debtor", "personality_profile": "cooperative", "outstanding_balance": "5000", "days_past_due": 30})
    db.add_all([agent, scenario])
    await db.flush()
    if not campaign:
        await db.commit()
        return agent, scenario, None, None
    campaign = Campaign(id=uuid.uuid4(), name="Pinned Campaign", status=CampaignStatus.ACTIVE.value)
    campaign.scenarios.append(scenario)
    campaign.agent_assignments.append(CampaignAgent(agent_id=agent.id, role=CampaignRole.PARTICIPANT.value))
    db.add(campaign)
    await db.commit()
    standard = await create_standard(db, campaign.id, agent.id, "Pinned Standard", "E2E standard", _two_block_standard())
    version = await publish_standard(db, campaign.id, agent.id, "E2E publication")
    assert version.standard_id == standard.id
    return agent, scenario, campaign, version


async def _seed_canonical_artifacts(db: AsyncSession, session: Session, version) -> CanonicalEvaluationResult:
    """Create canonical artifacts through real validation, scoring, and generators."""
    now = datetime.now(timezone.utc)
    transcript_rows = [
        Transcript(id=uuid.uuid4(), session_id=session.id, speaker=speaker, utterance_text=text, timestamp_ms=now + timedelta(seconds=index), sequence_number=index)
        for index, (speaker, text) in enumerate(_transcript_values())
    ]
    db.add_all(transcript_rows)
    await db.flush()
    transcript = [{"speaker": row.speaker, "text": row.utterance_text, "sequence_number": row.sequence_number} for row in transcript_rows]
    validated = validate_observation(_observation(), version.snapshot, transcript)
    canonical = calculate_rubric_score(validated)
    canonical = canonical.model_copy(update={"recommendations": build_rubric_recommendations(canonical, version.snapshot, version.id, version.version_number)})
    evaluation_result = EvaluationResult(
        session_id=session.id,
        category_scores=[],
        overall_score=float(canonical.weighted_total),
        strengths=[StrengthItem(description="Clear greeting", category=EvaluationCategory.CALL_OPENING, transcript_excerpt="Hello, this is Alex")],
        weaknesses=[WeaknessItem(description="Rushed resolution", category=EvaluationCategory.NEGOTIATION_RESOLUTION, transcript_excerpt="The full amount is due")],
        negotiation_standard_version_id=version.id,
        standard_name="Pinned Standard",
        standard_version_number=version.version_number,
        standard_snapshot=version.snapshot,
        weighted_total=float(canonical.weighted_total),
        passing_score=canonical.passing_score,
        passed=canonical.passed,
        rubric_result=canonical.model_dump(mode="json"),
    )
    coaching = await CoachingEngine(None).generate_report(session.id, transcript, evaluation_result)
    serialized_coaching = {"_rubric_coaching": coaching.rubric_coaching.model_dump(mode="json")}
    serialized_coaching["_rubric_recommendations"] = [item.model_dump(mode="json") for item in coaching.rubric_recommendations]
    serialized_coaching["_rubric_recommendations_by_block"] = {
        block_id: [item.model_dump(mode="json") for item in items]
        for block_id, items in coaching.rubric_recommendations_by_block.items()
    }
    db.add(Evaluation(
        session_id=session.id,
        overall_score=float(canonical.weighted_total),
        category_scores=[item.model_dump(mode="json") for item in canonical.categories],
        strengths=[],
        weaknesses=[],
        negotiation_standard_version_id=version.id,
        standard_snapshot=version.snapshot,
        weighted_total=float(canonical.weighted_total),
        passing_score=canonical.passing_score,
        passed=canonical.passed,
        rubric_result=canonical.model_dump(mode="json"),
        is_too_short=False,
    ))
    db.add(CoachingReport(session_id=session.id, mistakes_by_category=serialized_coaching, total_mistakes=coaching.total_mistakes, no_mistakes=coaching.no_mistakes))
    await db.commit()
    await LearningPlanGenerator().generate_and_persist(evaluation_result, session.id, session.agent_id, db=db)
    return canonical


async def _reload_session(db: AsyncSession, session_id: uuid.UUID) -> Session:
    result = await db.execute(
        select(Session).options(selectinload(Session.campaign), selectinload(Session.negotiation_standard_version)).where(Session.id == session_id)
    )
    return result.scalar_one()


async def _seed_report_variant(db: AsyncSession, *, variant: str) -> tuple[User, Session]:
    user, scenario, campaign, version = await _seed_identity(db)
    session = Session(id=uuid.uuid4(), scenario_id=scenario.id, agent_id=user.id, campaign_id=campaign.id, negotiation_standard_version_id=version.id, status="completed", created_at=datetime.now(timezone.utc), ended_at=datetime.now(timezone.utc), persona_context={"name": "Test Debtor", "communication_style": "cooperative", "emotional_state": 3})
    db.add(session)
    await db.flush()
    now = datetime.now(timezone.utc)
    if variant != "empty":
        db.add(Transcript(id=uuid.uuid4(), session_id=session.id, speaker="agent", utterance_text="Hello", timestamp_ms=now, sequence_number=0))
    if variant == "legacy":
        db.add(Evaluation(session_id=session.id, overall_score=60, category_scores=[{"category": "call_opening", "score": 60}], strengths=[], weaknesses=[], is_too_short=False))
    elif variant == "too_short":
        db.add(Evaluation(session_id=session.id, overall_score=0, category_scores=[], strengths=[], weaknesses=[], is_too_short=True))
    elif variant in {"empty", "no_evidence", "malformed"}:
        canonical = CanonicalEvaluationResult.model_validate({
            "status": "evaluated", "summary": "A canonical fixture.",
            "categories": [
                {"rubric_block_id": "opening", "category": "Opening", "raw_score": 80, "penalty_total": 0, "penalized_score": 80, "weight": 50, "weighted_contribution": 40, "passing_score": 70, "passed": True, "evidence": [], "strengths": [], "violations": [], "failed_criteria": [], "recommendation_inputs": []},
                {"rubric_block_id": "resolution", "category": "Resolution", "raw_score": 80, "penalty_total": 0, "penalized_score": 80, "weight": 50, "weighted_contribution": 40, "passing_score": 70, "passed": True, "evidence": [], "strengths": [], "violations": [], "failed_criteria": [], "recommendation_inputs": []},
            ],
            "weighted_total": 80, "passing_score": 70, "passed": True,
            "applied_techniques": {"techniques_used": [], "reason_if_empty": "None."},
            "missed_opportunities": {"missed_techniques": [], "reason_if_empty": "None."},
        })
        if variant == "malformed":
            raw = canonical.model_dump(mode="json")
            raw["categories"][0]["evidence"] = [{"sequence_number": 999, "speaker": "agent", "excerpt": "missing", "explanation": "bad reference"}]
            canonical = CanonicalEvaluationResult.model_validate(raw)
        db.add(Evaluation(session_id=session.id, overall_score=80, category_scores=[item.model_dump(mode="json") for item in canonical.categories], strengths=[], weaknesses=[], negotiation_standard_version_id=version.id, standard_snapshot=version.snapshot, weighted_total=80, passing_score=70, passed=True, rubric_result=canonical.model_dump(mode="json"), is_too_short=False))
    db.add(CoachingReport(session_id=session.id, mistakes_by_category={}, total_mistakes=0, no_mistakes=True))
    db.add(LearningPlan(session_id=session.id, agent_id=user.id, weak_competencies=[], all_passing=True))
    await db.commit()
    return user, session


@pytest.mark.asyncio
async def test_authenticated_session_report_full_lifecycle(e2e_client, async_db, monkeypatch):
    """Start/end, retrieve, status, export, regenerate, and isolate failure."""
    async def fake_llm(_self, messages, **kwargs):
        return _llm_response(messages, **kwargs)

    monkeypatch.setattr(LLMService, "chat_completion", fake_llm)
    agent, scenario, campaign, version = await _seed_identity(async_db)
    headers = await _login(e2e_client, agent)
    assert (await e2e_client.get("/api/auth/me", headers=headers)).status_code == 200

    started = await e2e_client.post(
        "/api/sessions",
        json={"scenario_id": str(scenario.id), "campaign_id": str(campaign.id)},
        headers=headers,
    )
    assert started.status_code == 201, started.text
    start_body = started.json()
    assert start_body["status"] == "pending"
    assert start_body["campaign_id"] == str(campaign.id)
    assert start_body["standard_version_id"] == str(version.id)

    session = await _reload_session(async_db, uuid.UUID(start_body["id"]))
    now = datetime.now(timezone.utc)
    async_db.add_all([
        Transcript(
            id=uuid.uuid4(),
            session_id=session.id,
            speaker=speaker,
            utterance_text=text,
            timestamp_ms=now + timedelta(seconds=index),
            sequence_number=index,
        )
        for index, (speaker, text) in enumerate(_transcript_values())
    ])
    await async_db.commit()

    ended = await e2e_client.post(f"/api/sessions/{session.id}/end", headers=headers)
    assert ended.status_code == 200, ended.text
    end_body = ended.json()
    assert end_body["id"] == str(session.id)
    assert end_body["status"] == "completed"
    assert end_body["ended_at"] is not None

    detail = await e2e_client.get(f"/api/sessions/{session.id}", headers=headers)
    assert detail.status_code == 200
    assert detail.json() == end_body
    transcript_response = await e2e_client.get(f"/api/sessions/{session.id}/transcript", headers=headers)
    assert transcript_response.status_code == 200
    assert [entry["sequence_number"] for entry in transcript_response.json()] == list(range(8))

    report_response = await e2e_client.get(f"/api/sessions/{session.id}/report", headers=headers)
    assert report_response.status_code == 200, report_response.text
    report_v1 = report_response.json()
    assert report_v1["report_version"] == 1
    payload_v1 = copy.deepcopy(report_v1["payload"])
    hash_v1 = report_v1["content_hash"]
    assert payload_v1["summary"]["campaign_name"] == "Pinned Campaign"
    assert payload_v1["summary"]["standard_version_number"] == 1
    assert payload_v1["evaluation"]["mode"] == "canonical"
    assert [item["penalty_total"] for item in payload_v1["evaluation"]["canonical"]["categories"]] == [10, 15]
    assert payload_v1["evaluation"]["canonical"]["categories"][1]["failed_criteria"] == ["no-plan"]
    assert payload_v1["evaluation"]["canonical"]["categories"][0]["evidence"][0]["sequence_number"] == 0
    assert [block["rubric_block_id"] for block in payload_v1["coaching"]["blocks"]] == ["opening", "resolution"]
    assert payload_v1["coaching"]["blocks"][0]["recommendations"][0]["source_speaker"] == "agent"
    assert all(item["scenario_id"] == str(scenario.id) for item in payload_v1["learning_plan"]["items"])

    status = await e2e_client.get(f"/api/sessions/{session.id}/report/status", headers=headers)
    assert status.status_code == 200
    assert status.json()["status"] == "ready"
    assert status.json()["report"]["content_hash"] == hash_v1

    export_expectations = {
        "json": ("application/json", b'"evaluation"'),
        "csv": ("text/csv", b"\r\n"),
        "pdf": ("application/pdf", b"%PDF"),
    }
    for export_format, (media_type, marker) in export_expectations.items():
        exported = await e2e_client.get(f"/api/sessions/{session.id}/report/export?format={export_format}", headers=headers)
        assert exported.status_code == 200, exported.text
        assert exported.headers["content-type"].startswith(media_type)
        assert exported.content and marker in exported.content

    row_v1 = (await async_db.execute(select(SessionReport).where(SessionReport.session_id == session.id, SessionReport.report_version == 1))).scalar_one()
    immutable_v1 = {"payload": copy.deepcopy(row_v1.payload), "content_hash": row_v1.content_hash, "status": row_v1.status, "generated_by": row_v1.generated_by, "created_at": row_v1.created_at, "updated_at": row_v1.updated_at}

    regenerated = await e2e_client.post(f"/api/sessions/{session.id}/report", headers=headers)
    assert regenerated.status_code == 201, regenerated.text
    assert regenerated.json()["report_version"] == 2
    assert regenerated.json()["payload"] == payload_v1
    assert regenerated.json()["content_hash"] == hash_v1
    row_v1_after = (await async_db.execute(select(SessionReport).where(SessionReport.session_id == session.id, SessionReport.report_version == 1))).scalar_one()
    assert {"payload": copy.deepcopy(row_v1_after.payload), "content_hash": row_v1_after.content_hash, "status": row_v1_after.status, "generated_by": row_v1_after.generated_by, "created_at": row_v1_after.created_at, "updated_at": row_v1_after.updated_at} == immutable_v1

    import app.services.session_report_service as report_service
    monkeypatch.setattr(report_service, "assemble_report_payload", AsyncMock(side_effect=RuntimeError("must not leak")))
    failed_regeneration = await e2e_client.post(f"/api/sessions/{session.id}/report", headers=headers)
    assert failed_regeneration.status_code == 500
    assert "must not leak" not in failed_regeneration.text
    assert "traceback" not in failed_regeneration.text.lower()
    current = await e2e_client.get(f"/api/sessions/{session.id}/report", headers=headers)
    assert current.status_code == 200
    assert current.json()["report_version"] == 2
    assert current.json()["payload"] == payload_v1
    failed_status = await e2e_client.get(f"/api/sessions/{session.id}/report/status", headers=headers)
    assert failed_status.status_code == 200
    assert failed_status.json()["status"] == "ready"
    assert failed_status.json()["report"]["report_version"] == 2
    assert failed_status.json()["latest_attempt"]["status"] == "failed"
    assert "payload" not in failed_status.json()["latest_attempt"]
    final_detail = await e2e_client.get(f"/api/sessions/{session.id}", headers=headers)
    assert final_detail.status_code == 200
    assert final_detail.json() == end_body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("variant", "expected_status"),
    [("legacy", "legacy_only"), ("too_short", "too_short"), ("empty", "empty_transcript"), ("no_evidence", "no_evidence")],
)
async def test_report_status_terminal_fixtures(e2e_client, async_db, variant, expected_status):
    user, session = await _seed_report_variant(async_db, variant=variant)
    headers = await _login(e2e_client, user)
    report = await generate_report(async_db, await _reload_session(async_db, session.id), generated_by=user.id)
    assert report.status == "ready"
    response = await e2e_client.get(f"/api/sessions/{session.id}/report/status", headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == expected_status
    assert body["report"]["payload"] is not None
    if expected_status == "legacy_only":
        assert body["report"]["payload"]["evaluation"]["canonical"] is None
        assert body["report"]["payload"]["evaluation"]["reason_code"] == "legacy_only"
    if expected_status == "too_short":
        assert body["report"]["payload"]["evaluation"]["mode"] == "too_short"
        assert body["report"]["payload"]["evaluation"]["weighted_total"] is None
    if expected_status == "empty_transcript":
        assert body["report"]["payload"]["transcript"]["entries"] == []
    if expected_status == "no_evidence":
        assert body["report"]["payload"]["evaluation"]["reason_code"] == "no_evidence"


@pytest.mark.asyncio
async def test_malformed_persisted_reference_becomes_safe_failed_report(e2e_client, async_db):
    user, session = await _seed_report_variant(async_db, variant="malformed")
    headers = await _login(e2e_client, user)
    response = await e2e_client.post(f"/api/sessions/{session.id}/report", headers=headers)
    assert response.status_code == 500
    assert "bad reference" not in response.text
    assert "traceback" not in response.text.lower()
    status = await e2e_client.get(f"/api/sessions/{session.id}/report/status", headers=headers)
    assert status.status_code == 200
    assert status.json()["status"] == "failed"
    assert status.json()["reason"]["code"] == "generation_failed"
    assert status.json()["report"] is None
    failed = (await async_db.execute(select(SessionReport).where(SessionReport.session_id == session.id))).scalar_one()
    assert failed.payload is None
    assert failed.content_hash is None
    assert failed.reason_code == "generation_failed"

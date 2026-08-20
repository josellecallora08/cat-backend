"""Session API router with endpoints for session lifecycle and artifact retrieval.

Validates: Requirements 4.1, 4.4, 5.1, 6.1, 7.8, 8.2
"""

import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_session as get_db_session
from app.models import CoachingReport, Evaluation, LearningPlan, Session, Transcript
from app.models.user import User, UserRole, UserType
from app.schemas import (
    SessionCreate,
    SessionResponse,
    PersonaSummary,
    SessionStatus,
    TranscriptEntry,
    EvaluationResult,
    CoachingReportSchema,
    LearningPlanSchema,
    CompetencyScore,
    StrengthItem,
    WeaknessItem,
    MistakeItem,
    LearningPlanItem,
    RubricRecommendation,
    RubricCoaching,
    EvaluationCategory,
)
from app.services.auth import get_current_user, require_auth
from app.services.debtor_simulator import (
    DebtorSimulatorService,
    EmotionalState,
    PersonaContext,
)
from app.services.evaluation_pipeline import EvaluationPipeline
from app.services.llm_service import LLMService
from app.services.script_content_loader import load_script_content
from app.services.session_access import get_authorized_session
from app.services.session_report_service import generate_report as generate_report_service
from app.services.session_service import (
    PublishedStandardRequiredError,
    create_session as create_session_service,
    get_session as get_session_service,
    end_session as end_session_service,
)
from app.services.campaign_validation_service import validate_campaign_context
from app.services.trainer_service import (
    get_trainer_campaign,
    get_trainer_campaign_agent_ids,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Session list models ---


class SessionListEntry(BaseModel):
    """A single session entry in the paginated list."""

    id: str
    scenario_id: str
    campaign_id: str | None = None
    campaign_name: str | None = None
    agent_id: str
    status: str
    persona_name: str | None = None
    overall_score: float | None = None
    created_at: str
    ended_at: str | None = None


class PaginatedSessionList(BaseModel):
    """Paginated response for session list."""

    items: list[SessionListEntry]
    total: int
    page: int
    page_size: int
    total_pages: int


# --- Session list endpoint ---


@router.get("", response_model=PaginatedSessionList)
async def list_sessions(
    page: int = 1,
    page_size: int = 20,
    agent_id: str | None = None,
    status: str | None = None,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_auth),
):
    """List sessions with role-based scoping, pagination, and filtering.

    Role-based behavior:
      - Admin: All sessions; optional agent_id and status filters.
      - Trainer: Sessions for agents in the trainer's campaign; optional
        agent_id filter (must be within campaign) and status filter.
      - Agent: Only sessions where agent_id matches the authenticated user.

    Query params:
      - page: Page number (default 1).
      - page_size: Items per page (default 20, max 100).
      - agent_id: Filter by agent UUID (admin/trainer only).
      - status: Filter by session status (pending, active, completed).
    """
    page = max(1, page)
    page_size = max(1, min(100, page_size))
    offset = (page - 1) * page_size

    # Determine role
    is_admin = current_user.role == UserRole.ADMIN.value
    is_trainer = (
        current_user.role == UserRole.USER.value
        and current_user.user_type == UserType.TRAINER.value
    )

    # Build filter conditions based on role
    conditions = []

    if is_admin:
        # Admin can optionally filter by agent_id
        if agent_id:
            conditions.append(Session.agent_id == UUID(agent_id))
    elif is_trainer:
        # Resolve trainer's campaign → get campaign agent IDs
        campaign = await get_trainer_campaign(db, current_user.id)
        if not campaign:
            # Trainer with no campaign: return empty result
            return PaginatedSessionList(
                items=[],
                total=0,
                page=page,
                page_size=page_size,
                total_pages=0,
            )
        campaign_agent_ids = await get_trainer_campaign_agent_ids(db, campaign.id)
        if not campaign_agent_ids:
            return PaginatedSessionList(
                items=[],
                total=0,
                page=page,
                page_size=page_size,
                total_pages=0,
            )
        # If trainer specifies an agent_id filter, validate it's within campaign
        if agent_id:
            requested_agent = UUID(agent_id)
            if requested_agent not in campaign_agent_ids:
                raise HTTPException(
                    status_code=403,
                    detail="Not authorized to view sessions for this agent",
                )
            conditions.append(Session.agent_id == requested_agent)
        else:
            conditions.append(Session.agent_id.in_(campaign_agent_ids))
    else:
        # Agent: force filter to own sessions, ignore agent_id param
        conditions.append(Session.agent_id == current_user.id)

    # Status filter (available to all roles)
    if status:
        conditions.append(Session.status == status)

    # Count total matching sessions
    count_stmt = select(func.count()).select_from(Session)
    for cond in conditions:
        count_stmt = count_stmt.where(cond)
    total = (await db.execute(count_stmt)).scalar_one()

    total_pages = max(1, -(-total // page_size))

    # Fetch page of sessions with evaluation scores
    stmt = (
        select(Session, Evaluation)
        .options(selectinload(Session.campaign))
        .outerjoin(Evaluation, Evaluation.session_id == Session.id)
    )
    for cond in conditions:
        stmt = stmt.where(cond)
    stmt = stmt.order_by(Session.created_at.desc()).offset(offset).limit(page_size)

    result = await db.execute(stmt)
    rows = result.all()

    items = []
    for session, evaluation in rows:
        persona_ctx = session.persona_context or {}
        score = None
        if evaluation and not evaluation.is_too_short:
            score = round(evaluation.overall_score, 1)

        campaign = getattr(session, "campaign", None)
        items.append(
            SessionListEntry(
                id=str(session.id),
                scenario_id=str(session.scenario_id),
                campaign_id=(str(session.campaign_id) if session.campaign_id else None),
                campaign_name=getattr(campaign, "name", None),
                agent_id=str(session.agent_id),
                status=session.status,
                persona_name=persona_ctx.get("name"),
                overall_score=score,
                created_at=(
                    session.created_at.isoformat() if session.created_at else ""
                ),
                ended_at=(session.ended_at.isoformat() if session.ended_at else None),
            )
        )

    return PaginatedSessionList(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


def _build_persona_summary(persona_context: dict | None) -> PersonaSummary | None:
    """Build a PersonaSummary from stored persona_context JSON."""
    if not isinstance(persona_context, dict) or not persona_context:
        return None
    return PersonaSummary(
        name=str(persona_context.get("name", "")),
        communication_style=str(persona_context.get("communication_style", "")),
        emotional_state=str(persona_context.get("emotional_state", "")),
    )


def _session_to_response(session: Session) -> SessionResponse:
    """Convert a Session model to a response with pinned standard metadata."""
    version = getattr(session, "negotiation_standard_version", None)
    standard = version.standard if version is not None else None
    campaign = getattr(session, "campaign", None)
    return SessionResponse(
        id=session.id,
        scenario_id=session.scenario_id,
        campaign_id=session.campaign_id,
        campaign_name=getattr(campaign, "name", None),
        persona=_build_persona_summary(session.persona_context),
        status=SessionStatus(session.status),
        created_at=session.created_at,
        ended_at=session.ended_at,
        standard_id=standard.id if standard is not None else None,
        standard_version_id=version.id if version is not None else None,
        standard_version_number=version.version_number if version is not None else None,
        standard_name=standard.name if standard is not None else None,
    )


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(
    body: SessionCreate,
    db: AsyncSession = Depends(get_db_session),
    current_user: User | None = Depends(get_current_user),
):
    """Create a new training session.

    Accepts a scenario_id, generates a debtor persona, and returns the new session.
    Uses the authenticated user's ID as the agent.
    """
    llm_service = LLMService()
    debtor_simulator = DebtorSimulatorService(llm_service)

    # Use the authenticated user's ID, or fallback to random UUID
    if current_user:
        agent_id = current_user.id
        logger.info(
            "Creating session for authenticated user: %s (%s)",
            current_user.email,
            current_user.id,
        )
    else:
        agent_id = uuid4()
        logger.warning(
            "Creating session without authenticated user — using random agent_id: %s",
            agent_id,
        )

    if body.campaign_id is not None:
        is_admin = (
            current_user is not None and current_user.role == UserRole.ADMIN.value
        )
        await validate_campaign_context(
            db=db,
            campaign_id=body.campaign_id,
            agent_id=agent_id,
            scenario_id=body.scenario_id,
            is_admin=is_admin,
        )

    try:
        session = await create_session_service(
            db=db,
            scenario_id=body.scenario_id,
            agent_id=agent_id,
            debtor_simulator=debtor_simulator,
            campaign_id=body.campaign_id,
        )
    except PublishedStandardRequiredError as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "published_standard_required",
                "campaign_id": str(error.campaign_id),
                "message": str(error),
            },
        ) from error
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return _session_to_response(session)


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_auth),
):
    """Get session details with role-based authorization.

    Role-based behavior:
      - Admin: Always allowed.
      - Trainer: Allowed only if session belongs to an agent in their campaign.
      - Agent: Allowed only if session belongs to them.
    """
    session = await get_authorized_session(db, session_id, current_user)
    return _session_to_response(session)


@router.post("/{session_id}/end", response_model=SessionResponse)
async def end_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """End an active session and trigger evaluation pipeline.

    The evaluation pipeline (evaluation → coaching → learning plan) is
    triggered after the session status transitions to completed.

    NOTE: In production, the pipeline should run as a background task
    to avoid blocking the HTTP response. For the MVP, it runs inline
    to ensure artifacts are available immediately after the response.
    """
    try:
        session = await end_session_service(db, session_id)
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg:
            raise HTTPException(status_code=404, detail=error_msg)
        # Invalid state transition
        raise HTTPException(status_code=400, detail=error_msg)

    # Trigger the evaluation pipeline
    # NOTE: In production, this would be dispatched as a background task.
    # Running inline here ensures artifacts are ready on response.
    try:
        llm_service = LLMService()
        pipeline = EvaluationPipeline(llm_service)
        await pipeline.run(
            session_id=session.id,
            agent_id=session.agent_id,
            db=db,
        )
    except Exception as exc:
        # Pipeline failure should not prevent the session from ending.
        # Log the error; artifacts can be regenerated later.
        logger.error(
            "Evaluation pipeline failed for session %s: %s",
            session_id,
            exc,
            exc_info=True,
        )

    # Generate the session report snapshot.
    # NOTE: Mirrors the pipeline's failure isolation above — a report
    # generation failure must never change session status or the response.
    try:
        await generate_report_service(db=db, session=session)
    except Exception as exc:
        logger.error(
            "Session report generation failed for session %s: %s",
            session_id,
            exc,
            exc_info=True,
        )

    try:
        return _session_to_response(session)
    except Exception:
        # The lifecycle transition has already committed. Optional persona or
        # pinned-standard metadata must not make a completed session appear to
        # have failed; return the durable core fields and log the bad metadata.
        logger.exception(
            "Session response serialization failed after ending session %s",
            session_id,
        )
        return SessionResponse(
            id=session.id,
            scenario_id=session.scenario_id,
            campaign_id=session.campaign_id,
            persona=None,
            status=SessionStatus.COMPLETED,
            created_at=session.created_at,
            ended_at=session.ended_at,
        )


@router.post("/{session_id}/evaluation/retry")
async def retry_session_evaluation(
    session_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_auth),
):
    """Re-run evaluation artifacts for a completed, authorized session.

    Session completion intentionally remains durable even when an LLM or
    persistence failure occurs. This endpoint is the explicit recovery path;
    the pipeline upserts its one-row artifacts, so retries are safe.
    """
    session = await get_authorized_session(db, session_id, current_user)
    if session.status != "completed":
        raise HTTPException(
            status_code=409,
            detail="Session must be completed before evaluation can be generated",
        )

    try:
        await EvaluationPipeline(LLMService()).run(
            session_id=session.id,
            agent_id=session.agent_id,
            db=db,
        )
    except Exception as exc:
        logger.error(
            "Evaluation retry failed for session %s: %s",
            session_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Evaluation generation failed. Please try again.",
        ) from None

    report_ready = True
    try:
        await generate_report_service(db=db, session=session)
    except Exception as exc:
        # The evaluation artifacts are already durable. Report generation can
        # be retried independently through POST /report.
        report_ready = False
        logger.error(
            "Report regeneration after evaluation retry failed for session %s: %s",
            session_id,
            exc,
            exc_info=True,
        )

    return {
        "session_id": str(session.id),
        "evaluation_ready": True,
        "report_ready": report_ready,
    }


@router.get("/{session_id}/transcript", response_model=list[TranscriptEntry])
async def get_session_transcript(
    session_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_auth),
):
    """Return transcript entries for an authorized session."""
    session = await get_authorized_session(db, session_id, current_user)

    stmt = (
        select(Transcript)
        .where(Transcript.session_id == session.id)
        .order_by(Transcript.sequence_number.asc())
    )
    result = await db.execute(stmt)
    transcripts = result.scalars().all()

    return [
        TranscriptEntry(
            speaker=t.speaker,
            text=t.utterance_text,
            timestamp=t.timestamp_ms,
            sequence_number=t.sequence_number,
        )
        for t in transcripts
    ]


@router.get("/{session_id}/evaluation", response_model=EvaluationResult)
async def get_session_evaluation(
    session_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_auth),
):
    """Return evaluation result for an authorized session."""
    session = await get_authorized_session(db, session_id, current_user)

    stmt = select(Evaluation).where(Evaluation.session_id == session.id)
    result = await db.execute(stmt)
    evaluation = result.scalar_one_or_none()

    if evaluation is None:
        raise HTTPException(
            status_code=404,
            detail=f"No evaluation found for session {session_id}",
        )

    # Canonical rubric category scores use a different shape than the legacy
    # competency contract. The canonical result is returned separately and
    # legacy category scores remain available for historical evaluations.
    rubric_result = evaluation.rubric_result or None
    if rubric_result and rubric_result.get("categories"):
        category_scores = []
    else:
        category_scores = [
            CompetencyScore(**cs) for cs in (evaluation.category_scores or [])
        ]
    strengths = [StrengthItem(**s) for s in (evaluation.strengths or [])]
    weaknesses = [WeaknessItem(**w) for w in (evaluation.weaknesses or [])]
    if rubric_result and rubric_result.get("categories"):
        # Canonical rubric evaluations may not have legacy finding arrays. Keep
        # the historical response contract valid without altering canonical data.
        fallback_excerpt = "See the canonical rubric evidence."
        if not strengths:
            strengths = [
                StrengthItem(
                    description="Canonical rubric result available.",
                    category=EvaluationCategory.CALL_OPENING,
                    transcript_excerpt=fallback_excerpt,
                )
            ]
        if not weaknesses:
            weaknesses = [
                WeaknessItem(
                    description="Review the canonical rubric findings.",
                    category=EvaluationCategory.CALL_OPENING,
                    transcript_excerpt=fallback_excerpt,
                )
            ]

    version = getattr(evaluation, "negotiation_standard_version", None)
    standard = getattr(version, "standard", None) if version is not None else None

    return EvaluationResult(
        session_id=evaluation.session_id,
        category_scores=category_scores,
        overall_score=evaluation.overall_score,
        strengths=strengths,
        weaknesses=weaknesses,
        is_too_short=evaluation.is_too_short,
        negotiation_standard_version_id=evaluation.negotiation_standard_version_id,
        standard_name=getattr(standard, "name", None),
        standard_version_number=getattr(version, "version_number", None),
        weighted_total=evaluation.weighted_total,
        passing_score=evaluation.passing_score,
        passed=evaluation.passed,
        standard_snapshot=evaluation.standard_snapshot,
        rubric_result=rubric_result,
    )


@router.get("/{session_id}/coaching", response_model=CoachingReportSchema)
async def get_session_coaching(
    session_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_auth),
):
    """Return coaching report for an authorized session."""
    session = await get_authorized_session(db, session_id, current_user)

    stmt = select(CoachingReport).where(CoachingReport.session_id == session.id)
    result = await db.execute(stmt)
    report = result.scalar_one_or_none()

    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"No coaching report found for session {session_id}",
        )

    # Parse legacy mistakes separately from rubric recommendation metadata.
    raw_mistakes = report.mistakes_by_category or {}
    recommendations = [
        RubricRecommendation.model_validate(item)
        for item in raw_mistakes.get("_rubric_recommendations", [])
    ]
    recommendations_by_block = {
        block_id: [RubricRecommendation.model_validate(item) for item in items]
        for block_id, items in raw_mistakes.get("_rubric_recommendations_by_block", {}).items()
    }
    rubric_coaching = None
    if raw_mistakes.get("_rubric_coaching") is not None:
        rubric_coaching = RubricCoaching.model_validate(raw_mistakes["_rubric_coaching"])
    if rubric_coaching is not None and not recommendations:
        recommendations = [
            recommendation
            for block in rubric_coaching.blocks
            for recommendation in block.recommendations
        ]
    if rubric_coaching is not None and not recommendations_by_block:
        recommendations_by_block = {
            block.rubric_block_id: block.recommendations
            for block in rubric_coaching.blocks
        }
    has_canonical_coaching = bool(
        rubric_coaching is not None
        or raw_mistakes.get("_rubric_recommendations")
        or raw_mistakes.get("_rubric_recommendations_by_block")
    )
    mistakes_by_category = {}
    if not has_canonical_coaching:
        for category_key, mistakes in raw_mistakes.items():
            if category_key.startswith("_"):
                continue
            try:
                cat = EvaluationCategory(category_key)
            except ValueError:
                continue
            mistakes_by_category[cat] = [MistakeItem(**m) for m in mistakes]

    canonical_total = len(recommendations)
    return CoachingReportSchema(
        session_id=report.session_id,
        mistakes_by_category=mistakes_by_category,
        total_mistakes=canonical_total if has_canonical_coaching else report.total_mistakes,
        no_mistakes=canonical_total == 0 if has_canonical_coaching else report.no_mistakes,
        rubric_coaching=rubric_coaching,
        rubric_recommendations=recommendations,
        rubric_recommendations_by_block=recommendations_by_block,
    )


@router.get("/{session_id}/learning-plan", response_model=LearningPlanSchema)
async def get_session_learning_plan(
    session_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_auth),
):
    """Return learning plan for an authorized session."""
    session = await get_authorized_session(db, session_id, current_user)

    stmt = select(LearningPlan).where(LearningPlan.session_id == session_id)
    result = await db.execute(stmt)
    plan = result.scalar_one_or_none()

    if plan is None:
        raise HTTPException(
            status_code=404,
            detail=f"No learning plan found for session {session_id}",
        )

    # Parse stored JSON into schema objects
    weak_competencies = [
        LearningPlanItem(**item) for item in (plan.weak_competencies or [])
    ]

    return LearningPlanSchema(
        session_id=plan.session_id,
        weak_competencies=weak_competencies,
        all_passing=plan.all_passing,
        standard_version_id=session.negotiation_standard_version_id,
    )


# --- Conversation endpoint for browser-based STT/TTS demo ---


class ConversationMessage(BaseModel):
    """Request body for sending a message in a conversation."""

    text: str


class ConversationResponse(BaseModel):
    """Response from the debtor simulator."""

    text: str
    emotional_state: str
    language: str
    call_ended: bool = False
    call_ended_reason: str | None = None
    interrupt: bool = False  # If true, debtor is interrupting the agent


# In-memory persona store for active conversations (keyed by session_id)
_active_personas: dict[UUID, PersonaContext] = {}


@router.post("/{session_id}/message", response_model=ConversationResponse)
async def send_message(
    session_id: UUID,
    body: ConversationMessage,
    db: AsyncSession = Depends(get_db_session),
):
    """Send a message in an active session and get the debtor's response.

    Uses browser-based STT/TTS: the frontend transcribes the agent's speech,
    sends text here, gets the debtor response text, and synthesizes it with
    browser TTS.
    """
    from app.services.debtor_simulator import (
        DebtorSimulatorService,
        Message,
        PersonaContext,
        select_opening_response,
    )
    from app.services.transcript_manager import TranscriptManager

    # Get the session
    session = await get_session_service(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    if session.status not in ("pending", "active"):
        raise HTTPException(status_code=400, detail="Session is not active")

    # Activate session if still pending.  First-turn detection is based on the
    # persona history as well, because a session may already be active when
    # the first message arrives (for example after reconnecting a client).
    is_first_message = False
    if session.status == "pending":
        session.status = "active"
        is_first_message = True
        await db.commit()
        await db.refresh(session)

    # Load pinned script content for script-driven behavior
    script_content = await load_script_content(db, session.script_version_id)

    # Get or create persona context for this session
    if session_id not in _active_personas:
        persona_ctx = session.persona_context or {}
        _active_personas[session_id] = PersonaContext(
            persona_id=uuid4(),
            name=persona_ctx.get("name", "Debtor"),
            communication_style=persona_ctx.get("communication_style", "cooperative"),
            financial_circumstances=persona_ctx.get("financial_circumstances", {}),
            emotional_state=EmotionalState(persona_ctx.get("emotional_state", 3)),
            language=persona_ctx.get("language", "TAGLISH"),
        )

    persona = _active_personas[session_id]
    if not is_first_message:
        is_first_message = not persona.conversation_history

    # Record agent transcript entry (skip system/initialization messages)
    transcript_manager = TranscriptManager(db)
    now = datetime.now(timezone.utc)
    is_system_prompt = body.text.startswith("[") and body.text.endswith("]")

    if not is_system_prompt:
        await transcript_manager.append_entry(
            session_id=session_id,
            speaker="agent",
            text=body.text,
            timestamp=now,
        )

    # A session can be activated by another transport before this endpoint is
    # called, so use the persona history as the source of truth as well.
    is_first_message = is_first_message or not persona.conversation_history

    # First-message detection: deliver opening response from script if available.
    # The call UI uses a bracketed system message to initialize the call; that
    # message must still trigger the scripted debtor opening response.
    if is_first_message and script_content is not None:
        opening_response = select_opening_response(script_content)
        if opening_response is not None:
            # Keep the in-memory simulator state aligned with the persisted
            # transcript. System initialization messages are not conversation
            # turns, but the scripted debtor response is.
            if not is_system_prompt:
                persona.conversation_history.append(
                    Message(role="agent", content=body.text)
                )
            persona.conversation_history.append(
                Message(role="debtor", content=opening_response)
            )
            # Record debtor opening response in transcript
            await transcript_manager.append_entry(
                session_id=session_id,
                speaker="debtor",
                text=opening_response,
                timestamp=datetime.now(timezone.utc),
            )
            await transcript_manager.persist(session_id)

            return ConversationResponse(
                text=opening_response,
                emotional_state=persona.emotional_state.name.lower(),
                language=persona.language,
                call_ended=False,
                call_ended_reason=None,
                interrupt=False,
            )

    # Script-driven escalation and goal completion checks (before LLM generation)
    if script_content is not None and not is_system_prompt:
        from app.services.debtor_simulator import (
            evaluate_escalation_conditions,
            evaluate_conversation_goal_completion,
        )

        # Check escalation conditions against agent's message
        escalation_result = evaluate_escalation_conditions(
            body.text, script_content.get("escalation_conditions")
        )
        if escalation_result is not None:
            escalation_behavior, ends_call = escalation_result
            if ends_call:
                # Record escalation behavior as debtor response in transcript
                await transcript_manager.append_entry(
                    session_id=session_id,
                    speaker="debtor",
                    text=escalation_behavior,
                    timestamp=datetime.now(timezone.utc),
                )
                await transcript_manager.persist(session_id)
                _active_personas.pop(session_id, None)

                return ConversationResponse(
                    text=escalation_behavior,
                    emotional_state=persona.emotional_state.name.lower(),
                    language=persona.language,
                    call_ended=True,
                    call_ended_reason=escalation_behavior,
                    interrupt=False,
                )

        # Check conversation goal completion against agent's message
        goal_outcome = evaluate_conversation_goal_completion(
            body.text, script_content.get("conversation_goal")
        )
        if goal_outcome is not None:
            # Goal met — end the call with a default completion message
            completion_message = "Salamat po. Maayos na po ang usapan natin."
            # Record completion message as debtor response in transcript
            await transcript_manager.append_entry(
                session_id=session_id,
                speaker="debtor",
                text=completion_message,
                timestamp=datetime.now(timezone.utc),
            )
            await transcript_manager.persist(session_id)
            _active_personas.pop(session_id, None)

            return ConversationResponse(
                text=completion_message,
                emotional_state=persona.emotional_state.name.lower(),
                language=persona.language,
                call_ended=True,
                call_ended_reason=goal_outcome,
                interrupt=False,
            )

    # Generate debtor response via LLM
    llm_service = LLMService()
    simulator = DebtorSimulatorService(llm_service)

    try:
        response = await simulator.generate_response(
            persona, body.text, script_content=script_content
        )
    except Exception as e:
        logger.error("Debtor response generation failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to generate response")

    # Record debtor transcript entry
    await transcript_manager.append_entry(
        session_id=session_id,
        speaker="debtor",
        text=response.text,
        timestamp=datetime.now(timezone.utc),
    )

    # Persist transcript entries
    await transcript_manager.persist(session_id)

    # Determine call-end and interrupt signals.
    # When script_content is present, skip hardcoded hang_up_signals entirely —
    # script-driven escalation/goal completion (evaluated above) handles call-end.
    # When no script is loaded, fall back to existing hardcoded detection.
    call_ended = False
    call_ended_reason = None
    display_text = response.text
    interrupt = False

    if script_content is None:
        # Fallback: detect if debtor wants to end the call via hardcoded signals
        hang_up_signals = [
            "hangs up",
            "ends the call",
            "slams the phone",
            "puts down the phone",
            "disconnects",
            "*hangs up*",
            "*ends call*",
            "*click*",
            "[end_call]",
        ]
        response_lower = response.text.lower()
        call_ended = any(signal in response_lower for signal in hang_up_signals)

        # Detect if debtor is interrupting (short, sharp interjection)
        interrupt_signals = [
            "wait",
            "teka",
            "sandali",
            "ano",
            "ha?",
            "huy",
            "excuse me",
            "hold on",
            "saglit",
            "wait lang",
        ]
        interrupt = len(response.text.split()) <= 8 and any(
            signal in response_lower for signal in interrupt_signals
        )

        if call_ended:
            call_ended_reason = "Debtor ended the call"
            _active_personas.pop(session_id, None)
            # Strip hang-up action markers from the displayed text
            import re

            display_text = re.sub(
                r"\s*\*(?:hangs up|ends call|click|slams the phone|puts down the phone|disconnects)\*\s*",
                "",
                display_text,
                flags=re.IGNORECASE,
            ).strip()
            # Remove [END_CALL] marker
            display_text = re.sub(
                r"\s*\[END_CALL\]\s*",
                "",
                display_text,
                flags=re.IGNORECASE,
            ).strip()
            # Also remove non-asterisk variants at the end of the message
            for signal in hang_up_signals:
                if not signal.startswith("*") and not signal.startswith("["):
                    display_text = re.sub(
                        rf",?\s*{re.escape(signal)}\.?\s*$",
                        "",
                        display_text,
                        flags=re.IGNORECASE,
                    ).strip()

    return ConversationResponse(
        text=display_text,
        emotional_state=response.emotional_state.name.lower(),
        language=response.language,
        call_ended=call_ended,
        call_ended_reason=call_ended_reason,
        interrupt=interrupt,
    )

"""Session API router with endpoints for session lifecycle and artifact retrieval.

Validates: Requirements 5.1, 6.1, 7.8, 8.2
"""

import logging
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session as get_db_session
from app.models import Session, Evaluation, CoachingReport, LearningPlan, Transcript
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
    EvaluationCategory,
)
from app.services.debtor_simulator import DebtorSimulatorService
from app.services.evaluation_pipeline import EvaluationPipeline
from app.services.llm_service import LLMService
from app.services.session_service import (
    create_session as create_session_service,
    get_session as get_session_service,
    end_session as end_session_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_persona_summary(persona_context: dict | None) -> PersonaSummary | None:
    """Build a PersonaSummary from stored persona_context JSON."""
    if not persona_context:
        return None
    return PersonaSummary(
        name=persona_context.get("name", ""),
        communication_style=persona_context.get("communication_style", ""),
        emotional_state=str(persona_context.get("emotional_state", "")),
    )


def _session_to_response(session: Session) -> SessionResponse:
    """Convert a Session model to a SessionResponse schema."""
    return SessionResponse(
        id=session.id,
        scenario_id=session.scenario_id,
        persona=_build_persona_summary(session.persona_context),
        status=SessionStatus(session.status),
        created_at=session.created_at,
        ended_at=session.ended_at,
    )


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(
    body: SessionCreate,
    db: AsyncSession = Depends(get_db_session),
):
    """Create a new training session.

    Accepts a scenario_id, generates a debtor persona, and returns the new session.
    """
    llm_service = LLMService()
    debtor_simulator = DebtorSimulatorService(llm_service)

    # Use a default agent_id for now (will be replaced with auth later)
    agent_id = uuid4()

    try:
        session = await create_session_service(
            db=db,
            scenario_id=body.scenario_id,
            agent_id=agent_id,
            debtor_simulator=debtor_simulator,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return _session_to_response(session)


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """Get session details."""
    session = await get_session_service(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

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

    return _session_to_response(session)


@router.get("/{session_id}/transcript", response_model=list[TranscriptEntry])
async def get_session_transcript(
    session_id: UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """Return transcript entries for a session."""
    # Verify session exists
    session = await get_session_service(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    stmt = (
        select(Transcript)
        .where(Transcript.session_id == session_id)
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
):
    """Return evaluation result for a session."""
    # Verify session exists
    session = await get_session_service(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    stmt = select(Evaluation).where(Evaluation.session_id == session_id)
    result = await db.execute(stmt)
    evaluation = result.scalar_one_or_none()

    if evaluation is None:
        raise HTTPException(
            status_code=404,
            detail=f"No evaluation found for session {session_id}",
        )

    # Parse stored JSON into schema objects
    category_scores = [
        CompetencyScore(**cs) for cs in (evaluation.category_scores or [])
    ]
    strengths = [StrengthItem(**s) for s in (evaluation.strengths or [])]
    weaknesses = [WeaknessItem(**w) for w in (evaluation.weaknesses or [])]

    return EvaluationResult(
        session_id=evaluation.session_id,
        category_scores=category_scores,
        overall_score=evaluation.overall_score,
        strengths=strengths,
        weaknesses=weaknesses,
        is_too_short=evaluation.is_too_short,
    )


@router.get("/{session_id}/coaching", response_model=CoachingReportSchema)
async def get_session_coaching(
    session_id: UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """Return coaching report for a session."""
    # Verify session exists
    session = await get_session_service(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    stmt = select(CoachingReport).where(CoachingReport.session_id == session_id)
    result = await db.execute(stmt)
    report = result.scalar_one_or_none()

    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"No coaching report found for session {session_id}",
        )

    # Parse stored JSON into schema objects
    mistakes_by_category = {}
    for category_key, mistakes in (report.mistakes_by_category or {}).items():
        cat = EvaluationCategory(category_key)
        mistakes_by_category[cat] = [MistakeItem(**m) for m in mistakes]

    return CoachingReportSchema(
        session_id=report.session_id,
        mistakes_by_category=mistakes_by_category,
        total_mistakes=report.total_mistakes,
        no_mistakes=report.no_mistakes,
    )


@router.get("/{session_id}/learning-plan", response_model=LearningPlanSchema)
async def get_session_learning_plan(
    session_id: UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """Return learning plan for a session."""
    # Verify session exists
    session = await get_session_service(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

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
    )


# --- Conversation endpoint for browser-based STT/TTS demo ---

from datetime import datetime, timezone
from pydantic import BaseModel


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
_active_personas: dict[UUID, "PersonaContext"] = {}


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
        EmotionalState,
        PersonaContext,
    )
    from app.services.transcript_manager import TranscriptManager

    # Get the session
    session = await get_session_service(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    if session.status not in ("pending", "active"):
        raise HTTPException(status_code=400, detail="Session is not active")

    # Activate session if still pending
    if session.status == "pending":
        session.status = "active"
        await db.commit()
        await db.refresh(session)

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

    # Generate debtor response via LLM
    llm_service = LLMService()
    simulator = DebtorSimulatorService(llm_service)

    try:
        response = await simulator.generate_response(persona, body.text)
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

    # Detect if debtor wants to end the call
    # Only trigger on very explicit hang-up phrases, not casual language
    hang_up_signals = [
        "hangs up", "ends the call", "slams the phone",
        "puts down the phone", "disconnects",
        "*hangs up*", "*ends call*", "*click*",
        "[end_call]",
    ]
    response_lower = response.text.lower()
    call_ended = any(signal in response_lower for signal in hang_up_signals)

    # Detect if debtor is interrupting (short, sharp interjection)
    interrupt_signals = [
        "wait", "teka", "sandali", "ano", "ha?", "huy",
        "excuse me", "hold on", "saglit", "wait lang",
    ]
    interrupt = (
        len(response.text.split()) <= 8 and
        any(signal in response_lower for signal in interrupt_signals)
    )

    call_ended_reason = None
    display_text = response.text
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

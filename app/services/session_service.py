"""Session repository and service layer.

Provides CRUD operations for training sessions including creation with
persona generation, status transitions, and session lifecycle management.

Validates: Requirements 3.5, 8.2
"""

import logging
import uuid as uuid_module
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Session, Scenario
from app.services.scenario_repository import get_scenario_by_id
from app.services.script_registry import get_active_published_version
from app.services.debtor_simulator import DebtorSimulatorService, PersonaContext, EmotionalState

logger = logging.getLogger(__name__)


async def create_session(
    db: AsyncSession,
    scenario_id: UUID,
    agent_id: UUID,
    debtor_simulator: DebtorSimulatorService,
) -> Session:
    """Create a new training session for a given scenario.

    Validates the scenario exists, generates a debtor persona via the
    DebtorSimulatorService, and persists a new session with status "pending".

    Args:
        db: Async database session.
        scenario_id: The UUID of the scenario to use.
        agent_id: The UUID of the agent starting the session.
        debtor_simulator: Service for generating the debtor persona.

    Returns:
        The newly created Session model instance.

    Raises:
        ValueError: If the scenario does not exist or is inactive.
        ValueError: If the scenario has no active Published_Script.
    """
    scenario = await get_scenario_by_id(db, scenario_id)
    if scenario is None:
        raise ValueError(f"Scenario with id {scenario_id} not found or inactive")

    script_version = await get_active_published_version(db, scenario_id)
    if script_version is None:
        raise ValueError(
            f"No Published_Script found for scenario {scenario_id}; "
            f"cannot start Training_Call"
        )

    # Build scenario dict for persona generation
    scenario_data = {
        "debtor_profile": scenario.debtor_profile,
        "scenario_type": scenario.scenario_type,
        "description": scenario.description or "",
    }

    persona: PersonaContext = await _generate_persona_with_fallback(
        debtor_simulator, scenario_data, scenario
    )

    # Serialize persona context to JSONB-compatible dict
    persona_dict = {
        "persona_id": str(persona.persona_id),
        "name": persona.name,
        "communication_style": persona.communication_style,
        "financial_circumstances": persona.financial_circumstances,
        "emotional_state": persona.emotional_state.value,
        "language": persona.language,
    }

    session = Session(
        scenario_id=scenario_id,
        agent_id=agent_id,
        status="pending",
        persona_context=persona_dict,
        script_version_id=script_version.id,
    )

    db.add(session)
    await db.commit()
    await db.refresh(session)

    return session


async def _generate_persona_with_fallback(
    debtor_simulator: DebtorSimulatorService,
    scenario_data: dict,
    scenario,
) -> PersonaContext:
    """Try LLM persona generation, fall back to template if LLM unavailable."""
    try:
        return await debtor_simulator.generate_persona(scenario_data)
    except Exception as e:
        logger.warning(
            "LLM persona generation failed, using fallback persona: %s", e
        )
        # Build a fallback persona from the scenario's debtor profile
        profile = scenario.debtor_profile or {}
        return PersonaContext(
            persona_id=uuid_module.uuid4(),
            name=profile.get("name", "Unknown Debtor"),
            communication_style=profile.get("personality_profile", "cooperative").split()[0].lower(),
            financial_circumstances={
                "income_level": "medium",
                "debt_amount": float(str(profile.get("outstanding_balance", 5000)).replace(",", "")),
                "reason_for_delinquency": "Financial difficulties",
            },
            emotional_state=EmotionalState.NEUTRAL,
            language="TAGLISH",
        )


async def get_session(db: AsyncSession, session_id: UUID) -> Optional[Session]:
    """Fetch a session by its ID.

    Args:
        db: Async database session.
        session_id: The UUID of the session to retrieve.

    Returns:
        The Session if found, otherwise None.
    """
    stmt = select(Session).where(Session.id == session_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def end_session(db: AsyncSession, session_id: UUID) -> Session:
    """End a training session by transitioning its status to completed.

    Sets ended_at to the current time and transitions status to "completed".
    The evaluation pipeline trigger will be wired in a later task (18.1).

    Args:
        db: Async database session.
        session_id: The UUID of the session to end.

    Returns:
        The updated Session model instance.

    Raises:
        ValueError: If the session does not exist.
        ValueError: If the session is not in a valid state for ending
                    (must be "pending" or "active").
    """
    session = await get_session(db, session_id)
    if session is None:
        raise ValueError(f"Session with id {session_id} not found")

    if session.status not in ("pending", "active"):
        raise ValueError(
            f"Cannot end session with status '{session.status}'. "
            f"Session must be 'pending' or 'active' to be ended."
        )

    session.status = "completed"
    session.ended_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(session)

    return session


async def activate_session(db: AsyncSession, session_id: UUID) -> Session:
    """Activate a pending session by transitioning its status to active.

    Args:
        db: Async database session.
        session_id: The UUID of the session to activate.

    Returns:
        The updated Session model instance.

    Raises:
        ValueError: If the session does not exist.
        ValueError: If the session is not in "pending" status.
    """
    session = await get_session(db, session_id)
    if session is None:
        raise ValueError(f"Session with id {session_id} not found")

    if session.status != "pending":
        raise ValueError(
            f"Cannot activate session with status '{session.status}'. "
            f"Session must be 'pending' to be activated."
        )

    session.status = "active"

    await db.commit()
    await db.refresh(session)

    return session

"""Session repository and service layer.

Provides CRUD operations for training sessions including creation with
persona generation, status transitions, and session lifecycle management.

Validates: Requirements 3.5, 8.2
"""

import logging
import uuid as uuid_module
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import (
    Campaign,
    CampaignAgent,
    NegotiationStandard,
    NegotiationStandardVersion,
    Session,
    campaign_scenarios,
)
from app.schemas.event import EventMetadata
from app.services.debtor_simulator import (
    DebtorSimulatorService,
    EmotionalState,
    PersonaContext,
)
from app.services.event_instances import event_broadcaster
from app.services.scenario_repository import get_scenario_by_id
from app.services.script_registry import get_active_published_version


logger = logging.getLogger(__name__)


class PublishedStandardRequiredError(ValueError):
    """Raised when a campaign simulation lacks a published rubric version."""

    def __init__(self, campaign_id: UUID) -> None:
        super().__init__(f"Campaign {campaign_id} requires a published negotiation standard")
        self.campaign_id = campaign_id


async def _resolve_published_version(
    db: AsyncSession,
    scenario_id: UUID,
    agent_id: UUID,
    campaign_id: UUID | None,
) -> NegotiationStandardVersion | None:
    """Resolve an assigned campaign and its current immutable published version."""
    campaign_statement = (
        select(Campaign)
        .join(campaign_scenarios, campaign_scenarios.c.campaign_id == Campaign.id)
        .where(campaign_scenarios.c.scenario_id == scenario_id)
    )
    if campaign_id is not None:
        campaign_statement = campaign_statement.where(Campaign.id == campaign_id)
    else:
        campaign_statement = campaign_statement.join(
            CampaignAgent, CampaignAgent.campaign_id == Campaign.id
        ).where(
            CampaignAgent.agent_id == agent_id,
            CampaignAgent.role != "trainer",
        )
    campaign_result = await db.execute(campaign_statement)
    campaigns = campaign_result.scalars().unique().all()
    if not campaigns:
        return None
    if campaign_id is None and len(campaigns) > 1:
        raise PublishedStandardRequiredError(campaigns[0].id)
    selected_campaign = campaigns[0]
    statement = (
        select(NegotiationStandardVersion)
        .join(
            NegotiationStandard,
            NegotiationStandard.current_version_id == NegotiationStandardVersion.id,
        )
        .where(
            NegotiationStandard.campaign_id == selected_campaign.id,
            NegotiationStandard.status == "published",
            NegotiationStandardVersion.snapshot.is_not(None),
        )
    )
    version = (await db.execute(statement)).scalar_one_or_none()
    if version is None:
        raise PublishedStandardRequiredError(selected_campaign.id)
    return version


async def create_session(
    db: AsyncSession,
    scenario_id: UUID,
    agent_id: UUID,
    debtor_simulator: DebtorSimulatorService,
    campaign_id: UUID | None = None,
) -> Session:
    """Create a session and pin an assigned campaign's published rubric version.

    Scenarios without a campaign remain readable for legacy/local sessions. When
    a campaign is explicitly selected or uniquely assigned, a published version
    is mandatory and is stored in the same transaction as the session.
    Validates the scenario exists, generates a debtor persona via the
    DebtorSimulatorService, and persists a new session with status "pending".

    Args:
        db: Async database session.
        scenario_id: The UUID of the scenario to use.
        agent_id: The UUID of the agent starting the session.
        debtor_simulator: Service for generating the debtor persona.
        campaign_id: Optional campaign context for the new session.

    Returns:
        The newly created Session model instance.

    Raises:
        ValueError: If the scenario does not exist or is inactive.
    """
    scenario = await get_scenario_by_id(db, scenario_id)
    if scenario is None:
        raise ValueError(f"Scenario with id {scenario_id} not found or inactive")

    standard_version = await _resolve_published_version(db, scenario_id, agent_id, campaign_id)
    script_version = await get_active_published_version(db, scenario_id)

    scenario_data = {
        "debtor_profile": scenario.debtor_profile,
        "scenario_type": scenario.scenario_type,
        "description": scenario.description or "",
    }

    # The reads above autobegin a transaction. Release its connection before
    # waiting on the external LLM; otherwise each concurrent session creation
    # holds a database connection for the full LLM timeout.
    script_version_id = script_version.id if script_version is not None else None
    negotiation_standard_version_id = standard_version.id if standard_version is not None else None
    await db.close()

    persona: PersonaContext = await _generate_persona_with_fallback(
        debtor_simulator, scenario_data, scenario
    )
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
        campaign_id=campaign_id,
        status="pending",
        persona_context=persona_dict,
        script_version_id=script_version_id,
        negotiation_standard_version_id=negotiation_standard_version_id,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    await event_broadcaster.emit(
        "session.created",
        session.id,
        EventMetadata(agent_id=agent_id),
    )
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
        logger.warning("LLM persona generation failed, using fallback persona: %s", e)
        # Build a fallback persona from the scenario's debtor profile
        profile = scenario.debtor_profile or {}
        return PersonaContext(
            persona_id=uuid_module.uuid4(),
            name=profile.get("name", "Unknown Debtor"),
            communication_style=profile.get("personality_profile", "cooperative")
            .split()[0]
            .lower(),
            financial_circumstances={
                "income_level": "medium",
                "debt_amount": float(
                    str(profile.get("outstanding_balance", 5000)).replace(",", "")
                ),
                "reason_for_delinquency": "Financial difficulties",
            },
            emotional_state=EmotionalState.NEUTRAL,
            language="TAGLISH",
        )


async def get_session(db: AsyncSession, session_id: UUID) -> Session | None:
    """Fetch a session by its ID.

    Args:
        db: Async database session.
        session_id: The UUID of the session to retrieve.

    Returns:
        The Session if found, otherwise None.
    """
    stmt = select(Session).options(selectinload(Session.campaign)).where(Session.id == session_id)
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
    session.ended_at = datetime.now(UTC)

    await db.commit()
    await db.refresh(session)

    await event_broadcaster.emit(
        "session.ended",
        session.id,
        EventMetadata(agent_id=session.agent_id),
    )

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

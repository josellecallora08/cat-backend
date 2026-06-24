"""Demo data seeder for the Collection Agent Trainer dashboards.

Populates realistic agents, completed sessions, transcripts, and evaluations
so that BOTH the admin dashboard (leaderboard, rankings, aggregate stats) and
the per-agent dashboard (score progression, competency radar, recent sessions)
have meaningful data to display.

Idempotent: keyed on a sentinel agent email + a marker so it won't duplicate
on repeated runs. Safe to call on startup or run manually.

Run manually:
    python -m app.services.seed_demo_data
"""

from __future__ import annotations

import logging
import random
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Scenario, Session, Transcript, Evaluation
from app.models.user import User, UserRole
from app.services.auth import hash_password

logger = logging.getLogger(__name__)

# Deterministic output so the demo looks the same each fresh seed
random.seed(42)

CATEGORIES = [
    "call_opening",
    "compliance",
    "empathy_communication",
    "negotiation_resolution",
]

CATEGORY_WEIGHTS = {
    "call_opening": 0.20,
    "compliance": 0.30,
    "empathy_communication": 0.25,
    "negotiation_resolution": 0.25,
}

# Demo agents — the default agent@cat.ph also gets data (handled separately).
DEMO_AGENTS = [
    {"email": "joselle@cat.ph", "full_name": "Joselle Callora", "skill": 0.88},
    {"email": "shanne@cat.ph", "full_name": "Shanne Rivera", "skill": 0.74},
    {"email": "marco@cat.ph", "full_name": "Marco Bautista", "skill": 0.63},
    {"email": "liza@cat.ph", "full_name": "Liza Fernandez", "skill": 0.80},
    {"email": "paolo@cat.ph", "full_name": "Paolo Mendoza", "skill": 0.55},
]

DEMO_PASSWORD = "agent123"

# A short, believable transcript template (agent/debtor turns)
SAMPLE_TURNS = [
    ("agent", "Good afternoon po, this is {agent} calling from CATS Collections regarding your account."),
    ("debtor", "Sino po ito? Anong account?"),
    ("agent", "I understand your concern po. May I verify I'm speaking with {debtor}?"),
    ("debtor", "Oo, ako nga. Pero wala akong pera ngayon."),
    ("agent", "I hear you po, and I appreciate your honesty. Pwede po nating pag-usapan ang options."),
    ("debtor", "Anong options? Hindi ko talaga kaya yung buong amount."),
    ("agent", "We can set up an installment plan that fits your budget po. Would PHP 2,000 monthly work?"),
    ("debtor", "Siguro kaya ko yun... pero next week pa ako sweldo."),
    ("agent", "That's perfectly fine po. Let's schedule the first payment after your payday."),
    ("debtor", "Sige po, salamat sa pag-intindi."),
    ("agent", "Thank you po for working with me today. I'll send the confirmation details."),
]

STRENGTHS_POOL = [
    ("Strong empathetic opening that built rapport", "empathy_communication"),
    ("Properly verified the debtor's identity before discussing the account", "compliance"),
    ("Offered a concrete, affordable installment plan", "negotiation_resolution"),
    ("Maintained a calm, professional tone throughout", "empathy_communication"),
    ("Clearly stated the purpose of the call upfront", "call_opening"),
]

WEAKNESSES_POOL = [
    ("Did not deliver the Mini-Miranda compliance disclosure", "compliance"),
    ("Missed an opportunity to confirm the payment date in writing", "negotiation_resolution"),
    ("Opening was slightly rushed before verifying identity", "call_opening"),
    ("Could have acknowledged the debtor's stress more directly", "empathy_communication"),
]


def _category_score(skill: float, drift: float) -> int:
    """Generate a believable per-category score around the agent's skill."""
    base = skill * 100 + drift
    noise = random.uniform(-8, 8)
    return max(35, min(99, int(round(base + noise))))


def _build_category_scores(skill: float, drift: float) -> tuple[list[dict], float]:
    """Build the category_scores JSON and weighted overall score."""
    scores: dict[str, int] = {}
    for cat in CATEGORIES:
        # Compliance tends to lag for lower-skill agents
        cat_bias = -6 if cat == "compliance" and skill < 0.7 else 0
        scores[cat] = _category_score(skill, drift + cat_bias)

    category_scores = [
        {"category": cat, "score": scores[cat], "strengths": [], "weaknesses": []}
        for cat in CATEGORIES
    ]
    overall = round(
        sum(scores[cat] * CATEGORY_WEIGHTS[cat] for cat in CATEGORIES), 1
    )
    return category_scores, overall


def _pick_strengths_weaknesses() -> tuple[list[dict], list[dict]]:
    s = random.sample(STRENGTHS_POOL, k=random.randint(1, 3))
    w = random.sample(WEAKNESSES_POOL, k=random.randint(1, 3))
    strengths = [
        {"description": desc, "category": cat, "transcript_excerpt": "I understand your concern po..."}
        for desc, cat in s
    ]
    weaknesses = [
        {"description": desc, "category": cat, "transcript_excerpt": "Anong options?"}
        for desc, cat in w
    ]
    return strengths, weaknesses


async def _get_or_create_agent(db: AsyncSession, email: str, full_name: str) -> User:
    stmt = select(User).where(User.email == email)
    existing = (await db.execute(stmt)).scalar_one_or_none()
    if existing:
        return existing
    user = User(
        id=uuid.uuid4(),
        email=email,
        hashed_password=hash_password(DEMO_PASSWORD),
        full_name=full_name,
        role=UserRole.AGENT.value,
    )
    db.add(user)
    await db.flush()
    return user


async def _seed_sessions_for_agent(
    db: AsyncSession,
    agent: User,
    scenarios: list[Scenario],
    num_sessions: int,
    skill: float,
) -> int:
    """Create completed sessions + transcripts + evaluations for one agent."""
    created = 0
    now = datetime.now(timezone.utc)

    for i in range(num_sessions):
        scenario = random.choice(scenarios)
        # Spread sessions across the last ~8 weeks, oldest first
        days_ago = (num_sessions - i) * random.randint(2, 4)
        created_at = now - timedelta(days=days_ago, hours=random.randint(0, 8))

        # Skill drifts upward over time to show an improvement trend
        progress_drift = (i / max(1, num_sessions - 1)) * 12 - 4

        persona = scenario.debtor_profile or {}
        session = Session(
            id=uuid.uuid4(),
            scenario_id=scenario.id,
            agent_id=agent.id,
            status="completed",
            persona_context={
                "name": persona.get("name", "Debtor"),
                "communication_style": "cooperative",
                "financial_circumstances": {"income_level": "medium"},
                "emotional_state": 3,
                "language": "TAGLISH",
            },
            created_at=created_at,
            ended_at=created_at + timedelta(minutes=random.randint(4, 12)),
        )
        db.add(session)
        await db.flush()

        # Transcript
        debtor_name = persona.get("name", "the debtor")
        for seq, (speaker, template) in enumerate(SAMPLE_TURNS):
            text = template.format(agent=agent.full_name.split()[0], debtor=debtor_name)
            db.add(
                Transcript(
                    id=uuid.uuid4(),
                    session_id=session.id,
                    speaker=speaker,
                    utterance_text=text,
                    timestamp_ms=created_at + timedelta(seconds=seq * 12),
                    sequence_number=seq,
                )
            )

        # Evaluation
        category_scores, overall = _build_category_scores(skill, progress_drift)
        strengths, weaknesses = _pick_strengths_weaknesses()
        db.add(
            Evaluation(
                id=uuid.uuid4(),
                session_id=session.id,
                overall_score=overall,
                category_scores=category_scores,
                strengths=strengths,
                weaknesses=weaknesses,
                is_too_short=False,
                created_at=created_at + timedelta(minutes=13),
            )
        )
        created += 1

    return created


async def seed_demo_data(db: AsyncSession, force: bool = False) -> None:
    """Seed demo dashboard data for admin + agents.

    Args:
        db: Async DB session.
        force: When False (default), skips seeding if demo data already exists.
    """
    # Idempotency guard — bail if a sentinel demo agent already has sessions
    sentinel = (
        await db.execute(select(User).where(User.email == "joselle@cat.ph"))
    ).scalar_one_or_none()
    if sentinel and not force:
        has_sessions = (
            await db.execute(select(Session.id).where(Session.agent_id == sentinel.id).limit(1))
        ).first()
        if has_sessions:
            logger.info("Demo data already present, skipping seed")
            return

    scenarios = list((await db.execute(select(Scenario))).scalars().all())
    if not scenarios:
        logger.warning("No scenarios found; seed scenarios first. Skipping demo data.")
        return

    total = 0

    # Give the default agent@cat.ph a personal history too (for agent dashboard demo)
    default_agent = (
        await db.execute(select(User).where(User.email == "agent@cat.ph"))
    ).scalar_one_or_none()
    if default_agent:
        total += await _seed_sessions_for_agent(
            db, default_agent, scenarios, num_sessions=9, skill=0.71
        )

    # Demo agents for the admin leaderboard / rankings
    for spec in DEMO_AGENTS:
        agent = await _get_or_create_agent(db, spec["email"], spec["full_name"])
        total += await _seed_sessions_for_agent(
            db,
            agent,
            scenarios,
            num_sessions=random.randint(7, 12),
            skill=spec["skill"],
        )

    await db.commit()
    logger.info("Seeded %d demo sessions across %d agents", total, len(DEMO_AGENTS) + (1 if default_agent else 0))


async def _main() -> None:
    """Standalone entrypoint for manual seeding."""
    logging.basicConfig(level=logging.INFO)
    from app.database import async_session_factory

    async with async_session_factory() as db:
        await seed_demo_data(db, force=False)


if __name__ == "__main__":
    import asyncio

    asyncio.run(_main())

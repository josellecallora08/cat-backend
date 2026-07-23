"""Seed script for campaign dashboard demo data.

Creates sample sessions and evaluations for agents in campaigns,
enabling the dashboard to display meaningful metrics and charts.

Usage:
    python -m scripts.seed_campaign_dashboard

Or import and call:
    from scripts.seed_campaign_dashboard import seed_campaign_dashboard_data
    await seed_campaign_dashboard_data(db)
"""

import asyncio
import logging
import random
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory
from app.models import Evaluation, Session
from app.models.campaign import (
    Campaign,
    CampaignAgent,
    CampaignStatus,
    campaign_scenarios,
)


logger = logging.getLogger(__name__)

SEED_MARKER = "campaign_dashboard"
CATEGORIES = [
    "call_opening",
    "compliance",
    "empathy_communication",
    "negotiation_resolution",
]


def _generate_category_scores(target_overall: float) -> list[dict]:
    """Generate category scores that roughly average to the target overall score."""
    scores = []
    for cat in CATEGORIES:
        variation = random.uniform(-15, 15)
        score = max(0, min(100, target_overall + variation))
        scores.append({"category": cat, "score": round(score, 1)})
    return scores


async def _has_existing_seed_data(db: AsyncSession) -> bool:
    """Check if seed data already exists by scanning persona_context for our marker."""
    stmt = select(Session).where(Session.status == "completed").limit(100)
    result = await db.execute(stmt)
    for session in result.scalars().all():
        if (
            isinstance(session.persona_context, dict)
            and session.persona_context.get("seeded") == SEED_MARKER
        ):
            return True
    return False


async def seed_campaign_dashboard_data(db: AsyncSession) -> None:
    """Seed demo sessions and evaluations for campaign dashboards.

    Idempotent: skips if seed data already exists.
    """
    if await _has_existing_seed_data(db):
        logger.info("Seed data already exists. Skipping.")
        return

    # Find campaigns that are not archived
    campaigns_stmt = select(Campaign).where(
        Campaign.status != CampaignStatus.ARCHIVED.value
    )
    campaigns_result = await db.execute(campaigns_stmt)
    campaigns = list(campaigns_result.scalars().all())

    if not campaigns:
        logger.info("No campaigns found. Create campaigns first.")
        return

    seeded_count = 0
    campaigns_seeded = 0

    for campaign in campaigns:
        # Get agents for this campaign
        agents_stmt = select(CampaignAgent.agent_id).where(
            CampaignAgent.campaign_id == campaign.id
        )
        agents_result = await db.execute(agents_stmt)
        agent_ids = list(agents_result.scalars().all())

        # Get scenarios for this campaign
        scenarios_stmt = select(campaign_scenarios.c.scenario_id).where(
            campaign_scenarios.c.campaign_id == campaign.id
        )
        scenarios_result = await db.execute(scenarios_stmt)
        scenario_ids = list(scenarios_result.scalars().all())

        if not agent_ids or not scenario_ids:
            continue

        campaigns_seeded += 1

        # Assign performance profiles: some pass (avg >= 70), some don't
        for i, agent_id in enumerate(agent_ids):
            is_high_performer = i < len(agent_ids) // 2 + 1

            base_score = (
                random.uniform(72, 88) if is_high_performer else random.uniform(48, 65)
            )
            num_sessions = random.randint(8, 15)

            now = datetime.now(timezone.utc)
            start_date = now - timedelta(days=30)

            for j in range(num_sessions):
                scenario_id = random.choice(scenario_ids)

                # Spread sessions across 30 days
                session_date = start_date + timedelta(
                    days=(30 * j / num_sessions) + random.uniform(-1, 1)
                )

                # High performers improve slightly over time
                score_drift = (
                    (j / num_sessions) * 5
                    if is_high_performer
                    else random.uniform(-3, 3)
                )
                overall_score = max(
                    20, min(98, base_score + score_drift + random.uniform(-8, 8))
                )
                overall_score = round(overall_score, 1)

                session = Session(
                    id=uuid.uuid4(),
                    scenario_id=scenario_id,
                    agent_id=agent_id,
                    status="completed",
                    persona_context={
                        "seeded": SEED_MARKER,
                        "name": f"Seed Debtor {j + 1}",
                    },
                    created_at=session_date,
                    ended_at=session_date + timedelta(minutes=random.randint(5, 20)),
                )
                db.add(session)
                await db.flush()

                category_scores = _generate_category_scores(overall_score)
                evaluation = Evaluation(
                    id=uuid.uuid4(),
                    session_id=session.id,
                    overall_score=overall_score,
                    category_scores=category_scores,
                    strengths=[{"text": "Good communication skills"}],
                    weaknesses=[{"text": "Needs work on compliance"}],
                    is_too_short=False,
                    created_at=session_date + timedelta(minutes=random.randint(1, 5)),
                )
                db.add(evaluation)
                seeded_count += 1

    await db.commit()
    logger.info(
        "Seeded %d sessions with evaluations across %d campaigns.",
        seeded_count,
        campaigns_seeded,
    )


async def main() -> None:
    """Run the seed script standalone."""
    logging.basicConfig(level=logging.INFO)
    async with async_session_factory() as db:
        await seed_campaign_dashboard_data(db)


if __name__ == "__main__":
    asyncio.run(main())

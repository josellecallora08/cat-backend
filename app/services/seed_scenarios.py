"""Default scenario seeder for the Collection Agent Trainer.

Seeds the database with realistic Filipino debt collection training scenarios
on application startup. Only inserts scenarios that don't already exist
(matched by name) to avoid duplicates on restart.
"""

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Scenario

logger = logging.getLogger(__name__)

DEFAULT_SCENARIOS = [
    {
        "name": "Financial Hardship — Job Loss",
        "scenario_type": "FINANCIAL_HARDSHIP",
        "description": (
            "The debtor recently lost their job and is struggling to make ends meet. "
            "Practice empathetic communication while still working toward a payment arrangement."
        ),
        "debtor_profile": {
            "name": "Maria Santos",
            "outstanding_balance": "45000.00",
            "days_past_due": 90,
            "personality_profile": (
                "Anxious and apologetic. Genuinely wants to pay but currently has no stable income. "
                "Will become cooperative if the agent shows understanding, but may shut down if pressured."
            ),
            "conversation_goal": "Negotiate a reduced payment plan that fits her current situation",
        },
    },
    {
        "name": "Angry Customer — Disputed Charges",
        "scenario_type": "ANGRY_CUSTOMER",
        "description": (
            "The debtor believes they were charged incorrectly and is frustrated with the company. "
            "Practice de-escalation techniques and complaint handling while maintaining professionalism."
        ),
        "debtor_profile": {
            "name": "Roberto Dela Cruz",
            "outstanding_balance": "28500.00",
            "days_past_due": 45,
            "personality_profile": (
                "Hostile and confrontational. Believes the charges are unfair and blames the company. "
                "Will raise his voice and threaten to report to authorities. Can be calmed with validation "
                "and clear explanation, but will hang up if the agent matches his aggression."
            ),
            "conversation_goal": "Get the agent to admit the charges are wrong and remove the debt",
        },
    },
    {
        "name": "Payment Extension — Medical Emergency",
        "scenario_type": "PAYMENT_EXTENSION",
        "description": (
            "The debtor had a family medical emergency and needs more time to pay. "
            "Practice balancing company policy with compassionate flexibility."
        ),
        "debtor_profile": {
            "name": "Jennifer Reyes",
            "outstanding_balance": "67000.00",
            "days_past_due": 60,
            "personality_profile": (
                "Cooperative but emotional. Her mother was recently hospitalized and medical bills "
                "took priority. She is honest about her situation and willing to pay, just needs "
                "a 2-week extension. May cry or get emotional when discussing her mother's health."
            ),
            "conversation_goal": "Secure a 2-week payment extension without additional penalties",
        },
    },
    {
        "name": "Balance Dispute — Partial Payment Claim",
        "scenario_type": "BALANCE_DISPUTE",
        "description": (
            "The debtor claims they already made a partial payment that wasn't reflected. "
            "Practice verification procedures and dispute resolution while maintaining trust."
        ),
        "debtor_profile": {
            "name": "Carlos Mendoza",
            "outstanding_balance": "35000.00",
            "days_past_due": 30,
            "personality_profile": (
                "Evasive but not hostile. Claims to have paid PHP 15,000 last month via bank transfer "
                "but has no receipt handy. Will provide vague details when pressed. Might be telling "
                "the truth or might be stalling — the agent needs to verify tactfully."
            ),
            "conversation_goal": "Convince the agent that PHP 15,000 was already paid and only owes the remainder",
        },
    },
    {
        "name": "Financial Hardship — Single Parent",
        "scenario_type": "FINANCIAL_HARDSHIP",
        "description": (
            "A single parent supporting three children on a minimum wage salary. "
            "Practice negotiating realistic payment plans while showing empathy for difficult circumstances."
        ),
        "debtor_profile": {
            "name": "Lorna Villanueva",
            "outstanding_balance": "52000.00",
            "days_past_due": 120,
            "personality_profile": (
                "Defensive initially but warms up to kind agents. She works two jobs and barely covers "
                "rent and school fees. She's been avoiding calls out of shame, not defiance. Will open up "
                "if the agent is non-judgmental and offers a genuinely affordable payment option."
            ),
            "conversation_goal": "Find a payment plan of no more than PHP 2,000 per month",
        },
    },
    {
        "name": "Angry Customer — Repeated Calls",
        "scenario_type": "ANGRY_CUSTOMER",
        "description": (
            "The debtor is furious about receiving multiple calls per day and threatens to file a complaint. "
            "Practice compliance awareness and professional tone under pressure."
        ),
        "debtor_profile": {
            "name": "Eduardo Garcia",
            "outstanding_balance": "18000.00",
            "days_past_due": 75,
            "personality_profile": (
                "Very hostile from the start. Has been receiving 3-4 calls daily and considers it harassment. "
                "Will immediately threaten legal action and demand a supervisor. Can only be calmed by "
                "acknowledging the issue, apologizing for the inconvenience, and offering to resolve the "
                "underlying debt in one conversation. Will absolutely hang up if threatened."
            ),
            "conversation_goal": "Get the agent to stop calling and acknowledge the harassment",
        },
    },
    {
        "name": "Payment Extension — Business Downturn",
        "scenario_type": "PAYMENT_EXTENSION",
        "description": (
            "A small business owner whose sari-sari store was affected by a typhoon. "
            "Practice understanding business circumstances while working toward resolution."
        ),
        "debtor_profile": {
            "name": "Ramon Aquino",
            "outstanding_balance": "95000.00",
            "days_past_due": 45,
            "personality_profile": (
                "Cooperative and straightforward. His store was damaged in a recent typhoon and "
                "he's waiting for insurance payout. He's a reliable payer with good history before "
                "this incident. Wants a 30-day extension and is confident he can pay in full after."
            ),
            "conversation_goal": "Get a 30-day extension until insurance money arrives",
        },
    },
    {
        "name": "Balance Dispute — Identity Concern",
        "scenario_type": "BALANCE_DISPUTE",
        "description": (
            "The debtor claims they never opened the account and suspects identity theft. "
            "Practice proper verification and escalation procedures."
        ),
        "debtor_profile": {
            "name": "Angela Torres",
            "outstanding_balance": "125000.00",
            "days_past_due": 15,
            "personality_profile": (
                "Anxious and confused. Genuinely believes this might be identity theft as she "
                "doesn't recall opening this credit line. She's cooperative but insistent that "
                "she shouldn't have to pay for something she didn't authorize. Will ask for "
                "documentation and dispute process details."
            ),
            "conversation_goal": "Understand the dispute process and get the collection paused while it's investigated",
        },
    },
]


async def seed_default_scenarios(db: AsyncSession) -> None:
    """Seed the database with default training scenarios.

    Only inserts scenarios whose names don't already exist in the database,
    making this safe to call on every startup.

    Args:
        db: An async database session.
    """
    # Get existing scenario names
    stmt = select(Scenario.name)
    result = await db.execute(stmt)
    existing_names = set(result.scalars().all())

    inserted = 0
    for scenario_data in DEFAULT_SCENARIOS:
        if scenario_data["name"] in existing_names:
            continue

        scenario = Scenario(
            id=uuid.uuid4(),
            name=scenario_data["name"],
            scenario_type=scenario_data["scenario_type"],
            description=scenario_data["description"],
            debtor_profile=scenario_data["debtor_profile"],
            is_active=True,
        )
        db.add(scenario)
        inserted += 1

    if inserted > 0:
        await db.commit()
        logger.info("Seeded %d default scenarios", inserted)
    else:
        logger.info("All default scenarios already exist, skipping seed")

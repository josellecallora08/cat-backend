"""Service layer for campaign dashboard analytics."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Evaluation,
    Scenario,
    Session,
)
from app.models.campaign import (
    Campaign,
    CampaignAgent,
    CampaignStatus,
    campaign_scenarios,
)
from app.models.user import User
from app.schemas.campaign_dashboard import (
    PASSING_THRESHOLD,
    AgentProgressResponse,
    AgentSessionItem,
    AgentSummary,
    CampaignDashboardResponse,
    CategoryAverage,
    ScenarioAverage,
    ScoreDataPoint,
)


def _compute_improvement_trend(scores: list[float]) -> float | None:
    """Compute improvement trend from chronological scores.

    Returns avg(last 3) - avg(first 3) when len >= 4, else None.
    """
    if len(scores) < 4:
        return None
    first_three = scores[:3]
    last_three = scores[-3:]
    return round(sum(last_three) / 3 - sum(first_three) / 3, 1)


async def _get_campaign_agent_ids(db: AsyncSession, campaign_id: UUID) -> list[UUID]:
    """Get all agent IDs assigned to a campaign."""
    stmt = select(CampaignAgent.agent_id).where(
        CampaignAgent.campaign_id == campaign_id
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def _get_campaign_scenario_ids(db: AsyncSession, campaign_id: UUID) -> list[UUID]:
    """Get all scenario IDs assigned to a campaign."""
    stmt = select(campaign_scenarios.c.scenario_id).where(
        campaign_scenarios.c.campaign_id == campaign_id
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def _get_campaign_or_none(db: AsyncSession, campaign_id: UUID) -> Campaign | None:
    """Fetch campaign by ID, excluding archived."""
    stmt = select(Campaign).where(
        Campaign.id == campaign_id,
        Campaign.status != CampaignStatus.ARCHIVED.value,
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def _get_agent_names(db: AsyncSession, agent_ids: list[UUID]) -> dict[UUID, str]:
    """Fetch display names for a list of agent UUIDs."""
    if not agent_ids:
        return {}
    stmt = select(User.id, User.full_name).where(User.id.in_(agent_ids))
    result = await db.execute(stmt)
    return {uid: name for uid, name in result.all()}


async def _get_score_history(
    db: AsyncSession,
    agent_ids: list[UUID],
    scenario_ids: list[UUID],
    agent_id_filter: UUID | None = None,
) -> list[ScoreDataPoint]:
    """Get chronological score data points for chart rendering."""
    stmt = (
        select(
            Evaluation.overall_score,
            Evaluation.created_at,
            Session.agent_id,
        )
        .join(Session, Session.id == Evaluation.session_id)
        .where(
            Session.agent_id.in_(agent_ids),
            Session.scenario_id.in_(scenario_ids),
            Evaluation.is_too_short == False,  # noqa: E712
        )
    )
    if agent_id_filter:
        stmt = stmt.where(Session.agent_id == agent_id_filter)

    stmt = stmt.order_by(Evaluation.created_at.asc()).limit(100)
    result = await db.execute(stmt)
    rows = result.all()

    return [
        ScoreDataPoint(
            session_number=i + 1,
            overall_score=round(row[0], 1),
            date=row[1].isoformat() if row[1] else "",
            agent_id=str(row[2]),
        )
        for i, row in enumerate(rows)
    ]


async def _get_category_averages(
    db: AsyncSession,
    agent_ids: list[UUID],
    scenario_ids: list[UUID],
) -> list[CategoryAverage]:
    """Compute per-category average scores across campaign sessions."""
    stmt = (
        select(Evaluation.category_scores)
        .join(Session, Session.id == Evaluation.session_id)
        .where(
            Session.agent_id.in_(agent_ids),
            Session.scenario_id.in_(scenario_ids),
            Evaluation.is_too_short == False,  # noqa: E712
        )
    )
    result = await db.execute(stmt)
    all_category_scores = result.scalars().all()

    category_totals: dict[str, list[float]] = {}
    for scores_list in all_category_scores:
        if isinstance(scores_list, list):
            for score_item in scores_list:
                if (
                    isinstance(score_item, dict)
                    and "category" in score_item
                    and "score" in score_item
                ):
                    cat = score_item["category"]
                    if cat not in category_totals:
                        category_totals[cat] = []
                    category_totals[cat].append(float(score_item["score"]))

    category_labels = {
        "call_opening": "Call Opening",
        "compliance": "Compliance",
        "empathy_communication": "Empathy & Communication",
        "negotiation_resolution": "Negotiation & Resolution",
    }

    return [
        CategoryAverage(
            category=category_labels.get(cat, cat),
            average_score=round(sum(scores) / len(scores), 1),
        )
        for cat, scores in category_totals.items()
        if scores
    ]


async def get_campaign_dashboard(
    db: AsyncSession,
    campaign_id: UUID,
    agent_id_filter: UUID | None = None,
) -> CampaignDashboardResponse:
    """Get aggregated campaign dashboard data.

    Args:
        db: Database session.
        campaign_id: The campaign to get dashboard for.
        agent_id_filter: Optional filter for score_history to a specific agent.

    Returns:
        CampaignDashboardResponse with KPIs, agent summaries, and chart data.

    Raises:
        ValueError: If campaign not found.
    """
    campaign = await _get_campaign_or_none(db, campaign_id)
    if campaign is None:
        raise ValueError("Campaign not found")

    agent_ids = await _get_campaign_agent_ids(db, campaign_id)
    scenario_ids = await _get_campaign_scenario_ids(db, campaign_id)

    if not agent_ids or not scenario_ids:
        # Campaign has no agents or scenarios — return empty dashboard
        agent_names = await _get_agent_names(db, agent_ids)
        return CampaignDashboardResponse(
            total_agents=len(agent_ids),
            average_score=None,
            agents_passed=0,
            agents_needing_improvement=len(agent_ids),
            agent_summaries=[
                AgentSummary(
                    agent_id=str(aid),
                    agent_name=agent_names.get(aid, "Unknown"),
                    sessions_completed=0,
                    average_score=None,
                    best_score=None,
                    improvement_trend=None,
                )
                for aid in agent_ids
            ],
            score_history=[],
            category_averages=[],
        )

    # Get all campaign-scoped evaluated sessions
    agent_names = await _get_agent_names(db, agent_ids)
    agent_summaries = []
    all_scores: list[float] = []
    agents_passed = 0
    agents_needing_improvement = 0

    for aid in agent_ids:
        # Get evaluated sessions for this agent in this campaign
        stmt = (
            select(Evaluation.overall_score)
            .join(Session, Session.id == Evaluation.session_id)
            .where(
                Session.agent_id == aid,
                Session.scenario_id.in_(scenario_ids),
                Evaluation.is_too_short == False,  # noqa: E712
            )
            .order_by(Evaluation.created_at.asc())
        )
        result = await db.execute(stmt)
        scores = list(result.scalars().all())

        avg_score = round(sum(scores) / len(scores), 1) if scores else None
        best_score = round(max(scores), 1) if scores else None
        trend = _compute_improvement_trend(scores)

        if avg_score is not None and avg_score >= PASSING_THRESHOLD:
            agents_passed += 1
        else:
            agents_needing_improvement += 1

        all_scores.extend(scores)

        agent_summaries.append(
            AgentSummary(
                agent_id=str(aid),
                agent_name=agent_names.get(aid, "Unknown"),
                sessions_completed=len(scores),
                average_score=avg_score,
                best_score=best_score,
                improvement_trend=trend,
            )
        )

    # Overall average
    overall_avg = round(sum(all_scores) / len(all_scores), 1) if all_scores else None

    # Score history (optionally filtered by agent)
    score_history = await _get_score_history(
        db, agent_ids, scenario_ids, agent_id_filter
    )

    # Category averages
    category_averages = await _get_category_averages(db, agent_ids, scenario_ids)

    return CampaignDashboardResponse(
        total_agents=len(agent_ids),
        average_score=overall_avg,
        agents_passed=agents_passed,
        agents_needing_improvement=agents_needing_improvement,
        agent_summaries=agent_summaries,
        score_history=score_history,
        category_averages=category_averages,
    )


async def get_agent_progress(
    db: AsyncSession,
    campaign_id: UUID,
    agent_id: UUID,
) -> AgentProgressResponse:
    """Get detailed progress for a single agent within a campaign.

    Args:
        db: Database session.
        campaign_id: The campaign context.
        agent_id: The agent to get progress for.

    Returns:
        AgentProgressResponse with score history, sessions, and scenario perf.

    Raises:
        ValueError: If campaign not found or agent not in campaign.
    """
    campaign = await _get_campaign_or_none(db, campaign_id)
    if campaign is None:
        raise ValueError("Campaign not found")

    agent_ids = await _get_campaign_agent_ids(db, campaign_id)
    if agent_id not in agent_ids:
        raise ValueError("Agent not found in this campaign")

    scenario_ids = await _get_campaign_scenario_ids(db, campaign_id)
    agent_names = await _get_agent_names(db, [agent_id])
    agent_name = agent_names.get(agent_id, "Unknown")

    # Score history for this agent
    stmt = (
        select(Evaluation.overall_score, Evaluation.created_at)
        .join(Session, Session.id == Evaluation.session_id)
        .where(
            Session.agent_id == agent_id,
            Session.scenario_id.in_(scenario_ids),
            Evaluation.is_too_short == False,  # noqa: E712
        )
        .order_by(Evaluation.created_at.asc())
    )
    result = await db.execute(stmt)
    score_rows = result.all()
    scores = [row[0] for row in score_rows]

    avg_score = round(sum(scores) / len(scores), 1) if scores else None
    trend = _compute_improvement_trend(scores)

    score_history = [
        ScoreDataPoint(
            session_number=i + 1,
            overall_score=round(row[0], 1),
            date=row[1].isoformat() if row[1] else "",
            agent_id=str(agent_id),
        )
        for i, row in enumerate(score_rows)
    ]

    # Session history (ordered by created_at desc)
    session_stmt = (
        select(Session, Evaluation, Scenario)
        .outerjoin(Evaluation, Evaluation.session_id == Session.id)
        .outerjoin(Scenario, Scenario.id == Session.scenario_id)
        .where(
            Session.agent_id == agent_id,
            Session.scenario_id.in_(scenario_ids),
        )
        .order_by(Session.created_at.desc())
    )
    session_result = await db.execute(session_stmt)
    session_rows = session_result.all()

    session_history = [
        AgentSessionItem(
            session_id=str(sess.id),
            scenario_name=scenario.name if scenario else "Unknown",
            date=sess.created_at.isoformat() if sess.created_at else "",
            overall_score=(
                round(eval_.overall_score, 1)
                if eval_ and not eval_.is_too_short
                else None
            ),
            status=sess.status,
        )
        for sess, eval_, scenario in session_rows
    ]

    # Per-scenario performance
    scenario_perf: dict[UUID, dict] = {}
    for sess, eval_, scenario in session_rows:
        if eval_ and not eval_.is_too_short and scenario:
            sid = scenario.id
            if sid not in scenario_perf:
                scenario_perf[sid] = {
                    "scenario_id": str(sid),
                    "scenario_name": scenario.name,
                    "scores": [],
                }
            scenario_perf[sid]["scores"].append(eval_.overall_score)

    scenario_performance = [
        ScenarioAverage(
            scenario_id=data["scenario_id"],
            scenario_name=data["scenario_name"],
            sessions_count=len(data["scores"]),
            average_score=round(sum(data["scores"]) / len(data["scores"]), 1),
        )
        for data in scenario_perf.values()
    ]

    return AgentProgressResponse(
        agent_id=str(agent_id),
        agent_name=agent_name,
        average_score=avg_score,
        sessions_completed=len(scores),
        improvement_trend=trend,
        score_history=score_history,
        session_history=session_history,
        scenario_performance=scenario_performance,
    )

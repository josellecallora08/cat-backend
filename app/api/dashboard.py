"""Dashboard API endpoint for aggregated training analytics."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import Session, Evaluation, Scenario, Transcript

logger = logging.getLogger(__name__)

router = APIRouter()


class CategoryAverage(BaseModel):
    category: str
    average_score: float


class RecentSession(BaseModel):
    id: str
    scenario_name: str
    persona_name: str
    status: str
    overall_score: Optional[float] = None
    created_at: str


class DashboardStats(BaseModel):
    """Aggregated dashboard statistics."""
    total_sessions: int
    completed_sessions: int
    active_sessions: int
    total_scenarios: int
    average_overall_score: Optional[float] = None
    category_averages: list[CategoryAverage]
    recent_sessions: list[RecentSession]
    total_conversations: int  # total transcript entries
    improvement_trend: Optional[float] = None  # score change over last 5 sessions


@router.get("/dashboard", response_model=DashboardStats)
async def get_dashboard(db: AsyncSession = Depends(get_session)):
    """Get aggregated dashboard statistics for the training platform.

    Returns:
    - Total/completed/active session counts
    - Average scores across all evaluations
    - Per-category score averages
    - Recent session history with scores
    - Total conversation count
    - Improvement trend (last 5 vs previous 5 sessions)
    """
    # Total sessions by status
    total_result = await db.execute(select(func.count()).select_from(Session))
    total_sessions = total_result.scalar_one()

    completed_result = await db.execute(
        select(func.count()).select_from(Session).where(Session.status == "completed")
    )
    completed_sessions = completed_result.scalar_one()

    active_result = await db.execute(
        select(func.count()).select_from(Session).where(Session.status.in_(["pending", "active"]))
    )
    active_sessions = active_result.scalar_one()

    # Total scenarios
    scenario_result = await db.execute(
        select(func.count()).select_from(Scenario).where(Scenario.is_active == True)
    )
    total_scenarios = scenario_result.scalar_one()

    # Average overall score
    avg_score_result = await db.execute(
        select(func.avg(Evaluation.overall_score)).where(Evaluation.is_too_short == False)
    )
    average_overall_score = avg_score_result.scalar_one()
    if average_overall_score is not None:
        average_overall_score = round(average_overall_score, 1)

    # Per-category averages from evaluations
    category_averages = []
    eval_result = await db.execute(
        select(Evaluation.category_scores).where(Evaluation.is_too_short == False)
    )
    all_category_scores = eval_result.scalars().all()

    if all_category_scores:
        category_totals: dict[str, list[float]] = {}
        for scores_list in all_category_scores:
            if isinstance(scores_list, list):
                for score_item in scores_list:
                    if isinstance(score_item, dict) and "category" in score_item and "score" in score_item:
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

        for cat, scores in category_totals.items():
            if scores:
                category_averages.append(CategoryAverage(
                    category=category_labels.get(cat, cat),
                    average_score=round(sum(scores) / len(scores), 1),
                ))

    # Total transcript entries (conversations)
    transcript_result = await db.execute(select(func.count()).select_from(Transcript))
    total_conversations = transcript_result.scalar_one()

    # Recent sessions with scores
    recent_sessions: list[RecentSession] = []
    stmt = (
        select(Session, Evaluation, Scenario)
        .outerjoin(Evaluation, Evaluation.session_id == Session.id)
        .outerjoin(Scenario, Scenario.id == Session.scenario_id)
        .order_by(desc(Session.created_at))
        .limit(10)
    )
    result = await db.execute(stmt)
    rows = result.all()

    for session, evaluation, scenario in rows:
        persona_ctx = session.persona_context or {}
        recent_sessions.append(RecentSession(
            id=str(session.id),
            scenario_name=scenario.name if scenario else "Unknown",
            persona_name=persona_ctx.get("name", "Unknown"),
            status=session.status,
            overall_score=round(evaluation.overall_score, 1) if evaluation and not evaluation.is_too_short else None,
            created_at=session.created_at.isoformat() if session.created_at else "",
        ))

    # Improvement trend: compare average of last 5 scored sessions vs previous 5
    improvement_trend = None
    scored_evals_stmt = (
        select(Evaluation.overall_score)
        .where(Evaluation.is_too_short == False)
        .order_by(desc(Evaluation.created_at))
        .limit(10)
    )
    scored_result = await db.execute(scored_evals_stmt)
    scored_list = [row for row in scored_result.scalars().all()]

    if len(scored_list) >= 6:
        recent_avg = sum(scored_list[:5]) / 5
        previous_avg = sum(scored_list[5:10]) / len(scored_list[5:10])
        improvement_trend = round(recent_avg - previous_avg, 1)

    return DashboardStats(
        total_sessions=total_sessions,
        completed_sessions=completed_sessions,
        active_sessions=active_sessions,
        total_scenarios=total_scenarios,
        average_overall_score=average_overall_score,
        category_averages=category_averages,
        recent_sessions=recent_sessions,
        total_conversations=total_conversations,
        improvement_trend=improvement_trend,
    )


class SessionListItem(BaseModel):
    id: str
    scenario_name: str
    persona_name: str
    status: str
    overall_score: Optional[float] = None
    created_at: str


@router.get("/sessions/list", response_model=list[SessionListItem])
async def list_all_sessions(db: AsyncSession = Depends(get_session)):
    """List all sessions with scenario and evaluation info.

    This replaces the localStorage-based session tracking on the frontend.
    """
    stmt = (
        select(Session, Evaluation, Scenario)
        .outerjoin(Evaluation, Evaluation.session_id == Session.id)
        .outerjoin(Scenario, Scenario.id == Session.scenario_id)
        .order_by(desc(Session.created_at))
        .limit(50)
    )
    result = await db.execute(stmt)
    rows = result.all()

    sessions = []
    for session, evaluation, scenario in rows:
        persona_ctx = session.persona_context or {}
        sessions.append(SessionListItem(
            id=str(session.id),
            scenario_name=scenario.name if scenario else "Unknown",
            persona_name=persona_ctx.get("name", "Unknown"),
            status=session.status,
            overall_score=round(evaluation.overall_score, 1) if evaluation and not evaluation.is_too_short else None,
            created_at=session.created_at.isoformat() if session.created_at else "",
        ))

    return sessions


class ScoreDataPoint(BaseModel):
    """A single data point for score progression charts."""
    session_number: int
    overall_score: float
    call_opening: Optional[float] = None
    compliance: Optional[float] = None
    empathy_communication: Optional[float] = None
    negotiation_resolution: Optional[float] = None
    date: str


@router.get("/dashboard/score-history", response_model=list[ScoreDataPoint])
async def get_score_history(db: AsyncSession = Depends(get_session)):
    """Get score progression over time for line/area charts.

    Returns chronological list of scored sessions with per-category breakdowns.
    Excludes too-short sessions.
    """
    stmt = (
        select(Evaluation)
        .where(Evaluation.is_too_short == False)
        .order_by(Evaluation.created_at.asc())
        .limit(50)
    )
    result = await db.execute(stmt)
    evaluations = result.scalars().all()

    data_points = []
    for i, eval in enumerate(evaluations):
        # Extract per-category scores
        cat_scores: dict[str, float] = {}
        if isinstance(eval.category_scores, list):
            for cs in eval.category_scores:
                if isinstance(cs, dict) and "category" in cs and "score" in cs:
                    cat_scores[cs["category"]] = float(cs["score"])

        data_points.append(ScoreDataPoint(
            session_number=i + 1,
            overall_score=round(eval.overall_score, 1),
            call_opening=cat_scores.get("call_opening"),
            compliance=cat_scores.get("compliance"),
            empathy_communication=cat_scores.get("empathy_communication"),
            negotiation_resolution=cat_scores.get("negotiation_resolution"),
            date=eval.created_at.isoformat() if eval.created_at else "",
        ))

    return data_points


class ScenarioPerformance(BaseModel):
    """Performance stats per scenario type for comparison charts."""
    scenario_type: str
    scenario_name: str
    sessions_count: int
    average_score: float
    best_score: float
    worst_score: float


@router.get("/dashboard/scenario-performance", response_model=list[ScenarioPerformance])
async def get_scenario_performance(db: AsyncSession = Depends(get_session)):
    """Get performance breakdown by scenario type for bar/comparison charts.

    Shows average, best, and worst scores per scenario.
    """
    stmt = (
        select(Scenario.name, Scenario.scenario_type, Evaluation.overall_score)
        .join(Session, Session.scenario_id == Scenario.id)
        .join(Evaluation, Evaluation.session_id == Session.id)
        .where(Evaluation.is_too_short == False)
    )
    result = await db.execute(stmt)
    rows = result.all()

    # Group by scenario
    scenario_data: dict[str, dict] = {}
    for name, s_type, score in rows:
        key = f"{s_type}|{name}"
        if key not in scenario_data:
            scenario_data[key] = {
                "scenario_type": s_type,
                "scenario_name": name,
                "scores": [],
            }
        scenario_data[key]["scores"].append(score)

    performances = []
    for data in scenario_data.values():
        scores = data["scores"]
        performances.append(ScenarioPerformance(
            scenario_type=data["scenario_type"],
            scenario_name=data["scenario_name"],
            sessions_count=len(scores),
            average_score=round(sum(scores) / len(scores), 1),
            best_score=round(max(scores), 1),
            worst_score=round(min(scores), 1),
        ))

    return sorted(performances, key=lambda p: p.average_score, reverse=True)


class AgentRanking(BaseModel):
    """Agent ranking entry for the leaderboard."""
    rank: int
    agent_id: str
    agent_name: str
    sessions_completed: int
    average_score: float
    best_score: float
    improvement: Optional[float] = None  # Score change over last sessions


# In-memory agent name registry (for MVP demo without auth)
_agent_names: dict[str, str] = {}


@router.post("/agents/register")
async def register_agent_name(
    body: dict,
    db: AsyncSession = Depends(get_session),
):
    """Register a display name for an agent_id (MVP demo, no auth)."""
    agent_id = body.get("agent_id", "")
    name = body.get("name", "")
    if agent_id and name:
        _agent_names[agent_id] = name
    return {"status": "ok"}


@router.get("/dashboard/leaderboard", response_model=list[AgentRanking])
async def get_leaderboard(db: AsyncSession = Depends(get_session)):
    """Get agent rankings sorted by average score.

    Aggregates evaluation scores per agent_id across all completed sessions.
    Returns top performers with session count, average/best scores, and trend.
    """
    # Get all evaluations joined with sessions for agent_id
    stmt = (
        select(Session.agent_id, Evaluation.overall_score)
        .join(Evaluation, Evaluation.session_id == Session.id)
        .where(Evaluation.is_too_short == False)
        .order_by(Session.agent_id, Evaluation.created_at.asc())
    )
    result = await db.execute(stmt)
    rows = result.all()

    if not rows:
        return []

    # Group by agent_id
    agent_data: dict[str, list[float]] = {}
    for agent_id, score in rows:
        aid = str(agent_id)
        if aid not in agent_data:
            agent_data[aid] = []
        agent_data[aid].append(score)

    # Build rankings
    rankings = []
    for agent_id, scores in agent_data.items():
        avg = sum(scores) / len(scores)
        best = max(scores)

        # Calculate improvement (last 3 vs first 3)
        improvement = None
        if len(scores) >= 4:
            recent = scores[-3:]
            earlier = scores[:3]
            improvement = round(sum(recent) / len(recent) - sum(earlier) / len(earlier), 1)

        # Get display name
        name = _agent_names.get(agent_id, f"Agent {agent_id[:8]}")

        rankings.append(AgentRanking(
            rank=0,  # Will be set after sorting
            agent_id=agent_id,
            agent_name=name,
            sessions_completed=len(scores),
            average_score=round(avg, 1),
            best_score=round(best, 1),
            improvement=improvement,
        ))

    # Sort by average score descending
    rankings.sort(key=lambda r: r.average_score, reverse=True)

    # Assign ranks
    for i, r in enumerate(rankings):
        r.rank = i + 1

    return rankings[:20]  # Top 20

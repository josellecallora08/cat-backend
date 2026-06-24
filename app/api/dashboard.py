"""Dashboard API endpoint for aggregated training analytics."""

import logging
from typing import Optional
from uuid import UUID as PyUUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import Session, Evaluation, Scenario, Transcript
from app.models.user import User

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
async def get_dashboard(
    agent_id: str | None = None,
    db: AsyncSession = Depends(get_session),
):
    """Get aggregated dashboard statistics for the training platform.

    Optional filter:
      - agent_id: When provided, only shows stats for that specific agent's sessions.

    Returns:
    - Total/completed/active session counts
    - Average scores across all evaluations
    - Per-category score averages
    - Recent session history with scores
    - Total conversation count
    - Improvement trend (last 5 vs previous 5 sessions)
    """
    # Build base session filter
    session_filter = []
    if agent_id:
        session_filter.append(Session.agent_id == PyUUID(agent_id))

    # Total sessions by status
    count_q = select(func.count()).select_from(Session)
    for f in session_filter:
        count_q = count_q.where(f)
    total_result = await db.execute(count_q)
    total_sessions = total_result.scalar_one()

    completed_q = select(func.count()).select_from(Session).where(Session.status == "completed")
    for f in session_filter:
        completed_q = completed_q.where(f)
    completed_result = await db.execute(completed_q)
    completed_sessions = completed_result.scalar_one()

    active_q = select(func.count()).select_from(Session).where(Session.status.in_(["pending", "active"]))
    for f in session_filter:
        active_q = active_q.where(f)
    active_result = await db.execute(active_q)
    active_sessions = active_result.scalar_one()

    # Total scenarios (not filtered by agent)
    scenario_result = await db.execute(
        select(func.count()).select_from(Scenario).where(Scenario.is_active == True)
    )
    total_scenarios = scenario_result.scalar_one()

    # Average overall score (filtered by agent if provided)
    avg_q = select(func.avg(Evaluation.overall_score)).where(Evaluation.is_too_short == False)
    if agent_id:
        avg_q = avg_q.join(Session, Session.id == Evaluation.session_id).where(Session.agent_id == PyUUID(agent_id))
    avg_score_result = await db.execute(avg_q)
    average_overall_score = avg_score_result.scalar_one()
    if average_overall_score is not None:
        average_overall_score = round(average_overall_score, 1)

    # Per-category averages from evaluations
    category_averages = []
    eval_q = select(Evaluation.category_scores).where(Evaluation.is_too_short == False)
    if agent_id:
        eval_q = eval_q.join(Session, Session.id == Evaluation.session_id).where(Session.agent_id == PyUUID(agent_id))
    eval_result = await db.execute(eval_q)
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

    # Total transcript entries (conversations) - filtered by agent if provided
    if agent_id:
        transcript_q = (
            select(func.count())
            .select_from(Transcript)
            .join(Session, Session.id == Transcript.session_id)
            .where(Session.agent_id == PyUUID(agent_id))
        )
    else:
        transcript_q = select(func.count()).select_from(Transcript)
    transcript_result = await db.execute(transcript_q)
    total_conversations = transcript_result.scalar_one()

    # Recent sessions with scores
    recent_sessions: list[RecentSession] = []
    recent_stmt = (
        select(Session, Evaluation, Scenario)
        .outerjoin(Evaluation, Evaluation.session_id == Session.id)
        .outerjoin(Scenario, Scenario.id == Session.scenario_id)
    )
    if agent_id:
        recent_stmt = recent_stmt.where(Session.agent_id == PyUUID(agent_id))
    recent_stmt = recent_stmt.order_by(desc(Session.created_at)).limit(10)
    result = await db.execute(recent_stmt)
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
    trend_stmt = (
        select(Evaluation.overall_score)
        .where(Evaluation.is_too_short == False)
    )
    if agent_id:
        trend_stmt = trend_stmt.join(Session, Session.id == Evaluation.session_id).where(Session.agent_id == PyUUID(agent_id))
    trend_stmt = trend_stmt.order_by(desc(Evaluation.created_at)).limit(10)
    scored_result = await db.execute(trend_stmt)
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
    agent_name: str
    agent_email: str
    status: str
    overall_score: Optional[float] = None
    created_at: str


class PaginatedSessions(BaseModel):
    items: list[SessionListItem]
    total: int
    page: int
    page_size: int
    total_pages: int


@router.get("/dashboard/sessions", response_model=PaginatedSessions)
async def list_all_sessions(
    page: int = 1,
    page_size: int = 20,
    agent_id: str | None = None,
    status: str | None = None,
    sort_by: str = "created_at",
    sort_dir: str = "desc",
    db: AsyncSession = Depends(get_session),
):
    """List all sessions with scenario, evaluation, and agent info (paginated).

    Optional filters:
      - agent_id: Filter by specific agent UUID
      - status: Filter by session status (pending, active, completed)

    Sorting:
      - sort_by: one of "created_at", "score", "scenario", "status"
      - sort_dir: "asc" or "desc"
    """
    # Clamp values
    page = max(1, page)
    page_size = max(1, min(100, page_size))
    offset = (page - 1) * page_size

    # Build filter conditions
    conditions = []
    if agent_id:
        conditions.append(Session.agent_id == PyUUID(agent_id))
    if status:
        conditions.append(Session.status == status)

    # Count total
    count_stmt = select(func.count()).select_from(Session)
    for cond in conditions:
        count_stmt = count_stmt.where(cond)
    total = (await db.execute(count_stmt)).scalar_one()

    # Fetch page
    stmt = (
        select(Session, Evaluation, Scenario, User)
        .outerjoin(Evaluation, Evaluation.session_id == Session.id)
        .outerjoin(Scenario, Scenario.id == Session.scenario_id)
        .outerjoin(User, User.id == Session.agent_id)
    )
    for cond in conditions:
        stmt = stmt.where(cond)

    # Resolve sort column from a whitelist (prevents injection / invalid columns)
    sort_columns = {
        "created_at": Session.created_at,
        "score": Evaluation.overall_score,
        "scenario": Scenario.name,
        "status": Session.status,
    }
    sort_col = sort_columns.get(sort_by, Session.created_at)
    direction = asc if sort_dir == "asc" else desc
    # NULLS LAST so unscored sessions don't dominate score sorts; stable tiebreak by created_at
    stmt = stmt.order_by(
        direction(sort_col).nulls_last(),
        desc(Session.created_at),
    ).offset(offset).limit(page_size)

    result = await db.execute(stmt)
    rows = result.all()

    items = []
    for session, evaluation, scenario, user in rows:
        persona_ctx = session.persona_context or {}
        items.append(SessionListItem(
            id=str(session.id),
            scenario_name=scenario.name if scenario else "Unknown",
            persona_name=persona_ctx.get("name", "Unknown"),
            agent_name=user.full_name if user else "Unknown Agent",
            agent_email=user.email if user else "",
            status=session.status,
            overall_score=round(evaluation.overall_score, 1) if evaluation and not evaluation.is_too_short else None,
            created_at=session.created_at.isoformat() if session.created_at else "",
        ))

    total_pages = max(1, -(-total // page_size))  # ceiling division

    return PaginatedSessions(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


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
async def get_score_history(
    agent_id: str | None = None,
    db: AsyncSession = Depends(get_session),
):
    """Get score progression over time for line/area charts.

    Optional filter:
      - agent_id: Only show scores for this agent's sessions.

    Returns chronological list of scored sessions with per-category breakdowns.
    Excludes too-short sessions.
    """
    stmt = (
        select(Evaluation)
        .where(Evaluation.is_too_short == False)
    )
    if agent_id:
        stmt = stmt.join(Session, Session.id == Evaluation.session_id).where(Session.agent_id == PyUUID(agent_id))
    stmt = stmt.order_by(Evaluation.created_at.asc()).limit(50)

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


class AgentListItem(BaseModel):
    """Simple agent info for filter dropdowns."""
    id: str
    full_name: str
    email: str


@router.get("/dashboard/agents", response_model=list[AgentListItem])
async def list_agents(db: AsyncSession = Depends(get_session)):
    """List all agents for filter dropdowns."""
    stmt = select(User).where(User.is_active == True).order_by(User.full_name)
    result = await db.execute(stmt)
    users = result.scalars().all()
    return [
        AgentListItem(id=str(u.id), full_name=u.full_name, email=u.email)
        for u in users
    ]


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

    # Fetch user names for all agent_ids
    agent_ids = list(agent_data.keys())
    user_stmt = select(User.id, User.full_name).where(User.id.in_(agent_ids))
    user_result = await db.execute(user_stmt)
    user_names = {str(uid): name for uid, name in user_result.all()}

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

        # Get display name from User table, fallback to in-memory registry
        name = user_names.get(agent_id) or _agent_names.get(agent_id, f"Agent {agent_id[:8]}")

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

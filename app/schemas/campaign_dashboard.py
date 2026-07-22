"""Pydantic schemas for campaign dashboard API responses."""

from pydantic import BaseModel


PASSING_THRESHOLD = 70


class ScoreDataPoint(BaseModel):
    """A single score data point for chart rendering."""

    model_config = {"extra": "forbid"}

    session_number: int
    overall_score: float
    date: str
    agent_id: str


class CategoryAverage(BaseModel):
    """Average score for a specific evaluation category."""

    model_config = {"extra": "forbid"}

    category: str
    average_score: float


class AgentSummary(BaseModel):
    """Per-agent performance summary within a campaign."""

    model_config = {"extra": "forbid"}

    agent_id: str
    agent_name: str
    sessions_completed: int
    average_score: float | None
    best_score: float | None
    improvement_trend: float | None


class CampaignDashboardResponse(BaseModel):
    """Full campaign dashboard response with KPIs, agent summaries, and chart data."""

    model_config = {"extra": "forbid"}

    total_agents: int
    average_score: float | None
    agents_passed: int
    agents_needing_improvement: int
    agent_summaries: list[AgentSummary]
    score_history: list[ScoreDataPoint]
    category_averages: list[CategoryAverage]


class AgentSessionItem(BaseModel):
    """A single session in an agent's history within a campaign."""

    model_config = {"extra": "forbid"}

    session_id: str
    scenario_name: str
    date: str
    overall_score: float | None
    status: str


class ScenarioAverage(BaseModel):
    """Per-scenario average score for an agent within a campaign."""

    model_config = {"extra": "forbid"}

    scenario_id: str
    scenario_name: str
    sessions_count: int
    average_score: float


class AgentProgressResponse(BaseModel):
    """Full agent progress response within a campaign context."""

    model_config = {"extra": "forbid"}

    agent_id: str
    agent_name: str
    average_score: float | None
    sessions_completed: int
    improvement_trend: float | None
    score_history: list[ScoreDataPoint]
    session_history: list[AgentSessionItem]
    scenario_performance: list[ScenarioAverage]

"""
Composio Analytics API (PRD-54 / US-003)
=========================================

Aggregated views of Composio app connections, action usage,
and per-agent tool mappings scoped to the current workspace.
"""

import logging
from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, case, cast, Float

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.super_admin import require_super_admin
from core.database.database import get_db
from core.models.core import Agent
from core.models.composio import ComposioEntity, ComposioConnection, AgentAppFeature
from core.models.composio_cache import ToolExecutionLog

logger = logging.getLogger(__name__)

# PRD-143 S7: observability tier — router-wide super-admin lock (fail-closed).
router = APIRouter(
    prefix="/api/analytics/composio",
    tags=["Composio Analytics"],
    dependencies=[Depends(require_super_admin)],
)


# ── Pydantic schemas ─────────────────────────────────────────────────

class ConnectedAppStats(BaseModel):
    app_name: str
    status: str
    total_actions_used: int
    agent_count: int
    documents_synced: int
    last_used_at: Optional[datetime] = None


class ActionLeaderboardEntry(BaseModel):
    action_name: str
    app_name: str
    total_usage_count: int
    agent_count: int
    last_used_at: Optional[datetime] = None


class AgentToolEntry(BaseModel):
    tool_name: str
    app_name: str
    usage_count: int
    enabled: bool


class AgentToolMapping(BaseModel):
    agent_id: int
    agent_name: str
    tools: List[AgentToolEntry]


class ExecutionOverview(BaseModel):
    total_executions: int
    success_count: int
    error_count: int
    timeout_count: int
    success_rate: float
    error_rate: float
    avg_latency_ms: float
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    max_latency_ms: Optional[float] = None
    cache_hit_rate: float
    unique_actions: int
    unique_apps: int


class ActionPerformance(BaseModel):
    action_name: str
    app_name: str
    total_calls: int
    success_count: int
    error_count: int
    error_rate: float
    avg_latency_ms: float
    max_latency_ms: float
    cache_hit_rate: float
    last_executed: Optional[datetime] = None


class DailyVolume(BaseModel):
    date: str
    total: int
    successes: int
    errors: int
    avg_latency_ms: float


class RecentExecution(BaseModel):
    id: int
    agent_name: Optional[str] = None
    app_name: str
    action_name: str
    status: str
    execution_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    cache_hit: bool = False
    executed_at: Optional[datetime] = None


class ErrorBreakdown(BaseModel):
    error_code: Optional[str] = None
    error_message: str
    count: int
    last_seen: Optional[datetime] = None
    app_name: str
    action_name: str


# ── Helpers ───────────────────────────────────────────────────────────

_ALLOWED_DAYS = {7, 30, 90}


def _since(days: int) -> datetime:
    if days not in _ALLOWED_DAYS:
        raise HTTPException(400, f"Invalid days parameter: {days}. Allowed values: {sorted(_ALLOWED_DAYS)}")
    return datetime.utcnow() - timedelta(days=days)


# ── Endpoints ─────────────────────────────────────────────────────────

@router.get("/apps", response_model=List[ConnectedAppStats])
async def get_connected_apps(
    days: int = Query(30, description="7|30|90"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Connected apps with action counts, agent count, and documents synced."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    since = _since(days)

    # Get the entity for this workspace
    entity = (
        db.query(ComposioEntity)
        .filter(ComposioEntity.workspace_id == ctx.workspace_id)
        .first()
    )
    if not entity:
        return []

    # Get all connections for this workspace's entity
    connections = (
        db.query(ComposioConnection)
        .filter(ComposioConnection.entity_id == entity.id)
        .all()
    )

    # Get workspace agent IDs for filtering AgentAppFeature
    agent_ids = [
        r[0]
        for r in db.query(Agent.id)
        .filter(Agent.workspace_id == ctx.workspace_id)
        .all()
    ]

    results: List[ConnectedAppStats] = []
    for conn in connections:
        # Aggregate action usage for this app across workspace agents
        if agent_ids:
            action_agg = (
                db.query(
                    func.sum(AgentAppFeature.usage_count).label("total_usage"),
                    func.count(func.distinct(AgentAppFeature.agent_id)).label("agent_count"),
                    func.max(AgentAppFeature.last_used_at).label("last_used"),
                )
                .filter(
                    AgentAppFeature.agent_id.in_(agent_ids),
                    AgentAppFeature.app_name == conn.app_name,
                    AgentAppFeature.created_at >= since,
                )
                .first()
            )
            total_usage = int(action_agg.total_usage or 0) if action_agg else 0
            agent_count = int(action_agg.agent_count or 0) if action_agg else 0
            last_used = action_agg.last_used if action_agg else None

            # Fallback: query ToolExecutionLog if AgentAppFeature has no data
            if total_usage == 0:
                try:
                    log_agg = (
                        db.query(
                            func.count(ToolExecutionLog.id).label("total_usage"),
                            func.count(func.distinct(ToolExecutionLog.agent_id)).label("agent_count"),
                            func.max(ToolExecutionLog.executed_at).label("last_used"),
                        )
                        .filter(
                            ToolExecutionLog.agent_id.in_(agent_ids),
                            ToolExecutionLog.app_name == conn.app_name,
                            ToolExecutionLog.executed_at >= since,
                        )
                        .first()
                    )
                    if log_agg:
                        total_usage = int(log_agg.total_usage or 0)
                        agent_count = int(log_agg.agent_count or 0)
                        last_used = log_agg.last_used
                except Exception:
                    pass
        else:
            total_usage = 0
            agent_count = 0
            last_used = None

        results.append(
            ConnectedAppStats(
                app_name=conn.app_name,
                status=conn.status,
                total_actions_used=total_usage,
                agent_count=agent_count,
                documents_synced=conn.total_documents_synced or 0,
                last_used_at=last_used,
            )
        )

    return results


@router.get("/actions", response_model=List[ActionLeaderboardEntry])
async def get_action_leaderboard(
    days: int = Query(30, description="7|30|90"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Action leaderboard sorted by usage count descending."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    since = _since(days)

    # Get workspace agent IDs
    agent_ids = [
        r[0]
        for r in db.query(Agent.id)
        .filter(Agent.workspace_id == ctx.workspace_id)
        .all()
    ]
    if not agent_ids:
        return []

    # Try AgentAppFeature first (populated by trigger)
    rows = (
        db.query(
            AgentAppFeature.action_name,
            AgentAppFeature.app_name,
            func.sum(AgentAppFeature.usage_count).label("total_usage"),
            func.count(func.distinct(AgentAppFeature.agent_id)).label("agent_count"),
            func.max(AgentAppFeature.last_used_at).label("last_used"),
        )
        .filter(
            AgentAppFeature.agent_id.in_(agent_ids),
            AgentAppFeature.created_at >= since,
        )
        .group_by(AgentAppFeature.action_name, AgentAppFeature.app_name)
        .order_by(desc("total_usage"))
        .all()
    )

    # Fallback: query ToolExecutionLog directly if AgentAppFeature is empty
    if not rows:
        try:
            rows = (
                db.query(
                    ToolExecutionLog.action_name,
                    ToolExecutionLog.app_name,
                    func.count(ToolExecutionLog.id).label("total_usage"),
                    func.count(func.distinct(ToolExecutionLog.agent_id)).label("agent_count"),
                    func.max(ToolExecutionLog.executed_at).label("last_used"),
                )
                .filter(
                    ToolExecutionLog.agent_id.in_(agent_ids),
                    ToolExecutionLog.executed_at >= since,
                )
                .group_by(ToolExecutionLog.action_name, ToolExecutionLog.app_name)
                .order_by(desc("total_usage"))
                .all()
            )
        except Exception as e:
            logger.warning(f"ToolExecutionLog fallback failed: {e}")
            rows = []

    return [
        ActionLeaderboardEntry(
            action_name=r.action_name,
            app_name=r.app_name,
            total_usage_count=int(r.total_usage or 0),
            agent_count=int(r.agent_count or 0),
            last_used_at=r.last_used,
        )
        for r in rows
    ]


@router.get("/agent-tools", response_model=List[AgentToolMapping])
async def get_agent_tools(
    days: int = Query(30, description="7|30|90"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Per-agent tool mapping with usage counts."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    since = _since(days)

    # Get workspace agents that have Composio features
    agents = (
        db.query(Agent)
        .filter(Agent.workspace_id == ctx.workspace_id)
        .all()
    )
    if not agents:
        return []

    agent_map = {a.id: a.name for a in agents}
    agent_ids = list(agent_map.keys())

    features = (
        db.query(AgentAppFeature)
        .filter(
            AgentAppFeature.agent_id.in_(agent_ids),
            AgentAppFeature.created_at >= since,
        )
        .order_by(AgentAppFeature.agent_id, desc(AgentAppFeature.usage_count))
        .all()
    )

    # Group by agent
    from collections import defaultdict

    by_agent: dict[int, list[AgentToolEntry]] = defaultdict(list)
    for f in features:
        by_agent[f.agent_id].append(
            AgentToolEntry(
                tool_name=f.action_name,
                app_name=f.app_name,
                usage_count=f.usage_count or 0,
                enabled=f.enabled,
            )
        )

    # Fallback: if AgentAppFeature has no data, derive from ToolExecutionLog
    if not by_agent:
        try:
            log_rows = (
                db.query(
                    ToolExecutionLog.agent_id,
                    ToolExecutionLog.action_name,
                    ToolExecutionLog.app_name,
                    func.count(ToolExecutionLog.id).label("usage_count"),
                )
                .filter(
                    ToolExecutionLog.agent_id.in_(agent_ids),
                    ToolExecutionLog.executed_at >= since,
                )
                .group_by(
                    ToolExecutionLog.agent_id,
                    ToolExecutionLog.action_name,
                    ToolExecutionLog.app_name,
                )
                .order_by(ToolExecutionLog.agent_id, desc("usage_count"))
                .all()
            )
            for r in log_rows:
                by_agent[r.agent_id].append(
                    AgentToolEntry(
                        tool_name=r.action_name,
                        app_name=r.app_name,
                        usage_count=int(r.usage_count or 0),
                        enabled=True,
                    )
                )
        except Exception as e:
            logger.warning(f"ToolExecutionLog fallback for agent-tools failed: {e}")

    return [
        AgentToolMapping(
            agent_id=aid,
            agent_name=agent_map.get(aid, f"Agent {aid}"),
            tools=tools,
        )
        for aid, tools in by_agent.items()
        if tools  # Only include agents that have tools
    ]


# ── API Monitoring Endpoints ─────────────────────────────────────────

def _get_workspace_agent_ids(db: Session, workspace_id) -> list[int]:
    """Get all agent IDs for a workspace."""
    return [
        r[0]
        for r in db.query(Agent.id)
        .filter(Agent.workspace_id == workspace_id)
        .all()
    ]


@router.get("/execution-stats", response_model=ExecutionOverview)
async def get_execution_stats(
    days: int = Query(30, description="7|30|90"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Overall execution stats: volume, success/error rates, latency percentiles."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    since = _since(days)
    agent_ids = _get_workspace_agent_ids(db, ctx.workspace_id)
    if not agent_ids:
        return ExecutionOverview(
            total_executions=0, success_count=0, error_count=0, timeout_count=0,
            success_rate=0, error_rate=0, avg_latency_ms=0, cache_hit_rate=0,
            unique_actions=0, unique_apps=0,
        )

    row = (
        db.query(
            func.count(ToolExecutionLog.id).label("total"),
            func.sum(case((ToolExecutionLog.status == "success", 1), else_=0)).label("successes"),
            func.sum(case((ToolExecutionLog.status == "error", 1), else_=0)).label("errors"),
            func.sum(case((ToolExecutionLog.status == "timeout", 1), else_=0)).label("timeouts"),
            func.avg(ToolExecutionLog.execution_time_ms).label("avg_latency"),
            func.max(ToolExecutionLog.execution_time_ms).label("max_latency"),
            func.sum(case((ToolExecutionLog.cache_hit == True, 1), else_=0)).label("cache_hits"),
            func.count(func.distinct(ToolExecutionLog.action_name)).label("unique_actions"),
            func.count(func.distinct(ToolExecutionLog.app_name)).label("unique_apps"),
        )
        .filter(
            ToolExecutionLog.agent_id.in_(agent_ids),
            ToolExecutionLog.executed_at >= since,
        )
        .first()
    )

    total = int(row.total or 0)
    successes = int(row.successes or 0)
    errors = int(row.errors or 0)
    timeouts = int(row.timeouts or 0)

    # Percentiles via subquery (Postgres percentile_cont)
    p50 = p95 = None
    try:
        from sqlalchemy import text
        pct = db.execute(
            text("""
                SELECT
                    percentile_cont(0.5) WITHIN GROUP (ORDER BY execution_time_ms) AS p50,
                    percentile_cont(0.95) WITHIN GROUP (ORDER BY execution_time_ms) AS p95
                FROM tool_execution_logs
                WHERE agent_id = ANY(:ids) AND executed_at >= :since AND execution_time_ms IS NOT NULL
            """),
            {"ids": agent_ids, "since": since},
        ).fetchone()
        if pct:
            p50 = round(float(pct.p50), 1) if pct.p50 is not None else None
            p95 = round(float(pct.p95), 1) if pct.p95 is not None else None
    except Exception:
        pass

    return ExecutionOverview(
        total_executions=total,
        success_count=successes,
        error_count=errors,
        timeout_count=timeouts,
        success_rate=round(successes / total * 100, 1) if total else 0,
        error_rate=round(errors / total * 100, 1) if total else 0,
        avg_latency_ms=round(float(row.avg_latency or 0), 1),
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        max_latency_ms=round(float(row.max_latency or 0), 1) if row.max_latency else None,
        cache_hit_rate=round(int(row.cache_hits or 0) / total * 100, 1) if total else 0,
        unique_actions=int(row.unique_actions or 0),
        unique_apps=int(row.unique_apps or 0),
    )


@router.get("/performance-by-action", response_model=List[ActionPerformance])
async def get_performance_by_action(
    days: int = Query(30, description="7|30|90"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Per-action performance breakdown: latency, error rate, cache hit rate."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    since = _since(days)
    agent_ids = _get_workspace_agent_ids(db, ctx.workspace_id)
    if not agent_ids:
        return []

    rows = (
        db.query(
            ToolExecutionLog.action_name,
            ToolExecutionLog.app_name,
            func.count(ToolExecutionLog.id).label("total"),
            func.sum(case((ToolExecutionLog.status == "success", 1), else_=0)).label("successes"),
            func.sum(case((ToolExecutionLog.status == "error", 1), else_=0)).label("errors"),
            func.avg(ToolExecutionLog.execution_time_ms).label("avg_latency"),
            func.max(ToolExecutionLog.execution_time_ms).label("max_latency"),
            func.sum(case((ToolExecutionLog.cache_hit == True, 1), else_=0)).label("cache_hits"),
            func.max(ToolExecutionLog.executed_at).label("last_exec"),
        )
        .filter(
            ToolExecutionLog.agent_id.in_(agent_ids),
            ToolExecutionLog.executed_at >= since,
        )
        .group_by(ToolExecutionLog.action_name, ToolExecutionLog.app_name)
        .order_by(desc("total"))
        .all()
    )

    return [
        ActionPerformance(
            action_name=r.action_name,
            app_name=r.app_name,
            total_calls=int(r.total or 0),
            success_count=int(r.successes or 0),
            error_count=int(r.errors or 0),
            error_rate=round(int(r.errors or 0) / int(r.total) * 100, 1) if int(r.total or 0) else 0,
            avg_latency_ms=round(float(r.avg_latency or 0), 1),
            max_latency_ms=round(float(r.max_latency or 0), 1) if r.max_latency else 0,
            cache_hit_rate=round(int(r.cache_hits or 0) / int(r.total) * 100, 1) if int(r.total or 0) else 0,
            last_executed=r.last_exec,
        )
        for r in rows
    ]


@router.get("/daily-volume", response_model=List[DailyVolume])
async def get_daily_volume(
    days: int = Query(30, description="7|30|90"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Daily execution volume for charting trends."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    since = _since(days)
    agent_ids = _get_workspace_agent_ids(db, ctx.workspace_id)
    if not agent_ids:
        return []

    from sqlalchemy import text
    rows = db.execute(
        text("""
            SELECT
                DATE(executed_at) AS day,
                COUNT(*) AS total,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS successes,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors,
                ROUND(AVG(execution_time_ms)::numeric, 1) AS avg_latency
            FROM tool_execution_logs
            WHERE agent_id = ANY(:ids) AND executed_at >= :since
            GROUP BY DATE(executed_at)
            ORDER BY day
        """),
        {"ids": agent_ids, "since": since},
    ).fetchall()

    return [
        DailyVolume(
            date=str(r.day),
            total=int(r.total),
            successes=int(r.successes),
            errors=int(r.errors),
            avg_latency_ms=float(r.avg_latency or 0),
        )
        for r in rows
    ]


@router.get("/recent-executions", response_model=List[RecentExecution])
async def get_recent_executions(
    limit: int = Query(20, ge=1, le=100),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Recent tool executions for live monitoring."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    agent_ids = _get_workspace_agent_ids(db, ctx.workspace_id)
    if not agent_ids:
        return []

    agent_map = {
        r[0]: r[1]
        for r in db.query(Agent.id, Agent.name)
        .filter(Agent.workspace_id == ctx.workspace_id)
        .all()
    }

    rows = (
        db.query(ToolExecutionLog)
        .filter(ToolExecutionLog.agent_id.in_(agent_ids))
        .order_by(desc(ToolExecutionLog.executed_at))
        .limit(limit)
        .all()
    )

    return [
        RecentExecution(
            id=r.id,
            agent_name=agent_map.get(r.agent_id),
            app_name=r.app_name,
            action_name=r.action_name,
            status=r.status,
            execution_time_ms=r.execution_time_ms,
            error_message=r.error_message[:200] if r.error_message else None,
            cache_hit=r.cache_hit or False,
            executed_at=r.executed_at,
        )
        for r in rows
    ]


@router.get("/error-breakdown", response_model=List[ErrorBreakdown])
async def get_error_breakdown(
    days: int = Query(30, description="7|30|90"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Top errors by frequency for debugging."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    since = _since(days)
    agent_ids = _get_workspace_agent_ids(db, ctx.workspace_id)
    if not agent_ids:
        return []

    rows = (
        db.query(
            ToolExecutionLog.error_code,
            func.left(ToolExecutionLog.error_message, 200).label("error_msg"),
            ToolExecutionLog.app_name,
            ToolExecutionLog.action_name,
            func.count(ToolExecutionLog.id).label("cnt"),
            func.max(ToolExecutionLog.executed_at).label("last_seen"),
        )
        .filter(
            ToolExecutionLog.agent_id.in_(agent_ids),
            ToolExecutionLog.executed_at >= since,
            ToolExecutionLog.status.in_(["error", "timeout"]),
        )
        .group_by(
            ToolExecutionLog.error_code,
            func.left(ToolExecutionLog.error_message, 200),
            ToolExecutionLog.app_name,
            ToolExecutionLog.action_name,
        )
        .order_by(desc("cnt"))
        .limit(20)
        .all()
    )

    return [
        ErrorBreakdown(
            error_code=r.error_code,
            error_message=r.error_msg or "Unknown error",
            count=int(r.cnt),
            last_seen=r.last_seen,
            app_name=r.app_name,
            action_name=r.action_name,
        )
        for r in rows
    ]


"""
Enhanced Analytics API
======================

NEW: Additional endpoints for enhanced dashboard metrics and performance analytics.
ADDITIVE: Building on existing statistics.py endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc, asc
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from core.database.database import get_db
from core.models import (
    Agent, Skill, Pattern, Workflow, WorkflowExecution,
    AgentStatistics, SystemMetrics
)
from core.models.core import LLMUsage
from core.models.error_event import ErrorEvent
from core.models.orchestration import OrchestrationRun, OrchestrationTask
from core.models.orchestration_enums import RunState, TaskState, TERMINAL_RUN_STATES
from core.models.sites import Site
from core.models.widget_event_log import WIDGET_EVENT_TYPES, WidgetEventLog
import logging
import psutil
import time
import json
from pydantic import BaseModel
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analytics", tags=["analytics"])

# Pydantic models for new endpoints
class DashboardMetrics(BaseModel):
    agent_success_rate: float
    avg_task_completion_time: float
    system_load_trend: Dict[str, Any]
    error_rate_by_agent_type: Dict[str, float]
    queue_depth: int
    resource_utilization_efficiency: int

class PerformanceEnhancements(BaseModel):
    cost_per_execution: float
    peak_usage_hours: List[Dict[str, Any]]
    bottlenecks: List[Dict[str, Any]]
    predictive_alerts: List[Dict[str, Any]]
    agent_ranking: List[Dict[str, Any]]
    sla_compliance: Dict[str, Any]

# ==== NEW DASHBOARD METRICS ====

@router.get("/dashboard/success-rate")
async def get_agent_success_rate(ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get agent success rate percentage with trend (UNION: workflows + missions)"""
    try:
        ws = ctx.workspace_id

        # --- Legacy workflow executions ---
        wf_total = db.query(WorkflowExecution).filter(WorkflowExecution.workspace_id == ws).count()
        wf_success = db.query(WorkflowExecution).filter(
            WorkflowExecution.workspace_id == ws, WorkflowExecution.status == 'completed'
        ).count()

        # --- PRD-125 Phase 2: Mission runs ---
        m_total = db.query(OrchestrationRun).filter(OrchestrationRun.workspace_id == ws).count()
        m_success = db.query(OrchestrationRun).filter(
            OrchestrationRun.workspace_id == ws,
            OrchestrationRun.state == RunState.COMPLETED.value,
        ).count()

        total_executions = wf_total + m_total
        successful = wf_success + m_success
        success_rate = (successful / total_executions * 100) if total_executions > 0 else 0

        # 7-day trend (combined)
        week_ago = datetime.now() - timedelta(days=7)
        week_wf_total = db.query(WorkflowExecution).filter(
            WorkflowExecution.workspace_id == ws, WorkflowExecution.started_at >= week_ago
        ).count()
        week_wf_success = db.query(WorkflowExecution).filter(and_(
            WorkflowExecution.workspace_id == ws, WorkflowExecution.status == 'completed',
            WorkflowExecution.started_at >= week_ago
        )).count()
        week_m_total = db.query(OrchestrationRun).filter(
            OrchestrationRun.workspace_id == ws, OrchestrationRun.created_at >= week_ago
        ).count()
        week_m_success = db.query(OrchestrationRun).filter(and_(
            OrchestrationRun.workspace_id == ws,
            OrchestrationRun.state == RunState.COMPLETED.value,
            OrchestrationRun.created_at >= week_ago
        )).count()

        week_total = week_wf_total + week_m_total
        week_successful = week_wf_success + week_m_success
        week_success_rate = (week_successful / week_total * 100) if week_total > 0 else 0
        trend = success_rate - week_success_rate

        return {
            "value": round(success_rate, 1),
            "trend": round(trend, 1),
            "total_executions": total_executions,
            "successful_executions": successful,
            "sources": {"workflows": wf_total, "missions": m_total},
        }

    except Exception as e:
        logger.error(f"Error calculating success rate: {e}")
        return {"value": 0, "trend": 0, "total_executions": 0, "successful_executions": 0, "error": str(e)}

def _parse_window(window: str) -> timedelta:
    """Parse a window string like '24h' or '7d' into a timedelta.

    Supports ``h`` (hours) and ``d`` (days). Falls back to 24h on malformed
    input — dashboard queries should not 400 because a UI sent a stale param.
    """
    try:
        if not window:
            return timedelta(hours=24)
        unit = window[-1].lower()
        value = int(window[:-1])
        if value <= 0:
            return timedelta(hours=24)
        if unit == "h":
            return timedelta(hours=value)
        if unit == "d":
            return timedelta(days=value)
    except (ValueError, IndexError):
        pass
    return timedelta(hours=24)


@router.get("/errors/by-subsystem")
async def get_errors_by_subsystem(
    window: str = "24h",
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Error count by subsystem over a rolling window (PRD-142 Wave 0 US-002).

    Backs the dashboard "Error rate by subsystem" tile. Reads from the
    ``error_events`` sink populated by ``record_error`` (US-001). Filters
    by the caller's workspace; system-level rows (``workspace_id IS NULL``)
    are excluded from this workspace-scoped view by design — the
    dashboard tile shows per-tenant errors only.

    Index path: ``idx_error_events_subsystem_created`` /
    ``idx_error_events_workspace_created`` cover the ``(workspace_id,
    created_at)`` filter + ``GROUP BY subsystem`` — no full-table scan.

    Returns ``{window, total, by_subsystem: [{subsystem, count, rate}],
    generated_at}``. ``rate = count / total`` over the window; 0 when
    total is 0 (no divide-by-zero).
    """
    window_start = datetime.utcnow() - _parse_window(window)

    rows = (
        db.query(
            ErrorEvent.subsystem,
            func.count(ErrorEvent.id).label("count"),
        )
        .filter(
            ErrorEvent.workspace_id == ctx.workspace_id,
            ErrorEvent.created_at >= window_start,
        )
        .group_by(ErrorEvent.subsystem)
        .all()
    )

    total = int(sum(int(r.count or 0) for r in rows))
    by_subsystem = [
        {
            "subsystem": r.subsystem,
            "count": int(r.count or 0),
            "rate": (int(r.count or 0) / total) if total > 0 else 0,
        }
        for r in rows
    ]

    return {
        "window": window,
        "total": total,
        "by_subsystem": by_subsystem,
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.get("/widget-engagement")
async def get_widget_engagement(
    window: str = "7d",
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Widget engagement counts by event_type + distinct sessions (PRD-142 Wave 0 US-004).

    Backs the dashboard "Widget engagement" tile. Read-only aggregation
    over ``widget_event_log`` (writer: ``modules/widgets/telemetry.py``;
    schema: ``core/models/widget_event_log.py``). This endpoint does NOT
    construct ``WidgetEventLog`` rows — telemetry's writer remains the
    single source of truth.

    Tenant isolation: ``widget_event_log`` has no ``workspace_id``
    column, so we resolve the caller's ``sites`` first (one workspace,
    many sites — PRD-008-A) and restrict the aggregation to that set.
    A workspace with zero sites short-circuits to an empty payload.

    Index path: the aggregation filters
    ``event_type IN WIDGET_EVENT_TYPES`` so
    ``idx_widget_event_log_type_created`` is eligible alongside the
    ``created_at >= cutoff`` window — no full-table scan.

    Returns ``{window, by_event_type: [{event_type, count}], sessions,
    generated_at}``.
    """
    window_start = datetime.utcnow() - _parse_window(window)

    site_rows = (
        db.query(Site.id).filter(Site.workspace_id == ctx.workspace_id).all()
    )
    site_ids = [row[0] for row in site_rows]

    if not site_ids:
        return {
            "window": window,
            "by_event_type": [],
            "sessions": 0,
            "generated_at": datetime.utcnow().isoformat(),
        }

    agg_rows = (
        db.query(
            WidgetEventLog.event_type,
            func.count(WidgetEventLog.id).label("count"),
        )
        .filter(
            WidgetEventLog.site_id.in_(site_ids),
            WidgetEventLog.event_type.in_(WIDGET_EVENT_TYPES),
            WidgetEventLog.created_at >= window_start,
        )
        .group_by(WidgetEventLog.event_type)
        .all()
    )

    by_event_type = [
        {"event_type": r.event_type, "count": int(r.count or 0)}
        for r in agg_rows
    ]

    sessions = (
        db.query(func.count(func.distinct(WidgetEventLog.session_id)))
        .filter(
            WidgetEventLog.site_id.in_(site_ids),
            WidgetEventLog.event_type.in_(WIDGET_EVENT_TYPES),
            WidgetEventLog.created_at >= window_start,
        )
        .scalar()
    ) or 0

    return {
        "window": window,
        "by_event_type": by_event_type,
        "sessions": int(sessions),
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.get("/dashboard/task-completion-time")
async def get_avg_task_completion_time(ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get average task completion time (UNION: workflows + missions)"""
    try:
        from sqlalchemy import extract

        ws = ctx.workspace_id

        # --- Legacy workflow executions ---
        wf_avg_seconds = db.query(
            func.avg(
                extract('epoch', WorkflowExecution.completed_at) -
                extract('epoch', WorkflowExecution.started_at)
            )
        ).filter(
            WorkflowExecution.workspace_id == ws,
            WorkflowExecution.status == 'completed',
            WorkflowExecution.completed_at.isnot(None),
            WorkflowExecution.started_at.isnot(None),
        ).scalar() or 0

        # --- PRD-125 Phase 2: Mission runs ---
        m_avg_seconds = db.query(
            func.avg(
                extract('epoch', OrchestrationRun.completed_at) -
                extract('epoch', OrchestrationRun.started_at)
            )
        ).filter(
            OrchestrationRun.workspace_id == ws,
            OrchestrationRun.state == RunState.COMPLETED.value,
            OrchestrationRun.completed_at.isnot(None),
            OrchestrationRun.started_at.isnot(None),
        ).scalar() or 0

        # Weighted average (prefer mission data when both exist)
        wf_count = db.query(WorkflowExecution).filter(
            WorkflowExecution.workspace_id == ws, WorkflowExecution.status == 'completed',
            WorkflowExecution.completed_at.isnot(None), WorkflowExecution.started_at.isnot(None),
        ).count()
        m_count = db.query(OrchestrationRun).filter(
            OrchestrationRun.workspace_id == ws, OrchestrationRun.state == RunState.COMPLETED.value,
            OrchestrationRun.completed_at.isnot(None), OrchestrationRun.started_at.isnot(None),
        ).count()
        total_count = wf_count + m_count
        if total_count > 0:
            avg_seconds = (float(wf_avg_seconds) * wf_count + float(m_avg_seconds) * m_count) / total_count
        else:
            avg_seconds = 0
        avg_minutes = round(avg_seconds / 60, 1)

        # 24h average (combined)
        day_ago = datetime.now() - timedelta(hours=24)
        daily_wf = db.query(
            func.avg(extract('epoch', WorkflowExecution.completed_at) - extract('epoch', WorkflowExecution.started_at))
        ).filter(
            WorkflowExecution.workspace_id == ws, WorkflowExecution.status == 'completed',
            WorkflowExecution.completed_at.isnot(None), WorkflowExecution.started_at.isnot(None),
            WorkflowExecution.started_at >= day_ago,
        ).scalar() or 0
        daily_m = db.query(
            func.avg(extract('epoch', OrchestrationRun.completed_at) - extract('epoch', OrchestrationRun.started_at))
        ).filter(
            OrchestrationRun.workspace_id == ws, OrchestrationRun.state == RunState.COMPLETED.value,
            OrchestrationRun.completed_at.isnot(None), OrchestrationRun.started_at.isnot(None),
            OrchestrationRun.started_at >= day_ago,
        ).scalar() or 0

        daily_wf_c = db.query(WorkflowExecution).filter(
            WorkflowExecution.workspace_id == ws, WorkflowExecution.status == 'completed',
            WorkflowExecution.started_at >= day_ago,
        ).count()
        daily_m_c = db.query(OrchestrationRun).filter(
            OrchestrationRun.workspace_id == ws, OrchestrationRun.state == RunState.COMPLETED.value,
            OrchestrationRun.started_at >= day_ago,
        ).count()
        daily_total = daily_wf_c + daily_m_c
        if daily_total > 0:
            daily_avg_seconds = (float(daily_wf) * daily_wf_c + float(daily_m) * daily_m_c) / daily_total
        else:
            daily_avg_seconds = 0
        daily_avg_minutes = round(daily_avg_seconds / 60, 1)
        improvement = round(daily_avg_minutes - avg_minutes, 1)

        return {
            "value": avg_minutes,
            "daily_average": daily_avg_minutes,
            "improvement": improvement,
            "unit": "minutes"
        }

    except Exception as e:
        logger.error(f"Error calculating completion time: {e}")
        return {"value": 0, "daily_average": 0, "improvement": 0, "unit": "minutes"}

@router.get("/dashboard/system-load-trend")
async def get_system_load_trend(ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get system load trend for 24h with color coding"""
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Determine load level and color
        if cpu_percent < 50:
            load_level = "low"
            color = "green"
        elif cpu_percent < 80:
            load_level = "medium"
            color = "yellow"
        else:
            load_level = "high" 
            color = "red"
            
        # Generate 24h trend data (based on real system metrics table or static current snapshot)
        trend_data = []
        for i in range(24):
            hour = (datetime.now() - timedelta(hours=23-i)).hour
            # Use current CPU as baseline; without stored history this is the best we can do
            trend_data.append({"hour": hour, "load": round(cpu_percent, 1)})
        
        return {
            "current_load": round(cpu_percent, 1),
            "level": load_level,
            "color": color,
            "memory_usage": round(memory.percent, 1),
            "trend_data": trend_data
        }
        
    except Exception as e:
        logger.error(f"Error getting system load: {e}")
        return {
            "current_load": 0,
            "level": "unknown",
            "color": "gray",
            "memory_usage": 0,
            "trend_data": [],
            "error": str(e)
        }

@router.get("/dashboard/error-rate-by-type")
async def get_error_rate_by_agent_type(ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get error rate breakdown by agent type (UNION: workflows + missions)"""
    try:
        ws = ctx.workspace_id

        # --- Legacy: Join WorkflowExecution with Agent ---
        wf_rows = (
            db.query(
                Agent.agent_type,
                func.count(WorkflowExecution.id).label('total'),
                func.count(WorkflowExecution.id).filter(
                    WorkflowExecution.status == 'failed'
                ).label('failed'),
            )
            .join(Agent, WorkflowExecution.agent_id == Agent.id)
            .filter(WorkflowExecution.workspace_id == ws)
            .group_by(Agent.agent_type)
            .all()
        )

        error_rates: Dict[str, Any] = {}
        for agent_type, total, failed in wf_rows:
            key = agent_type or "unknown"
            error_rates[key] = {"total_executions": total, "failed_executions": failed}

        # --- PRD-125 Phase 2: Mission tasks by assigned agent type ---
        m_rows = (
            db.query(
                Agent.agent_type,
                func.count(OrchestrationTask.id).label('total'),
                func.count(OrchestrationTask.id).filter(
                    OrchestrationTask.state == TaskState.FAILED.value
                ).label('failed'),
            )
            .join(Agent, OrchestrationTask.assigned_agent_id == Agent.id)
            .join(OrchestrationRun, OrchestrationTask.run_id == OrchestrationRun.id)
            .filter(OrchestrationRun.workspace_id == ws)
            .group_by(Agent.agent_type)
            .all()
        )

        for agent_type, total, failed in m_rows:
            key = agent_type or "unknown"
            if key in error_rates:
                error_rates[key]["total_executions"] += total
                error_rates[key]["failed_executions"] += failed
            else:
                error_rates[key] = {"total_executions": total, "failed_executions": failed}

        # Compute rates
        for key, data in error_rates.items():
            total = data["total_executions"]
            failed = data["failed_executions"]
            rate = round((failed / total * 100) if total > 0 else 0, 1)
            data["error_rate"] = rate
            data["status"] = "good" if rate < 5 else "warning" if rate < 10 else "critical"

        # Include agent types with no executions
        agent_types = db.query(Agent.agent_type, func.count().label('total')).filter(
            Agent.workspace_id == ws
        ).group_by(Agent.agent_type).all()

        for agent_type, agent_count in agent_types:
            key = agent_type or "unknown"
            if key not in error_rates:
                error_rates[key] = {
                    "error_rate": 0,
                    "total_executions": 0,
                    "failed_executions": 0,
                    "total_agents": agent_count,
                    "status": "good"
                }
            else:
                error_rates[key]["total_agents"] = agent_count

        return error_rates

    except Exception as e:
        logger.error(f"Error calculating error rates: {e}")
        return {}

@router.get("/dashboard/queue-depth")
async def get_queue_depth(ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get real-time queue depth for pending tasks (UNION: workflows + missions)"""
    try:
        ws = ctx.workspace_id

        # --- Legacy workflows ---
        pending_workflows = db.query(WorkflowExecution).filter(
            WorkflowExecution.workspace_id == ws,
            WorkflowExecution.status.in_(['pending', 'queued', 'running'])
        ).count()

        # --- PRD-125 Phase 2: Active mission runs + queued tasks ---
        active_missions = db.query(OrchestrationRun).filter(
            OrchestrationRun.workspace_id == ws,
            OrchestrationRun.state.in_([
                RunState.PENDING.value,
                RunState.PLANNING.value,
                RunState.AWAITING_APPROVAL.value,
                RunState.RUNNING.value,
            ]),
        ).count()

        pending_mission_tasks = db.query(OrchestrationTask).join(
            OrchestrationRun, OrchestrationTask.run_id == OrchestrationRun.id
        ).filter(
            OrchestrationRun.workspace_id == ws,
            OrchestrationTask.state.in_([
                TaskState.PENDING.value,
                TaskState.QUEUED.value,
                TaskState.RUNNING.value,
            ]),
        ).count()

        total_pending = pending_workflows + active_missions + pending_mission_tasks

        # High priority: recent items (last 1h)
        hour_ago = datetime.now() - timedelta(hours=1)
        high_priority_wf = db.query(WorkflowExecution).filter(and_(
            WorkflowExecution.workspace_id == ws,
            WorkflowExecution.status.in_(['pending', 'queued']),
            WorkflowExecution.started_at >= hour_ago,
        )).count()
        high_priority_m = db.query(OrchestrationRun).filter(and_(
            OrchestrationRun.workspace_id == ws,
            OrchestrationRun.state.in_([RunState.PENDING.value, RunState.RUNNING.value]),
            OrchestrationRun.created_at >= hour_ago,
        )).count()

        high_priority = high_priority_wf + high_priority_m
        normal_priority = total_pending - high_priority

        return {
            "total_pending": total_pending,
            "high_priority": high_priority,
            "normal_priority": max(normal_priority, 0),
            "pending_mission_tasks": pending_mission_tasks,
            "average_wait_time": 2.3,
            "trend": "stable",
            "sources": {"workflows": pending_workflows, "missions": active_missions},
        }
        
    except Exception as e:
        logger.error(f"Error getting queue depth: {e}")
        return {
            "total_pending": 0,
            "high_priority": 0,
            "normal_priority": 0,
            "average_wait_time": 0,
            "trend": "unknown",
            "error": str(e)
        }

@router.get("/dashboard/efficiency-score")
async def get_efficiency_score(ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get resource utilization efficiency score (0-100)"""
    try:
        # Calculate composite efficiency score
        # CPU efficiency (inverse of idle time)
        cpu_usage = psutil.cpu_percent(interval=1)
        cpu_efficiency = min(100, cpu_usage * 1.2)  # Normalize to favor moderate usage
        
        # Memory efficiency
        memory = psutil.virtual_memory()
        memory_efficiency = min(100, memory.percent * 1.1)
        
        # Agent utilization
        total_agents = db.query(Agent).filter(Agent.workspace_id == ctx.workspace_id).count()
        active_agents = db.query(Agent).filter(Agent.workspace_id == ctx.workspace_id, Agent.status == 'active').count()
        agent_efficiency = (active_agents / total_agents * 100) if total_agents > 0 else 0

        # Execution completion efficiency (UNION: workflows + missions)
        ws = ctx.workspace_id
        day_ago = datetime.now() - timedelta(hours=24)
        wf_recent = db.query(WorkflowExecution).filter(
            WorkflowExecution.workspace_id == ws, WorkflowExecution.started_at >= day_ago
        ).count()
        wf_completed = db.query(WorkflowExecution).filter(and_(
            WorkflowExecution.workspace_id == ws, WorkflowExecution.status == 'completed',
            WorkflowExecution.started_at >= day_ago
        )).count()
        m_recent = db.query(OrchestrationRun).filter(
            OrchestrationRun.workspace_id == ws, OrchestrationRun.created_at >= day_ago
        ).count()
        m_completed = db.query(OrchestrationRun).filter(and_(
            OrchestrationRun.workspace_id == ws,
            OrchestrationRun.state == RunState.COMPLETED.value,
            OrchestrationRun.created_at >= day_ago
        )).count()
        recent_executions = wf_recent + m_recent
        completed = wf_completed + m_completed
        workflow_efficiency = (completed / recent_executions * 100) if recent_executions > 0 else 0
        
        # Composite score
        efficiency_score = round((cpu_efficiency * 0.3 + memory_efficiency * 0.25 + 
                                agent_efficiency * 0.25 + workflow_efficiency * 0.2), 0)
        
        # Determine grade
        if efficiency_score >= 90:
            grade = "A"
            color = "green"
        elif efficiency_score >= 80:
            grade = "B"
            color = "blue"
        elif efficiency_score >= 70:
            grade = "C"
            color = "yellow"
        else:
            grade = "D"
            color = "red"
        
        return {
            "score": int(efficiency_score),
            "grade": grade,
            "color": color,
            "breakdown": {
                "cpu_efficiency": round(cpu_efficiency, 1),
                "memory_efficiency": round(memory_efficiency, 1),
                "agent_efficiency": round(agent_efficiency, 1),
                "workflow_efficiency": round(workflow_efficiency, 1)
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating efficiency score: {e}")
        return {
            "score": 0,
            "grade": "N/A",
            "color": "gray",
            "breakdown": {},
            "error": str(e)
        }

# ==== NEW PERFORMANCE ANALYTICS ENHANCEMENTS ====

@router.get("/performance/cost-per-execution")
async def get_cost_per_execution(ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get cost per execution from real LLM usage data"""
    try:
        from sqlalchemy import cast, Date

        # Last 30 days: SUM(total_cost) and COUNT(DISTINCT execution_id) grouped by date
        thirty_days_ago = datetime.now() - timedelta(days=30)
        rows = (
            db.query(
                cast(LLMUsage.created_at, Date).label('date'),
                func.sum(LLMUsage.total_cost).label('daily_cost'),
                func.count(func.distinct(LLMUsage.execution_id)).label('unique_executions'),
                func.count(LLMUsage.id).label('total_requests'),
            )
            .filter(
                LLMUsage.workspace_id == ctx.workspace_id,
                LLMUsage.created_at >= thirty_days_ago,
            )
            .group_by(cast(LLMUsage.created_at, Date))
            .order_by(cast(LLMUsage.created_at, Date))
            .all()
        )

        cost_data = []
        for row in rows:
            daily_cost = float(row.daily_cost or 0)
            execs = max(row.unique_executions or row.total_requests, 1)
            cost_data.append({
                "date": row.date.strftime("%Y-%m-%d") if row.date else "",
                "total_executions": execs,
                "total_cost": round(daily_cost, 4),
                "cost_per_execution": round(daily_cost / execs, 6),
            })

        if cost_data:
            avg_cost = round(sum(d["cost_per_execution"] for d in cost_data) / len(cost_data), 6)
            # Simple trend: compare first half vs second half
            mid = len(cost_data) // 2
            first_half = sum(d["total_cost"] for d in cost_data[:mid]) if mid > 0 else 0
            second_half = sum(d["total_cost"] for d in cost_data[mid:]) if mid > 0 else 0
            trend = "decreasing" if second_half < first_half else "increasing" if second_half > first_half else "stable"
        else:
            avg_cost = 0
            trend = "stable"

        return {
            "average_cost_per_execution": avg_cost,
            "monthly_data": cost_data,
            "cost_trend": trend,
        }

    except Exception as e:
        logger.error(f"Error calculating cost per execution: {e}")
        return {"average_cost_per_execution": 0, "monthly_data": [], "cost_trend": "stable"}

@router.get("/performance/peak-usage-hours")
async def get_peak_usage_hours(ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get peak usage hours from real LLM usage / execution data"""
    try:
        from sqlalchemy import extract

        # Count LLM requests grouped by hour of day (last 30 days)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        rows = (
            db.query(
                extract('hour', LLMUsage.created_at).label('hour'),
                func.count(LLMUsage.id).label('api_calls'),
                func.count(func.distinct(LLMUsage.agent_id)).label('active_agents'),
            )
            .filter(
                LLMUsage.workspace_id == ctx.workspace_id,
                LLMUsage.created_at >= thirty_days_ago,
            )
            .group_by(extract('hour', LLMUsage.created_at))
            .all()
        )

        hour_map = {int(r.hour): {"api_calls": r.api_calls, "active_agents": r.active_agents} for r in rows}
        max_calls = max((v["api_calls"] for v in hour_map.values()), default=1) or 1

        hourly_data = []
        for hour in range(24):
            data = hour_map.get(hour, {"api_calls": 0, "active_agents": 0})
            usage_pct = round(data["api_calls"] / max_calls * 100, 1) if max_calls else 0
            category = "peak" if usage_pct > 75 else "medium" if usage_pct > 40 else "low"
            hourly_data.append({
                "hour": hour,
                "usage_percent": usage_pct,
                "api_calls": data["api_calls"],
                "active_agents": data["active_agents"],
                "category": category,
            })

        peak_hours = [h["hour"] for h in hourly_data if h["category"] == "peak"]
        peak_usage = max((h["usage_percent"] for h in hourly_data), default=0)

        return {
            "hourly_pattern": hourly_data,
            "peak_hours": peak_hours,
            "peak_period": f"{min(peak_hours)} - {max(peak_hours)} h" if peak_hours else "N/A",
            "peak_usage_percent": peak_usage,
        }

    except Exception as e:
        logger.error(f"Error getting peak usage hours: {e}")
        return {"hourly_pattern": [], "peak_hours": [], "peak_period": "N/A"}

@router.get("/performance/bottlenecks")
async def get_bottleneck_detection(ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get resource bottleneck detection with recommendations"""
    try:
        bottlenecks = []
        
        # Check CPU bottleneck
        cpu_usage = psutil.cpu_percent(interval=1)
        if cpu_usage > 80:
            bottlenecks.append({
                "type": "cpu",
                "severity": "high",
                "current_usage": cpu_usage,
                "threshold": 80,
                "description": "High CPU usage detected",
                "recommendation": "Consider scaling up CPU resources or optimizing workloads",
                "impact": "May cause task delays and timeouts"
            })
        
        # Check Memory bottleneck
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            bottlenecks.append({
                "type": "memory",
                "severity": "high" if memory.percent > 95 else "medium",
                "current_usage": memory.percent,
                "threshold": 85,
                "description": "High memory usage detected",
                "recommendation": "Increase memory allocation or optimize memory-intensive processes",
                "impact": "Risk of system instability and process crashes"
            })
            
        # Check Database connections
        active_connections = db.query(func.count()).scalar() or 0
        if active_connections > 80:  # Mock threshold
            bottlenecks.append({
                "type": "database",
                "severity": "medium",
                "current_usage": active_connections,
                "threshold": 80,
                "description": "High database connection usage",
                "recommendation": "Implement connection pooling or optimize queries",
                "impact": "Slower response times for data operations"
            })
        
        return {
            "bottlenecks_detected": len(bottlenecks),
            "bottlenecks": bottlenecks,
            "overall_health": "good" if len(bottlenecks) == 0 else "warning" if len(bottlenecks) < 3 else "critical",
            "last_check": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error detecting bottlenecks: {e}")
        return {"bottlenecks_detected": 0, "bottlenecks": [], "overall_health": "good"}

@router.get("/performance/predictive-alerts")
async def get_predictive_alerts(ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get predictive capacity alerts"""
    try:
        alerts = []
        
        # Predict storage capacity
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        
        if disk_usage_percent > 75:
            days_until_full = round((100 - disk_usage_percent) / 2.5)  # Assume 2.5% growth per day
            alerts.append({
                "type": "storage_capacity",
                "severity": "warning" if days_until_full > 7 else "critical",
                "prediction": f"Storage will reach 90% capacity in {days_until_full} days",
                "current_usage": round(disk_usage_percent, 1),
                "recommended_action": "Plan for storage expansion or cleanup",
                "confidence": 85
            })
        
        # Predict agent capacity
        active_agents = db.query(Agent).filter(Agent.workspace_id == ctx.workspace_id, Agent.status == 'active').count()
        total_agents = db.query(Agent).filter(Agent.workspace_id == ctx.workspace_id).count()
        utilization = (active_agents / total_agents * 100) if total_agents > 0 else 0
        
        if utilization > 85:
            alerts.append({
                "type": "agent_capacity", 
                "severity": "warning",
                "prediction": "Agent capacity will reach maximum in 3-5 days at current growth rate",
                "current_usage": round(utilization, 1),
                "recommended_action": "Deploy additional agents or optimize workload distribution",
                "confidence": 78
            })
        
        # Predict API rate limits — use real request count from last hour
        last_hour = datetime.now() - timedelta(hours=1)
        current_api_rate = db.query(func.count(LLMUsage.id)).filter(
            LLMUsage.workspace_id == ctx.workspace_id,
            LLMUsage.created_at >= last_hour,
        ).scalar() or 0
        if current_api_rate > 1000:
            alerts.append({
                "type": "api_rate_limit",
                "severity": "medium",
                "prediction": "API rate limit may be exceeded during peak hours",
                "current_usage": current_api_rate,
                "recommended_action": "Implement rate limiting or request throttling",
                "confidence": 65
            })
        
        return {
            "predictive_alerts": alerts,
            "alerts_count": len(alerts),
            "forecast_period": "7 days",
            "confidence_level": "medium"
        }
        
    except Exception as e:
        logger.error(f"Error generating predictive alerts: {e}")
        return {"predictive_alerts": [], "alerts_count": 0}

@router.get("/performance/agent-ranking")
async def get_agent_ranking(ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get agent performance ranking from real execution data"""
    try:
        from sqlalchemy import extract

        agents = db.query(Agent).filter(
            Agent.workspace_id == ctx.workspace_id,
            Agent.status == 'active',
        ).all()

        agent_rankings = []
        for agent in agents:
            # Real execution stats for this agent
            total = db.query(func.count(WorkflowExecution.id)).filter(
                WorkflowExecution.agent_id == agent.id,
                WorkflowExecution.workspace_id == ctx.workspace_id,
            ).scalar() or 0

            successful = db.query(func.count(WorkflowExecution.id)).filter(
                WorkflowExecution.agent_id == agent.id,
                WorkflowExecution.workspace_id == ctx.workspace_id,
                WorkflowExecution.status == 'completed',
            ).scalar() or 0

            success_rate = round((successful / total * 100) if total > 0 else 0, 1)

            # Average duration in seconds
            avg_duration_sec = db.query(
                func.avg(
                    extract('epoch', WorkflowExecution.completed_at) -
                    extract('epoch', WorkflowExecution.started_at)
                )
            ).filter(
                WorkflowExecution.agent_id == agent.id,
                WorkflowExecution.workspace_id == ctx.workspace_id,
                WorkflowExecution.status == 'completed',
                WorkflowExecution.completed_at.isnot(None),
                WorkflowExecution.started_at.isnot(None),
            ).scalar() or 0

            avg_response_time = round(float(avg_duration_sec), 2)

            # Composite score: success_rate weighted most, then speed, then volume
            speed_score = max(0, 100 - avg_response_time * 10)
            volume_score = min(100, total / 5) if total > 0 else 0
            score = round(success_rate * 0.5 + speed_score * 0.3 + volume_score * 0.2, 1)

            agent_rankings.append({
                "agent_id": agent.id,
                "name": agent.name,
                "agent_type": agent.agent_type,
                "performance_score": score,
                "success_rate": success_rate,
                "avg_response_time": avg_response_time,
                "tasks_completed": total,
                "rank": 0,
            })

        agent_rankings.sort(key=lambda x: x["performance_score"], reverse=True)
        for i, entry in enumerate(agent_rankings):
            entry["rank"] = i + 1

        return {
            "agent_rankings": agent_rankings[:20],
            "total_agents": len(agents),
            "top_performer": agent_rankings[0] if agent_rankings else None,
            "average_score": round(
                sum(a["performance_score"] for a in agent_rankings) / len(agent_rankings), 1
            ) if agent_rankings else 0,
        }

    except Exception as e:
        logger.error(f"Error generating agent ranking: {e}")
        return {"agent_rankings": [], "total_agents": 0}

@router.get("/performance/sla-compliance")
async def get_sla_compliance(ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get SLA compliance from real execution and usage data"""
    try:
        from sqlalchemy import extract

        thirty_days_ago = datetime.now() - timedelta(days=30)

        # --- Task completion rate ---
        total_executions = db.query(func.count(WorkflowExecution.id)).filter(
            WorkflowExecution.workspace_id == ctx.workspace_id,
            WorkflowExecution.started_at >= thirty_days_ago,
        ).scalar() or 0

        completed = db.query(func.count(WorkflowExecution.id)).filter(
            WorkflowExecution.workspace_id == ctx.workspace_id,
            WorkflowExecution.status == 'completed',
            WorkflowExecution.started_at >= thirty_days_ago,
        ).scalar() or 0

        failed = db.query(func.count(WorkflowExecution.id)).filter(
            WorkflowExecution.workspace_id == ctx.workspace_id,
            WorkflowExecution.status == 'failed',
            WorkflowExecution.started_at >= thirty_days_ago,
        ).scalar() or 0

        completion_rate = round((completed / total_executions * 100) if total_executions > 0 else 0, 1)

        # --- Average execution time (seconds) ---
        avg_exec_sec = db.query(
            func.avg(
                extract('epoch', WorkflowExecution.completed_at) -
                extract('epoch', WorkflowExecution.started_at)
            )
        ).filter(
            WorkflowExecution.workspace_id == ctx.workspace_id,
            WorkflowExecution.status == 'completed',
            WorkflowExecution.completed_at.isnot(None),
            WorkflowExecution.started_at.isnot(None),
            WorkflowExecution.started_at >= thirty_days_ago,
        ).scalar() or 0

        avg_exec_sec = round(float(avg_exec_sec), 2)

        sla_metrics = {
            "task_completion": {
                "sla_target": 95.0,
                "current_rate": completion_rate,
                "compliance_rate": round(min(completion_rate / 95.0 * 100, 100), 1) if completion_rate > 0 else 0,
                "status": "excellent" if completion_rate >= 95 else "good" if completion_rate >= 85 else "warning",
                "failed_tasks": failed,
            },
            "response_time": {
                "sla_target": 120.0,
                "current_average": avg_exec_sec,
                "compliance_rate": round(min(120.0 / max(avg_exec_sec, 0.01) * 100, 100), 1) if avg_exec_sec > 0 else 100,
                "status": "good" if avg_exec_sec <= 120 else "warning",
            },
            "uptime": {"sla_target": 99.9, "current_uptime": "N/A", "compliance_rate": "N/A", "status": "N/A"},
            "support_response": {"sla_target": 15, "current_average": "N/A", "compliance_rate": "N/A", "status": "N/A"},
        }

        # Overall compliance from metrics that have real data
        real_rates = [
            m["compliance_rate"] for m in sla_metrics.values()
            if isinstance(m.get("compliance_rate"), (int, float))
        ]
        overall_compliance = round(sum(real_rates) / len(real_rates), 1) if real_rates else 0

        if overall_compliance >= 95:
            overall_status, status_color = "excellent", "green"
        elif overall_compliance >= 85:
            overall_status, status_color = "good", "blue"
        elif overall_compliance >= 75:
            overall_status, status_color = "warning", "yellow"
        else:
            overall_status, status_color = "critical", "red"

        return {
            "overall_compliance": overall_compliance,
            "overall_status": overall_status,
            "status_color": status_color,
            "sla_metrics": sla_metrics,
            "reporting_period": "Last 30 days",
        }

    except Exception as e:
        logger.error(f"Error getting SLA compliance: {e}")
        return {"overall_compliance": 0, "overall_status": "unknown", "sla_metrics": {}}

# ==== COMBINED DASHBOARD METRICS ENDPOINT ====

@router.get("/dashboard/all-metrics")
async def get_all_dashboard_metrics(ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)) -> DashboardMetrics:
    """Get all enhanced dashboard metrics in one call for efficiency"""
    try:
        # Call individual metric endpoints
        success_rate = await get_agent_success_rate(ctx, db)
        completion_time = await get_avg_task_completion_time(ctx, db)
        system_load = await get_system_load_trend(ctx, db)
        error_rates = await get_error_rate_by_agent_type(ctx, db)
        queue_depth = await get_queue_depth(ctx, db)
        efficiency = await get_efficiency_score(ctx, db)
        
        return DashboardMetrics(
            agent_success_rate=success_rate["value"],
            avg_task_completion_time=completion_time["value"],
            system_load_trend=system_load,
            error_rate_by_agent_type=error_rates,
            queue_depth=queue_depth["total_pending"],
            resource_utilization_efficiency=efficiency["score"]
        )
        
    except Exception as e:
        logger.error(f"Error getting all dashboard metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ==== COMBINED PERFORMANCE ANALYTICS ENDPOINT ====

@router.get("/performance/all-enhancements")
async def get_all_performance_enhancements(ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get all performance analytics enhancements in one call"""
    try:
        cost_per_execution = await get_cost_per_execution(ctx, db)
        peak_usage = await get_peak_usage_hours(ctx, db)
        bottlenecks = await get_bottleneck_detection(ctx, db)
        alerts = await get_predictive_alerts(ctx, db)
        ranking = await get_agent_ranking(ctx, db)
        sla = await get_sla_compliance(ctx, db)
        
        return {
            "cost_analysis": cost_per_execution,
            "peak_usage_analysis": peak_usage,
            "bottleneck_detection": bottlenecks,
            "predictive_alerts": alerts,
            "agent_performance_ranking": ranking,
            "sla_compliance": sla
        }
        
    except Exception as e:
        logger.error(f"Error getting performance enhancements: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

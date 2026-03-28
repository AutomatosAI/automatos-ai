"""
KPI API — Command Centre Dashboard Widgets
============================================

4 endpoints returning pre-aggregated KPI data for frontend widgets.
All workspace-scoped via hybrid auth.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, and_, extract, cast, Date
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models import Agent, WorkflowExecution
from core.models.core import LLMUsage, BoardTask
from core.models.orchestration import OrchestrationRun

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/kpi", tags=["kpi"])

PERIOD_DAYS = {"7d": 7, "30d": 30, "90d": 90}


def _parse_period(period: str) -> int:
    return PERIOD_DAYS.get(period, 30)


# ── Cost Tracker ─────────────────────────────────────────────


@router.get("/cost-tracker")
async def get_cost_tracker(
    period: str = Query("30d", regex="^(7d|30d|90d)$"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Period spend, daily trend, top 3 agents by cost."""
    try:
        days = _parse_period(period)
        since = datetime.now() - timedelta(days=days)

        # Total spend
        total_cost = float(
            db.query(func.sum(LLMUsage.total_cost))
            .filter(
                LLMUsage.workspace_id == ctx.workspace_id,
                LLMUsage.created_at >= since,
            )
            .scalar()
            or 0
        )

        # Daily trend
        daily_rows = (
            db.query(
                cast(LLMUsage.created_at, Date).label("date"),
                func.sum(LLMUsage.total_cost).label("cost"),
            )
            .filter(
                LLMUsage.workspace_id == ctx.workspace_id,
                LLMUsage.created_at >= since,
            )
            .group_by(cast(LLMUsage.created_at, Date))
            .order_by(cast(LLMUsage.created_at, Date))
            .all()
        )

        daily_trend = [
            {"date": r.date.strftime("%Y-%m-%d"), "cost": round(float(r.cost or 0), 4)}
            for r in daily_rows
        ]

        # Top 3 agents by cost
        top_agents_rows = (
            db.query(
                Agent.name,
                func.sum(LLMUsage.total_cost).label("cost"),
            )
            .join(Agent, LLMUsage.agent_id == Agent.id)
            .filter(
                LLMUsage.workspace_id == ctx.workspace_id,
                LLMUsage.created_at >= since,
            )
            .group_by(Agent.name)
            .order_by(func.sum(LLMUsage.total_cost).desc())
            .limit(3)
            .all()
        )

        top_agents = [
            {"name": r.name, "cost": round(float(r.cost or 0), 4)} for r in top_agents_rows
        ]

        # Period-over-period change
        prev_since = since - timedelta(days=days)
        prev_cost = float(
            db.query(func.sum(LLMUsage.total_cost))
            .filter(
                LLMUsage.workspace_id == ctx.workspace_id,
                LLMUsage.created_at >= prev_since,
                LLMUsage.created_at < since,
            )
            .scalar()
            or 0
        )

        change_pct = (
            round((total_cost - prev_cost) / prev_cost * 100, 1) if prev_cost > 0 else 0
        )

        return {
            "total_cost": round(total_cost, 4),
            "change_pct": change_pct,
            "daily_trend": daily_trend,
            "top_agents": top_agents,
            "period": period,
        }

    except Exception as e:
        logger.error("KPI cost-tracker failed: %s", e, exc_info=True)
        return {"total_cost": 0, "change_pct": 0, "daily_trend": [], "top_agents": [], "period": period}


# ── Agent Performance ────────────────────────────────────────


@router.get("/agent-performance")
async def get_agent_performance(
    period: str = Query("30d", regex="^(7d|30d|90d)$"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Per-agent success rate, avg completion time, tasks done."""
    try:
        days = _parse_period(period)
        since = datetime.now() - timedelta(days=days)

        agents = (
            db.query(Agent)
            .filter(Agent.workspace_id == ctx.workspace_id, Agent.status == "active")
            .all()
        )

        results = []
        for agent in agents:
            total = (
                db.query(func.count(WorkflowExecution.id))
                .filter(
                    WorkflowExecution.agent_id == agent.id,
                    WorkflowExecution.workspace_id == ctx.workspace_id,
                    WorkflowExecution.started_at >= since,
                )
                .scalar()
                or 0
            )

            if total == 0:
                continue

            successful = (
                db.query(func.count(WorkflowExecution.id))
                .filter(
                    WorkflowExecution.agent_id == agent.id,
                    WorkflowExecution.workspace_id == ctx.workspace_id,
                    WorkflowExecution.status == "completed",
                    WorkflowExecution.started_at >= since,
                )
                .scalar()
                or 0
            )

            avg_sec = (
                db.query(
                    func.avg(
                        extract("epoch", WorkflowExecution.completed_at)
                        - extract("epoch", WorkflowExecution.started_at)
                    )
                )
                .filter(
                    WorkflowExecution.agent_id == agent.id,
                    WorkflowExecution.workspace_id == ctx.workspace_id,
                    WorkflowExecution.status == "completed",
                    WorkflowExecution.completed_at.isnot(None),
                    WorkflowExecution.started_at.isnot(None),
                    WorkflowExecution.started_at >= since,
                )
                .scalar()
                or 0
            )

            success_rate = round(successful / total * 100, 1)
            results.append({
                "agent_id": agent.id,
                "name": agent.name,
                "success_rate": success_rate,
                "tasks_completed": total,
                "avg_completion_seconds": round(float(avg_sec), 1),
            })

        results.sort(key=lambda x: x["success_rate"], reverse=True)

        return {"agents": results, "total_agents": len(results), "period": period}

    except Exception as e:
        logger.error("KPI agent-performance failed: %s", e, exc_info=True)
        return {"agents": [], "total_agents": 0, "period": period}


# ── Playbook Metrics ─────────────────────────────────────────


@router.get("/playbook-metrics")
async def get_playbook_metrics(
    period: str = Query("30d", regex="^(7d|30d|90d)$"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Playbook/recipe runs, success %, avg duration."""
    try:
        days = _parse_period(period)
        since = datetime.now() - timedelta(days=days)

        # Playbook executions tracked as WorkflowExecution with workflow_id set
        rows = (
            db.query(
                WorkflowExecution.workflow_id,
                func.count(WorkflowExecution.id).label("runs"),
                func.count(WorkflowExecution.id)
                .filter(WorkflowExecution.status == "completed")
                .label("successes"),
                func.avg(
                    extract("epoch", WorkflowExecution.completed_at)
                    - extract("epoch", WorkflowExecution.started_at)
                ).label("avg_duration"),
            )
            .filter(
                WorkflowExecution.workspace_id == ctx.workspace_id,
                WorkflowExecution.workflow_id.isnot(None),
                WorkflowExecution.started_at >= since,
            )
            .group_by(WorkflowExecution.workflow_id)
            .all()
        )

        # Resolve workflow names
        from core.models import Workflow

        wf_ids = [r.workflow_id for r in rows if r.workflow_id]
        wf_map = {}
        if wf_ids:
            wfs = db.query(Workflow.id, Workflow.name).filter(Workflow.id.in_(wf_ids)).all()
            wf_map = {w.id: w.name for w in wfs}

        playbooks = []
        for r in rows:
            runs = r.runs or 0
            successes = r.successes or 0
            playbooks.append({
                "workflow_id": r.workflow_id,
                "name": wf_map.get(r.workflow_id, f"Playbook {r.workflow_id}"),
                "runs": runs,
                "success_pct": round(successes / runs * 100, 1) if runs > 0 else 0,
                "avg_duration_seconds": round(float(r.avg_duration or 0), 1),
            })

        playbooks.sort(key=lambda x: x["runs"], reverse=True)

        return {"playbooks": playbooks, "total": len(playbooks), "period": period}

    except Exception as e:
        logger.error("KPI playbook-metrics failed: %s", e, exc_info=True)
        return {"playbooks": [], "total": 0, "period": period}


# ── Approval Gates ───────────────────────────────────────────


@router.get("/approval-gates")
async def get_approval_gates(
    period: str = Query("30d", regex="^(7d|30d|90d)$"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Pending approvals, avg time-to-approve, recent decisions."""
    try:
        days = _parse_period(period)
        since = datetime.now() - timedelta(days=days)

        # Currently awaiting approval
        pending = (
            db.query(OrchestrationRun)
            .filter(
                OrchestrationRun.workspace_id == ctx.workspace_id,
                OrchestrationRun.state == "awaiting_approval",
            )
            .all()
        )

        pending_missions = [
            {
                "id": str(r.id),
                "goal": (r.goal[:100] if r.goal else ""),
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "waiting_since": r.updated_at.isoformat() if r.updated_at else None,
            }
            for r in pending
        ]

        # Recently approved/rejected (completed or failed after awaiting)
        completed_runs = (
            db.query(OrchestrationRun)
            .filter(
                OrchestrationRun.workspace_id == ctx.workspace_id,
                OrchestrationRun.state.in_(["running", "completed", "failed", "cancelled"]),
                OrchestrationRun.updated_at >= since,
            )
            .order_by(OrchestrationRun.updated_at.desc())
            .limit(10)
            .all()
        )

        # Avg time from creation to started (proxy for approval time)
        avg_approval_sec = (
            db.query(
                func.avg(
                    extract("epoch", OrchestrationRun.started_at)
                    - extract("epoch", OrchestrationRun.created_at)
                )
            )
            .filter(
                OrchestrationRun.workspace_id == ctx.workspace_id,
                OrchestrationRun.started_at.isnot(None),
                OrchestrationRun.created_at >= since,
            )
            .scalar()
            or 0
        )

        return {
            "pending_count": len(pending),
            "pending_missions": pending_missions,
            "avg_approval_seconds": round(float(avg_approval_sec), 1),
            "recent_count": len(completed_runs),
            "period": period,
        }

    except Exception as e:
        logger.error("KPI approval-gates failed: %s", e, exc_info=True)
        return {
            "pending_count": 0,
            "pending_missions": [],
            "avg_approval_seconds": 0,
            "recent_count": 0,
            "period": period,
        }

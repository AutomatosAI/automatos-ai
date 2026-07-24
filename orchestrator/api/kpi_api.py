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
from sqlalchemy import func, extract, cast, Date
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.super_admin import require_super_admin
from core.database.database import get_db
from core.models import Agent
from core.models.core import LLMUsage, BoardTask, RecipeExecution, WorkflowTemplate
from core.models.orchestration import OrchestrationRun

logger = logging.getLogger(__name__)

# PRD-143 S7: observability tier — router-wide super-admin lock (fail-closed).
router = APIRouter(
    prefix="/api/kpi",
    tags=["kpi"],
    dependencies=[Depends(require_super_admin)],
)

PERIOD_DAYS = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}


def _parse_period(period: str) -> int:
    return PERIOD_DAYS.get(period, 30)


# ── Cost Tracker ─────────────────────────────────────────────


@router.get("/cost-tracker")
async def get_cost_tracker(
    period: str = Query("30d", regex="^(1d|7d|30d|90d)$"),
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
    period: str = Query("30d", regex="^(1d|7d|30d|90d)$"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Per-agent success rate, avg completion time, tasks done.

    Sources: BoardTask (agent task board) for task counts + completion times.
    """
    try:
        days = _parse_period(period)
        since = datetime.now() - timedelta(days=days)

        rows = (
            db.query(
                BoardTask.assigned_agent_id,
                Agent.name,
                func.count(BoardTask.id).label("total"),
                func.count(BoardTask.id).filter(BoardTask.status == "done").label("done"),
                func.avg(
                    extract("epoch", BoardTask.completed_at)
                    - extract("epoch", BoardTask.started_at)
                ).filter(
                    BoardTask.status == "done",
                    BoardTask.completed_at.isnot(None),
                    BoardTask.started_at.isnot(None),
                ).label("avg_sec"),
            )
            .join(Agent, BoardTask.assigned_agent_id == Agent.id)
            .filter(
                BoardTask.workspace_id == ctx.workspace_id,
                BoardTask.assigned_agent_id.isnot(None),
                BoardTask.created_at >= since,
            )
            .group_by(BoardTask.assigned_agent_id, Agent.name)
            .all()
        )

        results = []
        for r in rows:
            total = r.total or 0
            if total == 0:
                continue
            done = r.done or 0
            success_rate = round(done / total * 100, 1)
            results.append({
                "agent_id": r.assigned_agent_id,
                "name": r.name,
                "success_rate": success_rate,
                "tasks_completed": total,
                "avg_completion_seconds": round(float(r.avg_sec or 0), 1),
            })

        results.sort(key=lambda x: x["success_rate"], reverse=True)

        return {"agents": results, "total_agents": len(results), "period": period}

    except Exception as e:
        logger.error("KPI agent-performance failed: %s", e, exc_info=True)
        return {"agents": [], "total_agents": 0, "period": period}


# ── Playbook Metrics ─────────────────────────────────────────


@router.get("/playbook-metrics")
async def get_playbook_metrics(
    period: str = Query("30d", regex="^(1d|7d|30d|90d)$"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Playbook/recipe runs, success %, avg duration.

    Sources: RecipeExecution (recipe_executions table) joined with
    WorkflowTemplate (workflow_recipes table) for names.
    """
    try:
        days = _parse_period(period)
        since = datetime.now() - timedelta(days=days)

        rows = (
            db.query(
                RecipeExecution.recipe_id,
                func.count(RecipeExecution.id).label("runs"),
                func.count(RecipeExecution.id)
                .filter(RecipeExecution.status == "completed")
                .label("successes"),
                func.avg(
                    extract("epoch", RecipeExecution.completed_at)
                    - extract("epoch", RecipeExecution.started_at)
                ).filter(
                    RecipeExecution.status == "completed",
                    RecipeExecution.completed_at.isnot(None),
                ).label("avg_duration"),
            )
            .filter(
                RecipeExecution.workspace_id == ctx.workspace_id,
                RecipeExecution.started_at >= since,
            )
            .group_by(RecipeExecution.recipe_id)
            .all()
        )

        recipe_ids = [r.recipe_id for r in rows if r.recipe_id]
        name_map = {}
        if recipe_ids:
            templates = (
                db.query(WorkflowTemplate.id, WorkflowTemplate.name)
                .filter(WorkflowTemplate.id.in_(recipe_ids))
                .all()
            )
            name_map = {t.id: t.name for t in templates}

        playbooks = []
        for r in rows:
            runs = r.runs or 0
            successes = r.successes or 0
            playbooks.append({
                "recipe_id": r.recipe_id,
                "name": name_map.get(r.recipe_id, f"Playbook {r.recipe_id}"),
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
    period: str = Query("30d", regex="^(1d|7d|30d|90d)$"),
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


# ── Decisions Needed (Wave 5) ────────────────────────────────


@router.get("/decisions-needed")
async def get_decisions_needed(
    limit: int = Query(10, ge=1, le=50),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Aggregate things that need Gerard's call but aren't already on the kanban.

    Specifically: reports flagged ``requires_approval=True`` and not yet
    acknowledged, plus orchestration runs in a BLOCKED state with
    escalation_level >= APPROVAL (or any blocked run when level is unset).

    Tasks themselves (priority urgent/high) are intentionally excluded —
    the kanban already surfaces those. Approval gates have their own widget.
    """
    from sqlalchemy import text

    try:
        # 1. Reports requiring approval that haven't been acknowledged.
        report_rows = db.execute(
            text(
                """
                SELECT id::text, title, summary, status, escalation_level,
                       agent_name, created_at, requires_approval
                  FROM agent_reports
                 WHERE workspace_id = :ws
                   AND requires_approval = TRUE
                   AND acknowledged_at IS NULL
              ORDER BY COALESCE(escalation_level, 0) DESC, created_at ASC
                 LIMIT :limit
                """
            ),
            {"ws": str(ctx.workspace_id), "limit": limit},
        ).fetchall()

        reports = [
            {
                "kind": "report",
                "id": r.id,
                "title": r.title,
                "summary": r.summary,
                "status": r.status,
                "escalation_level": r.escalation_level,
                "agent_name": r.agent_name,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in report_rows
        ]

        # 2. Missions blocked / awaiting human review.
        run_rows = (
            db.query(OrchestrationRun)
            .filter(
                OrchestrationRun.workspace_id == ctx.workspace_id,
                OrchestrationRun.state_type == "BLOCKED",
            )
            .order_by(OrchestrationRun.updated_at.desc())
            .limit(limit)
            .all()
        )

        missions = [
            {
                "kind": "mission",
                "id": str(r.id),
                "title": (r.goal[:120] if r.goal else "(no goal)"),
                "summary": r.stop_detail or r.stop_reason or "blocked",
                "status": r.state,
                "escalation_level": getattr(r, "escalation_level", None),
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "updated_at": r.updated_at.isoformat() if r.updated_at else None,
            }
            for r in run_rows
        ]

        merged = reports + missions
        merged.sort(
            key=lambda item: (
                -(item.get("escalation_level") or 0),
                item.get("created_at") or "",
            )
        )
        # Trim before counting — sub-queries each LIMIT independently, so
        # raw counts could double up. The widget's "total" must match the
        # number of rows it actually shows.
        items = merged[:limit]
        reports_in_items = sum(1 for i in items if i["kind"] == "report")
        missions_in_items = sum(1 for i in items if i["kind"] == "mission")

        return {
            "total": len(items),
            "reports_count": reports_in_items,
            "missions_count": missions_in_items,
            "items": items,
        }

    except Exception as e:
        logger.error("KPI decisions-needed failed: %s", e, exc_info=True)
        return {
            "total": 0,
            "reports_count": 0,
            "missions_count": 0,
            "items": [],
        }

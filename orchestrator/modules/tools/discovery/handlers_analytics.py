"""Analytics handlers for PlatformActionExecutor — LLM usage, cost, workspace stats, board summary."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def get_llm_usage(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import LLMUsage

    days = params.get("days", 30)
    since = datetime.now(timezone.utc) - timedelta(days=days)

    rows = (
        db.query(
            LLMUsage.model_id,
            LLMUsage.provider,
            func.count(LLMUsage.id).label("request_count"),
            func.sum(LLMUsage.input_tokens).label("total_input_tokens"),
            func.sum(LLMUsage.output_tokens).label("total_output_tokens"),
            func.sum(LLMUsage.total_tokens).label("total_tokens"),
        )
        .filter(
            LLMUsage.workspace_id == workspace_id,
            LLMUsage.created_at >= since,
        )
        .group_by(LLMUsage.model_id, LLMUsage.provider)
        .all()
    )

    models = []
    total_requests = 0
    total_tokens = 0
    for row in rows:
        models.append({
            "model": row.model_id,
            "provider": row.provider,
            "requests": row.request_count,
            "input_tokens": row.total_input_tokens or 0,
            "output_tokens": row.total_output_tokens or 0,
            "total_tokens": row.total_tokens or 0,
        })
        total_requests += row.request_count
        total_tokens += (row.total_tokens or 0)

    return {
        "success": True,
        "period_days": days,
        "total_requests": total_requests,
        "total_tokens": total_tokens,
        "by_model": models,
    }


async def get_cost_breakdown(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import LLMUsage

    days = params.get("days", 30)
    group_by = params.get("group_by", "model")
    since = datetime.now(timezone.utc) - timedelta(days=days)

    if group_by == "agent":
        group_col = LLMUsage.agent_id
    elif group_by == "day":
        group_col = func.date(LLMUsage.created_at)
    else:
        group_col = LLMUsage.model_id

    rows = (
        db.query(
            group_col.label("group_key"),
            func.sum(LLMUsage.total_cost).label("total_cost"),
            func.sum(LLMUsage.input_cost).label("input_cost"),
            func.sum(LLMUsage.output_cost).label("output_cost"),
            func.count(LLMUsage.id).label("request_count"),
        )
        .filter(
            LLMUsage.workspace_id == workspace_id,
            LLMUsage.created_at >= since,
        )
        .group_by(group_col)
        .order_by(func.sum(LLMUsage.total_cost).desc())
        .all()
    )

    breakdown = []
    total_cost = 0.0
    for row in rows:
        key = str(row.group_key) if row.group_key is not None else "unknown"
        cost = float(row.total_cost or 0)
        breakdown.append({
            group_by: key,
            "total_cost": round(cost, 6),
            "input_cost": round(float(row.input_cost or 0), 6),
            "output_cost": round(float(row.output_cost or 0), 6),
            "requests": row.request_count,
        })
        total_cost += cost

    return {
        "success": True,
        "period_days": days,
        "group_by": group_by,
        "total_cost": round(total_cost, 6),
        "breakdown": breakdown,
    }


async def workspace_stats(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Get workspace usage stats -- LLM usage, top models, top agents."""
    from core.models.core import LLMUsage
    from core.models import Agent, Document

    period = params.get("period", "7d")
    days = {"today": 1, "7d": 7, "30d": 30}.get(period, 7)
    since = datetime.now(timezone.utc) - timedelta(days=days)

    # LLM usage summary
    usage = (
        db.query(
            func.count(LLMUsage.id).label("total_requests"),
            func.sum(LLMUsage.total_tokens).label("total_tokens"),
            func.sum(LLMUsage.total_cost).label("total_cost"),
        )
        .filter(
            LLMUsage.workspace_id == workspace_id,
            LLMUsage.created_at >= since,
        )
        .first()
    )

    # Top models by usage
    top_models = (
        db.query(
            LLMUsage.model_id,
            func.count(LLMUsage.id).label("requests"),
            func.sum(LLMUsage.total_cost).label("cost"),
        )
        .filter(
            LLMUsage.workspace_id == workspace_id,
            LLMUsage.created_at >= since,
        )
        .group_by(LLMUsage.model_id)
        .order_by(func.count(LLMUsage.id).desc())
        .limit(5)
        .all()
    )

    # Top agents by cost
    top_agents = (
        db.query(
            LLMUsage.agent_id,
            func.count(LLMUsage.id).label("requests"),
            func.sum(LLMUsage.total_cost).label("cost"),
        )
        .filter(
            LLMUsage.workspace_id == workspace_id,
            LLMUsage.created_at >= since,
            LLMUsage.agent_id.isnot(None),
        )
        .group_by(LLMUsage.agent_id)
        .order_by(func.sum(LLMUsage.total_cost).desc())
        .limit(5)
        .all()
    )

    # Resource counts
    agent_count = (
        db.query(func.count(Agent.id))
        .filter(Agent.workspace_id == workspace_id, Agent.status == "active")
        .scalar()
    ) or 0
    doc_count = (
        db.query(func.count(Document.id))
        .filter(Document.workspace_id == workspace_id)
        .scalar()
    ) or 0

    return {
        "success": True,
        "period": period,
        "usage": {
            "total_requests": usage.total_requests or 0,
            "total_tokens": usage.total_tokens or 0,
            "total_cost": round(float(usage.total_cost or 0), 6),
        },
        "top_models": [
            {
                "model": r.model_id,
                "requests": r.requests,
                "cost": round(float(r.cost or 0), 6),
            }
            for r in top_models
        ],
        "top_agents": [
            {
                "agent_id": r.agent_id,
                "requests": r.requests,
                "cost": round(float(r.cost or 0), 6),
            }
            for r in top_agents
        ],
        "resources": {
            "agents": agent_count,
            "documents": doc_count,
        },
    }


# What one "how are things going?" answer needs, in one call (F028). Night 1
# answered that question by calling platform_list_tasks six times in a second
# plus a summary, an activity read and a schedule read — seven round trips and
# seven tool results in the prompt for one status line.
BOARD_SNAPSHOT_TASK_LIMIT = 200
BOARD_SNAPSHOT_RECENT = 10


async def board_snapshot(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Everything a status answer needs, in ONE call.

    Counts by status and priority, the open tickets, what changed recently, and
    what is scheduled — assembled here rather than by the model making six
    calls and stitching them together.
    """
    from sqlalchemy import text as sa_text

    from core.security.surface import widget_turn

    snapshot: Dict[str, Any] = {"success": True}
    # F155: a widget turn (tasks:read) gets where things stand: the counts and
    # each open or finished ticket's id, title and status. Never who is busiest,
    # the owner's scheduled routines, or anyone's errors.
    visitor = widget_turn()

    summary = await board_summary(db, workspace_id, params)
    snapshot["counts"] = {
        k: summary.get(k) for k in ("by_status", "by_priority", "busiest_agents", "total")
        if k in summary
    }

    limit = params.get("limit")
    try:
        limit = max(1, min(int(limit), BOARD_SNAPSHOT_TASK_LIMIT))
    except (TypeError, ValueError):
        limit = BOARD_SNAPSHOT_TASK_LIMIT

    try:
        rows = db.execute(sa_text("""
            SELECT bt.id, bt.title, bt.status, bt.priority, bt.updated_at, a.name AS agent_name
            FROM board_tasks bt
            LEFT JOIN agents a ON a.id = bt.assigned_agent_id
            WHERE bt.workspace_id = :ws
              AND bt.status NOT IN ('done', 'cancelled', 'closed')
            ORDER BY bt.updated_at DESC
            LIMIT :limit
        """), {"ws": str(workspace_id), "limit": limit}).fetchall()
        snapshot["open_tasks"] = [{
            "id": r.id, "title": (r.title or "")[:120], "status": r.status,
            **({} if visitor else {"priority": r.priority, "agent": r.agent_name}),
        } for r in rows]
        snapshot["open_task_count"] = len(snapshot["open_tasks"])
    except Exception as e:  # noqa: BLE001 — a partial snapshot beats no answer
        logger.warning("[board_snapshot] open tasks unavailable: %s", e)
        snapshot["open_tasks"] = []

    try:
        recent = db.execute(sa_text("""
            SELECT id, title, status, completed_at
            FROM board_tasks
            WHERE workspace_id = :ws AND completed_at IS NOT NULL
            ORDER BY completed_at DESC LIMIT :n
        """), {"ws": str(workspace_id), "n": BOARD_SNAPSHOT_RECENT}).fetchall()
        snapshot["recently_finished"] = [{
            "id": r.id, "title": (r.title or "")[:120], "status": r.status,
            "at": r.completed_at.isoformat() if r.completed_at else None,
        } for r in recent]
    except Exception as e:  # noqa: BLE001
        logger.warning("[board_snapshot] recent activity unavailable: %s", e)
        snapshot["recently_finished"] = []

    if visitor:
        return snapshot

    try:
        sched = db.execute(sa_text("""
            SELECT id, description, schedule, next_run_at
            FROM agent_scheduled_tasks
            WHERE workspace_id = :ws AND status = 'active'
            ORDER BY next_run_at NULLS LAST LIMIT :n
        """), {"ws": str(workspace_id), "n": BOARD_SNAPSHOT_RECENT}).fetchall()
        snapshot["scheduled"] = [{
            "id": r.id, "what": (r.description or "")[:100], "schedule": r.schedule,
            "next_run_at": r.next_run_at.isoformat() if r.next_run_at else None,
        } for r in sched]
    except Exception as e:  # noqa: BLE001
        logger.warning("[board_snapshot] schedule unavailable: %s", e)
        snapshot["scheduled"] = []

    return snapshot


async def board_summary(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Get a summary of the task board: counts, busiest agents, failures."""
    from core.models.core import BoardTask
    from core.models import Agent

    all_tasks = db.query(BoardTask).filter(
        BoardTask.workspace_id == workspace_id,
    ).all()

    # Counts by status
    by_status: Dict[str, int] = {}
    by_priority: Dict[str, int] = {}
    agent_task_counts: Dict[int, int] = {}
    failed_tasks = []

    for t in all_tasks:
        by_status[t.status] = by_status.get(t.status, 0) + 1
        by_priority[t.priority] = by_priority.get(t.priority, 0) + 1
        if t.assigned_agent_id:
            agent_task_counts[t.assigned_agent_id] = agent_task_counts.get(t.assigned_agent_id, 0) + 1
        if t.error_message:
            failed_tasks.append({"id": t.id, "title": t.title, "error": t.error_message[:200]})

    # Resolve agent names for busiest
    busiest_agents = []
    if agent_task_counts:
        sorted_agents = sorted(agent_task_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        agent_ids = [a[0] for a in sorted_agents]
        agents_map = {
            a.id: a.name
            for a in db.query(Agent).filter(Agent.id.in_(agent_ids)).all()
        }
        busiest_agents = [
            {"agent": agents_map.get(aid, f"Agent {aid}"), "task_count": count}
            for aid, count in sorted_agents
        ]

    from core.security.surface import widget_turn

    if widget_turn():
        # F155: a widget turn (tasks:read) gets the counts, never who is busiest
        # or what failed and why.
        return {"success": True, "total_tasks": len(all_tasks), "by_status": by_status, "by_priority": by_priority}
    return {
        "success": True,
        "total_tasks": len(all_tasks),
        "by_status": by_status,
        "by_priority": by_priority,
        "busiest_agents": busiest_agents,
        "failed_tasks": failed_tasks[:5],
    }

"""Analytics handlers for PlatformActionExecutor — LLM usage, cost, workspace stats, board summary."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

from modules.tools.discovery.board_waiting import failed_cards, with_whats_waiting
from services.ticket_refs import by_ticket_number

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


# 9 Oct (Auto's wishlist): a call on a subscription plan (a Claude Code session) is booked at
# $0 because the plan pays for it. Auto read the $0 as "free"; each row now says how it was billed.
BILLING_PLAN = "plan"
BILLING_METERED = "metered"
BILLING_MIXED = "plan and metered"
PLAN_NOTE = ("billing 'plan' means the calls ran on a subscription (such as a Claude Code session): "
             "the plan pays, so there is no dollar figure. Never call them free.")


def _cost_group_column(group_by: str):
    from core.models.core import LLMUsage

    if group_by == "agent":
        return LLMUsage.agent_id
    if group_by == "day":
        return func.date(LLMUsage.created_at)
    return LLMUsage.model_id


def _billing(requests: int, plan_requests: int) -> str:
    if plan_requests <= 0:
        return BILLING_METERED
    return BILLING_PLAN if plan_requests >= requests else BILLING_MIXED


def _breakdown_row(row: Any, group_by: str) -> Dict[str, Any]:
    plan_requests = int(row.plan_requests or 0)
    return {
        group_by: str(row.group_key) if row.group_key is not None else "unknown",
        "total_cost": round(float(row.total_cost or 0), 6),
        "input_cost": round(float(row.input_cost or 0), 6),
        "output_cost": round(float(row.output_cost or 0), 6),
        "requests": row.request_count,
        "plan_requests": plan_requests,
        "billing": _billing(row.request_count, plan_requests),
    }


async def get_cost_breakdown(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.llm.usage_tracker import TIER_SUBSCRIPTION
    from core.models.core import LLMUsage

    days = params.get("days", 30)
    group_by = params.get("group_by", "model")
    since = datetime.now(timezone.utc) - timedelta(days=days)
    group_col = _cost_group_column(group_by)

    rows = (
        db.query(
            group_col.label("group_key"),
            func.sum(LLMUsage.total_cost).label("total_cost"),
            func.sum(LLMUsage.input_cost).label("input_cost"),
            func.sum(LLMUsage.output_cost).label("output_cost"),
            func.count(LLMUsage.id).label("request_count"),
            func.count(LLMUsage.id).filter(LLMUsage.tier == TIER_SUBSCRIPTION).label("plan_requests"),
        )
        .filter(
            LLMUsage.workspace_id == workspace_id,
            LLMUsage.created_at >= since,
        )
        .group_by(group_col)
        .order_by(func.sum(LLMUsage.total_cost).desc())
        .all()
    )

    breakdown = [_breakdown_row(row, group_by) for row in rows]
    result = {
        "success": True,
        "period_days": days,
        "group_by": group_by,
        "total_cost": round(sum(float(row.total_cost or 0) for row in rows), 6),
        "plan_requests": sum(r["plan_requests"] for r in breakdown),
        "breakdown": breakdown,
    }
    return {**result, "note": PLAN_NOTE} if result["plan_requests"] else result


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


@with_whats_waiting  # F263: what waits for the owner, as the board's Needs you lists it
@by_ticket_number  # PRD-252 R4: each ticket listed with its number
async def board_snapshot(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Everything a status answer needs, in ONE call.

    Counts by status and priority, the open tickets, what changed recently, and
    what is scheduled — assembled here rather than by the model making six
    calls and stitching them together. Each part is read in its own savepoint:
    a part that fails (a workspace with no scheduled-tasks table) leaves the
    session usable for the rest, Needs you included (F263, night 9b CI).
    """
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
    open_tasks = _snapshot_part(db, "open tasks", _open_tasks, workspace_id, _snapshot_limit(params), visitor)
    snapshot["open_tasks"] = open_tasks
    snapshot["open_task_count"] = len(open_tasks)
    snapshot["recently_finished"] = _snapshot_part(db, "recent activity", _recently_finished, workspace_id)
    if not visitor:
        snapshot["scheduled"] = _snapshot_part(db, "schedule", _scheduled, workspace_id)
    return snapshot


def _snapshot_limit(params: Dict[str, Any]) -> int:
    try:
        return max(1, min(int(params.get("limit")), BOARD_SNAPSHOT_TASK_LIMIT))
    except (TypeError, ValueError):
        return BOARD_SNAPSHOT_TASK_LIMIT


def _snapshot_part(db: Session, label: str, read: Any, *args: Any) -> List[Dict[str, Any]]:
    """One part of the snapshot, read in a savepoint; [] when it can't be read
    (a partial snapshot beats no answer), with the session left usable."""
    try:
        with db.begin_nested():
            return read(db, *args)
    except Exception as e:  # noqa: BLE001 — logged; the rest of the snapshot still answers
        logger.warning("[board_snapshot] %s unavailable: %s", label, e)
        return []


def _open_tasks(db: Session, workspace_id: UUID, limit: int, visitor: bool) -> List[Dict[str, Any]]:
    from sqlalchemy import text as sa_text

    rows = db.execute(sa_text("""
        SELECT bt.id, bt.title, bt.status, bt.priority, bt.updated_at, a.name AS agent_name
        FROM board_tasks bt
        LEFT JOIN agents a ON a.id = bt.assigned_agent_id
        WHERE bt.workspace_id = :ws
          AND bt.status NOT IN ('done', 'cancelled', 'closed')
        ORDER BY bt.updated_at DESC
        LIMIT :limit
    """), {"ws": str(workspace_id), "limit": limit}).fetchall()
    return [{
        "id": r.id, "title": (r.title or "")[:120], "status": r.status,
        **({} if visitor else {"priority": r.priority, "agent": r.agent_name}),
    } for r in rows]


def _recently_finished(db: Session, workspace_id: UUID) -> List[Dict[str, Any]]:
    from sqlalchemy import text as sa_text

    recent = db.execute(sa_text("""
        SELECT id, title, status, completed_at
        FROM board_tasks
        WHERE workspace_id = :ws AND completed_at IS NOT NULL
        ORDER BY completed_at DESC LIMIT :n
    """), {"ws": str(workspace_id), "n": BOARD_SNAPSHOT_RECENT}).fetchall()
    return [{
        "id": r.id, "title": (r.title or "")[:120], "status": r.status,
        "at": r.completed_at.isoformat() if r.completed_at else None,
    } for r in recent]


def _scheduled(db: Session, workspace_id: UUID) -> List[Dict[str, Any]]:
    from sqlalchemy import text as sa_text

    sched = db.execute(sa_text("""
        SELECT id, description, schedule, next_run_at
        FROM agent_scheduled_tasks
        WHERE workspace_id = :ws AND status = 'active'
        ORDER BY next_run_at NULLS LAST LIMIT :n
    """), {"ws": str(workspace_id), "n": BOARD_SNAPSHOT_RECENT}).fetchall()
    return [{
        "id": r.id, "what": (r.description or "")[:100], "schedule": r.schedule,
        "next_run_at": r.next_run_at.isoformat() if r.next_run_at else None,
    } for r in sched]


@with_whats_waiting  # F263: what waits for the owner, as the board's Needs you lists it
async def board_summary(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Get a summary of the task board: counts, busiest agents, the cards failed now
    (F263: by their status, not by an error they once had)."""
    from core.models.core import BoardTask
    from core.models import Agent

    all_tasks = db.query(BoardTask).filter(
        BoardTask.workspace_id == workspace_id,
    ).all()

    # Counts by status
    by_status: Dict[str, int] = {}
    by_priority: Dict[str, int] = {}
    agent_task_counts: Dict[int, int] = {}

    for t in all_tasks:
        by_status[t.status] = by_status.get(t.status, 0) + 1
        by_priority[t.priority] = by_priority.get(t.priority, 0) + 1
        if t.assigned_agent_id:
            agent_task_counts[t.assigned_agent_id] = agent_task_counts.get(t.assigned_agent_id, 0) + 1

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
        "failed_tasks": failed_cards(db, workspace_id, all_tasks),
    }

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

    return {
        "success": True,
        "total_tasks": len(all_tasks),
        "by_status": by_status,
        "by_priority": by_priority,
        "busiest_agents": busiest_agents,
        "failed_tasks": failed_tasks[:5],
    }

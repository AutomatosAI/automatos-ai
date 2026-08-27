"""Report handlers for PlatformActionExecutor (PRD-76)."""

import logging
import re
from typing import Any, Dict, List
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def submit_report(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Submit a report: write file to workspace + insert DB row."""
    from services.report_service import ReportService

    title = params.get("title")
    content = params.get("content")
    report_type = params.get("report_type", "standup")
    status = params.get("status", "ok")

    if not title or not content:
        return {"success": False, "error": "title and content are required"}

    valid_types = {"standup", "research", "incident", "summary", "delivery", "audit", "onboarding"}
    if report_type not in valid_types:
        return {"success": False, "error": f"report_type must be one of: {', '.join(sorted(valid_types))}"}

    valid_statuses = {"ok", "warning", "critical", "info"}
    if status not in valid_statuses:
        return {"success": False, "error": f"status must be one of: {', '.join(sorted(valid_statuses))}"}

    # Optional: validate required sections in content
    required_sections: List[str] = params.get("required_sections", [])
    if required_sections:
        missing = _check_required_sections(content, required_sections)
        if missing:
            return {
                "success": False,
                "error": f"Report missing required sections: {', '.join(missing)}",
                "missing_sections": missing,
            }

    # Resolve agent context -- the calling agent's ID is passed via execution context
    agent_id = params.get("_agent_id")
    agent_name = params.get("_agent_name", "unknown")

    if not agent_id:
        # Fallback: try to find from params
        agent_id = params.get("agent_id")
        if not agent_id:
            return {"success": False, "error": "Could not determine calling agent"}

        from core.models import Agent
        agent = db.query(Agent).filter(
            Agent.id == agent_id,
            Agent.workspace_id == workspace_id,
        ).first()
        if not agent:
            return {"success": False, "error": f"Agent {agent_id} not found in workspace"}
        agent_name = agent.name

    svc = ReportService(db, workspace_id)
    result = await svc.create_report(
        agent_id=agent_id,
        agent_name=agent_name,
        title=title,
        content=content,
        report_type=report_type,
        status=status,
        summary=params.get("summary"),
        metrics=params.get("metrics"),
        attachments=params.get("attachments"),
        heartbeat_result_id=params.get("_heartbeat_result_id"),
        recommendations=params.get("recommendations"),
        action_items=params.get("action_items"),
        linked_task_ids=params.get("linked_task_ids"),
        requires_approval=bool(params.get("requires_approval", False)),
    )

    # PRD-164 S3: the knowledge flywheel inside ReportService.create_report now
    # owns BOTH the RAG ingest and the typed KG pending (with report text +
    # agent attribution, honoring the Q58 per-workspace opt-out). The old
    # direct schedule here sent a bare {"type": "report"} pending the
    # incremental build dropped — deleted with the path that replaced it.

    return result


async def acknowledge_report(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Wave 3 — mark a report as actioned. Stamps acknowledged_by/at."""
    report_id = params.get("report_id")
    if not report_id:
        return {"success": False, "error": "report_id is required"}

    user_id = params.get("user_id")
    try:
        result = db.execute(
            text(
                """
                UPDATE agent_reports
                   SET acknowledged_by = COALESCE(:user_id, acknowledged_by),
                       acknowledged_at = NOW(),
                       updated_at      = NOW()
                 WHERE id = :report_id
                   AND workspace_id = :workspace_id
                 RETURNING id
                """
            ),
            {
                "report_id": report_id,
                "workspace_id": str(workspace_id),
                "user_id": user_id,
            },
        ).fetchone()
        if result is None:
            return {"success": False, "error": "report not found in this workspace"}
        db.commit()
        return {"success": True, "data": {"report_id": str(result[0])}}
    except Exception as exc:
        logger.error("[acknowledge_report] failed: %s", exc, exc_info=True)
        db.rollback()
        return {"success": False, "error": str(exc)}


async def link_report_to_task(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Wave 3 — append a task_id to a report's linked_task_ids JSONB array."""
    report_id = params.get("report_id")
    task_id = params.get("task_id")
    if not report_id or task_id is None:
        return {"success": False, "error": "report_id and task_id are required"}

    try:
        task_id_int = int(task_id)
    except (TypeError, ValueError):
        return {"success": False, "error": "task_id must be an integer"}

    try:
        result = db.execute(
            text(
                """
                UPDATE agent_reports
                   SET linked_task_ids = COALESCE(linked_task_ids, '[]'::jsonb)
                                         || to_jsonb(:task_id::int),
                       updated_at = NOW()
                 WHERE id = :report_id
                   AND workspace_id = :workspace_id
                   AND NOT (linked_task_ids @> to_jsonb(:task_id::int))
                 RETURNING id, linked_task_ids
                """
            ),
            {
                "report_id": report_id,
                "workspace_id": str(workspace_id),
                "task_id": task_id_int,
            },
        ).fetchone()
        if result is None:
            existing = db.execute(
                text(
                    "SELECT linked_task_ids FROM agent_reports "
                    "WHERE id = :id AND workspace_id = :ws"
                ),
                {"id": report_id, "ws": str(workspace_id)},
            ).fetchone()
            if existing is None:
                return {"success": False, "error": "report not found in this workspace"}
            return {
                "success": True,
                "data": {
                    "report_id": report_id,
                    "linked_task_ids": existing[0],
                    "already_linked": True,
                },
            }
        db.commit()
        return {
            "success": True,
            "data": {"report_id": str(result[0]), "linked_task_ids": result[1]},
        }
    except Exception as exc:
        logger.error("[link_report_to_task] failed: %s", exc, exc_info=True)
        db.rollback()
        return {"success": False, "error": str(exc)}


async def get_latest_report(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Get the most recent report from a specific agent."""
    from services.report_service import ReportService

    agent_name = params.get("agent_name")
    agent_id = params.get("agent_id")
    report_type = params.get("report_type")

    if not agent_name and not agent_id:
        return {"success": False, "error": "Provide agent_name or agent_id"}

    svc = ReportService(db, workspace_id)
    return await svc.get_latest_report(
        agent_name=agent_name,
        agent_id=agent_id,
        report_type=report_type,
    )


async def browse_reports(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """List reports across the workspace with model/trigger/status filters.

    Joins onto the metrics jsonb so callers can filter on `trigger` and `model`
    (set by heartbeat / task / playbook auto-reports). Returns a compact rollup
    with per-row cost/duration/model so the system admin agent can analyse.
    """
    from sqlalchemy import text

    period = params.get("period", "7d")
    period_map = {
        "1d": "INTERVAL '1 day'",
        "7d": "INTERVAL '7 days'",
        "30d": "INTERVAL '30 days'",
        "90d": "INTERVAL '90 days'",
    }
    interval_sql = period_map.get(period, "INTERVAL '7 days'")
    has_period = period != "all"

    try:
        limit = max(1, min(int(params.get("limit", 50)), 200))
    except (TypeError, ValueError):
        limit = 50

    conditions = ["r.workspace_id = :workspace_id"]
    sql_params: Dict[str, Any] = {"workspace_id": str(workspace_id)}

    if has_period:
        conditions.append(f"r.created_at >= NOW() - {interval_sql}")

    if params.get("agent_id"):
        conditions.append("r.agent_id = :agent_id")
        sql_params["agent_id"] = params["agent_id"]

    if params.get("agent_name"):
        conditions.append("r.agent_name ILIKE :agent_name")
        sql_params["agent_name"] = f"%{params['agent_name']}%"

    if params.get("report_type"):
        conditions.append("r.report_type = :report_type")
        sql_params["report_type"] = params["report_type"]

    if params.get("status"):
        conditions.append("r.status = :status")
        sql_params["status"] = params["status"]

    if params.get("trigger"):
        conditions.append("r.metrics->>'trigger' = :trigger")
        sql_params["trigger"] = params["trigger"]

    if params.get("model"):
        conditions.append("r.metrics->>'model' = :model")
        sql_params["model"] = params["model"]

    where = " AND ".join(conditions)

    try:
        rows = db.execute(
            text(f"""
                SELECT
                    r.id,
                    r.agent_id,
                    r.agent_name,
                    r.report_type,
                    r.status,
                    r.title,
                    r.summary,
                    r.created_at,
                    r.metrics
                FROM agent_reports r
                WHERE {where}
                ORDER BY r.created_at DESC
                LIMIT :limit
            """),
            {**sql_params, "limit": limit},
        ).fetchall()
    except Exception as e:
        logger.error("[browse_reports] query failed: %s", e, exc_info=True)
        return {"success": False, "error": f"query failed: {e}"}

    items: list = []
    total_cost = 0.0
    total_tokens = 0
    total_duration_ms = 0
    by_model: Dict[str, Dict[str, Any]] = {}
    by_trigger: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        m = row.metrics if isinstance(row.metrics, dict) else {}
        cost = float(m.get("cost_usd") or 0)
        tokens = int(m.get("tokens_used") or 0)
        duration_ms = int(m.get("duration_ms") or 0)
        model = m.get("model") or "unknown"
        trigger = m.get("trigger") or row.report_type

        items.append({
            "id": str(row.id),
            "agent_id": row.agent_id,
            "agent_name": row.agent_name,
            "report_type": row.report_type,
            "status": row.status,
            "title": row.title,
            "summary": row.summary,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "model": model,
            "trigger": trigger,
            "tokens_used": tokens,
            "cost_usd": cost,
            "duration_ms": duration_ms,
            "llm_calls": int(m.get("llm_calls") or 0),
        })

        total_cost += cost
        total_tokens += tokens
        total_duration_ms += duration_ms

        bm = by_model.setdefault(model, {"reports": 0, "tokens": 0, "cost_usd": 0.0, "duration_ms": 0})
        bm["reports"] += 1
        bm["tokens"] += tokens
        bm["cost_usd"] += cost
        bm["duration_ms"] += duration_ms

        bt = by_trigger.setdefault(trigger, {"reports": 0, "tokens": 0, "cost_usd": 0.0, "duration_ms": 0})
        bt["reports"] += 1
        bt["tokens"] += tokens
        bt["cost_usd"] += cost
        bt["duration_ms"] += duration_ms

    return {
        "success": True,
        "period": period,
        "count": len(items),
        "totals": {
            "cost_usd": round(total_cost, 6),
            "tokens": total_tokens,
            "duration_ms": total_duration_ms,
        },
        "by_model": by_model,
        "by_trigger": by_trigger,
        "items": items,
    }


def _check_required_sections(content: str, required_sections: List[str]) -> List[str]:
    """Check that content contains markdown headers matching each required section.

    Matches ## or ### headers case-insensitively. Returns list of missing section names.
    """
    content_lower = content.lower()
    missing = []
    for section in required_sections:
        # Match markdown headers: ## Section Name or ### Section Name
        pattern = rf"^#{2,3}\s+{re.escape(section.lower())}"
        if not re.search(pattern, content_lower, re.MULTILINE):
            missing.append(section)
    return missing

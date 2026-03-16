"""Report handlers for PlatformActionExecutor (PRD-76)."""

import logging
from typing import Any, Dict
from uuid import UUID

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

    valid_types = {"standup", "research", "incident", "summary", "delivery", "audit"}
    if report_type not in valid_types:
        return {"success": False, "error": f"report_type must be one of: {', '.join(sorted(valid_types))}"}

    valid_statuses = {"ok", "warning", "critical", "info"}
    if status not in valid_statuses:
        return {"success": False, "error": f"status must be one of: {', '.join(sorted(valid_statuses))}"}

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
    return await svc.create_report(
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
    )


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

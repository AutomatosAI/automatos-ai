"""Report handlers for PlatformActionExecutor (PRD-76)."""

import logging
import re
from typing import Any, Dict, List
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
    )

    # PRD-126: Trigger knowledge graph update on report submission
    try:
        from modules.knowledge.graph_service import get_graph_service
        get_graph_service().schedule_incremental_update(
            int(workspace_id),
            [{"type": "report", "path": title, "id": result.get("report_id")}],
        )
    except Exception:
        logger.debug("Graph update skipped — service not available")

    return result


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

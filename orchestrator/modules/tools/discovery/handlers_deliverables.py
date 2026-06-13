"""Deliverable handlers for PlatformActionExecutor (PRD-164 S3).

Thin wrappers over the existing ``DeliverableService`` (v_workspace_outputs)
— list/get only; writes stay with the native services (PRD-133b).
"""

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_MAX_LIST_LIMIT = 50
_DEFAULT_LIST_LIMIT = 20


def _compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Trim a deliverable row to the fields an agent needs in-context."""
    return {
        "id": row.get("id"),
        "title": row.get("title"),
        "artifact_type": row.get("artifact_type"),
        "source_type": row.get("source_type"),
        "source_id": row.get("source_id"),
        "agent_id": row.get("agent_id"),
        "agent_name": row.get("agent_name"),
        "file_path": row.get("file_path"),
        "file_type": row.get("file_type"),
        "preview_url": row.get("preview_url"),
        "status": row.get("status"),
        "summary": row.get("summary"),
        "created_at": row.get("created_at"),
    }


async def list_deliverables(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """List workspace deliverables. ``mine=true`` scopes to the calling agent
    (``_agent_id`` is injected by the execution context, same as reports)."""
    from services.deliverable_service import DeliverableService

    agent_id = params.get("agent_id")
    if params.get("mine"):
        caller_id = params.get("_agent_id")
        if not caller_id:
            return {
                "success": False,
                "error": "mine=true but the calling agent could not be determined",
            }
        agent_id = caller_id

    if agent_id is not None:
        try:
            agent_id = int(agent_id)
        except (TypeError, ValueError):
            return {"success": False, "error": "agent_id must be an integer"}

    try:
        limit = max(1, min(int(params.get("limit") or _DEFAULT_LIST_LIMIT), _MAX_LIST_LIMIT))
    except (TypeError, ValueError):
        limit = _DEFAULT_LIST_LIMIT

    svc = DeliverableService(db, workspace_id)
    result = svc.list_deliverables(
        artifact_type=params.get("artifact_type"),
        source_type=params.get("source_type"),
        source_id=params.get("source_id"),
        agent_id=agent_id,
        search=params.get("search"),
        limit=limit,
    )
    if not result.get("success"):
        return {"success": False, "error": result.get("error", "list failed")}

    return {
        "success": True,
        "count": len(result.get("deliverables", [])),
        "total": result.get("total", 0),
        "deliverables": [_compact_row(r) for r in result.get("deliverables", [])],
    }


async def get_deliverable(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Fetch one deliverable; optionally include its text content."""
    from services.deliverable_service import DeliverableService

    deliverable_id = params.get("deliverable_id")
    if not deliverable_id or not str(deliverable_id).strip():
        return {"success": False, "error": "Missing required parameter: deliverable_id"}

    include_content = bool(params.get("include_content", False))

    svc = DeliverableService(db, workspace_id)
    result = await svc.get_deliverable(
        str(deliverable_id), include_content=include_content
    )
    if not result.get("success"):
        return {"success": False, "error": result.get("error", "Deliverable not found")}

    data = result.get("deliverable") or {}
    payload = _compact_row(data)
    if include_content:
        payload["content"] = data.get("content")
        payload["content_truncated"] = data.get("content_truncated", False)
        if data.get("content_error"):
            payload["content_error"] = data["content_error"]
    return {"success": True, "deliverable": payload}

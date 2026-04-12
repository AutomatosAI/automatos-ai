"""Resolve agent public_id (UUID) or legacy integer id to internal agent id.

Used by API endpoints that accept agent identifiers from external sources
(widgets, frontend, public API). Validates workspace ownership.
"""
import logging
from uuid import UUID as PyUUID

from fastapi import HTTPException
from sqlalchemy.orm import Session

from core.models.core import Agent

logger = logging.getLogger(__name__)


def resolve_agent_id(
    db: Session,
    agent_ref: "int | str",
    workspace_id: "PyUUID | str | None" = None,
) -> int:
    """Resolve a public_id (UUID string) or legacy integer id to the internal agent.id.

    Args:
        db: Database session
        agent_ref: Either a UUID string (public_id) or an integer (legacy id)
        workspace_id: If provided, validates the agent belongs to this workspace

    Returns:
        Internal integer agent id

    Raises:
        HTTPException 404 if agent not found or doesn't belong to workspace
    """
    agent_ref_str = str(agent_ref).strip()

    # Try UUID first
    try:
        uuid_val = PyUUID(agent_ref_str)
        query = db.query(Agent.id, Agent.workspace_id).filter(Agent.public_id == uuid_val)
        row = query.first()
        if row:
            if workspace_id and row.workspace_id and str(row.workspace_id) != str(workspace_id):
                logger.warning(
                    "Agent public_id=%s belongs to workspace %s, caller is %s",
                    agent_ref_str, row.workspace_id, workspace_id,
                )
                raise HTTPException(status_code=404, detail="Agent not found")
            return row.id
    except (ValueError, AttributeError):
        pass

    # Try integer (backward compat)
    try:
        int_id = int(agent_ref_str)
        query = db.query(Agent.id, Agent.workspace_id).filter(Agent.id == int_id)
        row = query.first()
        if row:
            if workspace_id and row.workspace_id and str(row.workspace_id) != str(workspace_id):
                logger.warning(
                    "Agent id=%s belongs to workspace %s, caller is %s",
                    int_id, row.workspace_id, workspace_id,
                )
                raise HTTPException(status_code=404, detail="Agent not found")
            return row.id
    except (ValueError, TypeError):
        pass

    raise HTTPException(status_code=404, detail="Agent not found")

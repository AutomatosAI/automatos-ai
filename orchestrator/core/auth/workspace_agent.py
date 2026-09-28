"""PRD-251 P251W1-RVW-1: a workspace route changes only the caller's own agents.

A marketplace or global agent has no workspace, and every install copies it into
the installing workspace: its persona, its skills and its plugin assignments go
onto the clone (``clone_agent_to_workspace``, ``cascade_agent_dependencies``). A
change a workspace made to one would run in every workspace that installs it
afterwards. The check these routes had, ``agent.workspace_id and agent.workspace_id
!= ctx.workspace_id``, skipped exactly those rows.

Pure (no request context of its own), so a route calls it in its body or wraps it
in a dependency, and it never imports the hybrid auth.
"""
from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from fastapi import HTTPException, status

AGENT_NOT_FOUND = "Agent not found"


def workspace_agent_or_404(db: Any, agent_id: int, workspace_id: Optional[UUID]):
    """The agent ``agent_id`` when it belongs to ``workspace_id``; a 404 for any other,
    a marketplace or global agent included.

    The workspace is checked before the query: ``Agent.workspace_id == None`` renders
    ``IS NULL`` and would match every marketplace row.
    """
    from core.models.core import Agent

    agent = None
    if workspace_id is not None:
        agent = db.query(Agent).filter(Agent.id == agent_id, Agent.workspace_id == workspace_id).first()
    if agent is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=AGENT_NOT_FOUND)
    return agent

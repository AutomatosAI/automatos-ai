"""F149 — an agent or playbook runs only for a message of its own workspace.

Routing rules, webhook overrides and cached routing decisions carry bare ids.
Before a channel, webhook or workflow dispatch runs one — and before a rule,
channel or API key stores one — it must belong to that workspace. An agent or
playbook with no workspace (a template) belongs to none. Both checks fail
closed: a missing or unreadable row is not in the workspace.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

_AGENT_SQL = "SELECT 1 FROM agents WHERE id = :id AND workspace_id = CAST(:ws AS uuid) LIMIT 1"
_PLAYBOOK_SQL = "SELECT 1 FROM workflow_recipes WHERE id = :id AND workspace_id = CAST(:ws AS uuid) LIMIT 1"


def _row_in_workspace(db: Any, sql: str, row_id: Any, workspace_id: Any, what: str) -> bool:
    if workspace_id is None or isinstance(row_id, bool):
        return False
    try:
        row_id = int(row_id)
    except (TypeError, ValueError):
        return False
    try:
        with db.begin_nested():
            return db.execute(text(sql), {"id": row_id, "ws": str(workspace_id)}).first() is not None
    except Exception:  # noqa: BLE001 — an unreadable row is not in the workspace
        logger.warning("[workspace-scope] %s %s lookup failed in %s", what, row_id, workspace_id, exc_info=True)
        return False


def agent_in_workspace(db: Any, agent_id: Any, workspace_id: Any) -> bool:
    """The agent exists and belongs to this workspace."""
    return _row_in_workspace(db, _AGENT_SQL, agent_id, workspace_id, "agent")


def playbook_in_workspace(db: Any, playbook_id: Any, workspace_id: Any) -> bool:
    """The playbook exists and belongs to this workspace."""
    return _row_in_workspace(db, _PLAYBOOK_SQL, playbook_id, workspace_id, "playbook")


def routing_target_error(db: Any, workspace_id: Any, agent_id: Any, playbook_id: Any) -> Optional[str]:
    """Why a routing rule's target is refused, or None when each one it names is
    this workspace's."""
    if agent_id is not None and not agent_in_workspace(db, agent_id, workspace_id):
        return "target_agent_id is not an agent of this workspace"
    if playbook_id is not None and not playbook_in_workspace(db, playbook_id, workspace_id):
        return "target_workflow_id is not a playbook of this workspace"
    return None

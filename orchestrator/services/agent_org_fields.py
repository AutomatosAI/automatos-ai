"""#831 — one validator for an agent's org fields (``team`` / ``reports_to_id``).

Before this, the REST API (``api/agents.py``) did not expose or accept these
fields at all, and Auto's tool handler (``modules/tools/discovery/
handlers_agents.py``) wrote them with no validation: any string became the
team, and ``reports_to_id`` was cast straight to ``int`` with no check that
the manager existed, lived in the same workspace, or that the edit didn't
point an agent at itself or its own descendant. Both paths now call this
module so the rule lives in one place.

``team`` is a free-text label (distinct from the ``Teams`` table that scopes
document access via ``core.team_access`` — same name, different concept): the
only rule is trim-and-clear-if-blank, the same convention ``job_title``
already uses on this model.
"""
from __future__ import annotations

from typing import Optional

# A real org chart never nests this deep; past it, treat the chain as broken
# rather than loop forever (same fail-closed stance as
# core.security.hierarchy_permissions._in_subtree).
MAX_CHAIN_DEPTH = 50


class AgentOrgFieldError(ValueError):
    """A team/manager value that fails validation. ``str(exc)`` is user-facing."""


def normalized_team_or_none(team: Optional[str]) -> Optional[str]:
    """Trim ``team``; ``None`` or blank clears it (mirrors ``job_title``)."""
    if team is None:
        return None
    trimmed = team.strip()
    return trimmed or None


def validate_manager(
    db,
    workspace_id,
    *,
    agent_id: Optional[int],
    reports_to_id: Optional[int],
) -> Optional[int]:
    """Validate a ``reports_to_id`` edit; return the value to store.

    ``agent_id`` is the agent being written (``None`` for a not-yet-created
    agent, where self-reference and cycles are impossible). A falsy
    ``reports_to_id`` (``None``/``0``) clears the manager. Raises
    :class:`AgentOrgFieldError` with a caller-facing message on:
    self-reporting, a manager outside the workspace (or that doesn't exist),
    or a manager whose own chain already leads back to ``agent_id``.
    """
    manager_id = _as_agent_id(reports_to_id)
    if not manager_id:
        return None
    if agent_id is not None and manager_id == agent_id:
        raise AgentOrgFieldError("An agent cannot report to itself.")
    if not _agent_exists_in_workspace(db, workspace_id, manager_id):
        raise AgentOrgFieldError(f"Manager agent {manager_id} was not found in this workspace.")
    if agent_id is not None and _chain_reaches(db, workspace_id, start=manager_id, target=agent_id):
        raise AgentOrgFieldError("That manager would create a reporting cycle.")
    return manager_id


def _as_agent_id(value: Optional[int]) -> Optional[int]:
    if value in (None, "", 0, "0"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AgentOrgFieldError("reports_to_id must be an integer agent id.") from exc


def _agent_exists_in_workspace(db, workspace_id, agent_id: int) -> bool:
    from core.models import Agent

    return (
        db.query(Agent.id)
        .filter(Agent.id == agent_id, Agent.workspace_id == workspace_id)
        .first()
        is not None
    )


def _chain_reaches(db, workspace_id, *, start: int, target: int) -> bool:
    """Walk ``reports_to_id`` from ``start`` toward the root; True if ``target`` is on it."""
    from core.models import Agent

    cursor: Optional[int] = start
    seen = set()
    for _ in range(MAX_CHAIN_DEPTH):
        if cursor == target:
            return True
        if cursor is None or cursor in seen:
            return False
        seen.add(cursor)
        cursor = (
            db.query(Agent.reports_to_id)
            .filter(Agent.id == cursor, Agent.workspace_id == workspace_id)
            .scalar()
        )
    return True  # depth cap reached — treat as a cycle rather than loop forever

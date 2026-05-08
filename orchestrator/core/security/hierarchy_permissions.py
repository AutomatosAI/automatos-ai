"""Hierarchy permissions — PRD-140 Phase 1.

Single helper that decides whether an actor agent may modify a target
resource based on the org chart (``Agent.reports_to_id``).

Defaults are conservative:

  - System agents (``is_system_agent=True``) bypass — Auto / CTO are the
    workspace authority.
  - Skills are out of scope at every tier — they affect behaviour beyond
    one agent and always escalate to Auto/ATLAS (PRD-140 §5.3, Q11).
  - All other resource types require the target to live inside the
    actor's reports-to subtree.
  - Default-deny on anything not explicitly allowed.

The helper is consulted from a single dispatch site
(``PlatformActionExecutor.execute``) so individual action handlers don't
each have to opt in. A CI audit-grep gate enforces that any new mutating
``platform_*`` action passes through that dispatcher (no bypass).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# Resource types we know how to scope. Anything else returns "not_scoped"
# and the dispatcher falls through to the existing admin/rate-limit gates.
TARGET_AGENT = "agent"
TARGET_HEARTBEAT = "heartbeat"
TARGET_PLAYBOOK = "playbook"
TARGET_TASK = "task"
TARGET_SKILL = "skill"
TARGET_TOOL_ASSIGNMENT = "tool_assignment"

KNOWN_TARGETS = frozenset({
    TARGET_AGENT,
    TARGET_HEARTBEAT,
    TARGET_PLAYBOOK,
    TARGET_TASK,
    TARGET_SKILL,
    TARGET_TOOL_ASSIGNMENT,
})


@dataclass(frozen=True)
class PermissionDecision:
    """Result of a hierarchy permission check.

    ``allowed`` is the only field most callers need. ``reason`` carries a
    short human-readable explanation suitable for a tool result and for
    audit / log output. ``escalation_target`` names where the actor
    should route the request when denied (e.g. Auto's agent id) so the
    caller can produce a useful error or queue the change for review.
    """

    allowed: bool
    reason: str
    escalation_target: Optional[str] = None  # "auto", "human", or None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_system_agent(db: Session, agent_id: int) -> Optional[bool]:
    """Return True/False for system flag, or None when the agent is missing."""
    row = db.execute(
        text("SELECT is_system_agent FROM agents WHERE id = :id"),
        {"id": agent_id},
    ).first()
    if row is None:
        return None
    return bool(row[0])


def _subtree_ids(db: Session, root_agent_id: int) -> set[int]:
    """Return the set of agent IDs that report (transitively) to ``root_agent_id``.

    The root itself is included so an actor can act on its own row when
    that's the relevant change (e.g. an agent updating its own job title
    is still constrained to itself).
    """
    rows = db.execute(
        text(
            """
            WITH RECURSIVE subtree AS (
                SELECT id FROM agents WHERE id = :root
                UNION ALL
                SELECT a.id
                FROM agents a
                JOIN subtree s ON a.reports_to_id = s.id
            )
            SELECT id FROM subtree
            """
        ),
        {"root": root_agent_id},
    ).fetchall()
    return {row[0] for row in rows}


def _agent_owner(db: Session, target_kind: str, target_id) -> Optional[int]:
    """Resolve a target's owning agent_id for hierarchy checks.

    Returns ``None`` when the target either doesn't exist or has no agent
    owner — caller should treat as not-scoped (default-deny in this module,
    falls through to the existing gates above).
    """
    if target_id is None:
        return None
    if target_kind == TARGET_AGENT:
        return int(target_id)
    if target_kind == TARGET_HEARTBEAT:
        # heartbeat_results is a wide log; the durable per-agent config
        # lives on the agent row itself. The only valid target_id for a
        # heartbeat-config change is the owning agent's id.
        return int(target_id)
    if target_kind == TARGET_TASK:
        row = db.execute(
            text("SELECT assigned_agent_id FROM board_tasks WHERE id = :id"),
            {"id": int(target_id)},
        ).first()
        return int(row[0]) if row and row[0] is not None else None
    if target_kind == TARGET_PLAYBOOK:
        # workflow_recipes (PRD-71 canonical name) — owner column varies
        # historically. Try the modern columns first; fall back to None
        # so the caller queues the change rather than auto-applying.
        try:
            row = db.execute(
                text(
                    "SELECT created_by_agent_id FROM workflow_recipes WHERE id = :id"
                ),
                {"id": int(target_id)},
            ).first()
            if row and row[0] is not None:
                return int(row[0])
        except Exception:
            pass
        return None
    if target_kind == TARGET_TOOL_ASSIGNMENT:
        # Tool assignments target an agent; target_id should be the agent's id.
        return int(target_id)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def can_actor_modify(
    db: Session,
    actor_agent_id: Optional[int],
    target_type: str,
    target_id,
    change_type: str = "update",
) -> PermissionDecision:
    """Decide whether ``actor_agent_id`` may perform ``change_type`` on the target.

    Args:
        db: Active SQLAlchemy session — used for the recursive subtree
            query and the system-agent lookup.
        actor_agent_id: The calling agent's id (``params._agent_id`` from
            the tool dispatcher). ``None`` means no agent context — e.g.
            a system call from the heartbeat scheduler — and is treated
            as system-level (allowed). Callers that want stricter handling
            should resolve the agent before calling.
        target_type: One of the TARGET_* constants. Unknown values return
            ``allowed=False, reason="unknown_target_type"`` so callers
            can fall back to the existing admin/rate-limit gates.
        target_id: Identifier for the target. Type varies by target_type
            (int agent_id, int task_id, etc.). May be ``None`` for create
            actions where the target doesn't exist yet — those default
            to the workspace-level gates rather than this helper.
        change_type: ``"create" | "update" | "delete" | "assign"``. Used
            for logging / audit; doesn't change the rule today, but the
            field is here so a future tier can vary the answer per change.

    Returns:
        PermissionDecision. ``allowed=True`` means the dispatcher should
        proceed; ``allowed=False`` means the dispatcher should return a
        permission_denied result and (optionally) queue the change to
        Auto via ``escalation_target``.
    """
    # No actor → system call (heartbeat scheduler, migration, console).
    # The platform-level admin / rate-limit gates already cover these.
    if actor_agent_id is None:
        return PermissionDecision(
            allowed=True,
            reason="no_actor_context_assumed_system",
        )

    if target_type not in KNOWN_TARGETS:
        return PermissionDecision(
            allowed=False,
            reason=f"unknown_target_type:{target_type}",
        )

    # Skills always escalate — never team-lead controlled (PRD-140 §5.3, Q11).
    # System agents still bypass below, so Auto / CTO can edit skills.
    if target_type == TARGET_SKILL:
        if _is_system_agent(db, actor_agent_id):
            return PermissionDecision(
                allowed=True,
                reason="system_agent_bypass",
            )
        return PermissionDecision(
            allowed=False,
            reason="skill_changes_route_to_auto",
            escalation_target="auto",
        )

    # System agents (Auto / CTO) bypass for every other target type too.
    is_system = _is_system_agent(db, actor_agent_id)
    if is_system is None:
        return PermissionDecision(
            allowed=False,
            reason="actor_agent_not_found",
        )
    if is_system:
        return PermissionDecision(
            allowed=True,
            reason="system_agent_bypass",
        )

    # Everything below is for non-system actors acting on a specific target.
    owner = _agent_owner(db, target_type, target_id)
    if owner is None:
        # Couldn't resolve an owning agent — playbook with no owner column,
        # task with no assignee, missing row. Queue rather than apply so
        # Auto can decide.
        return PermissionDecision(
            allowed=False,
            reason="target_owner_unresolved_route_to_auto",
            escalation_target="auto",
        )

    subtree = _subtree_ids(db, actor_agent_id)
    if owner in subtree:
        return PermissionDecision(
            allowed=True,
            reason="target_in_actor_subtree",
        )

    return PermissionDecision(
        allowed=False,
        reason="target_outside_actor_subtree",
        escalation_target="auto",
    )

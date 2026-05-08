"""PRD-140 — hierarchy permission helper.

Single chokepoint that says yes/no/why for "actor X wants to make change C
on target T". Designed to be called from service-layer mutations (the only
correct enforcement point) and from tool wrappers as a defence in depth.

Defaults are conservative:

  * Default DENY on broken state — missing manager, cycle in reports_to_id
    chain, unknown target, deleted/inactive actor.
  * Default DENY on ambiguity — if the helper cannot decide, the caller
    must NOT proceed. No "fail open" path.

System-agent bypass is **narrowed**: not every ``is_system_agent=True``
record gets a free pass. Only the specific named system actors registered
in :data:`SYSTEM_BYPASS_ALLOWLIST` may bypass, and every bypass is recorded
to ``permission_bypass_log`` (see :mod:`core.security.bypass_audit`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Set
from uuid import UUID

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- constants


# Narrowed bypass allowlist — these specific named actors may bypass the
# hierarchy. Adding to this list is a security-sensitive change. ``Auto``
# is the workspace orchestrator (slug auto-{workspace_id}); the others are
# platform service actors.
SYSTEM_BYPASS_ALLOWLIST: frozenset[str] = frozenset(
    {
        "Auto",                   # workspace orchestrator
        "HARNESS",                # weekly self-optimisation service actor
        "platform-admin",         # explicit admin agent
        "platform-system",        # internal service actor
    }
)

# Maximum subtree depth before we treat the chain as broken. Real org
# trees rarely exceed 6 levels; anything deeper is almost certainly a
# cycle the visited-set didn't catch (defence in depth).
MAX_SUBTREE_DEPTH: int = 16


# ----------------------------------------------------------------- types


@dataclass(frozen=True)
class PermissionDecision:
    """Result of a permission check. Always inspect ``allowed`` first."""

    allowed: bool
    reason: str
    bypass: bool = False
    bypass_kind: Optional[str] = None  # 'system_actor' | 'workspace_owner' | None
    actor_agent_id: Optional[int] = None
    actor_name: Optional[str] = None
    target_type: Optional[str] = None
    target_id: Optional[str] = None
    change_type: Optional[str] = None
    source: Optional[str] = None


# ----------------------------------------------------------------- main API


def can_actor_modify(
    db,
    *,
    actor_agent_id: Optional[int],
    target_type: str,
    target_id: Optional[str],
    change_type: str,
    workspace_id: UUID | str,
    source: Optional[str] = None,
) -> PermissionDecision:
    """Decide whether ``actor`` may apply ``change_type`` to ``target``.

    Parameters
    ----------
    db                : SQLAlchemy session.
    actor_agent_id    : ID of the agent attempting the change. ``None`` is
                        treated as an anonymous caller and almost always
                        denied (only the explicit platform-system flow may
                        pass with ``None``).
    target_type       : 'agent' | 'playbook' | 'task' | 'skill' | 'tool' | ...
    target_id         : ID of the target row (string for UUID tables, int as
                        string for integer PKs).
    change_type       : Free-form short verb e.g. 'update', 'delete',
                        'assign_skill', 'configure_heartbeat'.
    workspace_id      : Workspace scope — every check is workspace-bound.
                        Cross-workspace mutations are always denied here.
    source            : Optional caller identifier for audit ('platform_tool',
                        'api/agents', 'heartbeat_tick', etc.).

    Returns
    -------
    PermissionDecision (always non-None — never raises for permission state).
    """
    from core.models import Agent

    workspace_id_str = str(workspace_id)

    # --- Anonymous / missing actor: always deny (no ambient authority) ---
    if actor_agent_id is None:
        return PermissionDecision(
            allowed=False,
            reason="anonymous_actor",
            actor_agent_id=None,
            target_type=target_type,
            target_id=str(target_id) if target_id is not None else None,
            change_type=change_type,
            source=source,
        )

    actor = db.query(Agent).filter(Agent.id == actor_agent_id).first()
    if actor is None:
        return PermissionDecision(
            allowed=False,
            reason="actor_not_found",
            actor_agent_id=actor_agent_id,
            target_type=target_type,
            target_id=str(target_id) if target_id is not None else None,
            change_type=change_type,
            source=source,
        )

    # --- Workspace boundary check (cross-tenant attack surface) ---
    if str(actor.workspace_id) != workspace_id_str:
        return PermissionDecision(
            allowed=False,
            reason="cross_workspace_actor",
            actor_agent_id=actor.id,
            actor_name=actor.name,
            target_type=target_type,
            target_id=str(target_id) if target_id is not None else None,
            change_type=change_type,
            source=source,
        )

    # --- Inactive actor cannot mutate ---
    if (actor.status or "").lower() != "active":
        return PermissionDecision(
            allowed=False,
            reason="actor_inactive",
            actor_agent_id=actor.id,
            actor_name=actor.name,
            target_type=target_type,
            target_id=str(target_id) if target_id is not None else None,
            change_type=change_type,
            source=source,
        )

    # --- Narrowed system-agent bypass ---
    # Bypass is only granted when BOTH conditions hold: ``is_system_agent``
    # is set AND the agent's name is on the explicit allowlist. The flag
    # alone is not a golden key.
    if (
        bool(getattr(actor, "is_system_agent", False))
        and (actor.name or "") in SYSTEM_BYPASS_ALLOWLIST
    ):
        return PermissionDecision(
            allowed=True,
            reason="system_actor_bypass",
            bypass=True,
            bypass_kind="system_actor",
            actor_agent_id=actor.id,
            actor_name=actor.name,
            target_type=target_type,
            target_id=str(target_id) if target_id is not None else None,
            change_type=change_type,
            source=source,
        )

    # --- Subtree authority (team-lead path) ---
    # The actor may modify any agent in their reports-to subtree, plus
    # workspace-owned playbooks and board tasks they (or their reports)
    # own. Cross-team and shared-resource mutations are denied at this layer.
    if target_type == "agent":
        if target_id is None:
            return _deny(actor, target_type, target_id, change_type, source, "missing_target")
        try:
            target_int = int(target_id)
        except (TypeError, ValueError):
            return _deny(actor, target_type, target_id, change_type, source, "invalid_target_id")

        if target_int == actor.id:
            return PermissionDecision(
                allowed=True,
                reason="self_mutation",
                actor_agent_id=actor.id,
                actor_name=actor.name,
                target_type=target_type,
                target_id=str(target_id),
                change_type=change_type,
                source=source,
            )

        target = db.query(Agent).filter(Agent.id == target_int).first()
        if target is None:
            return _deny(actor, target_type, target_id, change_type, source, "target_not_found")
        if str(target.workspace_id) != workspace_id_str:
            return _deny(actor, target_type, target_id, change_type, source, "cross_workspace_target")

        in_subtree = _agent_in_subtree(db, root_id=actor.id, candidate_id=target_int)
        if in_subtree is None:
            # Cycle detected or chain too deep — default DENY.
            return _deny(actor, target_type, target_id, change_type, source, "broken_hierarchy")
        if in_subtree:
            return PermissionDecision(
                allowed=True,
                reason="subtree_authority",
                actor_agent_id=actor.id,
                actor_name=actor.name,
                target_type=target_type,
                target_id=str(target_id),
                change_type=change_type,
                source=source,
            )
        return _deny(actor, target_type, target_id, change_type, source, "out_of_subtree")

    # Other target_types fall through to default-deny here — wiring for
    # playbook / skill / tool / task is added per resource as the chokepoint
    # rolls out (Ticket 2). Default-deny means a new tool added before its
    # permission rules are coded gets blocked, not silently allowed.
    return _deny(
        actor,
        target_type,
        target_id,
        change_type,
        source,
        f"unsupported_target_type:{target_type}",
    )


# ----------------------------------------------------------------- helpers


def subtree_of(db, root_agent_id: int) -> Set[int]:
    """Return every agent id reachable via reports_to_id from ``root``.

    Cycle-safe (visited set + depth cap). Returns at minimum ``{root_id}``
    even when the agent has no direct reports. Returns an empty set when
    the chain is broken or the depth cap is exceeded.
    """
    from core.models import Agent

    out: Set[int] = {root_agent_id}
    frontier: Set[int] = {root_agent_id}

    for _ in range(MAX_SUBTREE_DEPTH):
        if not frontier:
            return out
        rows = (
            db.query(Agent.id)
            .filter(Agent.reports_to_id.in_(frontier))
            .all()
        )
        next_frontier: Set[int] = {row.id for row in rows} - out
        out.update(next_frontier)
        frontier = next_frontier

    # Hit depth cap — chain is suspect. Default deny by returning empty.
    logger.warning(
        "[hierarchy] subtree_of(%s) exceeded MAX_SUBTREE_DEPTH=%d — defaulting deny",
        root_agent_id,
        MAX_SUBTREE_DEPTH,
    )
    return set()


def _agent_in_subtree(db, root_id: int, candidate_id: int) -> Optional[bool]:
    """Return True / False / None.

    None signals broken hierarchy (cycle, depth cap, missing manager) so
    the caller can choose default deny.
    """
    from core.models import Agent

    # Walk UP from candidate toward root with cycle detection. This is
    # cheaper than expanding the full subtree of root when the tree is
    # wide — we only ever load the actor's reporting chain.
    visited: Set[int] = set()
    cursor: Optional[int] = candidate_id
    depth = 0

    while cursor is not None:
        if cursor in visited:
            logger.warning(
                "[hierarchy] cycle detected at agent_id=%s while walking from %s",
                cursor,
                candidate_id,
            )
            return None
        if depth >= MAX_SUBTREE_DEPTH:
            logger.warning(
                "[hierarchy] depth cap exceeded walking from %s",
                candidate_id,
            )
            return None
        visited.add(cursor)
        if cursor == root_id:
            return True
        row = (
            db.query(Agent.reports_to_id)
            .filter(Agent.id == cursor)
            .first()
        )
        if row is None:
            # candidate row vanished mid-walk — treat as broken
            return None
        # SQLAlchemy Row supports attribute access by column label.
        cursor = getattr(row, "reports_to_id", None)
        depth += 1

    return False


def _deny(actor, target_type, target_id, change_type, source, reason) -> PermissionDecision:
    return PermissionDecision(
        allowed=False,
        reason=reason,
        actor_agent_id=getattr(actor, "id", None),
        actor_name=getattr(actor, "name", None),
        target_type=target_type,
        target_id=str(target_id) if target_id is not None else None,
        change_type=change_type,
        source=source,
    )

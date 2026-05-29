"""PRD-140 — hierarchy permission helper.

Single chokepoint that answers "may actor X apply change C to target T?".
Called from the platform tool dispatcher (``platform_executor``) and intended
for service-layer mutations as defence in depth.

Model
-----
* **No actor** (``actor_agent_id is None``) → ALLOW. A missing actor means the
  call originates from a trusted platform/system flow (startup seeds, heartbeat
  ticks, agent factory) that has no agent identity to scope against.
* **System agents** (``agents.is_system_agent``) → ALLOW anything. These rows
  are platform-seeded only (Auto, Auto CTO, onboarding agents); users cannot
  set the flag, so the flag itself is the trust boundary.
* **Everyone else** is scoped to their org subtree: an actor may modify any
  agent in (or equal to) their ``reports_to_id`` subtree, plus tasks/playbooks
  owned by an agent in that subtree.
* Anything an actor may **not** do directly, but which Auto could arbitrate,
  is denied with ``escalation_target="auto"`` so the caller can route it to Auto
  instead of hard-failing. See :mod:`core.services.auto_cadence`.

Defaults are conservative: broken hierarchy (cycle / depth cap), unresolved
owner, or workspace-global resources (skills) deny-with-escalation rather than
fail open.

Queries use raw SQL against the minimal columns the decision needs
(``agents.id/is_system_agent/reports_to_id``, ``board_tasks.assigned_agent_id``,
``workflow_recipes.created_by_agent_id``) so the helper is cheap and does not
depend on the full ORM model being loadable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Set

from sqlalchemy import text

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- constants

# Target-type string constants — the public API for ``target_type``. Kept as
# plain strings so they survive JSON round-trips through the dispatcher.
TARGET_AGENT           = "agent"
TARGET_HEARTBEAT       = "heartbeat"
TARGET_PLAYBOOK        = "playbook"
TARGET_TASK            = "task"
TARGET_SKILL           = "skill"
TARGET_TOOL_ASSIGNMENT = "tool_assignment"

# Targets whose scope is the target agent row itself.
_AGENT_TARGETS = frozenset({TARGET_AGENT, TARGET_HEARTBEAT, TARGET_TOOL_ASSIGNMENT})

# Max reporting-chain depth before we treat the chain as broken. Real org
# trees rarely exceed a handful of levels; anything deeper is almost certainly
# a cycle the visited-set missed (defence in depth).
MAX_SUBTREE_DEPTH: int = 16


# ----------------------------------------------------------------- types


@dataclass(frozen=True)
class PermissionDecision:
    """Result of a permission check. Always inspect ``allowed`` first.

    ``escalation_target`` is set (to ``"auto"``) when the actor cannot make the
    change directly but Auto could arbitrate it — the caller should route the
    request to Auto rather than surfacing a hard failure.
    """

    allowed: bool
    reason: str
    escalation_target: Optional[str] = None
    bypass: bool = False
    bypass_kind: Optional[str] = None  # 'system_actor' | None
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
    target_id: Optional[object] = None,
    change_type: str = "update",
) -> PermissionDecision:
    """Decide whether ``actor`` may apply ``change_type`` to ``target``.

    Never raises for permission state — always returns a ``PermissionDecision``.
    """
    # Trusted system/platform flow with no agent identity.
    if actor_agent_id is None:
        return _allow("no_actor", target_type, target_id, change_type)

    actor = _agent_row(db, actor_agent_id)
    if actor is None:
        return _deny(
            "actor_not_found", target_type, target_id, change_type,
            actor_id=actor_agent_id, escalate=False,
        )

    if bool(actor.is_system_agent):
        return _allow(
            "system_actor_bypass", target_type, target_id, change_type,
            actor_id=actor_agent_id, bypass=True, bypass_kind="system_actor",
        )

    # Non-system actor — scope by org hierarchy.
    if target_type in _AGENT_TARGETS:
        return _scope_to_agent(db, actor_agent_id, target_id, target_type, change_type)

    if target_type == TARGET_SKILL:
        # Skills are workspace-global, not owned by a single agent — a
        # non-system actor never edits one directly; route to Auto.
        return _deny(
            "skill_requires_escalation", target_type, target_id, change_type,
            actor_id=actor_agent_id, escalate=True,
        )

    if target_type == TARGET_TASK:
        owner = _owner_id(db, "SELECT assigned_agent_id FROM board_tasks WHERE id = :id", target_id)
        return _scope_via_owner(db, actor_agent_id, owner, target_type, target_id, change_type)

    if target_type == TARGET_PLAYBOOK:
        owner = _owner_id(db, "SELECT created_by_agent_id FROM workflow_recipes WHERE id = :id", target_id)
        return _scope_via_owner(db, actor_agent_id, owner, target_type, target_id, change_type)

    return _deny(
        f"unknown_target_type:{target_type}", target_type, target_id, change_type,
        actor_id=actor_agent_id, escalate=False,
    )


# ----------------------------------------------------------------- scope helpers


def _scope_to_agent(db, actor_id, target_id, target_type, change_type) -> PermissionDecision:
    """Scope a change whose target *is* an agent row (agent / heartbeat / tool)."""
    if target_id is None:
        return _deny("missing_target", target_type, target_id, change_type,
                     actor_id=actor_id, escalate=True)
    target_int = _as_int(target_id)
    if target_int is None:
        return _deny("invalid_target_id", target_type, target_id, change_type,
                     actor_id=actor_id, escalate=False)
    if target_int == actor_id:
        return _allow("self_mutation", target_type, target_id, change_type, actor_id=actor_id)
    return _by_subtree(db, actor_id, target_int, target_type, target_id, change_type,
                       allow_reason="subtree_authority", deny_reason="out_of_subtree")


def _scope_via_owner(db, actor_id, owner_id, target_type, target_id, change_type) -> PermissionDecision:
    """Scope a change to a resource owned by an agent (task / playbook)."""
    if owner_id is None:
        # Orphaned / unresolvable owner (or the owner column is absent in this
        # deployment) — only Auto can safely arbitrate.
        return _deny("unresolved_owner", target_type, target_id, change_type,
                     actor_id=actor_id, escalate=True)
    owner_int = _as_int(owner_id)
    if owner_int is None:
        return _deny("invalid_owner", target_type, target_id, change_type,
                     actor_id=actor_id, escalate=True)
    if owner_int == actor_id:
        return _allow("owner_self", target_type, target_id, change_type, actor_id=actor_id)
    return _by_subtree(db, actor_id, owner_int, target_type, target_id, change_type,
                       allow_reason="owner_in_subtree", deny_reason="owner_out_of_subtree")


def _by_subtree(db, actor_id, candidate_id, target_type, target_id, change_type,
                *, allow_reason, deny_reason) -> PermissionDecision:
    rel = _in_subtree(db, root_id=actor_id, candidate_id=candidate_id)
    if rel is True:
        return _allow(allow_reason, target_type, target_id, change_type, actor_id=actor_id)
    if rel is None:
        return _deny("broken_hierarchy", target_type, target_id, change_type,
                     actor_id=actor_id, escalate=True)
    return _deny(deny_reason, target_type, target_id, change_type,
                 actor_id=actor_id, escalate=True)


# ----------------------------------------------------------------- db helpers


def _agent_row(db, agent_id):
    """Return a row with ``is_system_agent``/``reports_to_id`` or ``None``."""
    return db.execute(
        text("SELECT is_system_agent, reports_to_id FROM agents WHERE id = :id"),
        {"id": agent_id},
    ).first()


def _reports_to_id(db, agent_id) -> Optional[int]:
    return db.execute(
        text("SELECT reports_to_id FROM agents WHERE id = :id"),
        {"id": agent_id},
    ).scalar()


def _owner_id(db, sql: str, target_id) -> Optional[int]:
    """Resolve an owning-agent id, tolerant of a missing row or column.

    Wrapped in a SAVEPOINT so that a query failure (e.g. the column does not
    exist in this deployment) rolls back only the probe and leaves the caller's
    transaction intact; we then treat the owner as unresolved (→ escalate).
    """
    if target_id is None:
        return None
    try:
        with db.begin_nested():
            return db.execute(text(sql), {"id": target_id}).scalar()
    except Exception:
        logger.debug("[hierarchy] owner lookup failed (%s) — treating as unresolved", sql)
        return None


def _in_subtree(db, *, root_id: int, candidate_id: int) -> Optional[bool]:
    """Walk UP from ``candidate`` toward ``root`` via ``reports_to_id``.

    Returns True (candidate is root or a descendant), False (disjoint), or
    None (cycle / depth cap → broken hierarchy, caller should default deny).
    """
    visited: Set[int] = set()
    cursor: Optional[int] = candidate_id
    depth = 0
    while cursor is not None:
        if cursor == root_id:
            return True
        if cursor in visited:
            logger.warning("[hierarchy] cycle detected walking from %s", candidate_id)
            return None
        if depth >= MAX_SUBTREE_DEPTH:
            logger.warning("[hierarchy] depth cap exceeded walking from %s", candidate_id)
            return None
        visited.add(cursor)
        cursor = _reports_to_id(db, cursor)
        depth += 1
    return False


# ----------------------------------------------------------------- decision factories


def _as_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sid(value) -> Optional[str]:
    return str(value) if value is not None else None


def _allow(reason, target_type, target_id, change_type, *,
           actor_id=None, bypass=False, bypass_kind=None) -> PermissionDecision:
    return PermissionDecision(
        allowed=True,
        reason=reason,
        bypass=bypass,
        bypass_kind=bypass_kind,
        actor_agent_id=actor_id,
        target_type=target_type,
        target_id=_sid(target_id),
        change_type=change_type,
    )


def _deny(reason, target_type, target_id, change_type, *,
          actor_id=None, escalate=False) -> PermissionDecision:
    return PermissionDecision(
        allowed=False,
        reason=reason,
        escalation_target="auto" if escalate else None,
        actor_agent_id=actor_id,
        target_type=target_type,
        target_id=_sid(target_id),
        change_type=change_type,
    )

"""PRD-140 — hierarchy permission helper.

Single chokepoint that answers "may actor X apply change C to target T?".
Called from the platform tool dispatcher (``platform_executor``) and intended
for service-layer mutations as defence in depth.

Security model (deny-by-default; every check is workspace-bound)
----------------------------------------------------------------
The actor gate runs first and fails closed on anything suspicious:

* **No actor** (``actor_agent_id is None``) → DENY. Absence of identity is not
  proof of trust — a trusted system flow must pass an explicit (seeded) actor
  id, never rely on ambient authority.
* **Unknown actor** (no ``agents`` row) → DENY.
* **Cross-workspace actor** (actor's ``workspace_id`` ≠ the call's
  ``workspace_id``) → DENY. Blocks cross-tenant IDOR before any reasoning.
* **Inactive actor** (``status`` set and not ``active``) → DENY.
* **System bypass** is *narrowed*: a row gets the free pass only when
  ``is_system_agent`` is set **and** its ``name`` is on
  :data:`SYSTEM_BYPASS_ALLOWLIST`. The flag alone is not a master key — that
  way a stray/forgotten ``is_system_agent`` flag cannot escalate on its own.
  These rows are platform-seeded (Auto, Auto CTO, onboarding agents).

Everyone else is scoped to their org subtree: an actor may modify any agent in
(or equal to) their ``reports_to_id`` subtree, plus tasks owned by an agent in
that subtree, and only within their own workspace.

Playbooks belong to the person who made them, not to an agent (F133): an actor
may change one only while the call is made FOR its creator
(``workflow_recipes.created_by_user_id``) or for a workspace owner/admin
(:mod:`core.security.driving_user`), read from the server-built caller context.
The owner lookup used to query a ``created_by_agent_id`` column that never
existed; the error was swallowed, so every such edit was refused as
``unresolved_owner``.

Authority *limits* (out-of-subtree, unresolved owner, workspace-global skills,
broken hierarchy) deny with ``escalation_target="auto"`` so the caller can route
the request to Auto for arbitration instead of hard-failing. Security failures
(anonymous / unknown / cross-workspace / inactive) deny with **no** escalation —
those are not arbitrated, they are refused. See :mod:`core.services.auto_cadence`.

Queries use raw SQL against the minimal columns the decision needs so the helper
is cheap and does not depend on the full ORM model being loadable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Set, Tuple

from sqlalchemy import text

from core.security.driving_user import driver_is_workspace_admin, driving_user_id

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

# Narrowed system-agent bypass allowlist — only these *named*, platform-seeded
# actors bypass the hierarchy. ``is_system_agent=True`` is necessary but not
# sufficient; the name must also match. Adding to this list is a
# security-sensitive change. Keep in sync with the seeds in ``core/seeds/``
# (seed_auto_agent, seed_cto_agent).
SYSTEM_BYPASS_ALLOWLIST: frozenset = frozenset({
    "Auto",            # workspace orchestrator (seed_auto_agent)
    "Auto CTO",        # platform CTO agent (seed_cto_agent)
    "HARNESS",         # weekly self-optimisation service actor
    "platform-admin",  # explicit admin agent
    "platform-system", # internal service actor
})

# Max reporting-chain depth before we treat the chain as broken. Real org
# trees rarely exceed a handful of levels; anything deeper is almost certainly
# a cycle the visited-set missed (defence in depth).
MAX_SUBTREE_DEPTH: int = 16

# F133: what a refused playbook edit tells the agent, so it can get it done.
_PLAYBOOK_REFUSALS = {
    "no_driving_user": (
        "This change is not being made for anyone, and only the playbook's creator or a "
        "workspace owner or admin can change it. Ask for it in chat, or ask Auto to make it."
    ),
    "no_creator_recorded": (
        "This playbook has no recorded creator, so only a workspace owner or admin can change "
        "it. Ask one of them, or ask Auto."
    ),
    "not_the_creators_agent": (
        "Only the person who made this playbook, or a workspace owner or admin, can change it. "
        "Ask them, or ask Auto."
    ),
    "owner_lookup_failed": (
        "The playbook's creator could not be checked, so the change was refused. Try again, "
        "or ask Auto."
    ),
}


# ----------------------------------------------------------------- types


@dataclass(frozen=True)
class PermissionDecision:
    """Result of a permission check. Always inspect ``allowed`` first.

    ``escalation_target`` is set (to ``"auto"``) only when the actor is
    legitimate but lacks *authority* for this specific change and Auto could
    arbitrate it — the caller should route the request to Auto rather than
    surfacing a hard failure. Security failures (anonymous / cross-workspace /
    inactive) leave it ``None``: those are refused, not arbitrated.
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
    message: Optional[str] = None  # F133: the refusal in words the agent can act on


# ----------------------------------------------------------------- main API


def can_actor_modify(
    db,
    *,
    actor_agent_id: Optional[int],
    target_type: str,
    workspace_id: object,
    target_id: Optional[object] = None,
    change_type: str = "update",
    source: Optional[str] = None,
    caller_context: Optional[Mapping[str, Any]] = None,
) -> PermissionDecision:
    """Decide whether ``actor`` may apply ``change_type`` to ``target``.

    ``workspace_id`` scopes every check — cross-tenant mutations are denied
    here. ``caller_context`` is the server-built context of the call (who it is
    made for); only playbooks read it. Never raises for permission state —
    always returns a ``PermissionDecision``.
    """
    ws = _ws(workspace_id)

    # --- Actor gate (fail closed on anything suspicious) -----------------
    if actor_agent_id is None:
        # No ambient authority. Trusted flows must pass an explicit actor id.
        return _deny("anonymous_actor", target_type, target_id, change_type,
                     source=source, escalate=False)

    actor = _agent_row(db, actor_agent_id)
    if actor is None:
        return _deny("actor_not_found", target_type, target_id, change_type,
                     actor_id=actor_agent_id, source=source, escalate=False)

    if _ws(actor.workspace_id) != ws:
        return _deny("cross_workspace_actor", target_type, target_id, change_type,
                     actor_id=actor_agent_id, actor_name=actor.name,
                     source=source, escalate=False)

    if (actor.status or "active").strip().lower() != "active":
        return _deny("actor_inactive", target_type, target_id, change_type,
                     actor_id=actor_agent_id, actor_name=actor.name,
                     source=source, escalate=False)

    # --- Narrowed system bypass: flag AND allowlisted name ---------------
    # F155: never on a public widget turn — there Auto answers a visitor.
    from core.security.surface import widget_turn

    if bool(actor.is_system_agent) and (actor.name or "") in SYSTEM_BYPASS_ALLOWLIST and not widget_turn():
        return _allow(
            "system_actor_bypass", target_type, target_id, change_type,
            actor_id=actor_agent_id, actor_name=actor.name, source=source,
            bypass=True, bypass_kind="system_actor",
        )

    # --- Scope by org hierarchy (within the actor's workspace) -----------
    if target_type in _AGENT_TARGETS:
        return _scope_to_agent(db, actor, actor_agent_id, ws, target_id,
                               target_type, change_type, source)

    if target_type == TARGET_SKILL:
        # Skills are workspace-global, not owned by a single agent — a
        # non-system actor never edits one directly; route to Auto.
        return _deny("skill_requires_escalation", target_type, target_id, change_type,
                     actor_id=actor_agent_id, actor_name=actor.name,
                     source=source, escalate=True)

    if target_type == TARGET_TASK:
        owner = _owner_id(db, "SELECT assigned_agent_id FROM board_tasks WHERE id = :id", target_id)
        return _scope_via_owner(db, actor, actor_agent_id, ws, owner,
                                target_type, target_id, change_type, source)

    if target_type == TARGET_PLAYBOOK:
        return _scope_playbook(db, actor, actor_agent_id, ws, target_id, change_type, source,
                               caller_context)

    return _deny(
        f"unknown_target_type:{target_type}", target_type, target_id, change_type,
        actor_id=actor_agent_id, actor_name=actor.name, source=source, escalate=False,
    )


# ----------------------------------------------------------------- scope helpers


def _scope_to_agent(db, actor, actor_id, ws, target_id, target_type, change_type, source) -> PermissionDecision:
    """Scope a change whose target *is* an agent row (agent / heartbeat / tool)."""
    if target_id is None:
        return _deny("missing_target", target_type, target_id, change_type,
                     actor_id=actor_id, actor_name=actor.name, source=source, escalate=True)
    target_int = _as_int(target_id)
    if target_int is None:
        return _deny("invalid_target_id", target_type, target_id, change_type,
                     actor_id=actor_id, actor_name=actor.name, source=source, escalate=False)
    if target_int == actor_id:
        return _allow("self_mutation", target_type, target_id, change_type,
                      actor_id=actor_id, actor_name=actor.name, source=source)

    target = _agent_row(db, target_int)
    if target is None:
        return _deny("target_not_found", target_type, target_id, change_type,
                     actor_id=actor_id, actor_name=actor.name, source=source, escalate=False)
    if _ws(target.workspace_id) != ws:
        return _deny("cross_workspace_target", target_type, target_id, change_type,
                     actor_id=actor_id, actor_name=actor.name, source=source, escalate=False)

    return _by_subtree(db, actor, actor_id, target_int, target_type, target_id, change_type, source,
                       allow_reason="subtree_authority", deny_reason="out_of_subtree")


def _scope_via_owner(db, actor, actor_id, ws, owner_id, target_type, target_id, change_type, source) -> PermissionDecision:
    """Scope a change to a resource owned by an agent (task / playbook)."""
    if owner_id is None:
        # Orphaned / unresolvable owner (or the owner column is absent in this
        # deployment) — only Auto can safely arbitrate.
        return _deny("unresolved_owner", target_type, target_id, change_type,
                     actor_id=actor_id, actor_name=actor.name, source=source, escalate=True)
    owner_int = _as_int(owner_id)
    if owner_int is None:
        return _deny("invalid_owner", target_type, target_id, change_type,
                     actor_id=actor_id, actor_name=actor.name, source=source, escalate=True)
    if owner_int == actor_id:
        return _allow("owner_self", target_type, target_id, change_type,
                      actor_id=actor_id, actor_name=actor.name, source=source)

    owner = _agent_row(db, owner_int)
    if owner is not None and _ws(owner.workspace_id) != ws:
        return _deny("cross_workspace_owner", target_type, target_id, change_type,
                     actor_id=actor_id, actor_name=actor.name, source=source, escalate=False)

    return _by_subtree(db, actor, actor_id, owner_int, target_type, target_id, change_type, source,
                       allow_reason="owner_in_subtree", deny_reason="owner_out_of_subtree")


def _scope_playbook(db, actor, actor_id, ws, target_id, change_type, source,
                    caller_context) -> PermissionDecision:
    """F133 (B9): a playbook edit is made for its creator, or for a workspace
    owner/admin. Everything else is refused, fail-closed, with the real cause."""
    who = dict(actor_id=actor_id, actor_name=actor.name, source=source)

    def refuse(reason, *, escalate=True, cause=None):
        return _deny(f"{reason}:{cause}" if cause else reason, TARGET_PLAYBOOK, target_id, change_type,
                     escalate=escalate, message=_PLAYBOOK_REFUSALS.get(reason), **who)

    if target_id is None:
        return refuse("missing_target")
    try:
        found, creator = _playbook_creator(db, target_id, ws)
    except Exception as exc:  # noqa: BLE001 — refused with its cause, never swallowed
        logger.warning("[hierarchy] creator lookup for playbook %s failed: %r", target_id, exc)
        return refuse("owner_lookup_failed", cause=type(exc).__name__)
    if not found:
        return refuse("target_not_found", escalate=False)
    driver = driving_user_id(caller_context)
    if driver is None:
        return refuse("no_driving_user")
    if creator is not None and creator == driver:
        return _allow("acts_for_creator", TARGET_PLAYBOOK, target_id, change_type, **who)
    if driver_is_workspace_admin(db, ws, caller_context):
        return _allow("acts_for_workspace_admin", TARGET_PLAYBOOK, target_id, change_type, **who)
    return refuse("no_creator_recorded" if creator is None else "not_the_creators_agent")


def _playbook_creator(db, target_id, ws) -> Tuple[bool, Optional[int]]:
    """(the playbook exists in this workspace, its creator's users.id). A
    non-numeric id is no playbook. Raises when the lookup itself fails."""
    playbook_id = _as_int(target_id)
    if playbook_id is None:
        return False, None
    with db.begin_nested():
        row = db.execute(
            text("SELECT created_by_user_id FROM workflow_recipes "
                 "WHERE id = :id AND workspace_id = CAST(:ws AS uuid)"),
            {"id": playbook_id, "ws": ws},
        ).first()
    return (row is not None), (int(row[0]) if row is not None and row[0] is not None else None)


def _by_subtree(db, actor, actor_id, candidate_id, target_type, target_id, change_type, source,
                *, allow_reason, deny_reason) -> PermissionDecision:
    rel = _in_subtree(db, root_id=actor_id, candidate_id=candidate_id)
    if rel is True:
        return _allow(allow_reason, target_type, target_id, change_type,
                      actor_id=actor_id, actor_name=actor.name, source=source)
    if rel is None:
        return _deny("broken_hierarchy", target_type, target_id, change_type,
                     actor_id=actor_id, actor_name=actor.name, source=source, escalate=True)
    return _deny(deny_reason, target_type, target_id, change_type,
                 actor_id=actor_id, actor_name=actor.name, source=source, escalate=True)


# ----------------------------------------------------------------- db helpers


def _agent_row(db, agent_id):
    """Return a row with ``name``/``is_system_agent``/``reports_to_id``/
    ``workspace_id``/``status`` or ``None``."""
    return db.execute(
        text(
            "SELECT name, is_system_agent, reports_to_id, workspace_id, status "
            "FROM agents WHERE id = :id"
        ),
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


def _ws(value) -> Optional[str]:
    """Normalise a workspace id (UUID / str / None) for equality comparison."""
    return str(value) if value is not None else None


def _allow(reason, target_type, target_id, change_type, *,
           actor_id=None, actor_name=None, source=None,
           bypass=False, bypass_kind=None) -> PermissionDecision:
    return PermissionDecision(
        allowed=True,
        reason=reason,
        bypass=bypass,
        bypass_kind=bypass_kind,
        actor_agent_id=actor_id,
        actor_name=actor_name,
        target_type=target_type,
        target_id=_sid(target_id),
        change_type=change_type,
        source=source,
    )


def _deny(reason, target_type, target_id, change_type, *,
          actor_id=None, actor_name=None, source=None, escalate=False,
          message=None) -> PermissionDecision:
    return PermissionDecision(
        allowed=False,
        reason=reason,
        escalation_target="auto" if escalate else None,
        actor_agent_id=actor_id,
        actor_name=actor_name,
        target_type=target_type,
        target_id=_sid(target_id),
        change_type=change_type,
        source=source,
        message=message,
    )

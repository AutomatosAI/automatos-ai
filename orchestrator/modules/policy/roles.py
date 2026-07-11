"""Policy plane — one role + permission semantic (PRD-174 F042/F043, PRD-195 P2-14).

Two forks the plane closes:

- **F043** — seven+ routers gate admin functions with ``system_role == 'admin'``,
  which 403s the platform super-admin entirely. The fix is one shared
  ``super_admin ⊇ admin ⊇ user`` hierarchy: a super-admin satisfies every
  admin gate, an admin satisfies user gates. :func:`role_satisfies`.

- **F042** — RBAC god-key/null-key fork: an empty permission list means
  *allow-all* on the widget plane but *deny-all* on the board plane, so one
  no-permission key is a god-key on one plane and a null-key on the other. The
  fix is a single semantic — **empty permissions grant nothing** (least
  privilege) — applied on every plane. :func:`has_permission`.

PRD-195 (P2-14) makes this module the **single authorization authority**: the
workspace-role matrix (``owner ⊇ admin ⊇ editor ⊇ viewer``, previously
``core/workspaces/permissions.py``) lives here beside the system hierarchy, and
every checker — the admin routers' ``caller_is_admin``, the SDK-key service, the
widget plane's ``require_permission``, the ``require_workspace_permission``
gate — delegates to these functions. Permission strings follow the canonical
vocabulary (G1): ``resource:action`` over ``workspace | members | agents |
missions | playbooks | documents | knowledge | audit`` — the legacy
``workflows:*`` strings are renamed to ``missions:*`` + ``playbooks:*`` in the
same collapse (CLAUDE.md §10: Mission not Workflow, Playbook not Recipe).

Stdlib-only: no DB, no config, no imports beyond typing/enum. Callers pass the
already-resolved role string / permission list.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet, Iterable, Optional, Set

SUPER_ADMIN = "super_admin"
ADMIN = "admin"
USER = "user"

# Each role's *satisfied set*: the gate-roles this role is allowed to pass.
# super_admin ⊇ admin ⊇ user. Unknown/None roles satisfy nothing (fail-closed).
_SATISFIES: Dict[str, Set[str]] = {
    SUPER_ADMIN: {SUPER_ADMIN, ADMIN, USER},
    ADMIN: {ADMIN, USER},
    USER: {USER},
}


def role_satisfies(caller_role: Optional[str], required_role: str) -> bool:
    """True when ``caller_role`` is allowed to pass a gate requiring ``required_role``.

    ``super_admin`` passes ``admin`` and ``user`` gates; ``admin`` passes
    ``user`` gates. ``None`` / unknown roles satisfy nothing (fail-closed) —
    this is what keeps the seven ``system_role == 'admin'`` routers from 403'ing
    a super-admin once they route through here.
    """
    if not caller_role:
        return False
    return required_role in _SATISFIES.get(caller_role, frozenset())


def is_admin(caller_role: Optional[str]) -> bool:
    """True when ``caller_role`` satisfies an ``admin`` gate (admin or super_admin).

    Replaces the scattered ``getattr(user, 'system_role', 'user') == 'admin'``
    checks that lock super-admins out.
    """
    return role_satisfies(caller_role, ADMIN)


def is_super_admin(caller_role: Optional[str]) -> bool:
    """True only for the platform super-admin (observability tier)."""
    return role_satisfies(caller_role, SUPER_ADMIN)


def has_permission(
    permissions: Optional[Iterable[str]], required_permission: str
) -> bool:
    """True only when ``permissions`` **explicitly** grants ``required_permission``.

    **Empty = deny (F042).** A ``None`` or empty permission list grants nothing —
    the single least-privilege semantic that replaces the widget plane's
    "empty = unrestricted" god-key. There is no wildcard-by-omission; grants
    must be explicit. (An explicit ``"*"`` element, if a plane chooses to issue
    one, is honoured — that is a deliberate grant, not the absence of one.)
    """
    if not permissions:
        return False
    perms = set(permissions)
    return required_permission in perms or "*" in perms


# ---------------------------------------------------------------------------
# Workspace roles (PRD-195 S1 — absorbed from core/workspaces/permissions.py)
# ---------------------------------------------------------------------------

class WorkspaceRole(str, Enum):
    """Per-tenant role on ``workspace_members.role`` (PRD-37)."""

    OWNER = "owner"
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"


# The workspace-permission matrix — the ONE source the workspace gate reads.
# Canonical strings (G1): ``resource:action``; ``resource:*`` is the full grant
# on a resource. Deliberate shape (dossier auth-identity C.2, S3–S6 specs):
#
# - ``viewer`` is read-only: no create/update/delete/execute anywhere.
# - ``editor`` authors and runs but does not destroy or administer:
#   create/read/update on the five content/execution resources, plus
#   ``execute`` on agents/missions/playbooks (S3/S4: "editor can" create,
#   fire and cancel — execute is granted explicitly since the legacy matrix
#   predates Missions/Playbooks as runnable surfaces). No ``delete``, no
#   members/workspace/audit administration.
# - ``admin`` holds full resource wildcards + member invite/remove; only the
#   ``owner`` changes member roles, manages/deletes/bills the workspace.
ROLE_PERMISSIONS: Dict[WorkspaceRole, FrozenSet[str]] = {
    WorkspaceRole.OWNER: frozenset({
        "workspace:manage", "workspace:delete", "workspace:billing",
        "members:invite", "members:remove", "members:change_role",
        "members:read",
        "agents:*", "missions:*", "playbooks:*", "documents:*", "knowledge:*",
        "audit:view",
    }),
    WorkspaceRole.ADMIN: frozenset({
        "workspace:manage",
        "members:invite", "members:remove", "members:read",
        "agents:*", "missions:*", "playbooks:*", "documents:*", "knowledge:*",
        "audit:view",
    }),
    WorkspaceRole.EDITOR: frozenset({
        "members:read",
        "agents:create", "agents:read", "agents:update", "agents:execute",
        "missions:create", "missions:read", "missions:update", "missions:execute",
        "playbooks:create", "playbooks:read", "playbooks:update", "playbooks:execute",
        "documents:create", "documents:read", "documents:update",
        "knowledge:create", "knowledge:read", "knowledge:update",
    }),
    WorkspaceRole.VIEWER: frozenset({
        "members:read",
        "agents:read", "missions:read", "playbooks:read",
        "documents:read", "knowledge:read",
    }),
}


def workspace_has_permission(role: Optional[str], permission: str) -> bool:
    """True when workspace ``role`` grants ``permission``.

    Exact match or resource wildcard (``agents:*`` covers ``agents:create``).
    ``None`` / unknown roles grant nothing (fail-closed) — same posture as
    :func:`role_satisfies`. Accepts the enum or its string value.
    """
    if not role:
        return False
    try:
        role_enum = WorkspaceRole(role)
    except ValueError:
        return False
    permissions = ROLE_PERMISSIONS[role_enum]
    if permission in permissions:
        return True
    resource = permission.split(":", 1)[0]
    return f"{resource}:*" in permissions

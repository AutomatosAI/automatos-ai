"""One shared admin-role check for API routers (PRD-174 F043 · PRD-195 S1).

Seven+ routers duplicated ``getattr(user, "system_role", "user") == "admin"``,
which 403s the platform super-admin entirely (a super-admin is *not* literally
``"admin"``). This is the single choke point they now call instead.

PRD-195 S1 (P2-14, G2): the hierarchy is **unconditional**. The staged
``AUTOMATOS_POLICY_PLANE`` dial governs the governance legs only (budget,
act-vs-ask, fail-closed execution — PRD-192's territory); authorization
correctness does not ride a rollout flag. The plane-OFF branch that kept
super-admins locked out by default is deleted.

Pure role read: no DB, no config, no flag. ``modules.policy.roles`` is the one
authority (stdlib-only, safe to import from anywhere).
"""
from __future__ import annotations

from typing import Any

from modules.policy.roles import is_admin as _hierarchy_is_admin


def caller_role(user: Any) -> str:
    """Extract the principal's ``system_role`` (defaulting to ``"user"``)."""
    return getattr(user, "system_role", "user") or "user"


def caller_is_admin(user: Any) -> bool:
    """True when the principal satisfies an admin gate.

    ``super_admin ⊇ admin`` — unconditionally (F043 closed). Unknown / missing
    roles satisfy nothing.
    """
    return _hierarchy_is_admin(caller_role(user))

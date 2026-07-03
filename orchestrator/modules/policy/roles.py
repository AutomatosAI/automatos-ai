"""Policy plane — one role + permission semantic (PRD-174 F042/F043).

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

Stdlib-only: no DB, no config, no imports beyond typing. Callers pass the
already-resolved role string / permission list.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Set

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

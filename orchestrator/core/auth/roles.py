"""One shared admin-role check for API routers (PRD-174 F043).

Seven+ routers duplicated ``getattr(user, "system_role", "user") == "admin"``,
which 403s the platform super-admin entirely (a super-admin is *not* literally
``"admin"``). This is the single choke point they now call instead.

Behaviour is gated on the policy-plane flag so the rollout stays byte-for-byte:

- **Plane OFF** — exactly today's check: ``system_role == "admin"`` (super-admin
  still excluded, unchanged).
- **Plane ON** — the shared ``super_admin ⊇ admin ⊇ user`` hierarchy from
  :mod:`modules.policy.roles`, so a super-admin satisfies every admin gate.

Pure role read (no DB); the flag read is the only side input.
"""
from __future__ import annotations

from typing import Any

ADMIN_ROLE = "admin"


def caller_role(user: Any) -> str:
    """Extract the principal's ``system_role`` (defaulting to ``"user"``)."""
    return getattr(user, "system_role", "user") or "user"


def caller_is_admin(user: Any) -> bool:
    """True when the principal satisfies an admin gate.

    Plane OFF ⇒ ``system_role == "admin"`` (legacy, super-admin excluded).
    Plane ON  ⇒ super_admin ⊇ admin (super-admin passes admin gates).
    """
    role = caller_role(user)
    try:
        from modules.policy import policy_plane_enabled

        if policy_plane_enabled():
            from modules.policy.roles import is_admin as _hier_is_admin

            return _hier_is_admin(role)
    except Exception:
        # Fail-safe: fall back to the legacy exact check on any import/flag error.
        pass
    return role == ADMIN_ROLE

"""PRD-143: the one canonical super-admin gate for the observability tier.

A narrow dependency layered on the shared hybrid auth — ``core/auth/hybrid.py``
(657 call sites, PRD-09 precedent) must never be modified for tier checks.
Pure role check: no config reads, no env reads, no DB.
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, status

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid

SUPER_ADMIN_ROLE = "super_admin"


async def require_super_admin(
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> RequestContext:
    """403 unless the principal is literally ``system_role == 'super_admin'``.

    Fail-closed: missing user, missing/unknown system_role, API-key
    principals (system_role='admin', hybrid.py:783) and SDK/service
    principals (system_role='service', hybrid.py:616) all refuse.
    """
    user = getattr(ctx, "user", None)
    if user is None or getattr(user, "system_role", None) != SUPER_ADMIN_ROLE:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Super admin only")
    return ctx

"""PRD-185 S12: the own-workspace observability gate.

A narrow dependency layered on the shared hybrid auth — ``core/auth/hybrid.py``
must never be modified for tier checks (PRD-09 / PRD-143 precedent). It widens the
Command Center health-strip tiles from super-admin-only to the people who run a
workspace, WITHOUT widening the platform/cross-workspace analytics (those keep
``require_super_admin``).

Passes when the principal may read their OWN workspace's health tiles:

- the platform **super-admin** (they satisfy ``require_super_admin`` today), or
- an **owner/admin** of the request's workspace (owns it, or an active
  ``workspace_members`` row with role owner/admin).

Deliberately NARROWER than a generic admin check: API-key / SDK principals
(``system_role='admin'``, no workspace membership) stay refused, matching the
PRD-143 observability posture. Every tile behind this gate is already filtered by
``ctx.workspace_id``, so the gate only decides *who in a workspace* sees health —
it can never surface another tenant's data. Fail-closed: anonymous / plain-member
/ non-member principals refuse.
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from sqlalchemy import text

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.super_admin import SUPER_ADMIN_ROLE
from core.database.database import get_db

# workspace_members.role values that administer a workspace.
WORKSPACE_ADMIN_ROLES = ("owner", "admin")


def _is_workspace_admin_member(db, ctx: RequestContext) -> bool:
    """True when ``ctx.user`` owns, or is an active owner/admin member of,
    ``ctx.workspace_id``.

    Mirrors ``hybrid._user_has_workspace_access`` (owner OR active member) and
    narrows it to administering roles. Owning the workspace always qualifies —
    the personal-workspace owner is its admin by definition.
    """
    user = getattr(ctx, "user", None)
    clerk_uid = getattr(user, "clerk_user_id", None) if user else None
    if not clerk_uid or ctx.workspace_id is None:
        return False
    row = db.execute(
        text(
            "SELECT 1 FROM users u "
            "LEFT JOIN workspaces w ON w.owner_id = u.id AND w.id = :ws "
            "LEFT JOIN workspace_members wm ON wm.user_id = u.id "
            "  AND wm.workspace_id = :ws AND wm.is_active = true "
            "  AND wm.role IN ('owner', 'admin') "
            "WHERE u.clerk_user_id = :cid AND (w.id IS NOT NULL OR wm.id IS NOT NULL) "
            "LIMIT 1"
        ),
        {"cid": clerk_uid, "ws": str(ctx.workspace_id)},
    ).fetchone()
    return bool(row)


def may_see_own_workspace_health(db, ctx: RequestContext) -> bool:
    """Pure decision: may this principal read their own workspace's health tiles?

    Split out from the dependency so it is unit-testable without FastAPI.
    """
    user = getattr(ctx, "user", None)
    if user is None:
        return False
    # The platform super-admin sees every workspace's health (operator view).
    if getattr(user, "system_role", None) == SUPER_ADMIN_ROLE:
        return True
    # Everyone else must be an owner/admin of THIS workspace.
    return _is_workspace_admin_member(db, ctx)


async def require_workspace_admin(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db=Depends(get_db),
) -> RequestContext:
    """403 unless the caller may read their own workspace's health tiles."""
    if may_see_own_workspace_health(db, ctx):
        return ctx
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Workspace admin only",
    )

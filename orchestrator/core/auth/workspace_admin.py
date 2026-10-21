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

A paired phone or browser (PRD-266, ``auth_type`` ``device``) has no Clerk identity:
the same rule is read for its bound person by ``users.id``
(``core.auth.principal.resolve_user_pk``). The local operator's device is
``super_admin`` and passes as the operator does.

PRD-266 D16 (Gerard, 20 Oct; reverses PRD-196's owner/admin posture for these verbs):
``require_workspace_answerer`` is the same rule widened by one role. An owner, admin or
**editor** of the workspace (or the super-admin) answers its questions and grants or
denies its approvals; a viewer, an API key, an SDK key and the saas anonymous principal
do not. Revoking a grant and every health tile stay behind ``require_workspace_admin``.
"""
from __future__ import annotations

from typing import Sequence

from fastapi import Depends, HTTPException, status
from sqlalchemy import bindparam, text

from core.auth.dependencies import RequestContext
from core.auth.device import DEVICE_AUTH_TYPE
from core.auth.hybrid import get_request_context_hybrid
from core.auth.principal import resolve_user_pk
from core.auth.super_admin import SUPER_ADMIN_ROLE
from core.database.database import get_db
from modules.policy.roles import WorkspaceRole

# workspace_members.role values that administer a workspace.
WORKSPACE_ADMIN_ROLES = (WorkspaceRole.OWNER.value, WorkspaceRole.ADMIN.value)
# PRD-266 D16: the workspace_members.role values that answer questions and approvals.
WORKSPACE_ANSWERER_ROLES = (*WORKSPACE_ADMIN_ROLES, WorkspaceRole.EDITOR.value)
ANSWERER_ONLY = "Only owners, admins and editors can answer questions."

# The owner of ``:ws``, or an active member of it in one of ``:roles``: read by Clerk id,
# or by ``users.id`` for a paired device's person (PRD-266). Bound parameters only.
_MEMBER_IN_ROLES_BY_CLERK_SQL = text(
    "SELECT 1 FROM users u "
    "LEFT JOIN workspaces w ON w.owner_id = u.id AND w.id = :ws "
    "LEFT JOIN workspace_members wm ON wm.user_id = u.id "
    "  AND wm.workspace_id = :ws AND wm.is_active = true "
    "  AND wm.role IN :roles "
    "WHERE u.clerk_user_id = :cid AND (w.id IS NOT NULL OR wm.id IS NOT NULL) "
    "LIMIT 1"
).bindparams(bindparam("roles", expanding=True))
_MEMBER_IN_ROLES_BY_PK_SQL = text(
    "SELECT 1 FROM users u "
    "LEFT JOIN workspaces w ON w.owner_id = u.id AND w.id = :ws "
    "LEFT JOIN workspace_members wm ON wm.user_id = u.id "
    "  AND wm.workspace_id = :ws AND wm.is_active = true "
    "  AND wm.role IN :roles "
    "WHERE u.id = :uid AND (w.id IS NOT NULL OR wm.id IS NOT NULL) "
    "LIMIT 1"
).bindparams(bindparam("roles", expanding=True))


def _is_device_member_in(db, ctx: RequestContext, roles: Sequence[str]) -> bool:
    """``_is_member_in`` for a paired device: its bound person, by ``users.id``."""
    user_pk = resolve_user_pk(db, ctx) if ctx.workspace_id is not None else None
    if user_pk is None:
        return False
    params = {"uid": user_pk, "ws": str(ctx.workspace_id), "roles": list(roles)}
    return bool(db.execute(_MEMBER_IN_ROLES_BY_PK_SQL, params).fetchone())


def _is_member_in(db, ctx: RequestContext, roles: Sequence[str]) -> bool:
    """True when ``ctx.user`` owns ``ctx.workspace_id``, or is an active member of it
    whose role is one of ``roles``.

    Mirrors ``hybrid._user_has_workspace_access`` (owner OR active member) and
    narrows it to ``roles``. Owning the workspace always qualifies — the
    personal-workspace owner is its admin by definition. A caller with no
    ``clerk_user_id`` is read by ``users.id`` only on the device path.
    """
    user = getattr(ctx, "user", None)
    clerk_uid = getattr(user, "clerk_user_id", None) if user else None
    if not clerk_uid and getattr(ctx, "auth_type", None) == DEVICE_AUTH_TYPE:
        return _is_device_member_in(db, ctx, roles)
    if not clerk_uid or ctx.workspace_id is None:
        return False
    params = {"cid": clerk_uid, "ws": str(ctx.workspace_id), "roles": list(roles)}
    return bool(db.execute(_MEMBER_IN_ROLES_BY_CLERK_SQL, params).fetchone())


def _super_admin_or_member_in(db, ctx: RequestContext, roles: Sequence[str]) -> bool:
    """The platform super-admin (operator view of every workspace), or a member of
    THIS workspace in one of ``roles`` (its owner always)."""
    user = getattr(ctx, "user", None)
    if user is None:
        return False
    if getattr(user, "system_role", None) == SUPER_ADMIN_ROLE:
        return True
    return _is_member_in(db, ctx, roles)


def may_see_own_workspace_health(db, ctx: RequestContext) -> bool:
    """Pure decision: may this principal read their own workspace's health tiles?

    Split out from the dependency so it is unit-testable without FastAPI.
    """
    return _super_admin_or_member_in(db, ctx, WORKSPACE_ADMIN_ROLES)


def may_answer_workspace_questions(db, ctx: RequestContext) -> bool:
    """Pure decision (PRD-266 D16): may this principal answer the workspace's
    questions and grant or deny its approvals?

    Owner, admin or editor of ``ctx.workspace_id``, or the super-admin; a device by
    its bound person's role. Split out so the Needs-you counts read the same rule.
    """
    return _super_admin_or_member_in(db, ctx, WORKSPACE_ANSWERER_ROLES)


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


async def require_workspace_answerer(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db=Depends(get_db),
) -> RequestContext:
    """403 unless the caller may answer this workspace's questions and approvals (D16)."""
    if may_answer_workspace_questions(db, ctx):
        return ctx
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=ANSWERER_ONLY)

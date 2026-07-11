"""PRD-195 S2 (P2-14) — the one workspace-permission gate.

``require_workspace_permission(permission)`` is a Depends factory — the
platform's established gate idiom (``require_super_admin``,
``require_workspace_admin``, ``require_task_context``) — layered **after** the
shared hybrid dependency, which is never modified (PRD-09 precedent, 674 call
sites). It replaces the kwargs-sniffing ``@require_permission`` decorator that
was applied at exactly six endpoints, and is swept across every mutating
hybrid route by S3–S6 (the boundary-sweep test holds the coverage).

The auth-lane contract — explicit, and pinned by
``tests/test_p2w2_workspace_permission_gate.py``:

==============  =============================================================
lane            behaviour
==============  =============================================================
``clerk``       ``workspace_members.role`` for ``ctx.workspace_id`` (active
                row), with the legacy ``workspaces.owner_id`` fallback —
                checked against the matrix in ``modules/policy/roles.py``.
``super_admin`` (any lane) bypasses — the ONLY bypass. G3 narrowed the old
                decorator's ``system_role in (admin, super_admin)`` bypass:
                a plain ``admin`` (including the env-API-key principal,
                ``hybrid.py`` mints ``system_role='admin'``) now needs real
                membership like everyone else and is otherwise refused.
``api_key``     no bypass (see above) — no clerk identity ⇒ refused.
``anonymous``   treated as **owner of its resolved workspace**: only reachable
                when ``REQUIRE_AUTH=false`` / ``AUTH_EDITION=local`` — the
                local edition is a trusted single-user posture by definition
                (PRD-175) and must not 403 on every write.
``sdk_key``     never satisfies workspace-role gates — SDK scopes are the
                board/widget planes' own lane (untouched here).
==============  =============================================================

Cross-workspace drift guard: when the route carries a ``workspace_id`` path
parameter that differs from the caller's resolved ``ctx.workspace_id``, the
gate refuses (non-super-admin) — the role was resolved for ``ctx``'s tenant,
so it must not authorize writes addressed to another one.

Fail-closed: missing user, missing db, unknown role, no membership ⇒ 403,
with the missing permission named.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import text

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.super_admin import SUPER_ADMIN_ROLE
from core.database.database import get_db
from modules.policy.roles import WorkspaceRole, workspace_has_permission

# Marker attribute stamped on every gate dependency — the boundary-sweep test
# (tests/test_p2w2_authz_boundary_sweep.py) reads it off the live route table.
PERMISSION_MARKER_ATTR = "_workspace_permission"


def resolve_workspace_role(db: Any, ctx: Any) -> Optional[str]:
    """The caller's workspace role: active member row, else owner fallback.

    Lifted from the deleted ``core/workspaces/permissions.py`` decorator
    (member lookup + legacy ``workspaces.owner_id`` defence-in-depth). Returns
    ``None`` when the caller has no clerk identity, no workspace, or no
    membership — the gate fails closed on ``None``.
    """
    user = getattr(ctx, "user", None)
    clerk_uid = getattr(user, "clerk_user_id", None) if user else None
    ws_id = getattr(ctx, "workspace_id", None)
    if db is None or not clerk_uid or not ws_id:
        return None

    row = db.execute(
        text(
            "SELECT wm.role FROM workspace_members wm "
            "JOIN users u ON u.id = wm.user_id "
            "WHERE wm.workspace_id = :ws AND u.clerk_user_id = :cid "
            "  AND wm.is_active = true LIMIT 1"
        ),
        {"ws": str(ws_id), "cid": clerk_uid},
    ).fetchone()
    if row:
        return row[0]

    owner_row = db.execute(
        text(
            "SELECT 1 FROM workspaces w "
            "JOIN users u ON u.id = w.owner_id "
            "WHERE w.id = :ws AND u.clerk_user_id = :cid LIMIT 1"
        ),
        {"ws": str(ws_id), "cid": clerk_uid},
    ).fetchone()
    if owner_row:
        return WorkspaceRole.OWNER.value
    return None


def workspace_permission_granted(db: Any, ctx: Any, permission: str) -> bool:
    """Pure decision core: may this principal exercise ``permission`` in its
    resolved workspace? (Split from the dependency so it unit-tests without
    FastAPI — the ``may_see_own_workspace_health`` idiom.)"""
    user = getattr(ctx, "user", None)
    if user is None:
        return False

    # The ONLY bypass (G3): the platform super-admin.
    if getattr(user, "system_role", None) == SUPER_ADMIN_ROLE:
        return True

    auth_type = getattr(ctx, "auth_type", None)
    if auth_type == "sdk_key":
        # SDK keys have their own scope lane (board/widget) — a key is never
        # a workspace member.
        return False
    if auth_type == "anonymous":
        # Only reachable with REQUIRE_AUTH=false / AUTH_EDITION=local — the
        # trusted single-user posture owns its resolved workspace (PRD-175).
        return workspace_has_permission(WorkspaceRole.OWNER.value, permission)

    # clerk (and api_key, which has no clerk identity and therefore refuses):
    role = resolve_workspace_role(db, ctx)
    return workspace_has_permission(role, permission)


def require_workspace_permission(permission: str) -> Callable:
    """Dependency factory: 403 unless the caller's workspace role grants
    ``permission`` (matrix + wildcards in ``modules/policy/roles.py``)."""

    async def _dep(
        request: Request,
        ctx: RequestContext = Depends(get_request_context_hybrid),
        db=Depends(get_db),
    ) -> RequestContext:
        user = getattr(ctx, "user", None)
        is_super = bool(user) and getattr(user, "system_role", None) == SUPER_ADMIN_ROLE

        # Cross-workspace drift guard (non-su): a workspace_id PATH param must
        # match the tenant the role was resolved for.
        path_ws = request.path_params.get("workspace_id") if request.path_params else None
        if (
            not is_super
            and path_ws
            and getattr(ctx, "workspace_id", None) is not None
            and str(path_ws) != str(ctx.workspace_id)
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: workspace mismatch",
            )

        if workspace_permission_granted(db, ctx, permission):
            return ctx
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: {permission}",
        )

    setattr(_dep, PERMISSION_MARKER_ATTR, permission)
    _dep.__name__ = f"require_workspace_permission_{permission.replace(':', '_')}"
    return _dep

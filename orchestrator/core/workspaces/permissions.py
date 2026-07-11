"""Workspace-permission decorator — TRANSITIONAL (PRD-195 S1 → deleted in S2).

The role matrix and ``WorkspaceRole`` moved to the one authority,
``modules/policy/roles.py`` (PRD-195 S1). This module now only re-exports them
for its remaining importers and keeps the legacy kwargs-sniffing decorator
alive for ``api/team.py`` until S2 replaces it with the
``require_workspace_permission`` Depends factory and deletes this file.
"""
from functools import wraps
from typing import Optional

from fastapi import HTTPException

from modules.policy.roles import (  # noqa: F401 — re-exported for importers until S2
    ROLE_PERMISSIONS,
    WorkspaceRole,
    workspace_has_permission,
)


def require_permission(permission: str):
    """Decorator to require a specific workspace permission.

    Resolves the user's workspace role from ``workspace_members`` for the
    current ``ctx.workspace_id``. System admins (``system_role`` in
    {admin, super_admin}) bypass workspace checks. Falls back to
    ``workspaces.owner_id`` as defense-in-depth if the owner is missing
    from ``workspace_members`` (legacy data).
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from core.auth.dependencies import RequestContext
            from sqlalchemy.orm import Session
            from sqlalchemy import text

            ctx = kwargs.get("ctx")
            if not ctx:
                for arg in args:
                    if isinstance(arg, RequestContext):
                        ctx = arg
                        break

            if not ctx:
                raise HTTPException(status_code=403, detail="Permission denied: missing context")

            # System admins bypass workspace-scoped permission checks
            user = getattr(ctx, "user", None)
            if user is not None and getattr(user, "system_role", None) in ("admin", "super_admin"):
                return await func(*args, **kwargs)

            db = kwargs.get("db")
            if db is None:
                for arg in args:
                    if isinstance(arg, Session):
                        db = arg
                        break

            workspace_role: Optional[str] = None
            ws_id = getattr(ctx, "workspace_id", None)
            clerk_uid = getattr(user, "clerk_user_id", None) if user else None

            if db is not None and ws_id and clerk_uid:
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
                    workspace_role = row[0]

                # Defense-in-depth: legacy workspaces may lack a member row
                # for the owner. Treat workspace owner as 'owner'.
                if workspace_role is None:
                    owner_row = db.execute(
                        text(
                            "SELECT 1 FROM workspaces w "
                            "JOIN users u ON u.id = w.owner_id "
                            "WHERE w.id = :ws AND u.clerk_user_id = :cid LIMIT 1"
                        ),
                        {"ws": str(ws_id), "cid": clerk_uid},
                    ).fetchone()
                    if owner_row:
                        workspace_role = WorkspaceRole.OWNER.value

            if not workspace_role:
                raise HTTPException(status_code=403, detail=f"Permission denied: {permission}")

            if not workspace_has_permission(workspace_role, permission):
                raise HTTPException(status_code=403, detail=f"Permission denied: {permission}")

            return await func(*args, **kwargs)
        return wrapper
    return decorator

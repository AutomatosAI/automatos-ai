from enum import Enum
from typing import Set, Optional
from fastapi import HTTPException

class WorkspaceRole(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"

# Permission definitions
ROLE_PERMISSIONS: dict[WorkspaceRole, Set[str]] = {
    WorkspaceRole.OWNER: {
        "workspace:manage", "workspace:delete", "workspace:billing",
        "members:invite", "members:remove", "members:change_role",
        "members:read",
        "agents:*", "workflows:*", "documents:*", "knowledge:*",
        "audit:view"
    },
    WorkspaceRole.ADMIN: {
        "members:invite", "members:remove", "members:read",
        "agents:*", "workflows:*", "documents:*", "knowledge:*",
        "audit:view"
    },
    WorkspaceRole.EDITOR: {
        "members:read",
        "agents:create", "agents:read", "agents:update",
        "workflows:create", "workflows:read", "workflows:update",
        "documents:create", "documents:read", "documents:update",
        "knowledge:create", "knowledge:read", "knowledge:update"
    },
    WorkspaceRole.VIEWER: {
        "members:read",
        "agents:read", "workflows:read", "documents:read", "knowledge:read"
    }
}

def has_permission(role: str, permission: str) -> bool:
    """Check if a role has a specific permission."""
    try:
        role_enum = WorkspaceRole(role)
    except ValueError:
        return False
        
    permissions = ROLE_PERMISSIONS.get(role_enum, set())
    
    # Check exact match
    if permission in permissions:
        return True
    
    # Check wildcard (e.g., "agents:*" covers "agents:create")
    resource = permission.split(":")[0]
    if f"{resource}:*" in permissions:
        return True
    
    return False

from functools import wraps

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

            if not has_permission(workspace_role, permission):
                raise HTTPException(status_code=403, detail=f"Permission denied: {permission}")

            return await func(*args, **kwargs)
        return wrapper
    return decorator

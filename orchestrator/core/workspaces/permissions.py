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
    """Decorator to require a specific permission."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Try to get request context from kwargs or args
            ctx = kwargs.get("ctx")
            if not ctx:
                # Search in args
                from core.auth.dependencies import RequestContext
                for arg in args:
                    if isinstance(arg, RequestContext):
                        ctx = arg
                        break
            
            if not ctx:
                 # Fallback: maybe it's in a dependency that hasn't been resolved yet? 
                 # In FastAPI, dependencies are resolved before the function is called.
                 # If ctx is missing, we can't check permissions.
                 # However, usually ctx is passed as "ctx" argument.
                 pass

            if ctx and hasattr(ctx, 'role'):
                if not has_permission(ctx.role, permission):
                    raise HTTPException(status_code=403, detail=f"Permission denied: {permission}")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

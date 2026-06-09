from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from uuid import UUID


@dataclass(frozen=True)
class UserContext:
    """
    Minimal user principal for request-scoped authorization.

    This is intentionally lightweight: most handlers only need `id/email/role`.
    """

    id: Optional[str] = None
    email: Optional[str] = None
    role: str = "user"
    system_role: str = "user"

    # Auth-provider specific fields (optional)
    clerk_user_id: Optional[str] = None
    org_id: Optional[str] = None

    # Optional raw claims (useful for debugging/feature flags)
    raw_claims: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class RequestContext:
    """
    Request-scoped context injected via FastAPI dependencies.

    NOTE: `workspace_id` is a UUID in the DB models (PRD-37).
    """

    workspace_id: UUID
    user: UserContext
    auth_type: str = "anonymous"  # "clerk" | "api_key" | "sdk_key" | "anonymous"
    api_key_id: Optional[str] = None
    admin_all_workspaces: bool = False  # When True, endpoints should skip workspace filtering


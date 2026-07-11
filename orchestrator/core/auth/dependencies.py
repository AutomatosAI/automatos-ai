from __future__ import annotations

from dataclasses import InitVar, dataclass
from typing import Any, Dict, Optional
from uuid import UUID


@dataclass(frozen=True)
class UserContext:
    """
    Minimal user principal for request-scoped authorization.

    PRD-195 S8 (P2-14, G1): the ``role`` / ``system_role`` twins collapsed to
    ONE stored field — ``system_role`` (``super_admin | admin | user |
    service``). The dossier's C.1 finding was this exact confusion: callers
    read ``role`` expecting a workspace role and silently got the system twin.
    Workspace roles (owner/admin/editor/viewer) are resolved per-request from
    ``workspace_members`` (``core/auth/workspace_permission.py``), never
    stored here.

    ``role`` remains as an ACCEPTED-AND-IGNORED init parameter ONLY because
    ``core/auth/hybrid.py`` (the 674-site shared dependency this PRD must not
    modify) still passes ``role=`` alongside ``system_role=`` at its four
    mint sites. SUNSET: whichever PRD next owns ``core/auth/hybrid.py``
    (PRD-196's auth-consolidation lane is the natural owner) deletes those
    kwargs and this InitVar with them. Nothing may READ ``.role`` — there is
    no such attribute anymore.
    """

    id: Optional[str] = None
    email: Optional[str] = None
    role: InitVar[Optional[str]] = None  # deprecated twin — ignored (see docstring)
    system_role: str = "user"

    # Auth-provider specific fields (optional)
    clerk_user_id: Optional[str] = None
    org_id: Optional[str] = None

    # Optional raw claims (useful for debugging/feature flags)
    raw_claims: Optional[Dict[str, Any]] = None

    def __post_init__(self, role: Optional[str]) -> None:
        # The deprecated twin is deliberately discarded; system_role is the
        # one authority-facing field (modules/policy/roles.py).
        return None


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

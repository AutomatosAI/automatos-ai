from __future__ import annotations

import os
import logging
from typing import Optional
from uuid import UUID

from fastapi import HTTPException, Request, status
from sqlalchemy import text

from core.auth.clerk import get_clerk_auth
from core.auth.dependencies import RequestContext, UserContext
from core.database.database import SessionLocal

logger = logging.getLogger(__name__)


def _parse_uuid(value: Optional[str]) -> Optional[UUID]:
    if not value:
        return None
    try:
        return UUID(str(value).strip())
    except Exception:
        return None


def _get_workspace_id_from_request(request: Request) -> Optional[UUID]:
    """
    Resolve workspace_id from request headers/query/env.

    Supported (in priority order):
    - Header: x-workspace-id
    - Header: x-workspace
    - Query:  workspace_id
    - Env:    WORKSPACE_ID
    - Env:    DEFAULT_WORKSPACE_ID
    """

    # If some upstream middleware has already resolved workspace, trust it.
    override = getattr(getattr(request, "state", None), "workspace_id", None)
    parsed = _parse_uuid(str(override)) if override else None
    if parsed:
        return parsed

    header_candidates = [
        request.headers.get("x-workspace-id"),
        request.headers.get("x-workspace"),
    ]
    for candidate in header_candidates:
        parsed = _parse_uuid(candidate)
        if parsed:
            return parsed

    parsed = _parse_uuid(request.query_params.get("workspace_id"))
    if parsed:
        return parsed

    parsed = _parse_uuid(os.getenv("WORKSPACE_ID"))
    if parsed:
        return parsed

    parsed = _parse_uuid(os.getenv("DEFAULT_WORKSPACE_ID"))
    if parsed:
        return parsed

    return None


def _workspace_exists(db, workspace_id: UUID) -> bool:
    """Return True if workspace exists (and is active)."""
    row = db.execute(
        text("SELECT 1 FROM workspaces WHERE id = :id AND is_active = true LIMIT 1"),
        {"id": str(workspace_id)},
    ).fetchone()
    return bool(row)


def _resolve_workspace_for_clerk_user(db, clerk_user_id: Optional[str], org_id: Optional[str]) -> Optional[UUID]:
    """
    Resolve a user's workspace from the DB when the client didn't send X-Workspace-ID.

    Priority:
    - org workspace (workspaces.clerk_org_id == org_id)
    - personal workspace owned by mapped user (users.clerk_user_id -> workspaces.owner_id)
    - if exactly one active workspace exists globally, use it (dev-friendly)
    """
    if org_id:
        row = db.execute(
            text(
                "SELECT id FROM workspaces "
                "WHERE clerk_org_id = :org_id AND is_active = true "
                "ORDER BY updated_at DESC NULLS LAST "
                "LIMIT 1"
            ),
            {"org_id": org_id},
        ).fetchone()
        if row and row[0]:
            return row[0]

    if clerk_user_id:
        user_row = db.execute(
            text("SELECT id FROM users WHERE clerk_user_id = :cid LIMIT 1"),
            {"cid": clerk_user_id},
        ).fetchone()
        if user_row and user_row[0]:
            uid = user_row[0]
            ws_row = db.execute(
                text(
                    "SELECT id FROM workspaces "
                    "WHERE owner_id = :uid AND is_active = true "
                    "ORDER BY is_personal DESC, updated_at DESC NULLS LAST "
                    "LIMIT 1"
                ),
                {"uid": uid},
            ).fetchone()
            if ws_row and ws_row[0]:
                return ws_row[0]

    # Dev fallback (but still a real workspace from DB): only if unambiguous.
    rows = db.execute(
        text("SELECT id FROM workspaces WHERE is_active = true ORDER BY updated_at DESC NULLS LAST LIMIT 2")
    ).fetchall()
    if len(rows) == 1 and rows[0] and rows[0][0]:
        return rows[0][0]

    return None


def _get_api_key(request: Request) -> Optional[str]:
    # Common patterns
    api_key = request.headers.get("x-api-key") or request.headers.get("X-Api-Key")
    if api_key:
        return api_key.strip()

    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth:
        return None

    # Allow "ApiKey <key>" (optional)
    if auth.lower().startswith("apikey "):
        return auth.split(" ", 1)[1].strip()

    return None


def _get_bearer_token(request: Request) -> Optional[str]:
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth:
        return None
    if not auth.lower().startswith("bearer "):
        return None
    return auth.split(" ", 1)[1].strip()


async def get_request_context_hybrid(request: Request) -> RequestContext:
    """
    FastAPI dependency that supports:
    - Clerk JWT (Authorization: Bearer <jwt>) if Clerk JWKS is configured
    - API key (x-api-key) when ORCHESTRATOR_API_KEY/AUTOMATOS_API_KEY is set
    - Dev fallback (anonymous) when auth isn't configured
    """

    # Skip auth for OPTIONS requests (CORS preflight) - handled by CORS middleware
    # but we need to return early to avoid 401 errors in logs
    if request.method == "OPTIONS":
        # Return a minimal context for OPTIONS - won't be used but prevents errors
        return RequestContext(
            workspace_id=UUID("00000000-0000-0000-0000-000000000001"),
            user=UserContext(),
            auth_type="anonymous"
        )

    workspace_id = _get_workspace_id_from_request(request)

    # Secure-by-default: auth is required unless explicitly disabled
    _require_auth_raw = os.getenv("REQUIRE_AUTH", "true").strip().lower()
    require_auth = _require_auth_raw not in {"0", "false", "no", "off"}

    # Single DB session for all workspace resolution queries in this request
    db = SessionLocal()
    try:
        # 1) Clerk JWT
        bearer = _get_bearer_token(request)
        if bearer:
            clerk = get_clerk_auth()
            claims = clerk.verify_token(bearer)
            if claims:
                info = clerk.extract_user_info(claims)
                resolved = workspace_id or _resolve_workspace_for_clerk_user(
                    db,
                    clerk_user_id=info.get("clerk_user_id"),
                    org_id=info.get("org_id"),
                )
                if not resolved:
                    logger.warning(f"Auth failed: Workspace not resolved for user {info.get('email')}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Workspace not resolved. Ensure client sends X-Workspace-ID or user has a workspace.",
                    )
                if not _workspace_exists(db, resolved):
                    logger.warning(f"Auth failed: Workspace {resolved} does not exist")
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid workspace_id")

                user = UserContext(
                    id=info.get("clerk_user_id") or info.get("email"),
                    email=info.get("email"),
                    role=info.get("role") or "user",
                    system_role=info.get("system_role") or info.get("role") or "user",
                    clerk_user_id=info.get("clerk_user_id"),
                    org_id=info.get("org_id"),
                    raw_claims=claims,
                )
                return RequestContext(workspace_id=resolved, user=user, auth_type="clerk")

            # If a bearer token is present but invalid, treat as unauthorized.
            logger.warning("Auth failed: Invalid or expired Clerk token")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

        # 2) API key
        provided_key = _get_api_key(request)
        if provided_key:
            expected = (
                os.getenv("ORCHESTRATOR_API_KEY")
                or os.getenv("AUTOMATOS_API_KEY")
                or os.getenv("API_KEY")
            )
            import hmac as _hmac
            if expected and _hmac.compare_digest(provided_key, expected):
                user = UserContext(id="api_key", email=None, role="admin", system_role="admin")
                resolved = workspace_id
                if not resolved:
                    resolved = _parse_uuid(os.getenv("DEFAULT_WORKSPACE_ID")) or _resolve_workspace_for_clerk_user(
                        db, clerk_user_id=None, org_id=None
                    )
                if not resolved:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Workspace not resolved")
                if not _workspace_exists(db, resolved):
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid workspace_id")
                return RequestContext(
                    workspace_id=resolved, user=user, auth_type="api_key", api_key_id="env"
                )

            logger.warning("Auth failed: Invalid API key provided")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

        # 3) Dev fallback
        if require_auth:
            logger.warning("Auth failed: Authentication required but no credentials provided (REQUIRE_AUTH=true)")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

        # Keep noisy warnings down unless explicitly enabled
        if os.getenv("AUTH_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
            logger.info(
                "Auth note: No credentials, using anonymous context (REQUIRE_AUTH=false)."
            )

        # Anonymous requests (only reachable when REQUIRE_AUTH is explicitly disabled)
        resolved = workspace_id or _resolve_workspace_for_clerk_user(db, clerk_user_id=None, org_id=None)
        if not resolved:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Workspace not resolved. Send X-Workspace-ID header or configure DEFAULT_WORKSPACE_ID.",
            )
        return RequestContext(workspace_id=resolved, user=UserContext(), auth_type="anonymous")
    finally:
        db.close()


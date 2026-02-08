from __future__ import annotations

import os
import logging
import secrets
from typing import Optional
from uuid import UUID, uuid4

from fastapi import HTTPException, Request, status
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

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


def _workspace_exists(workspace_id: UUID) -> bool:
    """Return True if workspace exists (and is active)."""
    db = SessionLocal()
    try:
        row = db.execute(
            text("SELECT 1 FROM workspaces WHERE id = :id AND is_active = true LIMIT 1"),
            {"id": str(workspace_id)},
        ).fetchone()
        return bool(row)
    finally:
        db.close()


def _user_has_workspace_access(clerk_user_id: Optional[str], workspace_id: UUID) -> bool:
    """
    Verify that a Clerk user actually has access to the requested workspace.

    Prevents users from accessing other workspaces by spoofing X-Workspace-ID.
    Access is granted if the user owns the workspace OR is a member.
    """
    if not clerk_user_id:
        return False
    db = SessionLocal()
    try:
        row = db.execute(
            text(
                "SELECT 1 FROM users u "
                "LEFT JOIN workspaces w ON w.owner_id = u.id AND w.id = :ws_id "
                "LEFT JOIN workspace_members wm ON wm.user_id = u.id AND wm.workspace_id = :ws_id AND wm.is_active = true "
                "WHERE u.clerk_user_id = :cid AND (w.id IS NOT NULL OR wm.id IS NOT NULL) "
                "LIMIT 1"
            ),
            {"cid": clerk_user_id, "ws_id": str(workspace_id)},
        ).fetchone()
        return bool(row)
    finally:
        db.close()


def _provision_new_user_workspace(
    db, clerk_user_id: str, email: Optional[str] = None, name: Optional[str] = None,
) -> UUID:
    """
    Auto-provision a personal workspace for a new Clerk user.

    Creates:
    1. A `users` row (if missing) linked via clerk_user_id
    2. A personal `workspaces` row owned by that user
    3. A `workspace_members` row with role='owner'

    Returns the new workspace UUID.
    """
    # 1) Atomic upsert user — INSERT ON CONFLICT to avoid race conditions
    username = email or clerk_user_id
    try:
        db.execute(
            text(
                "INSERT INTO users (username, email, clerk_user_id, name, is_active) "
                "VALUES (:username, :email, :cid, :name, true) "
                "ON CONFLICT (clerk_user_id) DO NOTHING"
            ),
            {"username": username, "email": email or f"{clerk_user_id}@pending", "cid": clerk_user_id, "name": name},
        )
        db.flush()
    except IntegrityError:
        db.rollback()

    uid_row = db.execute(
        text("SELECT id FROM users WHERE clerk_user_id = :cid LIMIT 1"),
        {"cid": clerk_user_id},
    ).fetchone()
    uid = uid_row[0]
    logger.info("Resolved user record id=%s for clerk_user_id=%s", uid, clerk_user_id)

    # 2) Create personal workspace with slug collision handling
    ws_id = uuid4()
    ws_name = f"{name or email or 'My'}'s Workspace"
    base_slug = (email or clerk_user_id).split("@")[0].lower().replace(" ", "-")[:50]

    for attempt in range(5):
        slug = base_slug if attempt == 0 else f"{base_slug}-{secrets.token_hex(3)}"
        try:
            db.execute(
                text(
                    "INSERT INTO workspaces (id, name, slug, owner_id, is_personal, is_active, plan, plan_limits) "
                    "VALUES (:id, :name, :slug, :owner_id, true, true, 'starter', :plan_limits)"
                ),
                {
                    "id": str(ws_id),
                    "name": ws_name,
                    "slug": slug,
                    "owner_id": uid,
                    "plan_limits": '{"max_agents": 10, "max_workflows": 10, "max_documents": 100, "max_members": 5}',
                },
            )
            db.flush()
            break
        except IntegrityError:
            db.rollback()
            if attempt == 4:
                raise

    # 3) Create workspace_members row
    db.execute(
        text(
            "INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
            "VALUES (:ws_id, :uid, 'owner', true)"
        ),
        {"ws_id": str(ws_id), "uid": uid},
    )

    db.commit()
    logger.info(
        "Provisioned personal workspace %s (%s) for user %s (clerk=%s)",
        ws_id, ws_name, uid, clerk_user_id,
    )
    return ws_id


def _resolve_workspace_for_clerk_user(
    clerk_user_id: Optional[str],
    org_id: Optional[str],
    email: Optional[str] = None,
    name: Optional[str] = None,
) -> Optional[UUID]:
    """
    Resolve a user's workspace from the DB when the client didn't send X-Workspace-ID.

    Priority:
    1. Org workspace (workspaces.clerk_org_id == org_id)
    2. Personal workspace owned by mapped user
    3. Auto-provision a new personal workspace (new user signup)
    """
    db = SessionLocal()
    try:
        # 1) Org workspace
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

        # 2) Personal workspace via existing user record
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

        # 3) No workspace found — auto-provision for authenticated Clerk users
        if clerk_user_id:
            logger.info(
                "No workspace found for clerk_user_id=%s — provisioning new personal workspace",
                clerk_user_id,
            )
            return _provision_new_user_workspace(db, clerk_user_id, email=email, name=name)

        return None
    except Exception:
        db.rollback()
        logger.exception("Failed to resolve/provision workspace for clerk_user_id=%s", clerk_user_id)
        return None
    finally:
        db.close()


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

    require_auth = os.getenv("REQUIRE_AUTH", "").strip().lower() in {"1", "true", "yes", "on"}

    # 1) Clerk JWT
    bearer = _get_bearer_token(request)
    if bearer:
        clerk = get_clerk_auth()
        claims = clerk.verify_token(bearer)
        if claims:
            # ... (successful resolution logic)
            info = clerk.extract_user_info(claims)
            clerk_uid = info.get("clerk_user_id")

            # If client sent a workspace ID via header, verify the user has access
            if workspace_id:
                if not _workspace_exists(workspace_id):
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid workspace_id")
                if not _user_has_workspace_access(clerk_uid, workspace_id):
                    logger.warning("Access denied: user %s tried to access workspace %s", clerk_uid, workspace_id)
                    # Fall through to resolver instead of blocking — user may have a stale
                    # localStorage value from the old shared-workspace bug
                    workspace_id = None

            resolved = workspace_id or _resolve_workspace_for_clerk_user(
                clerk_user_id=clerk_uid,
                org_id=info.get("org_id"),
                email=info.get("email"),
                name=info.get("name"),
            )
            if not resolved:
                 logger.warning(f"Auth failed: Workspace not resolved for user {info.get('email')}")
                 raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Workspace not resolved. Ensure client sends X-Workspace-ID or user has a workspace.",
                )
            if not _workspace_exists(resolved):
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
        if expected and provided_key == expected:
            # ... (success logic)
            user = UserContext(id="api_key", email=None, role="admin", system_role="admin")
            resolved = workspace_id
            if not resolved:
                # Prefer env default if provided; otherwise use unambiguous DB workspace.
                resolved = _parse_uuid(os.getenv("DEFAULT_WORKSPACE_ID")) or _resolve_workspace_for_clerk_user(
                    clerk_user_id=None, org_id=None
                )
            if not resolved:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Workspace not resolved")
            if not _workspace_exists(resolved):
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

    # Anonymous requests
    resolved = workspace_id or _resolve_workspace_for_clerk_user(clerk_user_id=None, org_id=None)
    if not resolved:
        resolved = UUID("00000000-0000-0000-0000-000000000001")
    return RequestContext(workspace_id=resolved, user=UserContext(), auth_type="anonymous")


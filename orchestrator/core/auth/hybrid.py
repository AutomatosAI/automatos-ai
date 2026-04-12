from __future__ import annotations

import hmac as _hmac
import logging
import secrets
from typing import Optional
from uuid import UUID, uuid4

from fastapi import HTTPException, Request, status
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from config import config
from core.auth.clerk import get_clerk_auth
from core.auth.dependencies import RequestContext, UserContext
from core.database.database import SessionLocal

logger = logging.getLogger(__name__)


def _enrich_log_context(ctx: RequestContext) -> None:
    """Set ContextVars from resolved auth context for structured logging.

    Called after auth resolves so every downstream log entry
    carries workspace_id and user_id automatically.
    """
    try:
        from core.utils.logging_adapter import workspace_id_var, user_id_var, tenant_id_var
        if ctx.workspace_id:
            workspace_id_var.set(str(ctx.workspace_id))
            tenant_id_var.set(str(ctx.workspace_id))
        if ctx.user and ctx.user.id:
            user_id_var.set(str(ctx.user.id))
    except Exception:
        pass  # Never break auth for logging


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

    parsed = _parse_uuid(config.WORKSPACE_ID)
    if parsed:
        return parsed

    parsed = _parse_uuid(config.DEFAULT_WORKSPACE_ID)
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


def _user_is_workspace_member(db, workspace_id: UUID, clerk_user_id: Optional[str]) -> bool:
    """Return True if the Clerk user is an active member of the workspace."""
    if not clerk_user_id:
        return False
    row = db.execute(
        text(
            "SELECT 1 FROM workspace_members wm "
            "JOIN users u ON wm.user_id = u.id "
            "WHERE wm.workspace_id = :workspace_id "
            "AND u.clerk_user_id = :clerk_user_id "
            "AND wm.is_active = true "
            "LIMIT 1"
        ),
        {"workspace_id": str(workspace_id), "clerk_user_id": clerk_user_id},
    ).fetchone()
    return bool(row)


def _user_has_workspace_access(db, clerk_user_id: Optional[str], workspace_id: UUID) -> bool:
    """
    Verify that a Clerk user actually has access to the requested workspace.

    Prevents users from accessing other workspaces by spoofing X-Workspace-ID.
    Access is granted if the user owns the workspace OR is a member.
    """
    if not clerk_user_id:
        return False
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


def _provision_new_user_workspace(
    db, clerk_user_id: str, email: Optional[str] = None, name: Optional[str] = None,
) -> UUID:
    """
    Auto-provision a personal workspace for a new Clerk user.

    Creates:
    1. A ``users`` row (if missing) linked via clerk_user_id
    2. A personal ``workspaces`` row owned by that user
    3. A ``workspace_members`` row with role='owner'

    Returns the new workspace UUID.
    """
    # 1) Atomic upsert user -- INSERT ON CONFLICT to avoid race conditions
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

    webhook_key = uuid4().hex

    for attempt in range(5):
        slug = base_slug if attempt == 0 else f"{base_slug}-{secrets.token_hex(3)}"
        try:
            db.execute(
                text(
                    "INSERT INTO workspaces (id, name, slug, owner_id, is_personal, is_active, plan, plan_limits, webhook_key) "
                    "VALUES (:id, :name, :slug, :owner_id, true, true, 'starter', :plan_limits, :webhook_key)"
                ),
                {
                    "id": str(ws_id),
                    "name": ws_name,
                    "slug": slug,
                    "owner_id": uid,
                    "plan_limits": '{"max_agents": 10, "max_workflows": 10, "max_documents": 100, "max_members": 5}',
                    "webhook_key": webhook_key,
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

    # 4) Seed the Auto agent for this workspace (orchestrator default)
    try:
        from core.seeds.seed_auto_agent import seed_auto_agent
        seed_auto_agent(db, ws_id)
        db.commit()
    except Exception:
        logger.exception("Failed to seed Auto agent for workspace %s — non-fatal", ws_id)

    logger.info(
        "Provisioned personal workspace %s (%s) for user %s (clerk=%s)",
        ws_id, ws_name, uid, clerk_user_id,
    )
    return ws_id


def _resolve_workspace_for_clerk_user(
    db,
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

    # 3) No workspace found -- auto-provision for authenticated Clerk users
    if clerk_user_id:
        logger.info(
            "No workspace found for clerk_user_id=%s -- provisioning new personal workspace",
            clerk_user_id,
        )
        return _provision_new_user_workspace(db, clerk_user_id, email=email, name=name)

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

    # Check for admin "__all__" sentinel before UUID parsing discards it
    raw_ws_header = (request.headers.get("x-workspace-id") or "").strip()
    admin_all_workspaces = raw_ws_header == "__all__"

    # Secure-by-default: auth is required unless explicitly disabled
    require_auth = config.REQUIRE_AUTH

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
                clerk_uid = info.get("clerk_user_id")

                # Determine admin status
                system_role = info.get("system_role") or info.get("role") or "user"
                is_admin = system_role in ("admin", "super_admin")

                # Admin "__all__" sentinel: aggregate across all workspaces (no filter)
                if admin_all_workspaces and is_admin:
                    # Resolve admin's own workspace for the UserContext, but return
                    # workspace_id=None so endpoints return unfiltered (all workspaces)
                    admin_home_ws = _resolve_workspace_for_clerk_user(
                        db, clerk_user_id=clerk_uid,
                        org_id=info.get("org_id"),
                        email=info.get("email"),
                        name=info.get("name"),
                    )
                    user = UserContext(
                        id=info.get("clerk_user_id") or info.get("email"),
                        email=info.get("email"),
                        role=info.get("role") or "user",
                        system_role=system_role,
                        clerk_user_id=info.get("clerk_user_id"),
                        org_id=info.get("org_id"),
                        raw_claims=claims,
                    )
                    # workspace_id=None tells endpoints to skip workspace filter
                    result = RequestContext(workspace_id=admin_home_ws, user=user, auth_type="clerk", admin_all_workspaces=True)
                    _enrich_log_context(result)
                    return result

                # If client sent a workspace ID via header, verify the user has access
                if workspace_id:
                    if not _workspace_exists(db, workspace_id):
                        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid workspace_id")

                    # Admin bypass: admins can access any workspace
                    if not is_admin and not _user_has_workspace_access(db, clerk_uid, workspace_id):
                        logger.warning("Access denied: user %s tried to access workspace %s", clerk_uid, workspace_id)
                        # Fall through to resolver instead of blocking -- user may have a stale
                        # localStorage value from the old shared-workspace bug
                        workspace_id = None

                resolved = workspace_id or _resolve_workspace_for_clerk_user(
                    db,
                    clerk_user_id=clerk_uid,
                    org_id=info.get("org_id"),
                    email=info.get("email"),
                    name=info.get("name"),
                )
                if not resolved:
                    logger.warning("Auth failed: Workspace not resolved for user %s", info.get("email"))
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Workspace not resolved. Ensure client sends X-Workspace-ID or user has a workspace.",
                    )
                if not _workspace_exists(db, resolved):
                    logger.warning("Auth failed: Workspace %s does not exist", resolved)
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
                result = RequestContext(workspace_id=resolved, user=user, auth_type="clerk")
                _enrich_log_context(result)
                return result

            # If a bearer token is present but invalid, treat as unauthorized.
            logger.warning("Auth failed: Invalid or expired Clerk token")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

        # 2) API key
        provided_key = _get_api_key(request)
        if provided_key:
            expected = config.ORCHESTRATOR_API_KEY
            if expected and _hmac.compare_digest(provided_key, expected):
                user = UserContext(id="api_key", email=None, role="admin", system_role="admin")
                resolved = workspace_id
                if not resolved:
                    # Prefer env default if provided; otherwise resolve from DB.
                    resolved = _parse_uuid(config.DEFAULT_WORKSPACE_ID) or _resolve_workspace_for_clerk_user(
                        db, clerk_user_id=None, org_id=None
                    )
                if not resolved:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Workspace not resolved")
                if not _workspace_exists(db, resolved):
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid workspace_id")
                result = RequestContext(
                    workspace_id=resolved, user=user, auth_type="api_key", api_key_id="env"
                )
                _enrich_log_context(result)
                return result

            logger.warning("Auth failed: Invalid API key provided")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

        # 3) Dev fallback
        if require_auth:
            logger.warning("Auth failed: Authentication required but no credentials provided (REQUIRE_AUTH=true)")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

        # Keep noisy warnings down unless explicitly enabled
        if config.AUTH_DEBUG:
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
        result = RequestContext(workspace_id=resolved, user=UserContext(), auth_type="anonymous")
        _enrich_log_context(result)
        return result
    finally:
        db.close()

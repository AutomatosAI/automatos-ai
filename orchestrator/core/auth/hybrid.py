from __future__ import annotations

import hmac as _hmac
import logging
import secrets
import time
from typing import Callable, Optional
from urllib.parse import urlparse
from uuid import UUID, uuid4

from fastapi import HTTPException, Request, status
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from config import config
from core.auth.clerk import get_clerk_auth
from core.auth.dependencies import RequestContext, UserContext
from core.database.database import SessionLocal
from core.services.api_key_service import ApiKeyService

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
    """Return True if a non-deleted workspace row exists.

    Note: a *disabled* workspace (paused_at IS NOT NULL) still "exists" — the
    disabled-state gate is enforced separately via `_assert_workspace_usable`
    so we can return a clear 403 instead of a generic 400, and so admins can
    still load the workspace from the admin console.
    """
    row = db.execute(
        text(
            "SELECT 1 FROM workspaces "
            "WHERE id = :id AND deleted_at IS NULL LIMIT 1"
        ),
        {"id": str(workspace_id)},
    ).fetchone()
    return bool(row)


def _assert_workspace_usable(db, workspace_id: UUID, *, is_admin: bool) -> None:
    """Raise 403 if the workspace is disabled (paused) and the caller isn't admin.

    Admins always pass — they need to manage disabled workspaces from the admin
    console. Deleted workspaces are already filtered out by `_workspace_exists`.
    """
    if is_admin:
        return
    row = db.execute(
        text("SELECT paused_at FROM workspaces WHERE id = :id"),
        {"id": str(workspace_id)},
    ).fetchone()
    if row and row[0] is not None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Workspace is disabled. Contact an administrator.",
        )


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


def _has_pending_invitations(db, email: Optional[str]) -> bool:
    """Return True if the email has a pending (unaccepted, unexpired) invitation.

    Gates auto-provisioning so an invitee never silently lands in a personal
    workspace before the explicit /accept-invitation flow can run.
    """
    return _get_pending_invitation_token(db, email) is not None


def _get_pending_invitation_token(db, email: Optional[str]) -> Optional[str]:
    """Return the most recent pending invitation token for ``email``, or None.

    Used both to gate auto-provisioning and to surface the token in the 409
    response so the frontend can deep-link to /accept-invitation?token=… even
    if Clerk's redirect chain stripped query params from the original link.
    """
    if not email:
        return None
    row = db.execute(
        text(
            "SELECT token FROM workspace_invitations "
            "WHERE LOWER(email) = LOWER(:email) "
            "AND accepted_at IS NULL "
            "AND expires_at > NOW() "
            "ORDER BY created_at DESC "
            "LIMIT 1"
        ),
        {"email": email},
    ).fetchone()
    return row[0] if row else None


# PRD-128: Default notification routing seeded on workspace provisioning.
# Kept as a module-level constant so tests can import and assert against it
# without duplicating the list.
DEFAULT_NOTIFICATION_PREFERENCES: tuple[tuple[str, str], ...] = (
    ("heartbeat_complete", "in_app"),
    ("task_complete", "in_app"),
    ("mission_step_complete", "silent"),
    ("mission_complete", "in_app"),
    ("playbook_step_complete", "silent"),
    ("playbook_complete", "in_app"),
    ("trigger_fired", "in_app"),
    ("report_submitted", "in_app"),
    ("agent_error", "in_app"),
)


def _seed_default_notification_preferences(db, workspace_id: UUID) -> int:
    """Seed the 9 default notification preference rows for a new workspace.

    Idempotent: each row is inserted only when an identical
    ``(workspace_id, user_id IS NULL, event_type, destination)`` row does
    not already exist. There is no unique constraint on that tuple
    (fan-out to multiple destinations is allowed), so we guard with
    ``WHERE NOT EXISTS`` instead of ``ON CONFLICT``.

    Does **not** commit — the caller owns the transaction so preference
    inserts roll back with the workspace on failure.

    Returns the number of rows actually inserted (useful for logging /
    tests).
    """
    inserted = 0
    for event_type, destination in DEFAULT_NOTIFICATION_PREFERENCES:
        result = db.execute(
            text(
                "INSERT INTO notification_preferences "
                "(workspace_id, user_id, event_type, destination, enabled) "
                "SELECT :ws_id, NULL, :event_type, :destination, TRUE "
                "WHERE NOT EXISTS ("
                "  SELECT 1 FROM notification_preferences "
                "  WHERE workspace_id = :ws_id "
                "    AND user_id IS NULL "
                "    AND event_type = :event_type "
                "    AND destination = :destination"
                ")"
            ),
            {
                "ws_id": str(workspace_id),
                "event_type": event_type,
                "destination": destination,
            },
        )
        rowcount = getattr(result, "rowcount", 0) or 0
        inserted += rowcount
    return inserted


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

    # Serialize concurrent provisioning for the same user via transaction-scoped
    # advisory lock. Without this, two simultaneous requests for a brand-new
    # Clerk user both miss the SELECT and both INSERT a workspace — the slug
    # retry loop then silently succeeds with random suffixes, giving the user
    # N duplicate workspaces. Lock key uses a namespace (-hash of this call
    # site) to avoid colliding with other advisory locks.
    db.execute(
        text("SELECT pg_advisory_xact_lock(:ns, :uid)"),
        {"ns": 0x70726F76, "uid": int(uid)},  # 'prov'
    )

    # Re-check under the lock: another request may have just provisioned.
    existing = db.execute(
        text(
            "SELECT id FROM workspaces "
            "WHERE owner_id = :uid AND is_personal = true AND deleted_at IS NULL "
            "ORDER BY created_at ASC LIMIT 1"
        ),
        {"uid": uid},
    ).fetchone()
    if existing and existing[0]:
        logger.info(
            "Personal workspace already exists for user %s (id=%s) — skipping provisioning",
            uid, existing[0],
        )
        db.commit()
        return existing[0]

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
                    "VALUES (:id, :name, :slug, :owner_id, true, true, 'basic', :plan_limits, :webhook_key)"  # PRD-222 W2·S1: entry tier (renamed from 'starter')
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

    # PRD-128: Seed default notification preferences so the bell icon
    # starts working immediately without manual setup.
    seeded = _seed_default_notification_preferences(db, ws_id)
    logger.info("Seeded %d default notification preferences for workspace %s", seeded, ws_id)

    db.commit()

    # 4) Seed the Auto agent for this workspace (orchestrator default)
    try:
        from core.seeds.seed_auto_agent import seed_auto_agent
        seed_auto_agent(db, ws_id)
        db.commit()
    except Exception:
        logger.exception("Failed to seed Auto agent for workspace %s — non-fatal", ws_id)

    # 5) Seed starter document templates for this workspace
    try:
        from modules.documents.seed_templates import seed_starter_templates
        seed_starter_templates(db, ws_id)
        db.commit()
    except Exception:
        logger.exception("Failed to seed document templates for workspace %s — non-fatal", ws_id)

    # 5.5) 2026-08-29 (the Harbourline failure): every workspace starts with
    # WORKING models — base OpenRouter set in workspace_models + the primary in
    # settings.orchestrator. Zero-model workspaces broke Settings → Orchestrator
    # ("Failed to load LLM settings") and left chat with no sane default.
    try:
        from services.workspace_model_seeding import seed_workspace_models
        if seed_workspace_models(db, ws_id) is not None:
            db.commit()
    except Exception:
        logger.exception("Base-model seed failed for workspace %s — non-fatal", ws_id)

    # 6) PRD-222 US-004 (W1·S9): grant the one-time $5 onboarding trial credit so
    # the value moment lands before the BYOK ask. One per Clerk user; paused by
    # the global daily cap or the TRIAL_ENABLED kill switch (all config-driven).
    # Non-fatal — a workspace must provision even if the grant hiccups.
    try:
        from services.trial_ledger import grant_trial_at_provisioning
        if grant_trial_at_provisioning(db, ws_id, owner_id=uid) is not None:
            db.commit()
    except Exception:
        logger.exception("Trial grant failed for workspace %s — non-fatal", ws_id)
        try:
            db.rollback()
        except Exception:
            pass

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
    2. Any workspace the user can access (owned OR active membership), owner-biased
    3. Auto-provision a new personal workspace (only when no pending invitation)
    """
    # 1) Org workspace
    if org_id:
        row = db.execute(
            text(
                "SELECT id FROM workspaces "
                "WHERE clerk_org_id = :org_id AND is_active = true AND deleted_at IS NULL "
                "ORDER BY updated_at DESC NULLS LAST "
                "LIMIT 1"
            ),
            {"org_id": org_id},
        ).fetchone()
        if row and row[0]:
            return row[0]

    # 2) Workspace via ownership OR active membership.
    # Mirrors the LEFT JOIN access pattern in `_user_has_workspace_access` so
    # invitees who joined a workspace as members (not owners) resolve correctly
    # when no X-Workspace-ID header is present. Owner-bias keeps existing solo
    # users on their personal workspace; recency tiebreaker favours the most
    # recent join for users with multiple memberships.
    if clerk_user_id:
        ws_row = db.execute(
            text(
                "SELECT w.id FROM users u "
                "JOIN workspaces w "
                "  ON w.owner_id = u.id "
                "  OR EXISTS ("
                "    SELECT 1 FROM workspace_members wm "
                "    WHERE wm.workspace_id = w.id "
                "      AND wm.user_id = u.id "
                "      AND wm.is_active = true "
                "  ) "
                "LEFT JOIN workspace_members wm2 "
                "  ON wm2.workspace_id = w.id AND wm2.user_id = u.id AND wm2.is_active = true "
                "WHERE u.clerk_user_id = :cid "
                "  AND w.is_active = true "
                "  AND w.deleted_at IS NULL "
                "ORDER BY "
                "  CASE WHEN w.owner_id = u.id OR COALESCE(wm2.role, '') = 'owner' THEN 0 ELSE 1 END, "
                "  wm2.joined_at DESC NULLS LAST, "
                "  w.created_at DESC "
                "LIMIT 1"
            ),
            {"cid": clerk_user_id},
        ).fetchone()
        if ws_row and ws_row[0]:
            return ws_row[0]

    # 3) No workspace found -- auto-provision for authenticated Clerk users,
    # unless they have a pending invitation (which owns the consent UX).
    if clerk_user_id:
        if _has_pending_invitations(db, email):
            logger.info(
                "Pending invitation for %s — refusing auto-provision; caller must surface 409",
                email,
            )
            return None
        logger.info(
            "No workspace found for clerk_user_id=%s -- provisioning new personal workspace",
            clerk_user_id,
        )
        return _provision_new_user_workspace(db, clerk_user_id, email=email, name=name)

    return None


# ---------------------------------------------------------------------------
# PRD-233 S6 — the local operator's identity: ONE users row, resolved by email
# ---------------------------------------------------------------------------

# Backstop only: PUT /api/profile invalidates explicitly after every write; the
# TTL bounds staleness for writers that bypass the API (psql, a re-seed).
_LOCAL_OPERATOR_CACHE_TTL_SECONDS = 60.0
_LOCAL_OPERATOR_COLUMNS = ("id", "email", "name", "username", "avatar_url")
_local_operator_cache: dict = {}


def invalidate_local_operator_cache() -> None:
    """Drop the cached operator row so the next request re-reads ``users``."""
    _local_operator_cache.clear()


def _load_local_operator_row(db, email: str) -> Optional[dict]:
    """The operator's ``users`` row by email (the seed's lookup key) or None."""
    row = db.execute(
        text(
            "SELECT id, email, name, username, avatar_url FROM users "
            "WHERE email = :email LIMIT 1"
        ),
        {"email": email},
    ).fetchone()
    if not row:
        return None
    values = tuple(row[index] for index in range(len(_LOCAL_OPERATOR_COLUMNS)))
    # A real row carries an integer PK and the email it was looked up by;
    # anything else (a stubbed session in tests, a malformed row) is "no row".
    if not isinstance(values[0], int) or values[1] != email:
        return None
    return dict(zip(_LOCAL_OPERATOR_COLUMNS, values))


def _resolve_local_operator(db, email: str) -> Optional[dict]:
    """Cached read of the operator row; a miss is never cached (a seed that
    lands later must be picked up immediately) and a DB error never 500s the
    request — the caller falls back to PRD-209's email-only lane."""
    now = time.monotonic()
    cached = _local_operator_cache.get(email)
    if cached and (now - cached["fetched_at"]) < _LOCAL_OPERATOR_CACHE_TTL_SECONDS:
        return cached["row"]
    try:
        row = _load_local_operator_row(db, email)
    except Exception as exc:  # noqa: BLE001 — identity enrichment must never break auth
        logger.warning("Local operator row lookup failed (%s); using email-only lane", exc)
        try:
            db.rollback()
        except Exception:  # noqa: BLE001
            pass
        return None
    if row is None:
        logger.warning("Local operator row missing for %s — entrypoint seed did not run?", email)
        return None
    _local_operator_cache[email] = {"row": row, "fetched_at": now}
    return row


def _local_operator_user_context(db) -> Optional[UserContext]:
    """Bind the anonymous local session to the operator's ``users`` row.

    ``id`` carries the EMAIL, not the integer PK: every resolver in the tree
    maps ``ctx.user.id`` → ``users`` via ``clerk_user_id == ctx.user.id`` and
    then ``email == ctx.user.email`` (api/chat.py get_user_id, marketplace,
    notifications, workflow_recipes, harness, team), and the Clerk lane itself
    binds ``id = clerk_user_id or email``. An integer here would 500 the
    ``clerk_user_id == ctx.user.id`` sites (varchar = integer). The integer PK
    rides ``raw_claims["user_id"]`` for callers that want it without a query.
    """
    row = _resolve_local_operator(db, config.LOCAL_OPERATOR_EMAIL)
    if row is None:
        return None
    return UserContext(
        id=row["email"],
        email=row["email"],
        system_role="super_admin",
        raw_claims={
            "source": "local_operator",
            "user_id": row["id"],
            "name": row["name"],
            "username": row["username"],
            "avatar_url": row["avatar_url"],
        },
    )


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


def _extract_origin(request: Request) -> Optional[str]:
    """Return the request origin hostname from the Origin (or Referer) header.

    Shared by the board SDK path and the widget plane (``api/widgets/auth.py``
    imports this) so origin matching is identical across planes. Returns None
    when neither header is present.
    """
    origin = request.headers.get("Origin") or request.headers.get("Referer")
    if not origin:
        return None
    try:
        parsed = urlparse(origin)
        return parsed.hostname or origin
    except Exception:
        return origin


# ---------------------------------------------------------------------------
# SDK-key auth for the board plane (PRD-09 Slice 2, Step 0)
# ---------------------------------------------------------------------------
#
# A narrow, scope-gated path that lets a per-workspace SDK key (ak_pub_* /
# ak_srv_*) authenticate to the board task endpoints — WITHOUT modifying the
# shared `get_request_context_hybrid`, which backs ~96 routers / 657 call sites.
# Modifying that shared dependency would make every endpoint accept SDK keys,
# and since almost none of them check scopes, one key would grant full-platform
# access. So the SDK path is wired only into `api/board_tasks.py` reads via the
# `require_task_context` factory below.


def _sdk_key_has_scope(permissions: Optional[list], required_scope: str) -> bool:
    """Return True only when *permissions* EXPLICITLY grants *required_scope*.

    SECURITY — least privilege: an empty or null permission list grants
    NOTHING. The large population of already-issued publishable keys
    (chat/widget keys with null or unrelated permissions) must not be able to
    reach the task board. This board-plane rule is now the PLATFORM rule:
    PRD-195 S1 collapsed ``ApiKeyService.check_permissions`` and the widget
    plane's ``require_permission`` onto the same empty=deny semantic
    (``modules.policy.roles.has_permission``). No scope, no access.
    """
    if not permissions:
        return False
    return required_scope in permissions


def _resolve_sdk_key_context(
    request: Request, db, *, required_scope: str
) -> RequestContext:
    """Resolve an SDK key into a :class:`RequestContext` for the board plane.

    Mirrors the widget plane's key validation (``api/widgets/auth.py``) but adds
    the explicit scope gate and binds the workspace to the KEY only — never to
    env defaults, query params, or ``request.state`` (cross-tenant guard).
    """
    token = _get_bearer_token(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token"
        )

    record = ApiKeyService.validate_api_key(db, token)
    if record is None:
        logger.warning(
            "Board SDK auth: key rejected (prefix=%s…) — not found, revoked, or expired",
            token[:11],
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key",
        )

    # Least-privilege scope gate (see _sdk_key_has_scope).
    if not _sdk_key_has_scope(record.permissions, required_scope):
        logger.warning(
            "Board SDK auth: key %s lacks required scope %s",
            record.key_prefix, required_scope,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"API key lacks required scope: {required_scope}",
        )

    # Defense-in-depth origin check — only when a browser Origin is present.
    # (Desktop / ak_srv_ callers omit Origin; allowed_domains is a browser-CORS
    # control and gives server keys no protection — scope + workspace binding
    # are the real controls there.)
    origin = _extract_origin(request)
    if origin and not ApiKeyService.check_domain(record, origin):
        logger.warning(
            "Board SDK auth: origin %s not allowed for key %s",
            origin, record.key_prefix,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Origin not allowed for this API key",
        )

    # Cross-tenant guard: bind to the key's workspace. An X-Workspace-ID header
    # may only MATCH it, never override it. We never consult env defaults /
    # query params / request.state for an SDK key.
    workspace_id = record.workspace_id
    raw_ws_header = (request.headers.get("x-workspace-id") or "").strip()
    if raw_ws_header and _parse_uuid(raw_ws_header) != workspace_id:
        logger.warning(
            "Board SDK auth: workspace header mismatch for key %s", record.key_prefix
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Workspace mismatch between API key and header",
        )

    if not _workspace_exists(db, workspace_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid workspace_id"
        )
    _assert_workspace_usable(db, workspace_id, is_admin=False)

    result = RequestContext(
        workspace_id=workspace_id,
        user=UserContext(id=f"sdk:{record.id}", role="service", system_role="service"),
        auth_type="sdk_key",
        api_key_id=str(record.id),
    )
    _enrich_log_context(result)
    return result


def require_task_context(required_scope: str) -> Callable:
    """FastAPI dependency factory for the board plane.

    The returned dependency authenticates EITHER:
    - a per-workspace SDK key (``ak_pub_*`` / ``ak_srv_*``) bearing
      *required_scope* (via :func:`_resolve_sdk_key_context`), OR
    - anything ``get_request_context_hybrid`` already accepts (Clerk JWT, env
      admin key, anonymous, CORS preflight) — delegated UNCHANGED.

    Purely additive: existing Clerk/env callers behave exactly as before. Wired
    only into ``api/board_tasks.py``; the shared dependency is not modified, so
    SDK-key acceptance does not leak to the other routers.
    """

    async def _dep(request: Request) -> RequestContext:
        token = _get_bearer_token(request)
        if token and token.startswith(("ak_pub_", "ak_srv_")):
            db = SessionLocal()
            try:
                return _resolve_sdk_key_context(
                    request, db, required_scope=required_scope
                )
            finally:
                db.close()
        return await get_request_context_hybrid(request)

    return _dep


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
                    pending_token = _get_pending_invitation_token(db, info.get("email"))
                    if pending_token:
                        raise HTTPException(
                            status_code=status.HTTP_409_CONFLICT,
                            detail={
                                "code": "pending_invitation",
                                "redirect": f"/accept-invitation?token={pending_token}",
                                "token": pending_token,
                                "message": "Accept your pending invitation before continuing.",
                            },
                        )
                    logger.warning("Auth failed: Workspace not resolved for user %s", info.get("email"))
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Workspace not resolved. Ensure client sends X-Workspace-ID or user has a workspace.",
                    )
                if not _workspace_exists(db, resolved):
                    logger.warning("Auth failed: Workspace %s does not exist", resolved)
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid workspace_id")
                _assert_workspace_usable(db, resolved, is_admin=is_admin)

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
                # API keys are admin-equivalent (system_role="admin" above) — bypass disabled gate
                _assert_workspace_usable(db, resolved, is_admin=True)
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
        _assert_workspace_usable(db, resolved, is_admin=False)
        # PRD-209 local edition: there is exactly one operator and it is their
        # machine — the anonymous local session is the instance's SUPER admin
        # (there is no platform above the operator of a self-hosted instance:
        # system settings, credentials, admin analytics all belong to them).
        # In saas, the anonymous dev-fallback remains a plain user.
        if config.AUTH_EDITION == "local":
            # PRD-233 S6: bind the operator's seeded users row (email-keyed; see
            # _local_operator_user_context for why `id` is the email, never the
            # PK). A missing row keeps PRD-209's email-only lane — never a 500.
            anon_user = _local_operator_user_context(db) or \
                UserContext(email=config.LOCAL_OPERATOR_EMAIL, system_role="super_admin")
        else:
            anon_user = UserContext()
        result = RequestContext(workspace_id=resolved, user=anon_user, auth_type="anonymous")
        _enrich_log_context(result)
        return result
    finally:
        db.close()

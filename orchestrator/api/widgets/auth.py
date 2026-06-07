"""
Widget API Auth Middleware
==========================

FastAPI dependencies that authenticate incoming SDK requests using either:

1. **JWT session tokens** — short-lived tokens previously exchanged from an
   API key via the token-exchange endpoint (preferred).
2. **Raw API keys** — validated directly against the ``sdk_api_keys`` table
   as a fallback when no JWT is present.

Both paths produce a :class:`WidgetAuthContext` that downstream route
handlers receive via ``Depends(widget_auth)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from core.auth.hybrid import _extract_origin
from core.database.database import get_db
from core.services.api_key_service import ApiKeyService
from config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WIDGET_TOKEN_SECRET: str = config.WIDGET_TOKEN_SECRET or ""
WIDGET_TOKEN_ALGORITHM: str = "HS256"


# ---------------------------------------------------------------------------
# Auth context returned to route handlers
# ---------------------------------------------------------------------------

@dataclass
class WidgetAuthContext:
    """Resolved identity & permissions for an authenticated widget request."""

    workspace_id: UUID
    api_key_id: UUID
    permissions: List[str] = field(default_factory=list)
    default_agent_id: Optional[int] = None
    # PRD-124: Team lock — scope all requests through this key to a specific team
    team: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_bearer_token(request: Request) -> Optional[str]:
    """Pull the Bearer token from the Authorization header, if present."""
    auth_header: Optional[str] = request.headers.get("Authorization")
    if not auth_header:
        return None
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1]


def _try_jwt(token: str) -> Optional[dict]:
    """Attempt to decode *token* as a JWT.

    Returns the decoded payload on success, ``None`` on any failure
    (invalid signature, expiry, missing secret, etc.).
    """
    if not WIDGET_TOKEN_SECRET:
        return None
    try:
        payload = jwt.decode(
            token,
            WIDGET_TOKEN_SECRET,
            algorithms=[WIDGET_TOKEN_ALGORITHM],
        )
        return payload
    except jwt.ExpiredSignatureError:
        logger.debug("Widget JWT expired")
        return None
    except jwt.InvalidTokenError as exc:
        logger.debug("Widget JWT invalid: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Primary dependency — widget_auth
# ---------------------------------------------------------------------------

async def widget_auth(
    request: Request,
    db: Session = Depends(get_db),
) -> WidgetAuthContext:
    """FastAPI dependency that authenticates widget / SDK requests.

    Resolution order:

    1. Extract Bearer token from ``Authorization`` header.
    2. Try decoding it as a JWT session token (fast path).
    3. Fall back to raw API key validation via :class:`ApiKeyService`.
    4. Check the request origin against the key's ``allowed_domains``.
    5. Populate ``request.state`` and return :class:`WidgetAuthContext`.

    Raises:
        HTTPException 401: Missing / invalid / expired credentials.
        HTTPException 403: Origin not in the key's allowed domains.
    """

    token = _extract_bearer_token(request)
    origin_for_log = _extract_origin(request) or "?"
    if not token:
        logger.warning(
            "widget_auth: missing/invalid Authorization header (origin=%s, has_header=%s)",
            origin_for_log,
            bool(request.headers.get("Authorization")),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    workspace_header = request.headers.get("X-Workspace-ID")
    token_preview = token[:11] + "…" if len(token) > 11 else "(short)"

    # ----- 1. Try JWT session token first --------------------------------
    jwt_payload = _try_jwt(token)
    if jwt_payload is not None:
        try:
            workspace_id = UUID(jwt_payload["workspace_id"])
            api_key_id = UUID(jwt_payload["api_key_id"])
            permissions: List[str] = jwt_payload.get("permissions", [])
        except (KeyError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Malformed widget session token: {exc}",
            )

        # If both header and JWT carry a workspace, they must agree.
        if workspace_header and UUID(workspace_header) != workspace_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Workspace mismatch between token and header",
            )

        default_agent_id = jwt_payload.get("default_agent_id")
        ctx = WidgetAuthContext(
            workspace_id=workspace_id,
            api_key_id=api_key_id,
            permissions=permissions,
            default_agent_id=int(default_agent_id) if default_agent_id else None,
            team=jwt_payload.get("team"),
        )
        request.state.workspace_id = workspace_id
        request.state.api_key_id = api_key_id
        request.state.permissions = permissions
        return ctx

    # ----- 2. Fall back to raw API key validation -------------------------
    api_key_record = ApiKeyService.validate_api_key(db, token)
    if api_key_record is None:
        logger.warning(
            "widget_auth: API key rejected (prefix=%s, origin=%s) — not found, revoked, or expired",
            token_preview,
            origin_for_log,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key",
        )

    # ----- 3. Domain / origin check ---------------------------------------
    origin = _extract_origin(request)
    if origin and not ApiKeyService.check_domain(api_key_record, origin):
        logger.warning(
            "widget_auth: origin %s not in allowed_domains for key %s",
            origin,
            api_key_record.key_prefix,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Origin not allowed for this API key",
        )

    # If a workspace header was provided, it must match the key's workspace.
    if workspace_header:
        if UUID(workspace_header) != api_key_record.workspace_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Workspace mismatch between API key and header",
            )

    permissions = api_key_record.permissions or []
    ctx = WidgetAuthContext(
        workspace_id=api_key_record.workspace_id,
        api_key_id=api_key_record.id,
        permissions=permissions,
        default_agent_id=api_key_record.default_agent_id,
        team=getattr(api_key_record, "team", None),
    )
    request.state.workspace_id = api_key_record.workspace_id
    request.state.api_key_id = api_key_record.id
    request.state.permissions = permissions
    return ctx


# ---------------------------------------------------------------------------
# Permission guard — require_permission
# ---------------------------------------------------------------------------

def require_permission(permission: str) -> Callable:
    """Dependency factory that enforces a specific permission.

    Usage::

        @router.post("/chat")
        async def chat(
            auth: WidgetAuthContext = Depends(widget_auth),
            _perm=Depends(require_permission("widget:chat")),
        ):
            ...

    Raises:
        HTTPException 403: If the authenticated context lacks *permission*.
    """

    async def _check(
        auth: WidgetAuthContext = Depends(widget_auth),
    ) -> WidgetAuthContext:
        # An empty permissions list means unrestricted (all permissions).
        if auth.permissions and permission not in auth.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permission: {permission}",
            )
        return auth

    return _check

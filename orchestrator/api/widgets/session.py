"""
Widget Session Token Exchange
==============================

Lets SDK users exchange a **server-type** API key for a short-lived JWT
session token.  The resulting token can then be passed to browser clients
so they never need direct access to the secret key.

    POST /widgets/auth
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from uuid import UUID

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from core.database.database import get_db
from core.services.api_key_service import ApiKeyService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WIDGET_TOKEN_SECRET: str = os.getenv("WIDGET_TOKEN_SECRET", "")
WIDGET_TOKEN_ALGORITHM: str = "HS256"
MAX_EXPIRES_IN: int = 86_400   # 24 hours
DEFAULT_EXPIRES_IN: int = 3_600  # 1 hour

router = APIRouter(tags=["Widget Auth"])

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class SessionTokenRequest(BaseModel):
    """Body sent by a backend server to obtain a browser-safe session token."""

    api_key: str
    user_id: Optional[str] = None
    permissions: Optional[List[str]] = None
    expires_in: int = DEFAULT_EXPIRES_IN

    @field_validator("expires_in")
    @classmethod
    def clamp_expires_in(cls, v: int) -> int:
        return max(60, min(v, MAX_EXPIRES_IN))


class SessionTokenResponse(BaseModel):
    """Returned to the caller after a successful token exchange."""

    session_token: str
    expires_at: str
    permissions: List[str]
    workspace_id: str


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/auth", response_model=SessionTokenResponse)
async def exchange_session_token(
    request: SessionTokenRequest,
    db: Session = Depends(get_db),
) -> SessionTokenResponse:
    """Exchange a server API key for a short-lived JWT session token."""

    # --- 0. Ensure signing is configured --------------------------------
    if not WIDGET_TOKEN_SECRET:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Widget token signing not configured",
        )

    # --- 1. Validate the API key ----------------------------------------
    api_key_record = ApiKeyService.validate_api_key(db, request.api_key)
    if api_key_record is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key",
        )

    # --- 2. Only server keys may exchange for session tokens -------------
    if api_key_record.key_type != "server":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only server-type API keys can exchange for session tokens",
        )

    # --- 3. Resolve effective permissions --------------------------------
    key_permissions: List[str] = api_key_record.permissions or []

    if request.permissions is not None:
        # Requested permissions must be a subset of the key's permissions.
        # An empty key_permissions list means unrestricted (all allowed).
        if key_permissions:
            extra = set(request.permissions) - set(key_permissions)
            if extra:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Requested permissions not granted by API key: {sorted(extra)}",
                )
        effective_permissions = request.permissions
    else:
        effective_permissions = key_permissions

    # --- 4. Build and sign the JWT --------------------------------------
    now = datetime.now(timezone.utc)
    exp = now + timedelta(seconds=request.expires_in)

    payload: dict = {
        "workspace_id": str(api_key_record.workspace_id),
        "api_key_id": str(api_key_record.id),
        "permissions": effective_permissions,
        "exp": exp,
        "iat": now,
    }

    if request.user_id is not None:
        payload["user_id"] = request.user_id

    if api_key_record.default_agent_id is not None:
        payload["default_agent_id"] = api_key_record.default_agent_id

    session_token = jwt.encode(
        payload,
        WIDGET_TOKEN_SECRET,
        algorithm=WIDGET_TOKEN_ALGORITHM,
    )

    # --- 5. Return response ---------------------------------------------
    return SessionTokenResponse(
        session_token=session_token,
        expires_at=exp.isoformat(),
        permissions=effective_permissions,
        workspace_id=str(api_key_record.workspace_id),
    )

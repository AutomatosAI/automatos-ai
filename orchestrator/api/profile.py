"""PRD-233 S6 — the profile behind the session: who Auto is talking to.

GET /api/profile (both editions)
    Identity from the request context plus the resolved ``users`` row when one
    exists. Resolution order is the tree's own (api/chat.py ``get_user_id``):
    ``users.clerk_user_id == ctx.user.id`` then ``users.email == ctx.user.email``
    — NEVER ``ctx.user.id`` as ``users.id`` (it is a Clerk string in saas and
    the operator's email in local).

PUT /api/profile (LOCAL edition only)
    Edits the operator's row — ``name`` / ``username`` / ``avatar_url``. Email is
    READ-ONLY: it is the lookup key the anonymous lane resolves the row by
    (``LOCAL_OPERATOR_EMAIL`` in ``.env``), so changing it here would orphan
    the session from its own row. In saas the profile is managed by the
    identity provider (Clerk) → 403. Every write invalidates hybrid.py's
    operator-row cache so the next request already carries the new name.
"""
from __future__ import annotations

import logging
import re
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid, invalidate_local_operator_cache
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.core import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/profile", tags=["profile"])

# Editing the operator's identity is a workspace-owner act in the single-operator
# edition (the anonymous local session IS the owner — PRD-175 lane contract).
# Module constant so tests override the exact dependency object.
_PROFILE_WRITE_GATE = require_workspace_permission("workspace:manage")

LOCAL_EDITION = "local"
EMAIL_NOTE_LOCAL = (
    "Email is the operator lookup key — it is set by LOCAL_OPERATOR_EMAIL in .env "
    "and cannot be changed here."
)
EMAIL_NOTE_SAAS = "Managed by your identity provider."
MANAGED_BY_IDENTITY_PROVIDER = "Profile is managed by your identity provider"
OPERATOR_ROW_MISSING = "Local operator row not found — the entrypoint seed did not run"
USERNAME_TAKEN = "username is already taken"

_HTML_MARKERS = re.compile(r"[<>]")
_ALLOWED_AVATAR_SCHEMES = ("http", "https")
_USERNAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
# Columns a blank value clears (NULL); username is NOT NULL + unique and may not be blanked.
_CLEARABLE_FIELDS = frozenset({"name", "avatar_url"})


class ProfileOut(BaseModel):
    edition: str
    editable: bool
    id: Optional[int] = None
    email: Optional[str] = None
    name: Optional[str] = None
    username: Optional[str] = None
    avatar_url: Optional[str] = None
    system_role: str
    email_note: str


class ProfileUpdate(BaseModel):
    """Omitted fields are unchanged; ``name`` / ``avatar_url`` accept "" to clear."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: Optional[str] = Field(default=None, max_length=255)
    username: Optional[str] = Field(
        default=None, min_length=1, max_length=255, pattern=_USERNAME_PATTERN
    )
    avatar_url: Optional[str] = Field(default=None, max_length=500)

    @field_validator("name")
    @classmethod
    def _name_has_no_markup(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and _HTML_MARKERS.search(value):
            raise ValueError("name must not contain HTML")
        return value

    @field_validator("avatar_url")
    @classmethod
    def _avatar_is_http_url_or_blank(cls, value: Optional[str]) -> Optional[str]:
        if not value:
            return value
        parsed = urlparse(value)
        if parsed.scheme not in _ALLOWED_AVATAR_SCHEMES or not parsed.netloc:
            raise ValueError("avatar_url must be an http(s) URL")
        return value


def _resolve_user_row(db: Session, ctx: RequestContext) -> Optional[User]:
    """The caller's ``users`` row, or None — same order as api/chat.py get_user_id."""
    user = ctx.user
    if user is None:
        return None
    clerk_uid = user.clerk_user_id or (user.id if isinstance(user.id, str) else None)
    row = None
    if clerk_uid:
        row = db.query(User).filter(User.clerk_user_id == clerk_uid).first()
    if row is None and user.email:
        row = db.query(User).filter(User.email == user.email).first()
    return row


def _to_out(ctx: RequestContext, row: Optional[User]) -> ProfileOut:
    is_local = config.AUTH_EDITION == LOCAL_EDITION
    return ProfileOut(
        edition=config.AUTH_EDITION,
        editable=is_local,
        id=row.id if row is not None else None,
        email=row.email if row is not None else ctx.user.email,
        name=row.name if row is not None else None,
        username=row.username if row is not None else None,
        avatar_url=row.avatar_url if row is not None else None,
        system_role=ctx.user.system_role,
        email_note=EMAIL_NOTE_LOCAL if is_local else EMAIL_NOTE_SAAS,
    )


def _update_values(body: ProfileUpdate) -> dict:
    """Column values for the fields the client actually sent (blank clears)."""
    changes = body.model_dump(exclude_unset=True)
    return {
        field: (value or None) if field in _CLEARABLE_FIELDS else value
        for field, value in changes.items()
    }


@router.get("", response_model=ProfileOut)
async def get_profile(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> ProfileOut:
    """Who the platform thinks is talking (both editions)."""
    return _to_out(ctx, _resolve_user_row(db, ctx))


@router.put("", response_model=ProfileOut, dependencies=[Depends(_PROFILE_WRITE_GATE)])
async def update_profile(
    body: ProfileUpdate,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> ProfileOut:
    """Edit the local operator's profile row (local edition only)."""
    if config.AUTH_EDITION != LOCAL_EDITION:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=MANAGED_BY_IDENTITY_PROVIDER)

    operator = User.email == config.LOCAL_OPERATOR_EMAIL
    if db.query(User.id).filter(operator).first() is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=OPERATOR_ROW_MISSING)

    values = _update_values(body)
    if values:
        try:
            db.query(User).filter(operator).update(values, synchronize_session=False)
            db.commit()
        except IntegrityError:
            db.rollback()
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=USERNAME_TAKEN)
        invalidate_local_operator_cache()
        logger.info("Local operator profile updated: %s", sorted(values))

    return _to_out(ctx, db.query(User).filter(operator).first())

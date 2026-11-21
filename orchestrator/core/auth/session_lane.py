"""PRD-277 Wave 3 (§5.3, US-301, US-302): the session lane, a signed-in person's ``ams_`` token.

Reached only through the bearer-prefix registry (``core/auth/bearer_registry.py``), and only where
the edition issues sessions (``issues_sessions``: enterprise). In ``local`` and ``saas`` the
registry registers no ``ams_`` lane and reads no cookie, so those editions answer an ``ams_``
token exactly as they did before this lane existed.

* **Where the token comes from.** ``Authorization: Bearer ams_…`` (``resolve_session_bearer``),
  or, when the request has no ``Authorization`` header and no API key, the session cookie
  (``resolve_session_cookie``). A cookie-authenticated request that changes something must pass
  ``session_cookie.assert_cookie_request_allowed`` first, before anything is looked up.
* **The session.** Found by its token's digest and confirmed in constant time
  (``core.auth.sessions.find_session``). Unknown, revoked or expired is
  ``401 {"code": "session_ended", "reason": "unknown" | "revoked" | "expired", "message": …}``,
  which the web app recognises; the reason goes to the log too, never the token. When the token
  came from the cookie, the answer also clears the cookie (``session_cookie.CookieSessionEnded``).
* **The person** is the session's ``users`` row, by ``users.id``; a deactivated one is
  ``401 session_ended`` (``revoked``). Their install-wide role is the one the install stores
  (``core.auth.system_role.stored_system_role``), else ``user``.
* **The workspace** is the one the request names (``X-Workspace-ID``, ``X-Workspace`` or
  ``?workspace_id=``), else ``DEFAULT_WORKSPACE_ID``. The person must own it or be an active
  member of it, read by ``users.id`` (``core.auth.person.may_open_workspace``, the rule the Clerk
  lane applies), or be an install admin, who may open any workspace as on the Clerk lane; otherwise
  403 (``workspace_required`` when nothing was named).
  It must exist (400) and be usable (hybrid's ``_assert_workspace_usable``: paused is 403 for a
  non-admin). An install admin's ``X-Workspace-ID: __all__`` asks for every workspace, as there.
* **The context** binds the person as the device lane binds its person: ``id`` and ``email`` are
  the email, and ``raw_claims`` carries ``{source: "auth_session", session_id, user_id,
  auth_time}``, so ``core.auth.principal.resolve_user_pk`` reads the ``users.id`` without a query
  and step-up can read ``auth_time`` (``session_auth_time``).
* ``last_seen_at`` and the idle clock are written at most once every five minutes
  (``sessions.touch_session``), never on every request.

The lookup runs in the threadpool on its own session, always closed. ``core.auth.hybrid`` imports
the registry, which imports this module, so hybrid's helpers are imported at call time.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from types import ModuleType
from typing import Any, NamedTuple, Optional, Tuple
from uuid import UUID

from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from config import config
from core.auth import person as person_resolver
from core.auth import session_cookie, sessions
from core.auth.dependencies import RequestContext, UserContext
from core.auth.device import WORKSPACE_HEADERS, WORKSPACE_QUERY_PARAM
from core.auth.principal import AUTH_SESSION_CLAIM_SOURCE
from core.auth.roles import caller_is_admin
from core.auth.system_role import stored_system_role
from core.database.database import SessionLocal
from core.models.core import User

logger = logging.getLogger(__name__)

SESSION_AUTH_TYPE = "session"
PERSON_ROLE = "user"
ALL_WORKSPACES = "__all__"

SESSION_ENDED = "session_ended"
SESSION_ENDED_MESSAGE = "Your session has ended. Sign in again."
WORKSPACE_REQUIRED = "workspace_required"
WORKSPACE_REQUIRED_MESSAGE = "You are signed in, and not in a workspace yet. Ask an administrator to add you."
NOT_A_MEMBER = "You are not a member of this workspace."
INVALID_WORKSPACE = "Invalid workspace_id"
ADMINS_ONLY = "Only an install admin can open every workspace."


class Person(NamedTuple):
    """The session's person: ``users.id``, email and install-wide role."""

    user_pk: int
    email: str
    system_role: str


def _hybrid() -> ModuleType:
    """``core.auth.hybrid``, imported at call time (it imports the registry that imports us)."""
    from core.auth import hybrid

    return hybrid


def _now() -> datetime:
    return datetime.now(timezone.utc)


def session_ended(reason: str) -> HTTPException:
    """The 401 the web app recognises; the reason goes to the log, the token never does."""
    logger.warning("Session lane: session refused (%s)", reason)
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"code": SESSION_ENDED, "reason": reason, "message": SESSION_ENDED_MESSAGE},
    )


def _person(db: Session, user_pk: int) -> Optional[Person]:
    """The session's person, or None when the row is gone or deactivated."""
    row = db.query(User.id, User.email).filter(User.id == user_pk, User.is_active.isnot(False)).first()
    if row is None or not row[1]:
        return None
    return Person(int(row[0]), row[1], stored_system_role(db, int(row[0])) or PERSON_ROLE)


def _named_workspaces(request: Any) -> Tuple[str, ...]:
    """Every workspace the request names, as sent (blank values name none)."""
    sent = [request.headers.get(header) for header in WORKSPACE_HEADERS]
    sent.append(request.query_params.get(WORKSPACE_QUERY_PARAM))
    return tuple(value.strip() for value in sent if isinstance(value, str) and value.strip())


def _requested_workspace(request: Any, hybrid: ModuleType) -> Tuple[Optional[UUID], bool]:
    """The workspace asked for and whether the request named it; 400 for an unparsable or split ask."""
    named = _named_workspaces(request)
    if not named:
        return hybrid._parse_uuid(config.DEFAULT_WORKSPACE_ID), False
    parsed = {hybrid._parse_uuid(value) for value in named}
    if None in parsed or len(parsed) != 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=INVALID_WORKSPACE)
    return parsed.pop(), True


def _not_in_workspace(named: bool) -> HTTPException:
    if named:
        return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=NOT_A_MEMBER)
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                         detail={"code": WORKSPACE_REQUIRED, "message": WORKSPACE_REQUIRED_MESSAGE})


def _bound_workspace(db: Session, request: Any, person: Person, is_admin: bool, hybrid: ModuleType) -> UUID:
    """The workspace this request runs in, which the person may use, or 400/403."""
    workspace_id, named = _requested_workspace(request, hybrid)
    if workspace_id is None:
        raise _not_in_workspace(named)
    if not hybrid._workspace_exists(db, workspace_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=INVALID_WORKSPACE)
    if not is_admin and not person_resolver.may_open_workspace(db, person_resolver.by_user_pk(person.user_pk),
                                                               workspace_id):
        raise _not_in_workspace(named)
    hybrid._assert_workspace_usable(db, workspace_id, is_admin=is_admin)
    return workspace_id


def _user_context(row: Any, person: Person) -> UserContext:
    """The session's person, bound as the device lane binds its person."""
    auth_time = row.auth_time if row.auth_time.tzinfo else row.auth_time.replace(tzinfo=timezone.utc)
    return UserContext(
        id=person.email,
        email=person.email,
        system_role=person.system_role,
        raw_claims={"source": AUTH_SESSION_CLAIM_SOURCE, "session_id": str(row.id),
                    "user_id": person.user_pk, "auth_time": auth_time.timestamp()},
    )


def _context(db: Session, request: Any, row: Any, person: Person, hybrid: ModuleType) -> RequestContext:
    """The request's context: one workspace, or every workspace for an install admin who asks."""
    user = _user_context(row, person)
    is_admin = caller_is_admin(user)
    if (request.headers.get("x-workspace-id") or "").strip() == ALL_WORKSPACES:
        if not is_admin:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=ADMINS_ONLY)
        home = hybrid._parse_uuid(config.DEFAULT_WORKSPACE_ID)
        return RequestContext(workspace_id=home, user=user, auth_type=SESSION_AUTH_TYPE, admin_all_workspaces=True)
    workspace_id = _bound_workspace(db, request, person, is_admin, hybrid)
    return RequestContext(workspace_id=workspace_id, user=user, auth_type=SESSION_AUTH_TYPE)


def _resolve_in_own_session(request: Any, token: str) -> RequestContext:
    """Find, check and bind the session on a database session of its own, always closed."""
    hybrid = _hybrid()
    now = _now()
    db = SessionLocal()
    try:
        row = sessions.find_session(db, token)
        state = sessions.session_state(row, now)
        if state != sessions.LIVE:
            raise session_ended(state)
        person = _person(db, row.user_id)
        if person is None:
            raise session_ended(sessions.REVOKED)
        context = _context(db, request, row, person, hybrid)
        sessions.touch_session(db, row, now)
        return context
    finally:
        db.close()


async def _resolve(request: Request, token: str) -> RequestContext:
    context = await run_in_threadpool(_resolve_in_own_session, request, token)
    _hybrid()._enrich_log_context(context)
    return context


async def resolve_session_bearer(request: Request, token: str) -> RequestContext:
    """The registry's resolver for ``Authorization: Bearer ams_…``: never subject to the origin check."""
    return await _resolve(request, token)


def _is_session_ended(refused: HTTPException) -> bool:
    detail = refused.detail
    return refused.status_code == status.HTTP_401_UNAUTHORIZED and isinstance(detail, dict) \
        and detail.get("code") == SESSION_ENDED


async def resolve_session_cookie(request: Request, token: str) -> RequestContext:
    """The registry's resolver for the session cookie: the origin check first, then the session.

    An ended session is ``session_cookie.CookieSessionEnded``: the same 401, whose answer also
    clears the cookie. Every other refusal is raised as it is.
    """
    session_cookie.assert_cookie_request_allowed(request)
    try:
        return await _resolve(request, token)
    except HTTPException as refused:
        if not _is_session_ended(refused):
            raise
        ended = session_cookie.CookieSessionEnded(status_code=refused.status_code, detail=refused.detail)
    raise ended


def session_auth_time(ctx: Any) -> Optional[float]:
    """When a session context's person last proved who they are (epoch seconds), else None.

    The value step-up reads in an edition whose people step up by signing in again (US-406).
    Only a context this lane made carries one.
    """
    if getattr(ctx, "auth_type", None) != SESSION_AUTH_TYPE:
        return None
    claims = getattr(getattr(ctx, "user", None), "raw_claims", None)
    if not isinstance(claims, dict) or claims.get("source") != AUTH_SESSION_CLAIM_SOURCE:
        return None
    value = claims.get("auth_time")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


__all__ = [
    "SESSION_AUTH_TYPE",
    "SESSION_ENDED",
    "WORKSPACE_REQUIRED",
    "resolve_session_bearer",
    "resolve_session_cookie",
    "session_auth_time",
    "session_ended",
]

"""PRD-277 US-303 (§5.6): signing out, and an install admin ending sessions (``/api/v1/auth``).

Unless the edition issues sessions (``issues_sessions``: enterprise), each route answers its
declared method with 404 before any authentication or database work. A method a path does not
declare still answers 405, as FastAPI answers it in every edition, so the paths are not hidden.

* ``POST /sign-out`` — the caller's own session, presented as ``Authorization: Bearer ams_…`` or,
  with no ``Authorization`` header, as the session cookie (which must pass the cross-site check,
  US-302). A live session is revoked, one ``auth:sign_out`` row is written, and the answer is 204
  with the cookie cleared. An unknown, revoked or expired one is ``401 session_ended``, and the
  cookie is cleared all the same. Its credential is the session itself, never the shared request
  context, so a person in no workspace can still sign out.
* ``GET /people/{user_id}/sessions`` — an install admin lists a person's sessions, newest first:
  their clocks, address, browser and end, never a token or a hash.
* ``POST /sessions/{session_id}/revoke`` — an install admin ends one session.
* ``POST /people/{user_id}/sessions/revoke`` — an install admin ends every session of a person.
* ``POST /people/{user_id}/deactivate`` — an install admin deactivates a person: their sessions and
  their devices end in the same transaction (``core.auth.people.deactivate_person``).

The three admin changes require step-up (``require_step_up(STEP_UP_MAX_AGE_MINUTES)``). An install
admin is a person on the session lane whose stored install-wide role is ``admin`` or
``super_admin`` (``_require_admin(ctx)``): no key, device or other lane passes. An admin acts only
on a person whose install-wide role theirs satisfies, and never deactivates themselves. An unknown
person or session is 404. Each change is one ``audit_logs`` row in the workspace the admin acted
from. Routes are plain ``def``: the sync session runs in the threadpool. Registered through
``RouterSpec("api.auth_sessions")`` in ``router_manifest.py``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from api.credentials import get_client_ip
from config import config
from core import editions
from core.auth import people, session_cookie, sessions
from core.auth.bearer_registry import bearer_token, session_cookie_token
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.principal import claimed_user_pk
from core.auth.roles import caller_is_admin, caller_role
from core.auth.session_lane import SESSION_AUTH_TYPE, SESSION_ENDED, SESSION_ENDED_MESSAGE
from core.auth.step_up import require_step_up
from core.auth.system_role import UnknownPerson
from core.database.database import get_db
from core.models.auth_sessions import AuthSession
from core.models.core import User
from modules.policy.roles import USER, role_satisfies

# What FastAPI answers for a path it does not serve: the body a closed route answers with.
NOT_FOUND = "Not Found"
ADMINS_ONLY = "Only an install admin signed in to Automatos can end sessions or deactivate people."
OUTRANKED = "This person's install role is above yours."
NOT_YOURSELF = "You cannot deactivate yourself."
NO_SUCH_PERSON = "No such person."
NO_SUCH_SESSION = "No such session."
LISTED_SESSIONS_MAX = 100


def _sessions_issued() -> None:
    """404 each route's declared method where the edition issues no sessions, before anything else."""
    if not editions.current().issues_sessions:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=NOT_FOUND)


router = APIRouter(prefix="/api/v1/auth", tags=["auth"], dependencies=[Depends(_sessions_issued)])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _browser(request: Request) -> Tuple[Optional[str], Optional[str]]:
    """The caller's address and agent, for the audit row."""
    return get_client_ip(request), request.headers.get("user-agent")


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


# --------------------------------------------------------------------------- signing out


def _presented_session_token(request: Request) -> Tuple[Optional[str], bool]:
    """The session token the caller presents, and whether it came from the cookie."""
    if request.headers.get("authorization"):
        return bearer_token(request), False
    cookie = session_cookie_token(request)
    return cookie, cookie is not None


def _ended(reason: str) -> Response:
    """401 ``session_ended``, the cookie cleared so the browser stops sending it."""
    response = JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={
        "detail": {"code": SESSION_ENDED, "reason": reason, "message": SESSION_ENDED_MESSAGE},
    })
    session_cookie.clear_session_cookie(response)
    return response


@router.post("/sign-out", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def sign_out(request: Request, db: Session = Depends(get_db)) -> Response:
    """Revoke the caller's own session and clear the cookie (US-303)."""
    token, from_cookie = _presented_session_token(request)
    if from_cookie:
        session_cookie.assert_cookie_request_allowed(request)
    row = sessions.find_session(db, token)
    state = sessions.session_state(row, _now())
    if state != sessions.LIVE:
        return _ended(state)
    ip, agent = _browser(request)
    people.sign_out(db, row.id, people.Requester(user_id=row.user_id, workspace_id=None,
                                                 ip_address=ip, user_agent=agent))
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    session_cookie.clear_session_cookie(response)
    return response


# --------------------------------------------------------------------------- an install admin


def _require_admin(ctx: RequestContext) -> int:
    """The acting install admin's ``users.id``; 403 for anyone else (a key, a device, a member)."""
    actor = claimed_user_pk(ctx) if ctx.auth_type == SESSION_AUTH_TYPE else None
    if actor is None or not caller_is_admin(ctx.user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=ADMINS_ONLY)
    return actor


def _assert_may_act_on(db: Session, ctx: RequestContext, user_id: int) -> None:
    """404 for an unknown person; 403 when their install-wide role is above the admin's."""
    row = db.query(User.id, User.system_role).filter(User.id == user_id).first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=NO_SUCH_PERSON)
    if not role_satisfies(caller_role(ctx.user), row[1] or USER):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=OUTRANKED)


def _requester(request: Request, ctx: RequestContext, actor: int) -> people.Requester:
    ip, agent = _browser(request)
    return people.Requester(user_id=actor, workspace_id=ctx.workspace_id, ip_address=ip, user_agent=agent)


def _view(row: AuthSession) -> Dict[str, Any]:
    """A session as an admin sees it: never its token or its hash."""
    return {
        "id": str(row.id), "user_id": row.user_id, "created_at": _iso(row.created_at),
        "last_seen_at": _iso(row.last_seen_at), "idle_expires_at": _iso(row.idle_expires_at),
        "absolute_expires_at": _iso(row.absolute_expires_at), "auth_time": _iso(row.auth_time),
        "state": sessions.session_state(row, _now()), "revoked_at": _iso(row.revoked_at),
        "revoked_reason": row.revoked_reason, "ip_address": row.ip_address, "user_agent": row.user_agent,
    }


@router.get("/people/{user_id}/sessions")
def list_sessions(user_id: int, ctx: RequestContext = Depends(get_request_context_hybrid),
                  db: Session = Depends(get_db)) -> Dict[str, List[Dict[str, Any]]]:
    """A person's sessions, newest first, for an install admin."""
    _require_admin(ctx)
    _assert_may_act_on(db, ctx, user_id)
    rows = (db.query(AuthSession).filter(AuthSession.user_id == user_id)
            .order_by(AuthSession.created_at.desc(), AuthSession.id).limit(LISTED_SESSIONS_MAX).all())
    return {"sessions": [_view(row) for row in rows]}


@router.post("/sessions/{session_id}/revoke",
             dependencies=[Depends(require_step_up(config.STEP_UP_MAX_AGE_MINUTES))])
def revoke_session(session_id: UUID, request: Request, ctx: RequestContext = Depends(get_request_context_hybrid),
                   db: Session = Depends(get_db)) -> Dict[str, bool]:
    """End one session (US-303); ``revoked`` is false when it had already ended."""
    actor = _require_admin(ctx)
    owner = db.query(AuthSession.user_id).filter(AuthSession.id == session_id).first()
    if owner is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=NO_SUCH_SESSION)
    _assert_may_act_on(db, ctx, owner[0])
    return {"revoked": people.end_session(db, session_id, _requester(request, ctx, actor))}


@router.post("/people/{user_id}/sessions/revoke",
             dependencies=[Depends(require_step_up(config.STEP_UP_MAX_AGE_MINUTES))])
def revoke_persons_sessions(user_id: int, request: Request,
                            ctx: RequestContext = Depends(get_request_context_hybrid),
                            db: Session = Depends(get_db)) -> Dict[str, int]:
    """End every session of a person (US-303); how many were live."""
    actor = _require_admin(ctx)
    _assert_may_act_on(db, ctx, user_id)
    return {"revoked": people.end_sessions_of(db, user_id, _requester(request, ctx, actor))}


@router.post("/people/{user_id}/deactivate",
             dependencies=[Depends(require_step_up(config.STEP_UP_MAX_AGE_MINUTES))])
def deactivate_person(user_id: int, request: Request, ctx: RequestContext = Depends(get_request_context_hybrid),
                      db: Session = Depends(get_db)) -> Dict[str, int]:
    """Deactivate a person: their sessions and devices end in the same transaction (US-303)."""
    actor = _require_admin(ctx)
    if actor == user_id:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=NOT_YOURSELF)
    _assert_may_act_on(db, ctx, user_id)
    try:
        done = people.deactivate_person(db, user_id, _requester(request, ctx, actor))
    except UnknownPerson:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=NO_SUCH_PERSON) from None
    return {"sessions_revoked": done.sessions_ended, "devices_revoked": done.devices_revoked}

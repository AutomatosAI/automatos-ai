"""PRD-277 US-501, US-502 (§5.3): what the web app asks before and after a person signs in (``/api/v1/auth``).

Unless the edition issues sessions (``issues_sessions``: enterprise), both routes answer 404 before
any authentication or database work, so ``local`` and ``saas`` serve exactly what they served before.

* ``GET /me`` — who the browser's session belongs to: the person (``id``, ``email``, ``name``,
  their install-wide ``system_role``), every usable workspace they can open with their role in each
  (``core.auth.person.workspaces_of``), whether they are in no workspace yet, the install's default
  workspace and the edition. Its credential is the session itself, as sign-out's is
  (``api.auth_sessions``): ``Authorization: Bearer ams_…`` or, with no ``Authorization`` header, the
  session cookie. It never rides the shared request context, because that context binds one
  workspace and refuses a person who is in none (``workspace_required``, US-404) before any handler
  runs: the web app must still learn who they are, to say they are waiting. An unknown, revoked or
  expired session, or a deactivated person, is ``401 session_ended`` with the cookie cleared. A GET
  changes nothing the person asked for, so it is not subject to the cross-site check; it moves the
  session's idle clock as every request does (``sessions.touch_session``). ``Cache-Control:
  no-store``. The role comes from the install's own store (``stored_system_role``), never the
  browser.
* ``GET /sign-in-page`` — what the sign-in page shows before anyone is signed in: the provider's
  name for its one button (``OIDC_DISPLAY_NAME``, checked at start-up to be short plain text). It
  reads one setting and nothing else, so it is public. The discovery document cannot carry it: it
  refuses in enterprise until the phone's sign-in is built (US-602).

Both are plain ``def``. Registered through ``RouterSpec("api.auth_web")`` in ``router_manifest.py``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

# Sign-out's own credential, its session_ended answer and its 404 where no session is issued, shared
# so the routes agree.
from api.auth_sessions import _ended as session_ended_answer
from api.auth_sessions import _presented_session_token as presented_session_token
from api.auth_sessions import _sessions_issued
from config import config
from core import editions
from core.auth import person as person_resolver
from core.auth import sessions
from core.auth.session_lane import PERSON_ROLE
from core.auth.system_role import stored_system_role
from core.auth.user_link import PENDING_EMAIL_DOMAIN
from core.database.database import get_db
from core.models.core import User

NO_STORE = {"Cache-Control": "no-store"}

router = APIRouter(prefix="/api/v1/auth", tags=["auth"], dependencies=[Depends(_sessions_issued)])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _shown_email(email: Optional[str]) -> Optional[str]:
    """The person's email, or None for the placeholder of an identity with no verified address."""
    return email if email and not email.lower().endswith(PENDING_EMAIL_DOMAIN) else None


def _who(db: Session, person: User) -> Dict[str, Any]:
    """The answer for a live session's active person."""
    held = person_resolver.workspaces_of(db, person_resolver.by_user_pk(person.id))
    return {
        "edition": editions.current_name(),
        "person": {
            "id": person.id,
            "email": _shown_email(person.email),
            "name": person.name or None,
            "system_role": stored_system_role(db, person.id) or PERSON_ROLE,
        },
        "workspaces": [{"id": one.id, "name": one.name, "role": one.role} for one in held],
        "in_no_workspace": not held,
        "default_workspace_id": (config.DEFAULT_WORKSPACE_ID or "").strip() or None,
    }


@router.get("/me", response_class=Response)
def who_am_i(request: Request, db: Session = Depends(get_db)) -> Response:
    """The session's person, their install role and their workspaces (US-501); 401 once it has ended."""
    token, _ = presented_session_token(request)
    row = sessions.find_session(db, token)
    now = _now()
    state = sessions.session_state(row, now)
    if state != sessions.LIVE:
        return session_ended_answer(state)
    person = db.query(User).filter(User.id == row.user_id, User.is_active.isnot(False)).first()
    if person is None:
        return session_ended_answer(sessions.REVOKED)
    answer = _who(db, person)
    sessions.touch_session(db, row, now)
    return JSONResponse(content=answer, headers=NO_STORE)


@router.get("/sign-in-page")
def sign_in_page() -> JSONResponse:
    """The sign-in page's one button names the provider (US-502); public, one setting, nothing else."""
    return JSONResponse(content={"display_name": config.OIDC_DISPLAY_NAME}, headers=NO_STORE)


__all__ = ["router"]

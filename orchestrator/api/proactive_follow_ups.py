"""PRD-268 §6.3 'In the thread', K8, K22 — the person's follow-ups (``/api/proactive/follow-ups``).

The follow-ups Auto promised in a person's chat ("I'll check again at 15:10"), as that person
sees them, and nobody else (K8):

* ``GET /follow-ups?chat_id=&status=pending|all`` — the caller's own follow-ups in this
  workspace, pending by default (soonest first); ``all`` adds those that fired or were
  cancelled (newest first); ``chat_id`` keeps one chat's; at most ``follow_ups.LIST_LIMIT``
  (100: the pending cap is 5, so only old history is ever left out). ``{follow_ups: [{id, at,
  reason, status, chat_id}]}``: ``at`` is the UTC ISO time and ``status`` pending, fired or
  cancelled.
  The reason was written by the model in the person's own chat: it is theirs to read, and a
  client draws it as plain text.
* ``POST /follow-ups/{follow_up_id}/cancel`` — cancel the caller's own pending follow-up:
  ``{follow_up}``, now cancelled, its job removed and the cancel audited with the person as
  its actor. One that already fired is 409 ``follow_up_fired``, one already cancelled 409
  ``follow_up_cancelled``.

Another person's follow-up and another workspace's are 404 ``follow_up_not_found``, never
403: the list never shows them, and the cancel cannot tell them from none. A caller with no
person behind it is 403 ``follow_up_no_person``, and a call with no workspace 404
``no_workspace``. A database that cannot be read or written is 503. A refusal is ``{"code",
"detail"}``, its sentence in English or Portuguese (Brazil) by ``Accept-Language`` (K13).

Reading is every role's (``members:read``, as ``GET /api/proactive/me``). The cancel is
``require_workspace_permission("agents:execute")`` (K22, the authz sweep's class (a)): the
scope a chat with Auto needs (``api/chat.py``), so whoever could have been promised a
follow-up can cancel it. Plain ``def`` routes on the sync session, registered through
``RouterSpec("api.proactive_follow_ups")`` in ``router_manifest.py``.
``services.proactive.follow_ups`` holds the rules.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Path, Query
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.principal import resolve_user_pk
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from modules.proactive.locale import request_locale
from services.proactive import follow_ups
from services.proactive.decision_text import say

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/proactive", tags=["proactive"])

# Every workspace role holds it: the caller's own list.
READ_PERMISSION = "members:read"
# The scope a chat with Auto needs, which schedules every follow-up: the caller's own cancel.
CANCEL_PERMISSION = "agents:execute"
# A follow-up's number, as the follow-up tools take it.
MAX_FOLLOW_UP_ID = 10 ** 18
NO_WORKSPACE = "no_workspace"
NO_PERSON = "follow_up_no_person"
NOT_FOUND = "follow_up_not_found"
FIRED = "follow_up_fired"
CANCELLED = "follow_up_cancelled"
UNREADABLE = "follow_ups_unreadable"
UNAVAILABLE = "follow_up_unavailable"
HTTP_FORBIDDEN, HTTP_NOT_FOUND, HTTP_CONFLICT, HTTP_UNAVAILABLE = 403, 404, 409, 503
# Why a cancel did not happen, by what the service says: not the caller's (or none), or already done.
NOT_CANCELLED = {
    follow_ups.NOT_FOUND: (NOT_FOUND, HTTP_NOT_FOUND),
    follow_ups.FIRED: (FIRED, HTTP_CONFLICT),
    follow_ups.CANCELLED: (CANCELLED, HTTP_CONFLICT),
}

ListStatus = Literal["pending", "all"]


def _refused(locale: str, code: str, status_code: int) -> JSONResponse:
    return JSONResponse({"code": code, "detail": say(locale, code)}, status_code=status_code)


def _unavailable(db: Session, locale: str, what: str, failure: Exception, code: str) -> JSONResponse:
    """Roll back and refuse with 503: the failure and its stack go to the server log, never the body."""
    db.rollback()
    logger.error("follow-up %s failed", what, exc_info=failure)
    return _refused(locale, code, HTTP_UNAVAILABLE)


@router.get("/follow-ups", response_model=None)
def list_my_follow_ups(
    chat_id: Optional[UUID] = Query(None),
    status: ListStatus = Query(follow_ups.PENDING),
    accept_language: Optional[str] = Header(None),
    ctx: RequestContext = Depends(require_workspace_permission(READ_PERMISSION)),
    db: Session = Depends(get_db),
) -> Union[Dict[str, Any], JSONResponse]:
    """The caller's own follow-ups in this workspace: pending ones by default, or all; in one chat when named."""
    locale = request_locale(accept_language)
    if ctx.workspace_id is None:
        return _refused(locale, NO_WORKSPACE, HTTP_NOT_FOUND)
    try:
        user_pk = resolve_user_pk(db, ctx)
        if user_pk is None:
            return _refused(locale, NO_PERSON, HTTP_FORBIDDEN)
        found = follow_ups.list_follow_ups(db, ctx.workspace_id, user_pk, chat_id=chat_id,
                                           only_pending=status == follow_ups.PENDING)
    except SQLAlchemyError as failure:
        return _unavailable(db, locale, "list", failure, UNREADABLE)
    return {"follow_ups": [item.to_dict() for item in found]}


@router.post("/follow-ups/{follow_up_id}/cancel", response_model=None)
def cancel_my_follow_up(
    follow_up_id: int = Path(..., gt=0, lt=MAX_FOLLOW_UP_ID),
    accept_language: Optional[str] = Header(None),
    ctx: RequestContext = Depends(require_workspace_permission(CANCEL_PERMISSION)),
    db: Session = Depends(get_db),
) -> Union[Dict[str, Any], JSONResponse]:
    """Cancel the caller's own pending follow-up; 409 when it already fired or was cancelled."""
    locale = request_locale(accept_language)
    if ctx.workspace_id is None:
        return _refused(locale, NO_WORKSPACE, HTTP_NOT_FOUND)
    try:
        user_pk = resolve_user_pk(db, ctx)
        if user_pk is None:
            return _refused(locale, NO_PERSON, HTTP_FORBIDDEN)
        outcome, cancelled = follow_ups.cancel_follow_up(db, ctx.workspace_id, user_pk, follow_up_id,
                                                         actor_type=follow_ups.ACTOR_PERSON)
    except SQLAlchemyError as failure:
        return _unavailable(db, locale, "cancel", failure, UNAVAILABLE)
    if cancelled is None:
        code, status_code = NOT_CANCELLED.get(outcome, NOT_CANCELLED[follow_ups.NOT_FOUND])
        return _refused(locale, code, status_code)
    return {"follow_up": cancelled.to_dict()}

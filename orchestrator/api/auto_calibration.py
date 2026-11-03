"""PRD-268 §6.11.5, K15, K16, K22 — calibration on day one (``/api/auto/calibration``).

Auto asks once, then says back what it thinks it knows about the person's work, from the titles
of their recent work only, so they can strike what is wrong and keep what is right:

* ``GET /calibration`` — ``{available, offer, consent_line, status, calibration, rejected_lines,
  accepted_lines}``: ``offer`` is true while the person was never offered it (or their offer was
  abandoned: its claimant died before posting, ``claim.abandoned``) and may answer it;
  ``status`` is null before that, else ``declined``, ``open`` or ``answered``; ``calibration``
  is the ``Calibration@1`` once built.
* ``POST /calibration`` ``{consent, chat_id?, locale?}`` — the answer to the consent line, once
  per person per workspace (``services.calibration.offer``): false records ``declined`` and
  reads nothing; true builds one calibration and posts it as one Auto message into ``chat_id``
  when that is the caller's own chat, else their Auto thread, answering ``{status,
  calibration, chat_id, message_id}``. A second POST is 409 ``already_offered`` with the
  ``record`` as it is; a build or post that fails is 503 ``calibration_unavailable`` and the
  person is offered again.
* ``POST /calibration/{calibration_id}/lines/{line_id}/reject`` and ``POST
  /calibration/{calibration_id}/accept`` (Sounds right) — the person's answer, remembered through
  the memory service (``services.calibration.answers``, K16): the record, ``not_stored`` (the
  lines an exclusion rule kept out of memory) and a ``note`` that says so. Each is idempotent;
  rejecting a line once the calibration was answered is 409 ``calibration_answered``; a memory
  that was not written is 503 ``memory_unavailable`` and changes nothing.

Ships OFF (K15, ``CALIBRATION_ENABLED``): while it is off GET answers ``available`` false and
reads no record, and every POST is 404 ``not_available``. Another person's calibration and
another workspace's are 404 ``calibration_not_found``, never 403. A caller with no person behind
it, or who does not belong to the workspace, is 403; a call with no workspace 404; a database
that cannot be read or written 503. A refusal is ``{"code", "detail"}``, its sentence in the
caller's language (``locale``, else ``Accept-Language``: W2's K7 chooser).

GET is every role's (``members:read``). The POSTs are ``require_workspace_permission("agents:execute")``
(K22, the authz sweep's class (a)): the scope a chat with Auto needs (``api/chat.py``), since
each acts on the caller's own record only. Plain ``def`` routes on the sync session (the build
makes its one model call synchronously; a memory write runs on the app's loop through
``anyio.from_thread``), registered through ``RouterSpec("api.auto_calibration")`` in
``router_manifest.py``.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Path, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, StrictBool, field_validator
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.chat_models import clean_locale
from core.auth.dependencies import RequestContext
from core.auth.principal import resolve_user_pk
from core.auth.workspace_permission import require_workspace_permission, workspace_permission_granted
from core.database.database import get_db
from core.models.proactive import ProactiveCalibration
from modules.proactive.locale import request_locale
from services.calibration import answers, claim, offer
from services.calibration.strings import calibration_language, consent_line, say
from services.calibration.switch import calibration_enabled
from services.mobile.digest import belongs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auto", tags=["auto"])

# Every workspace role holds it: the caller's own calibration, read.
READ_PERMISSION = "members:read"
# The scope a chat with Auto needs: the caller's own consent and answers.
ANSWER_PERMISSION = "agents:execute"
# A line's id, as Calibration@1 gives it ("line-1").
LINE_ID_MAX = 32
LINE_ID_PATTERN = r"^[A-Za-z0-9_-]+$"
LOCALE_MAX = 35
NOT_AVAILABLE = "not_available"
NO_WORKSPACE = "calibration_no_workspace"
NO_PERSON = "calibration_no_person"
NOT_A_MEMBER = "calibration_not_a_member"
ALREADY_OFFERED = "already_offered"
ANSWERED = "calibration_answered"
UNAVAILABLE = "calibration_unavailable"
MEMORY_UNAVAILABLE = "memory_unavailable"
NOT_STORED = "not_stored"
HTTP_FORBIDDEN, HTTP_NOT_FOUND, HTTP_CONFLICT, HTTP_UNAVAILABLE = 403, 404, 409, 503


class CalibrationConsent(BaseModel):
    """``POST /calibration``: the answer to the consent line, the chat the person is in, their language."""

    model_config = ConfigDict(extra="forbid")

    consent: StrictBool
    chat_id: Optional[UUID] = None
    locale: Optional[str] = None

    @field_validator("locale", mode="before")
    @classmethod
    def _a_tag_or_nothing(cls, value: Any) -> Optional[str]:
        return clean_locale(value)


class _Refused(Exception):
    """A refusal the route answers as ``{"code", "detail"}``."""

    def __init__(self, code: str, status_code: int) -> None:
        super().__init__(code)
        self.code, self.status_code = code, status_code


def _refused(language: str, code: str, status_code: int, **more: Any) -> JSONResponse:
    return JSONResponse({"code": code, "detail": say(language, code), **more}, status_code=status_code)


def _unavailable(db: Session, language: str, what: str, failure: Exception) -> JSONResponse:
    """Roll back and refuse with 503: the failure and its stack go to the server log, never the body."""
    db.rollback()
    logger.error("calibration %s failed", what, exc_info=failure)
    return _refused(language, UNAVAILABLE, HTTP_UNAVAILABLE)


def _person(db: Session, ctx: RequestContext) -> int:
    """The caller's ``users.id`` while they belong to the workspace (403 otherwise; 404 with no workspace)."""
    if ctx.workspace_id is None:
        raise _Refused(NO_WORKSPACE, HTTP_NOT_FOUND)
    user_pk = resolve_user_pk(db, ctx)
    if user_pk is None:
        raise _Refused(NO_PERSON, HTTP_FORBIDDEN)
    if not belongs(db, ctx.workspace_id, user_pk):
        raise _Refused(NOT_A_MEMBER, HTTP_FORBIDDEN)
    return user_pk


def _read_view(language: str, record: Optional[ProactiveCalibration], may_answer: bool) -> Dict[str, Any]:
    saved = claim.view(record) if record is not None else None
    offered = record is None or claim.abandoned(record, datetime.now(timezone.utc))
    return {
        "available": True,
        "offer": offered and may_answer,
        "consent_line": consent_line(language),
        "status": saved["status"] if saved else None,
        "calibration": saved["calibration"] if saved else None,
        "rejected_lines": saved["rejected_lines"] if saved else [],
        "accepted_lines": saved["accepted_lines"] if saved else [],
    }


def _not_available_view(language: str) -> Dict[str, Any]:
    return {"available": False, "offer": False, "consent_line": consent_line(language), "status": None,
            "calibration": None, "rejected_lines": [], "accepted_lines": []}


@router.get("/calibration", response_model=None)
def read_calibration(
    locale: Optional[str] = Query(None, max_length=LOCALE_MAX),
    accept_language: Optional[str] = Header(None),
    ctx: RequestContext = Depends(require_workspace_permission(READ_PERMISSION)),
    db: Session = Depends(get_db),
) -> Union[Dict[str, Any], JSONResponse]:
    """The caller's calibration in this workspace, and whether to offer it."""
    said = calibration_language(clean_locale(locale) or request_locale(accept_language))
    if not calibration_enabled():
        return _not_available_view(said)
    try:
        user_pk = _person(db, ctx)
        record = claim.record_of(db, ctx.workspace_id, user_pk)
        may_answer = workspace_permission_granted(db, ctx, ANSWER_PERMISSION)
    except _Refused as refusal:
        return _refused(said, refusal.code, refusal.status_code)
    except SQLAlchemyError as failure:
        return _unavailable(db, said, "read", failure)
    return _read_view(said, record, may_answer)


@router.post("/calibration", response_model=None)
def offer_calibration(
    body: CalibrationConsent,
    accept_language: Optional[str] = Header(None),
    ctx: RequestContext = Depends(require_workspace_permission(ANSWER_PERMISSION)),
    db: Session = Depends(get_db),
) -> Union[Dict[str, Any], JSONResponse]:
    """The caller's answer to the consent line: declined, or one calibration posted."""
    language = body.locale or request_locale(accept_language)
    said = calibration_language(language)
    if not calibration_enabled():
        return _refused(said, NOT_AVAILABLE, HTTP_NOT_FOUND)
    now = datetime.now(timezone.utc)
    try:
        user_pk = _person(db, ctx)
        if not body.consent:
            return offer.decline(db, ctx.workspace_id, user_pk, now)
        return offer.consent(db, ctx.workspace_id, user_pk, language=language, chat_id=body.chat_id,
                             now=now).to_dict()
    except _Refused as refusal:
        return _refused(said, refusal.code, refusal.status_code)
    except offer.AlreadyOffered as taken:
        return _refused(said, ALREADY_OFFERED, HTTP_CONFLICT, record=taken.record)
    except offer.OfferFailed:
        return _refused(said, UNAVAILABLE, HTTP_UNAVAILABLE)
    except SQLAlchemyError as failure:
        return _unavailable(db, said, "offer", failure)


def _answer(db: Session, ctx: RequestContext, said: str,
            work: Callable[[int], answers.Answered]) -> Union[Dict[str, Any], JSONResponse]:
    """Run an answer for the caller and say what came of it."""
    if not calibration_enabled():
        return _refused(said, NOT_AVAILABLE, HTTP_NOT_FOUND)
    try:
        answered = work(_person(db, ctx))
    except _Refused as refusal:
        return _refused(said, refusal.code, refusal.status_code)
    except answers.NotFound as missing:
        return _refused(said, missing.code, HTTP_NOT_FOUND)
    except answers.AlreadyAnswered:
        return _refused(said, ANSWERED, HTTP_CONFLICT)
    except answers.NotRemembered:
        return _refused(said, MEMORY_UNAVAILABLE, HTTP_UNAVAILABLE)
    except SQLAlchemyError as failure:
        return _unavailable(db, said, "answer", failure)
    note = say(said, NOT_STORED) if answered.not_stored else None
    return {**answered.record, "not_stored": list(answered.not_stored), "note": note}


@router.post("/calibration/{calibration_id}/lines/{line_id}/reject", response_model=None)
def reject_calibration_line(
    calibration_id: UUID,
    line_id: str = Path(..., min_length=1, max_length=LINE_ID_MAX, pattern=LINE_ID_PATTERN),
    accept_language: Optional[str] = Header(None),
    ctx: RequestContext = Depends(require_workspace_permission(ANSWER_PERMISSION)),
    db: Session = Depends(get_db),
) -> Union[Dict[str, Any], JSONResponse]:
    """Reject one line of the caller's calibration; Auto remembers the correction."""
    said = calibration_language(request_locale(accept_language))
    return _answer(db, ctx, said, lambda user_pk: answers.reject_line(
        db, ctx.workspace_id, user_pk, calibration_id, line_id))


@router.post("/calibration/{calibration_id}/accept", response_model=None)
def accept_calibration(
    calibration_id: UUID,
    accept_language: Optional[str] = Header(None),
    ctx: RequestContext = Depends(require_workspace_permission(ANSWER_PERMISSION)),
    db: Session = Depends(get_db),
) -> Union[Dict[str, Any], JSONResponse]:
    """Sounds right: accept every line of the caller's calibration not rejected; Auto remembers each."""
    said = calibration_language(request_locale(accept_language))
    return _answer(db, ctx, said, lambda user_pk: answers.accept(
        db, ctx.workspace_id, user_pk, calibration_id, datetime.now(timezone.utc)))

"""PRD-268 §6.4, K7, K11, K21 — the brief on open (``POST /api/proactive/brief/open``).

``{local_date, timezone, locale?}``: the client calls it once per local date, when the web app
loads or the phone comes to the foreground. It claims the caller's day in their workspace,
shared with the scheduled digest, every device and every tab:

* the claim won: the brief is built, posted into the caller's own Auto thread (origin
  ``daily_digest``, via ``open``, never announced) and answered ``{chat_id, message_id,
  posted: true}``;
* the claim taken: the day's message, ``{chat_id, message_id, posted: false}``, or
  ``{posted: false, pending: true}`` while the winner is still building.

The person is the caller and nobody else: the body names no person, no chat and no workspace.
``local_date`` more than a day from the server's UTC date is 422 ``open_out_of_window``, and a
``timezone`` that is not an IANA name is 422 (the body's own validation). ``timezone`` is the
greeting's clock. ``locale`` (a BCP 47 tag, as ``ChatRequest.locale``; one that is not a tag is
ignored) is the brief's language, else ``Accept-Language`` (W1's K13 chooser), else English (K7).

Every member may open their brief, a viewer too, whose brief has no question or approval rows
(PRD-266 D16): ``require_workspace_permission("members:read")``, the scope every workspace role
holds (``modules/policy/roles.py``), as W1's ``PUT /api/proactive/me`` uses for a member's own
settings. A caller with no role in the workspace is refused by that gate (403). Past it (the
super admin's bypass), a call with no workspace is 404, and one with no person behind it, or
one who does not belong to the workspace, 403. A brief that cannot be built or posted is 503 ``open_unavailable`` (the
claim given back, so the next open tries again), and so is a database that cannot be read. A
refusal is ``{"code", "detail"}``, its sentence in the brief's language.

A plain ``def`` on the sync session (the build runs Auto's Read under ``asyncio.run`` in the
threadpool thread), registered through ``RouterSpec("api.proactive_brief")`` in
``router_manifest.py``. ``services.brief.on_open`` holds the rules.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, Optional, Union

from fastapi import APIRouter, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.chat_models import clean_locale
from api.mobile_devices import TimeZoneName
from core.auth.dependencies import RequestContext
from core.auth.principal import resolve_user_pk
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from modules.proactive.locale import request_locale
from services.brief import on_open
from services.brief.strings import brief_language, say
from services.mobile.digest import DuePerson, belongs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/proactive", tags=["proactive"])

# Every workspace role holds it, a viewer too: the brief is the caller's own (K21, §6.4).
BRIEF_PERMISSION = "members:read"
NO_WORKSPACE = "open_no_workspace"
NO_PERSON = "open_no_person"
NOT_A_MEMBER = "open_not_a_member"
OUT_OF_WINDOW = "open_out_of_window"
UNAVAILABLE = "open_unavailable"
HTTP_FORBIDDEN, HTTP_NOT_FOUND, HTTP_UNPROCESSABLE, HTTP_UNAVAILABLE = 403, 404, 422, 503


class BriefOpenRequest(BaseModel):
    """``POST /brief/open``: the caller's local date, their IANA time zone and, optionally, their language."""

    model_config = ConfigDict(extra="forbid")

    local_date: date
    timezone: TimeZoneName
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


def _refused(language: str, code: str, status_code: int) -> JSONResponse:
    return JSONResponse({"code": code, "detail": say(language, code)}, status_code=status_code)


def _person(db: Session, ctx: RequestContext, body: BriefOpenRequest, language: str) -> DuePerson:
    """The caller's day: their ``users.id`` while they belong to the workspace (403 otherwise)."""
    user_pk = resolve_user_pk(db, ctx)
    if user_pk is None:
        raise _Refused(NO_PERSON, HTTP_FORBIDDEN)
    if not belongs(db, ctx.workspace_id, user_pk):
        raise _Refused(NOT_A_MEMBER, HTTP_FORBIDDEN)
    return DuePerson(ctx.workspace_id, user_pk, body.local_date, body.timezone, language)


@router.post("/brief/open", response_model=None)
def open_brief(
    body: BriefOpenRequest,
    ctx: RequestContext = Depends(require_workspace_permission(BRIEF_PERMISSION)),
    accept_language: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> Union[Dict[str, Any], JSONResponse]:
    """The caller's brief for their local date: posted now, the day's message, or pending."""
    language = body.locale or request_locale(accept_language)
    said = brief_language(language)
    if ctx.workspace_id is None:
        return _refused(said, NO_WORKSPACE, HTTP_NOT_FOUND)
    try:
        opened = on_open.open_brief(db, _person(db, ctx, body, language))
    except _Refused as refusal:
        return _refused(said, refusal.code, refusal.status_code)
    except on_open.OutOfWindow:
        return _refused(said, OUT_OF_WINDOW, HTTP_UNPROCESSABLE)
    except on_open.NotPosted:
        return _refused(said, UNAVAILABLE, HTTP_UNAVAILABLE)
    except SQLAlchemyError as failure:
        db.rollback()
        logger.error("brief on open: the database could not be read (%s)", type(failure).__name__)
        return _refused(said, UNAVAILABLE, HTTP_UNAVAILABLE)
    return opened.to_dict()

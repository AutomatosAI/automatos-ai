"""PRD-268 §6.2.5, §6.9, FR-7, FR-13, FR-14 — Proactive Auto's suggestions, runs and settings (``/api/proactive``).

* ``GET /suggestions?status=open`` — the suggestions in one state (``open`` by default),
  newest first: an owner or admin sees every person's in the workspace, anyone else only the
  ones addressed to them (W1 addresses every suggestion to the workspace owner, K11). A caller
  with no person behind it, or none in this workspace, is 403. An open suggestion past its 7
  days lapses first, here and in Start and Hide: it lists as ``expired`` and either decision
  of it is 409 ``expired``.
* ``POST /suggestions/{suggestion_id}/start`` ``{via?}`` — Start (owners, admins and editors,
  the PRD-266 D16 rule; a viewer is 403): ``{suggestion, opens}``, where ``opens`` is the ticket
  Start filed (``{type: 'board_task', board_task_id}``) or the chat to open
  (``{type: 'chat', suggestion_id}``). One Start wins (409 for the rest); a Mission or a
  playbook run is 409 ``not_available`` in W1 (K12); the day's spend ceiling is 409
  ``spend_ceiling`` and leaves the suggestion open.
* ``POST /suggestions/{suggestion_id}/hide`` ``{reason?, via?}`` — Hide, never gated, with
  an optional reason ``not_now | not_useful | wrong_agent``: the person it is addressed to,
  or an owner or admin (a viewer is 403, even as the addressee). ``{suggestion}``.
* ``GET /runs?since=`` — the workspace's check-in runs since a time (the last 7 days by
  default), newest first: owners and admins.
* ``GET /settings`` and ``PUT /settings`` ``{enabled?, caps?}`` — the workspace switch and the
  §6.9 caps, each cap over its default (owners and admins, the workspace manage permission):
  ``{enabled, caps, platform_caps}``. A workspace can only tighten the platform's caps: a cap
  looser than ``platform_caps`` is 422 ``caps_looser``, and caps that would not fit together
  422 ``caps_wrong``; nothing is saved. A cap sent as null is back at its default. A change
  is audited.
* ``GET /me`` and ``PUT /me`` ``{paused?, quiet_start?, quiet_end?, timezone?}`` — the
  caller's own settings in this workspace (any member; no person behind the caller is 403):
  "Pause suggestions" and their quiet hours, ``HH:MM`` to ``HH:MM`` in an IANA time zone (the
  phone's; null is the workspace's), read with the workspace's window
  (``services.proactive.quiet``). What a ``PUT`` leaves out stays; the two times are sent
  together, both null to clear them, and never the same. ``{paused, quiet_start, quiet_end,
  timezone}``. A change is audited.

Another workspace's suggestion is 404. ``via`` says where the person decided (``web``,
``go``, ``push``, ``chat``, ``telegram``): left out, ``go`` for a paired device and ``web``
otherwise. A refusal is ``{"code", "detail"}``, its sentence in English or Portuguese
(Brazil) by ``Accept-Language`` (K13). The rules are ``services.proactive.decisions``,
``services.proactive.listing`` and ``services.proactive.settings``. Plain ``def`` routes on
the sync session, registered through ``RouterSpec("api.proactive")`` in ``router_manifest.py``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Dict, Literal, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Body, Depends, Header, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.mobile_devices import DIGEST_TIME_PATTERN, TimeZoneName
from core.auth.dependencies import RequestContext
from core.auth.device import DEVICE_AUTH_TYPE
from core.auth.hybrid import get_request_context_hybrid
from core.auth.principal import resolve_user_pk
from core.auth.workspace_admin import require_workspace_answerer
from core.auth.workspace_permission import (
    require_workspace_permission,
    resolve_workspace_role,
    workspace_permission_granted,
)
from core.database.database import get_db
from core.models.proactive import SUGGESTION_OPEN
from modules.proactive.locale import request_locale
from services.board_consent import actor_ref
from services.proactive import decisions, listing
from services.proactive import settings as proactive_settings
from services.proactive.decision_text import say

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/proactive", tags=["proactive"])

# Owners and admins: every person's suggestions, any suggestion's Hide, the runs, the settings.
MANAGE_PERMISSION = "workspace:manage"
# Every workspace role holds it: a member's own pause.
MEMBER_PERMISSION = "members:read"
NO_WORKSPACE = "no_workspace"
NO_PERSON = "no_person"
UNREADABLE = "unreadable"
UNAVAILABLE = "unavailable"
SETTINGS_UNREADABLE = "settings_unreadable"
SETTINGS_UNAVAILABLE = "settings_unavailable"
CAPS_WRONG = "caps_wrong"
CAPS_LOOSER = "caps_looser"
CAPS_STORED_WRONG = "caps_stored_wrong"
HTTP_FORBIDDEN, HTTP_NOT_FOUND, HTTP_CONFLICT, HTTP_UNPROCESSABLE, HTTP_UNAVAILABLE = 403, 404, 409, 422, 503
VIA_WEB, VIA_GO = "web", "go"

SuggestionStatus = Literal["open", "started", "hidden", "expired"]
Via = Literal["web", "go", "push", "chat", "telegram"]
HideReason = Literal["not_now", "not_useful", "wrong_agent"]
QuietTime = Annotated[str, Field(pattern=DIGEST_TIME_PATTERN)]  # HH:MM, 00:00 to 23:59


class StartRequest(BaseModel):
    """``POST …/start``: where the person pressed Start, nothing else."""

    model_config = ConfigDict(extra="forbid")

    via: Optional[Via] = None


class HideRequest(BaseModel):
    """``POST …/hide``: why the person hid it, and where."""

    model_config = ConfigDict(extra="forbid")

    reason: Optional[HideReason] = None
    via: Optional[Via] = None


class CapsChange(BaseModel):
    """The caps a ``PUT …/settings`` changes: each a whole number, or null for its default."""

    model_config = ConfigDict(extra="forbid")

    max_model_runs_per_day: Optional[StrictInt] = Field(None, ge=0)
    max_suggestions_per_run: Optional[StrictInt] = Field(None, ge=0)
    max_new_per_person_per_day: Optional[StrictInt] = Field(None, ge=0)
    max_open_per_person: Optional[StrictInt] = Field(None, ge=0)
    max_open_per_agent: Optional[StrictInt] = Field(None, ge=0)
    min_interval_minutes: Optional[StrictInt] = Field(None, ge=1)
    max_interval_minutes: Optional[StrictInt] = Field(None, ge=1)


class SettingsRequest(BaseModel):
    """``PUT …/settings``: the workspace switch and the caps to change; what is left out stays."""

    model_config = ConfigDict(extra="forbid")

    enabled: Optional[StrictBool] = None
    caps: Optional[CapsChange] = None


class MeRequest(BaseModel):
    """``PUT …/me``: the caller's pause and quiet hours to change; what is left out stays."""

    model_config = ConfigDict(extra="forbid")

    paused: Optional[StrictBool] = None
    quiet_start: Optional[QuietTime] = None
    quiet_end: Optional[QuietTime] = None
    timezone: Optional[TimeZoneName] = None

    @model_validator(mode="after")
    def _a_whole_change(self) -> "MeRequest":
        sent = self.model_fields_set
        if not sent:
            raise ValueError("send paused, the quiet hours or a timezone")
        if "paused" in sent and self.paused is None:
            raise ValueError("paused is true or false")
        if ("quiet_start" in sent) != ("quiet_end" in sent) or (self.quiet_start is None) != (self.quiet_end is None):
            raise ValueError("quiet_start and quiet_end are sent together: two times, or both null to clear them")
        if self.quiet_start is not None and self.quiet_start == self.quiet_end:
            raise ValueError("quiet hours end at a different time from when they start")
        return self


@dataclass(frozen=True)
class _Viewer:
    """Who lists suggestions: an owner or admin (``manages``), or the person ``user_pk``."""

    manages: bool
    user_pk: Optional[int]


def _refused(locale: str, code: str, status_code: int) -> JSONResponse:
    return JSONResponse({"code": code, "detail": say(locale, code)}, status_code=status_code)


def _manages(db: Session, ctx: RequestContext) -> bool:
    return workspace_permission_granted(db, ctx, MANAGE_PERMISSION)


def _viewer(db: Session, ctx: RequestContext) -> Optional[_Viewer]:
    """An owner or admin, else a person of this workspace; None for anyone else."""
    if _manages(db, ctx):
        return _Viewer(manages=True, user_pk=resolve_user_pk(db, ctx))
    user_pk = resolve_user_pk(db, ctx)
    if user_pk is None or resolve_workspace_role(db, ctx) is None:
        return None
    return _Viewer(manages=False, user_pk=user_pk)


def _decider(db: Session, ctx: RequestContext, via: Optional[str]) -> decisions.Decider:
    user = ctx.user
    created_by = getattr(user, "clerk_user_id", None) or getattr(user, "id", None)
    default_via = VIA_GO if ctx.auth_type == DEVICE_AUTH_TYPE else VIA_WEB
    return decisions.Decider(user_pk=resolve_user_pk(db, ctx), created_by=str(created_by) if created_by else None,
                             actor=actor_ref(ctx), via=via or default_via)


def _unavailable(db: Session, locale: str, what: str, failure: Exception, code: str) -> JSONResponse:
    """Roll back and refuse with 503: the failure and its stack go to the server log, never the body."""
    db.rollback()
    logger.error("proactive %s failed", what, exc_info=failure)
    return _refused(locale, code, HTTP_UNAVAILABLE)


@router.get("/suggestions", response_model=None)
def list_suggestions(
    status: SuggestionStatus = Query(SUGGESTION_OPEN),
    limit: int = Query(listing.DEFAULT_SUGGESTIONS, ge=1, le=listing.MAX_SUGGESTIONS),
    accept_language: Optional[str] = Header(None),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Union[Dict[str, Any], JSONResponse]:
    """The suggestions in ``status``, newest first: the workspace's for an owner or admin, else the caller's."""
    locale = request_locale(accept_language)
    if ctx.workspace_id is None:
        return _refused(locale, NO_WORKSPACE, HTTP_NOT_FOUND)
    try:
        viewer = _viewer(db, ctx)
        if viewer is None:
            return _refused(locale, NO_PERSON, HTTP_FORBIDDEN)
        rows = listing.suggestions(db, ctx.workspace_id, status=status, limit=limit,
                                   everyone=viewer.manages, user_id=viewer.user_pk)
    except SQLAlchemyError as failure:
        return _unavailable(db, locale, "suggestion list", failure, UNREADABLE)
    return {"suggestions": [row.to_dict() for row in rows]}


@router.post("/suggestions/{suggestion_id}/start", response_model=None)
def start_suggestion(
    suggestion_id: UUID,
    body: Optional[StartRequest] = Body(None),
    accept_language: Optional[str] = Header(None),
    ctx: RequestContext = Depends(require_workspace_answerer),
    db: Session = Depends(get_db),
) -> Union[Dict[str, Any], JSONResponse]:
    """Start the suggestion: the ticket it files, or the chat to open; 409 when it can't start now."""
    locale = request_locale(accept_language)
    if ctx.workspace_id is None:
        return _refused(locale, NO_WORKSPACE, HTTP_NOT_FOUND)
    try:
        decider = _decider(db, ctx, body.via if body else None)
        started = decisions.start(db, ctx.workspace_id, suggestion_id, decider, locale)
    except decisions.Refused as refusal:
        return _refused(locale, refusal.code, refusal.http_status)
    except SQLAlchemyError as failure:
        return _unavailable(db, locale, "start", failure, UNAVAILABLE)
    return {"suggestion": started.suggestion, "opens": started.opens}


@router.post("/suggestions/{suggestion_id}/hide", response_model=None)
def hide_suggestion(
    suggestion_id: UUID,
    body: Optional[HideRequest] = Body(None),
    accept_language: Optional[str] = Header(None),
    ctx: RequestContext = Depends(require_workspace_answerer),
    db: Session = Depends(get_db),
) -> Union[Dict[str, Any], JSONResponse]:
    """Hide the suggestion, never gated: its addressee, or an owner or admin."""
    locale = request_locale(accept_language)
    if ctx.workspace_id is None:
        return _refused(locale, NO_WORKSPACE, HTTP_NOT_FOUND)
    asked = body or HideRequest()
    try:
        decider = _decider(db, ctx, asked.via)
        hidden = decisions.hide(db, ctx.workspace_id, suggestion_id, decider, asked.reason, any_row=_manages(db, ctx))
    except decisions.Refused as refusal:
        return _refused(locale, refusal.code, refusal.http_status)
    except SQLAlchemyError as failure:
        return _unavailable(db, locale, "hide", failure, UNAVAILABLE)
    return {"suggestion": hidden}


@router.get("/runs", response_model=None)
def list_runs(
    since: Optional[datetime] = Query(None),
    limit: int = Query(listing.DEFAULT_RUNS, ge=1, le=listing.MAX_RUNS),
    accept_language: Optional[str] = Header(None),
    ctx: RequestContext = Depends(require_workspace_permission(MANAGE_PERMISSION)),
    db: Session = Depends(get_db),
) -> Union[Dict[str, Any], JSONResponse]:
    """The workspace's check-in runs since ``since`` (the last 7 days by default), newest first."""
    locale = request_locale(accept_language)
    if ctx.workspace_id is None:
        return _refused(locale, NO_WORKSPACE, HTTP_NOT_FOUND)
    try:
        rows = listing.runs(db, ctx.workspace_id, since=since, limit=limit)
    except SQLAlchemyError as failure:
        return _unavailable(db, locale, "run list", failure, UNREADABLE)
    return {"runs": [row.to_dict() for row in rows]}


@router.get("/settings", response_model=None)
def get_settings(
    accept_language: Optional[str] = Header(None),
    ctx: RequestContext = Depends(require_workspace_permission(MANAGE_PERMISSION)),
    db: Session = Depends(get_db),
) -> Union[Dict[str, Any], JSONResponse]:
    """The workspace switch and its caps, each over its default."""
    locale = request_locale(accept_language)
    if ctx.workspace_id is None:
        return _refused(locale, NO_WORKSPACE, HTTP_NOT_FOUND)
    try:
        current = proactive_settings.read_workspace(db, ctx.workspace_id)
    except proactive_settings.NoWorkspace:
        return _refused(locale, NO_WORKSPACE, HTTP_NOT_FOUND)
    except proactive_settings.CapsWrong:
        logger.warning("proactive caps of workspace %s are stored wrong", ctx.workspace_id)
        return _refused(locale, CAPS_STORED_WRONG, HTTP_CONFLICT)
    except SQLAlchemyError as failure:
        return _unavailable(db, locale, "settings read", failure, SETTINGS_UNREADABLE)
    return current.to_dict()


@router.put("/settings", response_model=None)
def put_settings(
    body: SettingsRequest,
    accept_language: Optional[str] = Header(None),
    ctx: RequestContext = Depends(require_workspace_permission(MANAGE_PERMISSION)),
    db: Session = Depends(get_db),
) -> Union[Dict[str, Any], JSONResponse]:
    """Switch the workspace's check-ins on or off and change its caps."""
    locale = request_locale(accept_language)
    if ctx.workspace_id is None:
        return _refused(locale, NO_WORKSPACE, HTTP_NOT_FOUND)
    caps = body.caps.model_dump(exclude_unset=True) if body.caps is not None else {}
    try:
        changed = proactive_settings.change_workspace(db, ctx.workspace_id, body.enabled, caps,
                                                      resolve_user_pk(db, ctx), actor_ref(ctx))
    except proactive_settings.NoWorkspace:
        return _refused(locale, NO_WORKSPACE, HTTP_NOT_FOUND)
    except proactive_settings.CapsLooser:
        return _refused(locale, CAPS_LOOSER, HTTP_UNPROCESSABLE)
    except proactive_settings.CapsWrong:
        return _refused(locale, CAPS_WRONG, HTTP_UNPROCESSABLE)
    except SQLAlchemyError as failure:
        return _unavailable(db, locale, "settings write", failure, SETTINGS_UNAVAILABLE)
    return changed.to_dict()


@router.get("/me", response_model=None)
def get_my_settings(
    accept_language: Optional[str] = Header(None),
    ctx: RequestContext = Depends(require_workspace_permission(MEMBER_PERMISSION)),
    db: Session = Depends(get_db),
) -> Union[Dict[str, Any], JSONResponse]:
    """The caller's own proactive settings in this workspace: ``{paused, quiet_start, quiet_end, timezone}``."""
    locale = request_locale(accept_language)
    if ctx.workspace_id is None:
        return _refused(locale, NO_WORKSPACE, HTTP_NOT_FOUND)
    try:
        user_pk = resolve_user_pk(db, ctx)
        if user_pk is None:
            return _refused(locale, NO_PERSON, HTTP_FORBIDDEN)
        mine = proactive_settings.read_person(db, ctx.workspace_id, user_pk)
    except SQLAlchemyError as failure:
        return _unavailable(db, locale, "person settings read", failure, SETTINGS_UNREADABLE)
    return mine.to_dict()


@router.put("/me", response_model=None)
def put_my_settings(
    body: MeRequest,
    accept_language: Optional[str] = Header(None),
    ctx: RequestContext = Depends(require_workspace_permission(MEMBER_PERMISSION)),
    db: Session = Depends(get_db),
) -> Union[Dict[str, Any], JSONResponse]:
    """Pause or resume the caller's suggestions in this workspace, and set or clear their quiet hours."""
    locale = request_locale(accept_language)
    if ctx.workspace_id is None:
        return _refused(locale, NO_WORKSPACE, HTTP_NOT_FOUND)
    try:
        user_pk = resolve_user_pk(db, ctx)
        if user_pk is None:
            return _refused(locale, NO_PERSON, HTTP_FORBIDDEN)
        mine = proactive_settings.change_person(db, ctx.workspace_id, user_pk, body.model_dump(exclude_unset=True),
                                                actor_ref(ctx))
    except SQLAlchemyError as failure:
        return _unavailable(db, locale, "person settings write", failure, SETTINGS_UNAVAILABLE)
    return mine.to_dict()

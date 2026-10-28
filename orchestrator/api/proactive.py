"""PRD-268 §6.2.5, FR-7, FR-14 — Proactive Auto's suggestions and runs (``/api/proactive``).

* ``GET /suggestions?status=open`` — the suggestions in one state (``open`` by default),
  newest first: an owner or admin sees every person's in the workspace, anyone else only the
  ones addressed to them (W1 addresses every suggestion to the workspace owner, K11). A caller
  with no person behind it, or none in this workspace, is 403.
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

Another workspace's suggestion is 404. ``via`` says where the person decided (``web``,
``go``, ``push``, ``chat``, ``telegram``): left out, ``go`` for a paired device and ``web``
otherwise. A refusal is ``{"code", "detail"}``, its sentence in English or Portuguese
(Brazil) by ``Accept-Language`` (K13). The rules are ``services.proactive.decisions`` and
``services.proactive.listing``. Plain ``def`` routes on the sync session, registered through
``RouterSpec("api.proactive")`` in ``router_manifest.py``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Literal, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Body, Depends, Header, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

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
from services.proactive.decision_text import say

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/proactive", tags=["proactive"])

# Owners and admins: every person's suggestions, any suggestion's Hide, the runs.
MANAGE_PERMISSION = "workspace:manage"
NO_WORKSPACE = "no_workspace"
NO_PERSON = "no_person"
UNREADABLE = "unreadable"
UNAVAILABLE = "unavailable"
HTTP_FORBIDDEN, HTTP_NOT_FOUND, HTTP_UNAVAILABLE = 403, 404, 503
VIA_WEB, VIA_GO = "web", "go"

SuggestionStatus = Literal["open", "started", "hidden", "expired"]
Via = Literal["web", "go", "push", "chat", "telegram"]
HideReason = Literal["not_now", "not_useful", "wrong_agent"]


class StartRequest(BaseModel):
    """``POST …/start``: where the person pressed Start, nothing else."""

    model_config = ConfigDict(extra="forbid")

    via: Optional[Via] = None


class HideRequest(BaseModel):
    """``POST …/hide``: why the person hid it, and where."""

    model_config = ConfigDict(extra="forbid")

    reason: Optional[HideReason] = None
    via: Optional[Via] = None


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


def _unavailable(db: Session, locale: str, what: str, failure: SQLAlchemyError, code: str) -> JSONResponse:
    db.rollback()
    logger.error("proactive %s failed: %s", what, type(failure).__name__)
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
        return decisions.start(db, ctx.workspace_id, suggestion_id, decider, locale)
    except decisions.Refused as refusal:
        return _refused(locale, refusal.code, refusal.http_status)
    except SQLAlchemyError as failure:
        return _unavailable(db, locale, "start", failure, UNAVAILABLE)


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
        return decisions.hide(db, ctx.workspace_id, suggestion_id, decider, asked.reason, any_row=_manages(db, ctx))
    except decisions.Refused as refusal:
        return _refused(locale, refusal.code, refusal.http_status)
    except SQLAlchemyError as failure:
        return _unavailable(db, locale, "hide", failure, UNAVAILABLE)


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

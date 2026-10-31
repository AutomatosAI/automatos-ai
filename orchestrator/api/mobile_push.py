"""PRD-266 §4.4 (W2 Push) — the Automatos Go push routes (``/api/v1/mobile``).

* ``GET /push-consent`` — the operator's answer to "Send notifications through Expo's push
  service?" (D21, K13): ``{agreed, answered, answered_at}``. A settings read that fails is
  503, never "not answered", so Studio does not ask again over a stored answer.
* ``PUT /push-consent`` ``{agreed: bool}`` — record the answer (and when), committed; answers
  the same body. Studio asks once at the first "Pair a phone" and keeps a switch on
  Settings → Phones & devices; the app asks at pairing with its device token.

Both are local edition only: 404 on SaaS, before any authentication runs, where push always
goes through the platform's Expo project. Both require ``require_workspace_admin``: the local
operator passes as super_admin, and so does a phone bound to the operator through W1's
device lane (``core/auth/device.py``); a plain member or a viewer is 403. The answer is
install-wide (``services.push.consent``), not a workspace row.

* ``PUT /push-settings`` ``{redact_questions: bool}`` — US-014b (D20): with it on, this
  workspace's question pushes say only that a question is waiting (no text, no options, Open
  only). Both editions; ``workspace:manage`` (owners and admins; editors and viewers 403).
  Saved as ``workspace.settings['push']`` (new dicts, the ``save_voice_live_settings``
  pattern), off by default; ``GET /api/workspaces/current`` shows it with the settings, and
  the fan-out reads it at send time (``services.push.payload.redact_questions_enabled``).

* ``POST /digest/now`` — "Send me today's brief now" (no body): the daily digest (US-013),
  made and posted for the caller now, so the phone's morning brief can be tried without
  waiting a day. Both editions; hybrid auth like every mobile route; the caller only, in the
  ``X-Workspace-ID`` workspace, any role (a caller with no workspace is 404; no person behind
  it, or one who does not belong there, 403). ``202 {"status": "queued"}`` at once: Auto's
  Read is made and posted into the caller's Auto thread in the background, exactly as the
  scheduled digest posts it (origin ``daily_digest``, so ``auto_message`` and its push), with
  no ``mobile_digest_posts`` claim, so today's and tomorrow's scheduled digests still go out.
  The person's ``daily_digest`` switch is not read (an explicit ask); the push follows their
  ``auto_message`` preference. One ask per person every ``DIGEST_NOW_MIN_INTERVAL_SECONDS``:
  past it, ``429 {"code": "digest_rate_limited", "retry_after_seconds": N}`` with
  ``Retry-After``. A digest that cannot be made is still 202 (the job logs the failure's class
  only); a request that cannot start the job (its database read fails, or no event loop takes
  it) is ``503 {"code": "digest_unavailable"}``. ``services.mobile.digest_now`` holds the rules.

W1's ``api/mobile_devices.py`` stays under 400 lines: the wave's routes live here. Plain
``def`` routes (the sync session runs in the threadpool), registered through
``RouterSpec("api.mobile_push")`` in ``router_manifest.py``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Union

import anyio
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, StrictBool
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.principal import resolve_user_pk
from core.auth.workspace_admin import require_workspace_admin
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.workspaces import Workspace
from services.mobile import digest, digest_now
from services.push import consent
from services.push.payload import PUSH_SETTINGS_KEY, REDACT_QUESTIONS_KEY, with_redact_questions

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/mobile", tags=["mobile"])

# What FastAPI answers for a route that is not there, so a closed route looks absent.
NOT_FOUND = "Not Found"
UNREADABLE = "The answer could not be read right now. Try again in a minute."
UNAVAILABLE = "The answer could not be saved right now. Try again in a minute."
# The push settings are the workspace's: owners and admins change them (modules/policy/roles.py).
MANAGE_PERMISSION = "workspace:manage"
WORKSPACE_NOT_FOUND = "Workspace not found"
SETTINGS_UNAVAILABLE = "The push settings could not be saved right now. Try again in a minute."
DIGEST_QUEUED = "queued"
DIGEST_RATE_LIMITED = "digest_rate_limited"
DIGEST_UNAVAILABLE = "digest_unavailable"
NO_WORKSPACE = "Choose a workspace first."
NO_PERSON = "Today's brief is posted to a person, and this caller is not one."
NOT_A_MEMBER = "Today's brief is only for the people in this workspace."


class PushConsentRequest(BaseModel):
    """``PUT /push-consent``: yes or no, nothing else."""

    model_config = ConfigDict(extra="forbid")

    agreed: StrictBool


class PushSettingsRequest(BaseModel):
    """``PUT /push-settings``: whether this workspace's question pushes are sent generic."""

    model_config = ConfigDict(extra="forbid")

    redact_questions: StrictBool


def _require_local_edition() -> None:
    """404 outside the local edition: SaaS always pushes through the platform's Expo project."""
    if not config.IS_LOCAL_EDITION:
        raise HTTPException(status_code=404, detail=NOT_FOUND)


@router.get("/push-consent", dependencies=[Depends(_require_local_edition)])
def get_push_consent(
    ctx: RequestContext = Depends(require_workspace_admin),
) -> Dict[str, Any]:
    """The operator's answer: ``{agreed, answered, answered_at}``; 503 when it cannot be read."""
    try:
        return consent.consent_state().to_dict()
    except SQLAlchemyError as failure:
        logger.error("push consent read failed: %s", type(failure).__name__)
        raise HTTPException(status_code=503, detail=UNREADABLE) from failure


@router.put("/push-consent", dependencies=[Depends(_require_local_edition)])
def put_push_consent(
    body: PushConsentRequest,
    ctx: RequestContext = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Record whether this install may send notifications through Expo, and when it was asked."""
    try:
        state = consent.set_push_consent(db, body.agreed)
        db.commit()
    except SQLAlchemyError as failure:
        db.rollback()
        logger.error("push consent write failed: %s", type(failure).__name__)
        raise HTTPException(status_code=503, detail=UNAVAILABLE) from failure
    logger.info("push through Expo %s for this install (workspace %s, auth %s)",
                "agreed" if state.agreed else "declined", ctx.workspace_id, ctx.auth_type)
    return state.to_dict()


@router.put("/push-settings")
def put_push_settings(
    body: PushSettingsRequest,
    ctx: RequestContext = Depends(require_workspace_permission(MANAGE_PERMISSION)),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Send the caller's workspace's question pushes generic, or with their text again."""
    workspace = db.get(Workspace, ctx.workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail=WORKSPACE_NOT_FOUND)
    workspace.settings = with_redact_questions(workspace.settings, body.redact_questions)
    flag_modified(workspace, "settings")
    try:
        db.commit()
    except SQLAlchemyError as failure:
        db.rollback()
        logger.error("push settings write failed for workspace %s: %s",
                     ctx.workspace_id, type(failure).__name__)
        raise HTTPException(status_code=503, detail=SETTINGS_UNAVAILABLE) from failure
    logger.info("question pushes %s for workspace %s",
                "redacted" if body.redact_questions else "with their text", ctx.workspace_id)
    return {PUSH_SETTINGS_KEY: {REDACT_QUESTIONS_KEY: body.redact_questions}}


def _digest_asker(db: Session, ctx: RequestContext) -> int:
    """The caller's ``users.id`` while they belong to the workspace: 404 without one, else 403."""
    if ctx.workspace_id is None:
        raise HTTPException(status_code=404, detail=NO_WORKSPACE)
    user_pk = resolve_user_pk(db, ctx)
    if user_pk is None:
        raise HTTPException(status_code=403, detail=NO_PERSON)
    if not digest.belongs(db, ctx.workspace_id, user_pk):
        raise HTTPException(status_code=403, detail=NOT_A_MEMBER)
    return user_pk


def _digest_unavailable() -> JSONResponse:
    return JSONResponse({"code": DIGEST_UNAVAILABLE}, status_code=503)


@router.post("/digest/now", status_code=202, response_model=None)
def send_digest_now(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Union[Dict[str, str], JSONResponse]:
    """Today's brief for the caller, now: 202 queued; 429 within the interval; 503 when it cannot start."""
    try:
        user_pk = _digest_asker(db, ctx)
    except SQLAlchemyError as failure:
        db.rollback()
        logger.error("digest now: the caller could not be read (%s)", type(failure).__name__)
        return _digest_unavailable()
    wait = digest_now.cooldown.take(user_pk)
    if wait is not None:
        return JSONResponse({"code": DIGEST_RATE_LIMITED, "retry_after_seconds": wait},
                            status_code=429, headers={"Retry-After": str(wait)})
    try:
        anyio.from_thread.run_sync(digest_now.launch, ctx.workspace_id, user_pk)
    except RuntimeError as failure:
        digest_now.cooldown.give_back(user_pk)
        logger.error("digest now: the job could not start for person %s in %s (%s)",
                     user_pk, ctx.workspace_id, type(failure).__name__)
        return _digest_unavailable()
    logger.info("digest now: queued for person %s in %s", user_pk, ctx.workspace_id)
    return {"status": DIGEST_QUEUED}

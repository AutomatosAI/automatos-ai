"""
Socials retake (PRD-251B US-B111)
=================================

``POST /api/socials/posts/{post_id}/retake`` (``socials:approve``; body ``{"guidance"?}``;
202 with the post): Auto makes another take of a post. It is allowed on a post in ``draft``,
``needs_approval``, ``changes_requested`` or ``failed`` (F378: a draft or a failed render can
be rescued too) and nowhere else (409). A post with no brief, no copy and no fields has
nothing to start from: 422 ``nothing_to_retake``, nothing composed.

F378 (night 11, 7 Oct): the composer is given the post's current take (its copy and its
fields' values, ``modules/socials/retakes.py``) and changes only what the guidance asks; the
take it replaces is kept in the post's history, and
``POST /api/socials/posts/{post_id}/retake/undo`` restores it (409 ``nothing_to_undo`` when
there is none), then renders it again as a retake does. The response carries what the
composer had to say (``take``: its warnings, and its questions for the owner).

The post is composed again through the composer (``api/socials_compose.py``,
``modules/socials/compose.py``) with its own brief, format, channels (its targets), template
and length (US-B103); the reviewer's guidance, when given, is appended to the brief as
"Changes requested: ...". The proposal's copy, variables and sources are written through the
one edit path (``api/socials.py edit_post``: a new content hash, the compare-and-set commit;
the planned slot is kept), then the post renders again through the render lifecycle and
returns to ``needs_approval`` when the render ends. A post without a template has nothing to
render: one in ``changes_requested`` is sent for approval again at once. The model's failure
is the composer's: 502 (504 when it does not answer in time), and nothing is changed.

A plain ``def``: its database work runs in the threadpool (F105), and the model call and the
render start run on the event loop from there, as ``/compose`` does.

This router has no prefix and no gate of its own: ``api/socials.py`` includes it in the
Socials router, whose ``require_socials_enabled`` answers 404 unless both switches are on
(D1). Never mount it in the app directly. It reuses that module's post helpers, imported
when a request runs, because that module includes this one.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional
from uuid import UUID

import anyio
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from api import socials_compose as compose_api
from config import config
from core import media_render_quota as render_quota
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.socials import SocialPost
from modules.socials import compose, retakes, service

logger = logging.getLogger(__name__)
router = APIRouter()

CAN_REVIEW = Depends(require_workspace_permission("socials:approve"))
RETAKE_STATUSES = frozenset({service.DRAFT, service.NEEDS_APPROVAL, service.CHANGES_REQUESTED, service.FAILED})
ACTION_RETAKE = "retake"
ACTION_UNDO = "undo_the_retake_of"  # IllegalTransition reads it as "cannot undo the retake of a post that is …"
GUIDANCE_PREFIX = "Changes requested: "
GUIDANCE_MAX_CHARS = 2000
RETAKE_NOTE = "Auto made another take."
NOTHING_TO_RETAKE, NOTHING_TO_UNDO = "nothing_to_retake", "nothing_to_undo"


class RetakeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    guidance: Optional[str] = Field(None, max_length=GUIDANCE_MAX_CHARS)


def _posts_api() -> Any:
    import api.socials as posts_api  # that module includes this router

    return posts_api


def retake_brief(post: SocialPost, guidance: Optional[str]) -> str:
    """The post's own brief (its title when it has none), with the reviewer's guidance."""
    brief = (post.brief or post.title or "").strip()
    note = (guidance or "").strip()
    return f"{brief}\n\n{GUIDANCE_PREFIX}{note}" if note else brief


def retake_request(post: SocialPost, guidance: Optional[str]) -> compose_api.ComposeRequest:
    """What the composer is asked: the post's brief, format, channels, template and length."""
    channels = sorted({target.toolkit for target in post.targets}) or None
    return compose_api.ComposeRequest(
        brief=retake_brief(post, guidance)[: compose_api.BRIEF_MAX_CHARS],
        channels=channels,
        format=post.format,
        template_id=post.template_id,
        length_seconds=post.length_seconds,
    )


def _propose(db: Session, post: SocialPost, guidance: Optional[str]) -> Dict[str, Any]:
    """The composer's new take, started from the post's current one (F378), or the composer's
    own refusal (422, 502, 504)."""
    try:
        context = compose_api.compose_context(
            db, post.workspace_id, retake_request(post, guidance), current_take=retakes.current_take(post),
        )
    except compose_api.ChoiceRefused as exc:
        raise HTTPException(status_code=422, detail={"code": exc.code, "message": str(exc)}) from exc
    timeout = float(config.SOCIALS_COMPOSE_TIMEOUT_SECONDS)
    try:
        return anyio.from_thread.run(compose.propose, context, compose.llm_factory(post.workspace_id), timeout)
    except compose.ComposeTimedOut as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except compose.ComposeFailed as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("[Socials] the retake of post %s could not be composed", post.id)
        raise HTTPException(status_code=502, detail="The model could not be reached. Try again.") from exc


def take_changes(proposal: Dict[str, Any]) -> Dict[str, Any]:
    """The proposal as an edit: its copy (base and each channel's), variables and sources only."""
    return {
        "copy": dict(proposal.get("copy") or {}),  # the composer's shape is the one a save takes (F378)
        "variables": dict(proposal.get("variables") or {}),
        "sources": dict(proposal.get("sources") or {}),
    }


def take_notes(proposal: Mapping[str, Any]) -> Dict[str, List[str]]:
    """F378: what the composer had to say about the new take: its warnings and its questions
    for the owner, returned with the post instead of being thrown away."""
    return {"warnings": list(proposal.get("warnings") or []), "questions": list(proposal.get("questions") or [])}


def _after_take(db: Session, ctx: RequestContext, post: SocialPost, actor: str, saved: Dict[str, Any]) -> Dict[str, Any]:
    """A new take's next step: a post with a template renders again; one in changes_requested
    without one goes back for approval; any other is as saved."""
    posts_api = _posts_api()
    if post.template_id is not None:
        return anyio.from_thread.run(posts_api.render_post, db, posts_api._workspace(db, ctx), post, actor)
    if post.status == service.CHANGES_REQUESTED:
        return posts_api.submit_post(db, post, actor, note=RETAKE_NOTE)
    return saved


@router.post("/posts/{post_id}/retake", status_code=202, dependencies=[CAN_REVIEW])
def retake_social_post(
    post_id: UUID,
    body: Optional[RetakeRequest] = None,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """Another take of a post, kept undoable (the module docstring)."""
    posts_api = _posts_api()
    post = posts_api._load(db, ctx, post_id)
    actor = posts_api._actor(ctx)
    if post.status not in RETAKE_STATUSES:
        posts_api._raise_for(service.IllegalTransition(post.status, ACTION_RETAKE))
    if retakes.nothing_to_retake(post):
        raise HTTPException(status_code=422, detail={"code": NOTHING_TO_RETAKE, "message": retakes.NOTHING_TO_RETAKE})
    guidance = body.guidance if body is not None else None
    proposal = _propose(db, post, guidance)
    try:
        retakes.record_previous(post, actor, guidance)  # F378: the take it replaces stays in the history
        saved = anyio.from_thread.run(posts_api.edit_post, db, post, actor, take_changes(proposal))
        return {**_after_take(db, ctx, post, actor, saved), "take": take_notes(proposal)}
    except (service.SocialsError, render_quota.RenderQuotaExceeded) as exc:
        posts_api._raise_for(exc)


@router.post("/posts/{post_id}/retake/undo", status_code=202, dependencies=[CAN_REVIEW])
def undo_social_post_retake(
    post_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """F378: the take before Auto's last one, restored and rendered again as a retake is;
    409 ``nothing_to_undo`` when the post has no earlier take kept."""
    posts_api = _posts_api()
    post = posts_api._load(db, ctx, post_id)
    actor = posts_api._actor(ctx)
    # A post a retake may change is one its undo may restore, so the restore always renders.
    if post.status not in RETAKE_STATUSES:
        posts_api._raise_for(service.IllegalTransition(post.status, ACTION_UNDO))
    entry = retakes.restoring(post)
    if entry is None:
        raise HTTPException(status_code=409, detail={"code": NOTHING_TO_UNDO, "message": retakes.NOTHING_TO_UNDO})
    try:
        retakes.record_undo(post, actor, entry)
        saved = anyio.from_thread.run(posts_api.edit_post, db, post, actor, retakes.restore_changes(entry))
        return _after_take(db, ctx, post, actor, saved)
    except (service.SocialsError, render_quota.RenderQuotaExceeded) as exc:
        posts_api._raise_for(exc)

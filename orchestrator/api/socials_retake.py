"""
Socials retake (PRD-251B US-B111)
=================================

``POST /api/socials/posts/{post_id}/retake`` (``socials:approve``; body ``{"guidance"?}``;
202 with the post): Auto makes another take of a post waiting in the Queue. It is allowed
on a post in ``needs_approval`` or ``changes_requested`` and nowhere else (409).

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
from typing import Any, Dict, Optional
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
from modules.socials import compose, service

logger = logging.getLogger(__name__)
router = APIRouter()

CAN_REVIEW = Depends(require_workspace_permission("socials:approve"))
RETAKE_STATUSES = frozenset({service.NEEDS_APPROVAL, service.CHANGES_REQUESTED})
ACTION_RETAKE = "retake"
GUIDANCE_PREFIX = "Changes requested: "
GUIDANCE_MAX_CHARS = 2000
RETAKE_NOTE = "Auto made another take."


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
    """The composer's new take, or the composer's own refusal (422, 502, 504)."""
    try:
        context = compose_api.compose_context(db, post.workspace_id, retake_request(post, guidance))
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


@router.post("/posts/{post_id}/retake", status_code=202, dependencies=[CAN_REVIEW])
def retake_social_post(
    post_id: UUID,
    body: Optional[RetakeRequest] = None,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """Another take of a post waiting for approval (the module docstring)."""
    posts_api = _posts_api()
    post = posts_api._load(db, ctx, post_id)
    actor = posts_api._actor(ctx)
    if post.status not in RETAKE_STATUSES:
        posts_api._raise_for(service.IllegalTransition(post.status, ACTION_RETAKE))
    proposal = _propose(db, post, body.guidance if body is not None else None)
    try:
        saved = anyio.from_thread.run(posts_api.edit_post, db, post, actor, take_changes(proposal))
        if post.template_id is not None:
            return anyio.from_thread.run(posts_api.render_post, db, posts_api._workspace(db, ctx), post, actor)
        if post.status == service.CHANGES_REQUESTED:
            return posts_api.submit_post(db, post, actor, note=RETAKE_NOTE)
        return saved
    except (service.SocialsError, render_quota.RenderQuotaExceeded) as exc:
        posts_api._raise_for(exc)

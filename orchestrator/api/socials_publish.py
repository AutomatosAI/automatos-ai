"""
Socials publishing (PRD-251 US-301, S3.3)
=========================================

``POST /api/socials/posts/{post_id}/publish-now`` publishes an approved, scheduled or
missed post, and ``POST /api/socials/posts/{post_id}/retry`` publishes again the targets that
failed of a failed or partially published post. Each answers 202 with the post in
``publishing``; the publish runs in the background (``modules/socials/publishing.py``)
and ends the post ``published``, ``partially_published`` or ``failed``, each target
with its receipt.

``GET /api/socials/posts/{post_id}/ai-labels`` (PRD-264 EUAI-05) answers each channel's
AI label decision for the receipt (``modules/socials/ai_media.py``): the line a target's
publish used, once one ran, else the line its publish will use, the post's AI media, and
its AI-generated label setting (PRD-264 EUAI-08a, ``modules/socials/deepfake_label.py``).
``PUT /api/socials/posts/{post_id}/ai-label`` (EUAI-08a) is where a person switches that
label on or off for the post: approved content, so an approved post goes back for approval.

The approval guard runs first (D6): a post whose approval no longer matches its
content answers 409 and nothing reaches Composio. The claim is a compare-and-set
(``modules/socials/publisher.py``): a second request for a post already claimed
answers 409 and publishes nothing.

Plain ``def`` routes: FastAPI runs their synchronous database work in the threadpool
(F105), and the background publish is launched on the event loop through
``anyio.from_thread``. No prefix and no gate of their own: ``api/socials.py``
includes this router in the Socials router, whose ``require_socials_enabled`` answers
404 unless both switches are on (D1). Never mount it in the app directly.
"""

from __future__ import annotations

from typing import Any, Callable, Dict
from uuid import UUID

import anyio
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, StrictBool
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from modules.socials import ai_media, deepfake_label, publisher, schedule_jobs, service

router = APIRouter()

CAN_UPDATE = Depends(require_workspace_permission("documents:update"))
CAN_READ = Depends(require_workspace_permission("documents:read"))


def _posts_api() -> Any:
    """``api/socials.py``: its router includes this one, so it is imported when a request runs."""
    from api import socials

    return socials


def _start(db: Session, ctx: RequestContext, post_id: UUID, begin: Callable[..., publisher.PublishJob]) -> Dict[str, Any]:
    posts_api = _posts_api()
    post = posts_api._load(db, ctx, post_id)
    actor = posts_api._actor(ctx)
    try:
        job = begin(db, post, actor)
    except service.SocialsError as exc:
        posts_api._raise_for(exc)
    schedule_jobs.sync_job(post)  # a scheduled post published now has no slot left
    anyio.from_thread.run_sync(publisher.launch, job)
    return post.to_dict()


@router.post("/posts/{post_id}/publish-now", status_code=202, dependencies=[CAN_UPDATE])
def publish_social_post_now(
    post_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """Publish an approved or scheduled post now: 202 with it in ``publishing``. 409
    when its approval no longer matches its content, it is not approved or scheduled,
    or another request claimed it first; 409 too when it has no channels."""
    return _start(db, ctx, post_id, publisher.begin_publish)


@router.post("/posts/{post_id}/retry", status_code=202, dependencies=[CAN_UPDATE])
def retry_social_post(
    post_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """Publish again the failed targets of a failed or partially published post: 202
    with it in ``publishing``. A published target is never run again. 409 unless the
    post's approval still matches its content and one of its targets failed."""
    return _start(db, ctx, post_id, publisher.begin_retry)


@router.get("/posts/{post_id}/ai-labels", dependencies=[CAN_READ])
def social_post_ai_labels(
    post_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """Each channel's AI label decision for the post (PRD-264 EUAI-05), and its AI media:
    ``{"ai_media": {...}, "targets": [{"id", "toolkit", "post_kind", "ai_label", "at_publish"}]}``.
    404 for a post outside the caller's workspace."""
    return ai_media.labels_for(_posts_api()._load(db, ctx, post_id))


class AiLabelRequest(BaseModel):
    """The post's AI-generated label: on or off (unknown fields are refused)."""

    model_config = ConfigDict(extra="forbid")
    on: StrictBool


@router.put("/posts/{post_id}/ai-label", dependencies=[CAN_UPDATE])
def set_social_post_ai_label(
    post_id: UUID,
    body: AiLabelRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """Switch the post's "Realistic AI people, places, products or voice: label as
    AI-generated" on or off (PRD-264 EUAI-08a), as the signed-in person; the post as
    saved. The setting is approved content: an approved or scheduled post goes back for
    approval. 404 for a post outside the caller's workspace; 409 while it renders or
    publishes, or when another write committed first."""
    posts_api = _posts_api()
    post = posts_api._load(db, ctx, post_id)
    actor = posts_api._actor(ctx)
    status, content_hash = post.status, post.content_hash
    try:
        deepfake_label.set_label(post, actor, body.on)
        return posts_api._commit_unchanged(db, post, status=status, content_hash=content_hash)
    except service.SocialsError as exc:
        posts_api._raise_for(exc)

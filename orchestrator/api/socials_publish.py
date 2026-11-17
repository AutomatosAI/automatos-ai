"""
Socials publishing (PRD-251 US-301, S3.3)
=========================================

``POST /api/socials/posts/{post_id}/publish-now`` publishes an approved, scheduled or
missed post, and ``POST /api/socials/posts/{post_id}/retry`` publishes again the targets that
failed of a failed or partially published post. Each answers 202 with the post in
``publishing``; the publish runs in the background (``modules/socials/publishing.py``)
and ends the post ``published``, ``partially_published`` or ``failed``, each target
with its receipt.

One Approve (17 Nov 2026): the approve route (``api/socials.py``) calls
:func:`publish_on_approval` once its approval has committed. A post is published then only when
the click asked for it (``post_now``: its button said "Approve & post now") AND the server agrees
(``publish_lifecycle.publishes_on_approval``: no planned slot, no schedule, a channel), through
publish-now's claim (``publisher.begin_publish``, the job synced, ``publisher.launch``): the
approve answers with the post in ``publishing``. A post whose planned slot, else a schedule kept
from before an edit, is still ahead was scheduled there by the approval, and any other stays
approved: neither is published. An edit that won the race since is publish-now's 409; any other
claim refused, or a launch that failed, leaves the approval standing: logged, and the answer says
so (``posting_not_started``). A post claimed whose run never started is ended at once as the
lost-publish sweep ends one (``publish_records.end_lost``: failed, each channel never tried, so
Retry publishes it), never left publishing. The agent tools never reach it (D6, D14).

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

import logging
from typing import Any, Callable, Dict, Optional
from uuid import UUID

import anyio
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, StrictBool
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.socials import SocialPost
from modules.socials import ai_media, deepfake_label, notify, publish_lifecycle, publish_records, publisher, schedule_jobs, service

router = APIRouter()
logger = logging.getLogger(__name__)
# In an approve's answer: the approval stands, but the publish it asked for did not start.
POSTING_NOT_STARTED = "posting_not_started"

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


def publish_on_approval(db: Session, ctx: RequestContext, post: SocialPost, *, post_now: bool) -> Optional[Dict[str, Any]]:
    """The approve route's last step, after its approval committed: with ``post_now`` (the click
    said "Approve & post now") and the server's own check agreeing, the post is published now
    through publish-now's claim, and answered as it is then (``publishing``). None for any other
    post (scheduled, or left approved), which the approve answers as it saved it. An edit that
    voided the approval since is publish-now's 409, nothing launched; any other failure after the
    commit is logged and answered as :func:`_not_started` says."""
    if not (post_now and publish_lifecycle.publishes_on_approval(post)):
        return None
    posts_api = _posts_api()
    post_id, approved_hash, actor = post.id, post.approved_hash, posts_api._actor(ctx)
    try:
        job = publisher.begin_publish(db, post, actor)
    except service.SocialsError as exc:
        current = posts_api._load(db, ctx, post_id)
        if current.approved_hash != approved_hash or not publish_lifecycle.approval_matches(current):
            posts_api._raise_for(exc)  # an edit voided the approval: the 409 publish-now answers
        logger.exception("[Socials] post %s: approved, but its publish claim was refused", post_id)
        return _not_started(current)
    try:
        schedule_jobs.sync_job(post)
        anyio.from_thread.run_sync(publisher.launch, job)
    except Exception:
        logger.exception("[Socials] post %s: approved and claimed, but its publish did not start", post_id)
        return _not_started(_end_unstarted(db, post, actor))
    return post.to_dict()


def _end_unstarted(db: Session, post: SocialPost, actor: str) -> SocialPost:
    """A post claimed for publishing whose run never started, ended now as the lost-publish sweep
    ends one whose run is gone (``schedule_jobs.end_lost_publishes``): failed, each channel never
    tried, so Retry publishes it, and the workspace is told, as the sweep tells it. A compare-and-set,
    so an end written first stands."""
    content_hash = post.content_hash
    status = publish_records.end_lost(post, actor)
    if service.claim_unchanged(db, post, status=service.PUBLISHING, content_hash=content_hash):
        db.commit()
        notify.notify_publish_outcome(post.workspace_id, post.id, post.title or "", status)
    else:
        db.rollback()
    db.refresh(post)
    return post


def _not_started(post: SocialPost) -> Dict[str, Any]:
    """The approve's answer when the publish it asked for did not start: the post as it is now, and
    ``posting_not_started`` unless another request is publishing it (the web says "Approved, but
    posting didn't start", with the button to press: Publish now, or Retry for a post ended failed)."""
    return {**post.to_dict(), POSTING_NOT_STARTED: post.status != service.PUBLISHING}


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

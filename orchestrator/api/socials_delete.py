"""
Delete a post (3 Oct 2026, Gerard: "I need to be able to add channels, edit a post ... delete")
================================================================================================

``DELETE /api/socials/posts/{post_id}``: the post is gone, with its channel rows (they
cascade). Owners and admins only (``documents:delete``): it cannot be undone.

* Refused (409) while it renders or publishes, and once any of it is live on a channel
  (published, partly published, or a channel row with a published time or a remote id):
  what went out keeps its record.
* A scheduled post's publish job goes with it (``schedule_jobs.cancel_job``; the reconcile
  pass would drop it too).
* A plan's post: its slot is marked skipped, so the plan does not make it again, and the
  idea it used is unused again.
* The files the post made, uploaded or was offered leave Deliverables (a soft delete, after
  the post's own commit; a failure is logged and the delete stands). A Library picture it
  borrowed carries its own source and stays.

This router has no prefix and no gate of its own: ``api/socials.py`` includes it in the
Socials router, whose ``require_socials_enabled`` answers 404 unless both switches are on.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import or_, text
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.socials import SocialCampaign, SocialPost, SocialPostTarget, SocialTopic
from modules.socials import plan_store, plans, schedule_jobs, service
from services.deliverable_service import DeliverableService

logger = logging.getLogger(__name__)

router = APIRouter()

CAN_DELETE = Depends(require_workspace_permission("documents:delete"))
BUSY = frozenset({service.RENDERING, service.PUBLISHING})
WENT_OUT = frozenset({service.PUBLISHED, service.PARTIALLY_PUBLISHED})
BUSY_MESSAGE = "This post is rendering or publishing: delete it once that ends."
WENT_OUT_MESSAGE = "This post has gone out on a channel, so it stays: its receipts are its record."
_ITS_FILES = text(
    "SELECT id FROM deliverables WHERE workspace_id = :workspace_id AND source_id = :post_id AND deleted_at IS NULL"
)


def _posts_api() -> Any:
    """``api/socials.py``: it includes this module, so it is imported when a request runs."""
    from api import socials

    return socials


def _refusal(db: Session, post: SocialPost) -> Optional[str]:
    """Why ``post`` cannot be deleted now, or ``None``."""
    live = db.query(SocialPostTarget.id).filter(
        SocialPostTarget.post_id == post.id,
        or_(SocialPostTarget.published_at.isnot(None), SocialPostTarget.remote_id.isnot(None)),
    ).first()
    if post.status in WENT_OUT or live is not None:
        return WENT_OUT_MESSAGE
    return BUSY_MESSAGE if post.status in BUSY else None


def _free_its_plan_slot(db: Session, post: SocialPost) -> None:
    """A plan's post: the idea it used is unused again, and its slot is skipped."""
    for topic in db.query(SocialTopic).filter(SocialTopic.used_post_id == post.id):
        topic.used_post_id, topic.used_at = None, None
    if not (post.slot_key and post.campaign_id):
        return
    plan = db.get(SocialCampaign, post.campaign_id)
    if plan is None or plan.kind != plans.PLAN:
        return
    try:
        plan_store.move_slot(plan, post.slot_key, to=None, skip=True)
    except plans.InvalidPlan as exc:  # the slot left the plan (its cadence changed): nothing to skip
        logger.warning("[Socials] slot %s of plan %s was not skipped: %s", post.slot_key, plan.id, exc)


def _retire_its_files(db: Session, workspace_id: UUID, post_id: UUID) -> int:
    """Soft-delete the Deliverables the post's own files are (``soft_delete`` logs its failures)."""
    rows = db.execute(_ITS_FILES, {"workspace_id": str(workspace_id), "post_id": str(post_id)}).fetchall()
    deliverables = DeliverableService(db, workspace_id)
    return sum(1 for row in rows if deliverables.soft_delete(str(row.id)).get("success"))


@router.delete("/posts/{post_id}", status_code=204, dependencies=[CAN_DELETE])
def delete_social_post(
    post_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Response:
    """Delete the post (the module docstring): 204; 404 outside the caller's workspace;
    409 while it renders or publishes, or once it went out. A plain ``def`` (F105)."""
    post = _posts_api()._load(db, ctx, post_id)
    refusal = _refusal(db, post)
    if refusal:
        raise HTTPException(status_code=409, detail=refusal)
    had_files = bool(post.media)
    _free_its_plan_slot(db, post)
    db.delete(post)
    db.commit()
    schedule_jobs.cancel_job(post_id)
    retired = _retire_its_files(db, ctx.workspace_id, post_id) if had_files else 0
    logger.info("[Socials] post %s deleted in workspace %s (%d files retired)", post_id, ctx.workspace_id, retired)
    return Response(status_code=204)

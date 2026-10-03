"""
Socials planned slots (PRD-251B B11, US-B105)
=============================================

``PUT /api/socials/posts/{post_id}/slot`` sets the slot a post is planned for, stored in
UTC with its zone. A slot is not content: the hash and the approval are untouched, so
moving a slot never voids an approval. Before approval the slot is set alone (``null``
clears it); an approved or scheduled post, or a missed one whose approval stands, is
rescheduled through the one schedule path, its planned slot kept in step; a missed post
without an approval restarts as a draft at the new slot; the publishing states refuse
(409). Approving a post with a future slot schedules it there (``service.approve``).

This router has no prefix and no gate of its own: ``api/socials.py`` includes it in the
Socials router, whose ``require_socials_enabled`` answers 404 unless both switches are
on (D1). Never mount it in the app directly. It reuses that module's post helpers,
imported when a request runs, because that module includes this one.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.socials import SocialPost
from modules.socials import publish_lifecycle, service

router = APIRouter()

CAN_UPDATE = Depends(require_workspace_permission("documents:update"))
DEFAULT_TZ = "UTC"
SLOT_EDIT_STATUSES = frozenset({service.DRAFT, service.NEEDS_APPROVAL, service.CHANGES_REQUESTED})
SLOT_RESCHEDULE_STATUSES = frozenset({service.APPROVED, service.SCHEDULED, service.MISSED})


class SlotRequest(BaseModel):
    """The slot a post is planned for; ``null`` clears it (before approval)."""

    model_config = ConfigDict(extra="forbid")

    planned_for: Optional[datetime] = None
    timezone: Optional[str] = None


def _posts_api() -> Any:
    import api.socials as posts_api  # that module includes this router

    return posts_api


def apply_slot(post: SocialPost, actor: str, body: SlotRequest) -> None:
    """Before approval the slot is set alone (never a content change); an approved or
    scheduled post, or a missed one whose approval stands, is rescheduled through the one
    schedule path; a missed post without an approval restarts as a draft at the new
    slot; the publishing states refuse."""
    if post.status in SLOT_EDIT_STATUSES:
        service.set_planned_for(post, body.planned_for, body.timezone)
        return
    if post.status == service.MISSED and not publish_lifecycle.approval_matches(post):
        service.reslot(post, actor, body.planned_for, body.timezone)
        return
    if post.status in SLOT_RESCHEDULE_STATUSES:
        if body.planned_for is None:
            raise service.InvalidPost("an approved post keeps its slot: unschedule it instead")
        service.schedule(post, actor, body.planned_for, body.timezone or post.timezone or DEFAULT_TZ)
        service.set_planned_for(post, body.planned_for, body.timezone)
        return
    raise service.IllegalTransition(post.status, "slot")


@router.put("/posts/{post_id}/slot", dependencies=[CAN_UPDATE])
def set_social_post_slot(
    post_id: UUID,
    body: SlotRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Set the post's planned slot (the module docstring). A plain ``def`` (F105)."""
    posts_api = _posts_api()
    post = posts_api._load(db, ctx, post_id)
    status, content_hash = post.status, post.content_hash
    try:
        apply_slot(post, posts_api._actor(ctx), body)
        return posts_api._commit_unchanged(db, post, status=status, content_hash=content_hash)
    except service.SocialsError as exc:
        posts_api._raise_for(exc)

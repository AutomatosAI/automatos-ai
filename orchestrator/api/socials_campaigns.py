"""
Socials campaigns (PRD-251 D6, S2.4)
====================================

Campaigns and their series approval (``modules/socials/campaigns.py``):

* ``GET /api/socials/campaigns``: the workspace's campaigns, newest first, each
  with how many posts it holds; ``POST`` creates one (``per_post`` by default).
* ``GET /api/socials/campaigns/{campaign_id}``: the campaign with its posts;
  ``PATCH`` renames it or changes its ``approval_mode``.
* ``POST`` / ``DELETE /api/socials/campaigns/{campaign_id}/posts/{post_id}``: put a
  post in the campaign, or take it out (sets or clears its ``campaign_id``).
* ``POST /api/socials/campaigns/{campaign_id}/approve`` (``socials:approve``): the
  series approval. The body lists every post the approver was shown, each with
  the ``content_hash`` of the version shown; each still waiting for approval with
  that hash is approved as a single approval would approve it (D6, D7), and the
  answer reports the approved posts and each post left unapproved, with why.
  409 when the workspace's series approval switch is off, or the campaign
  approves post by post.

Every read and write is scoped to the caller's workspace: another workspace's
campaign or post is a 404. Every route is a plain ``def``: FastAPI runs its
synchronous database work in the threadpool (F105).

This router has no prefix and no gate of its own: ``api/socials.py`` includes it
in the Socials router, whose ``require_socials_enabled`` answers 404 unless both
switches are on (D1). Never mount it in the app directly. It reuses that
module's helpers, imported when a request runs, because that module includes
this one.
"""

from __future__ import annotations

from typing import Any, Dict, List, NoReturn, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.socials import SocialCampaign, SocialPost
from modules.socials import campaigns, service

router = APIRouter()

CAN_CREATE = Depends(require_workspace_permission("documents:create"))
CAN_UPDATE = Depends(require_workspace_permission("documents:update"))
CAN_REVIEW = Depends(require_workspace_permission("socials:approve"))

CAMPAIGN_NOT_FOUND = "Campaign not found"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateCampaignRequest(_Strict):
    name: str = Field(..., min_length=1, max_length=campaigns.CAMPAIGN_NAME_MAX_CHARS)
    approval_mode: str = campaigns.PER_POST


class UpdateCampaignRequest(_Strict):
    name: Optional[str] = Field(None, min_length=1, max_length=campaigns.CAMPAIGN_NAME_MAX_CHARS)
    approval_mode: Optional[str] = None


class ShownPostRequest(_Strict):
    post_id: UUID
    # The hash of the version the approver was shown (the post's content_hash).
    content_hash: str = Field(..., pattern=service.CONTENT_HASH_PATTERN)
    # D7's second confirmation, for this post's unsourced claims.
    override_unsourced: bool = False


class ApproveSeriesRequest(_Strict):
    posts: List[ShownPostRequest] = Field(..., min_length=1, max_length=campaigns.SERIES_MAX_POSTS)
    comment: Optional[str] = Field(None, max_length=service.COMMENT_MAX_CHARS)


def _posts_api() -> Any:
    """``api/socials.py``: its router includes this one, so it is imported when a request runs."""
    from api import socials

    return socials


def _raise_for(exc: Exception) -> NoReturn:
    if isinstance(exc, campaigns.CampaignNotFound):
        raise HTTPException(status_code=404, detail=CAMPAIGN_NOT_FOUND)
    if isinstance(exc, campaigns.SeriesApprovalRefused):
        raise HTTPException(status_code=409, detail=str(exc))
    _posts_api()._raise_for(exc)


def _load(db: Session, ctx: RequestContext, campaign_id: UUID) -> SocialCampaign:
    campaign = campaigns.get_campaign(db, ctx.workspace_id, campaign_id)
    if campaign is None:
        raise HTTPException(status_code=404, detail=CAMPAIGN_NOT_FOUND)
    return campaign


def _with_posts(db: Session, campaign: SocialCampaign) -> Dict[str, Any]:
    posts = campaigns.campaign_posts(db, campaign.workspace_id, campaign.id)
    return {**campaign.to_dict(), "posts": [post.to_dict() for post in posts]}


def _saved_post(db: Session, post: SocialPost) -> Dict[str, Any]:
    db.commit()
    db.refresh(post)
    return post.to_dict()


@router.get("/campaigns")
def list_social_campaigns(
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """The workspace's campaigns, newest first, each with ``post_count``."""
    counts = campaigns.post_counts(db, ctx.workspace_id)
    rows = [
        {**campaign.to_dict(), "post_count": counts.get(campaign.id, 0)}
        for campaign in campaigns.list_campaigns(db, ctx.workspace_id)
    ]
    return {"campaigns": rows, "total": len(rows)}


@router.post("/campaigns", status_code=201, dependencies=[CAN_CREATE])
def create_social_campaign(
    body: CreateCampaignRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """Create a campaign with no posts (``per_post`` unless ``approval_mode`` says series)."""
    actor = _posts_api()._actor(ctx)
    try:
        campaign = campaigns.create_campaign(
            db, workspace_id=ctx.workspace_id, created_by=actor, name=body.name, approval_mode=body.approval_mode
        )
    except service.SocialsError as exc:
        _raise_for(exc)
    db.commit()
    db.refresh(campaign)
    return _with_posts(db, campaign)


@router.get("/campaigns/{campaign_id}")
def get_social_campaign(
    campaign_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """The campaign with its posts, oldest first, each with its status and content_hash."""
    return _with_posts(db, _load(db, ctx, campaign_id))


@router.patch("/campaigns/{campaign_id}", dependencies=[CAN_UPDATE])
def update_social_campaign(
    campaign_id: UUID,
    body: UpdateCampaignRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """Rename the campaign or change its approval mode."""
    campaign = _load(db, ctx, campaign_id)
    try:
        campaigns.update_campaign(campaign, body.model_dump(exclude_unset=True))
    except service.SocialsError as exc:
        _raise_for(exc)
    db.commit()
    db.refresh(campaign)
    return _with_posts(db, campaign)


@router.post("/campaigns/{campaign_id}/posts/{post_id}", dependencies=[CAN_UPDATE])
def add_social_campaign_post(
    campaign_id: UUID,
    post_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """Put the post in the campaign (moving it from another). Its approval is untouched:
    a series approval given before it joined does not cover it."""
    campaign = _load(db, ctx, campaign_id)
    post = _posts_api()._load(db, ctx, post_id)
    try:
        campaigns.add_post(campaign, post)
    except service.SocialsError as exc:
        _raise_for(exc)
    return _saved_post(db, post)


@router.delete("/campaigns/{campaign_id}/posts/{post_id}", dependencies=[CAN_UPDATE])
def remove_social_campaign_post(
    campaign_id: UUID,
    post_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """Take the post out of the campaign; a post the campaign does not hold is a 404."""
    campaign = _load(db, ctx, campaign_id)
    post = _posts_api()._load(db, ctx, post_id)
    try:
        campaigns.remove_post(campaign, post)
    except service.SocialsError as exc:
        _raise_for(exc)
    return _saved_post(db, post)


@router.post("/campaigns/{campaign_id}/approve", dependencies=[CAN_REVIEW])
def approve_social_campaign(
    campaign_id: UUID,
    body: ApproveSeriesRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """Approve the posts the approver was shown as a series (D6). 409 unless the
    workspace's series approval switch is on and the campaign is in series mode."""
    posts_api = _posts_api()
    campaign = _load(db, ctx, campaign_id)
    actor = posts_api._actor(ctx)
    shown = [
        campaigns.ShownPost(item.post_id, item.content_hash, item.override_unsourced) for item in body.posts
    ]
    try:
        campaigns.assert_series_allowed(posts_api._workspace(db, ctx).settings, campaign)
        return campaigns.approve_series(db, campaign, actor, shown, comment=body.comment)
    except service.SocialsError as exc:
        _raise_for(exc)

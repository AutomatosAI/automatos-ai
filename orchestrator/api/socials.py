"""
Socials API (PRD-251 S0.3b)
===========================

The post lifecycle over HTTP. Every route sits behind ``require_socials_enabled``
(a router-level dependency, so a new route cannot skip it): 404 unless the
platform master switch AND the caller's workspace switch are on (D1).

Every read and write is scoped to the caller's workspace: another workspace's
post is a 404. Review actions (approve, request changes, reject) need
``socials:approve`` (D6); other writes mirror ``api/blog.py``.

The lifecycle lives in ``modules/socials/service.py``; this module maps its
errors: IllegalTransition → 409, StaleContent → 409 (giving the current
``content_hash``), UnsourcedClaims → 422 (naming the claims), NotPublishable →
409, PublishingUnavailable → 501, InvalidPost → 422.

An approval binds to the content the approver saw (D6): the approve request
carries that version's ``content_hash``, and the write is a compare-and-set on
the post's status and hash (``service.claim_unchanged``). A post that changed
before the click, or while the request runs, answers 409 and is not written.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated, Any, Dict, List, NoReturn, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.core import DocumentTemplate
from core.models.socials import SOCIAL_POST_STATUSES, SocialPost
from modules.socials import service
from modules.socials.publisher import PublishingUnavailable, publish_post
from modules.socials.settings import require_socials_enabled

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/socials",
    tags=["Socials"],
    dependencies=[Depends(require_socials_enabled)],
)

CAN_CREATE = Depends(require_workspace_permission("documents:create"))
CAN_UPDATE = Depends(require_workspace_permission("documents:update"))
CAN_REVIEW = Depends(require_workspace_permission("socials:approve"))

POST_NOT_FOUND = "Post not found"


# ---------------------------------------------------------------------------
# Request schemas (unknown fields are refused)
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ``copy`` is aliased: a field literally named ``copy`` would shadow
# ``BaseModel.copy``. Dump with ``by_alias=True`` to get the service's names.
# The alias rides ``Annotated`` metadata (pydantic's supported form).
PostCopy = Annotated[Optional[Dict[str, Any]], Field(alias="copy")]


class CreateSocialPostRequest(_Strict):
    title: str = Field(..., min_length=1, max_length=500)
    brief: Optional[str] = None
    post_copy: PostCopy = None
    format: Optional[str] = None
    template_id: Optional[UUID] = None
    variables: Optional[Dict[str, Any]] = None
    sources: Optional[Dict[str, Any]] = None
    media: Optional[Dict[str, Any]] = None


class UpdateSocialPostRequest(_Strict):
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    brief: Optional[str] = None
    post_copy: PostCopy = None
    format: Optional[str] = None
    template_id: Optional[UUID] = None
    variables: Optional[Dict[str, Any]] = None
    sources: Optional[Dict[str, Any]] = None
    media: Optional[Dict[str, Any]] = None


class ApproveRequest(_Strict):
    # The hash of the version the approver was shown (the post's content_hash).
    content_hash: str = Field(..., pattern=service.CONTENT_HASH_PATTERN)
    override_unsourced: bool = False
    comment: Optional[str] = None


class RequestChangesRequest(_Strict):
    comment: str = Field(..., min_length=1)


class RejectRequest(_Strict):
    reason: Optional[str] = None


class ScheduleRequest(_Strict):
    scheduled_for: datetime
    timezone: str = "UTC"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _actor(ctx: RequestContext) -> str:
    user = getattr(ctx, "user", None)
    actor = getattr(user, "id", None) or getattr(user, "email", None)
    if not actor:
        raise HTTPException(status_code=401, detail="A signed-in user is required")
    return str(actor)


def _raise_for(exc: service.SocialsError) -> NoReturn:
    if isinstance(exc, service.UnsourcedClaims):
        raise HTTPException(status_code=422, detail={"message": str(exc), "claims": exc.names})
    if isinstance(exc, service.StaleContent):
        raise HTTPException(status_code=409, detail={"message": str(exc), "content_hash": exc.current_hash})
    if isinstance(exc, PublishingUnavailable):
        raise HTTPException(status_code=501, detail=str(exc))
    if isinstance(exc, (service.IllegalTransition, service.NotPublishable)):
        raise HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, service.InvalidPost):
        raise HTTPException(status_code=422, detail=str(exc))
    raise HTTPException(status_code=400, detail=str(exc))


def _load(db: Session, ctx: RequestContext, post_id: UUID) -> SocialPost:
    post = service.get_post(db, ctx.workspace_id, post_id)
    if post is None:
        raise HTTPException(status_code=404, detail=POST_NOT_FOUND)
    return post


def _check_template(db: Session, ctx: RequestContext, template_id: Optional[UUID]) -> None:
    """A template must be one of the caller's own (never another workspace's)."""
    if template_id is None:
        return
    found = (
        db.query(DocumentTemplate.id)
        .filter(DocumentTemplate.id == template_id, DocumentTemplate.workspace_id == ctx.workspace_id)
        .first()
    )
    if found is None:
        raise HTTPException(status_code=422, detail="template_id is not a template in this workspace")


def _save(db: Session, post: SocialPost) -> Dict[str, Any]:
    db.commit()
    db.refresh(post)
    return post.to_dict()


def _parse_statuses(status: Optional[str]) -> Optional[List[str]]:
    if not status:
        return None
    statuses = [s.strip() for s in status.split(",") if s.strip()]
    unknown = [s for s in statuses if s not in SOCIAL_POST_STATUSES]
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown status {unknown!r}")
    return statuses


# ---------------------------------------------------------------------------
# Posts
# ---------------------------------------------------------------------------


@router.get("/posts")
async def list_social_posts(
    status: Optional[str] = Query(None, description="One status, or several comma-separated"),
    window_from: Optional[datetime] = Query(None, alias="from"),
    window_to: Optional[datetime] = Query(None, alias="to"),
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """The workspace's posts, newest first. ``from``/``to`` bound a post's date:
    its slot when scheduled, otherwise when it was created."""
    posts = service.list_posts(
        db,
        ctx.workspace_id,
        statuses=_parse_statuses(status),
        window_from=window_from,
        window_to=window_to,
    )
    return {"posts": [p.to_dict() for p in posts], "total": len(posts)}


@router.post("/posts", status_code=201, dependencies=[CAN_CREATE])
async def create_social_post(
    body: CreateSocialPostRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Create a draft."""
    _check_template(db, ctx, body.template_id)
    try:
        post = service.create_draft(
            db, workspace_id=ctx.workspace_id, created_by=_actor(ctx), **body.model_dump(by_alias=True)
        )
    except service.SocialsError as exc:
        _raise_for(exc)
    return _save(db, post)


@router.get("/posts/{post_id}")
async def get_social_post(
    post_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    return _load(db, ctx, post_id).to_dict()


@router.patch("/posts/{post_id}", dependencies=[CAN_UPDATE])
async def update_social_post(
    post_id: UUID,
    body: UpdateSocialPostRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Edit a post. A content change voids an approval (D6)."""
    post = _load(db, ctx, post_id)
    changes = body.model_dump(exclude_unset=True, by_alias=True)
    if "title" in changes and changes["title"] is None:
        raise HTTPException(status_code=422, detail="title cannot be empty")
    _check_template(db, ctx, changes.get("template_id"))
    try:
        service.update_post(post, _actor(ctx), changes)
    except service.SocialsError as exc:
        _raise_for(exc)
    return _save(db, post)


# ---------------------------------------------------------------------------
# Lifecycle actions — each path is declared explicitly (never swallowed by /{id})
# ---------------------------------------------------------------------------


@router.post("/posts/{post_id}/submit", dependencies=[CAN_UPDATE])
async def submit_social_post(
    post_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    post = _load(db, ctx, post_id)
    try:
        service.submit(post, _actor(ctx))
    except service.SocialsError as exc:
        _raise_for(exc)
    return _save(db, post)


@router.post("/posts/{post_id}/approve", dependencies=[CAN_REVIEW])
async def approve_social_post(
    post_id: UUID,
    body: ApproveRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Approve the version the reviewer saw (D6). ``content_hash`` differs from
    the post's → 409 with the current hash. An edit another worker commits
    after the load is caught by the compare-and-set → the same 409, nothing
    written."""
    post = _load(db, ctx, post_id)
    try:
        service.approve(
            post,
            _actor(ctx),
            content_hash=body.content_hash,
            override_unsourced=body.override_unsourced,
            comment=body.comment,
        )
    except service.SocialsError as exc:
        _raise_for(exc)
    if not service.claim_unchanged(db, post, status=service.NEEDS_APPROVAL, content_hash=body.content_hash):
        db.rollback()
        _raise_for(service.StaleContent(service.compute_content_hash(_load(db, ctx, post_id))))
    return _save(db, post)


@router.post("/posts/{post_id}/request-changes", dependencies=[CAN_REVIEW])
async def request_changes_social_post(
    post_id: UUID,
    body: RequestChangesRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    post = _load(db, ctx, post_id)
    try:
        service.request_changes(post, _actor(ctx), body.comment)
    except service.SocialsError as exc:
        _raise_for(exc)
    return _save(db, post)


@router.post("/posts/{post_id}/reject", dependencies=[CAN_REVIEW])
async def reject_social_post(
    post_id: UUID,
    body: Optional[RejectRequest] = None,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    body = body or RejectRequest()
    post = _load(db, ctx, post_id)
    try:
        service.reject(post, _actor(ctx), body.reason)
    except service.SocialsError as exc:
        _raise_for(exc)
    return _save(db, post)


@router.post("/posts/{post_id}/schedule", dependencies=[CAN_UPDATE])
async def schedule_social_post(
    post_id: UUID,
    body: ScheduleRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    post = _load(db, ctx, post_id)
    try:
        service.schedule(post, _actor(ctx), body.scheduled_for, body.timezone)
    except service.SocialsError as exc:
        _raise_for(exc)
    return _save(db, post)


@router.post("/posts/{post_id}/unschedule", dependencies=[CAN_UPDATE])
async def unschedule_social_post(
    post_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    post = _load(db, ctx, post_id)
    try:
        service.unschedule(post, _actor(ctx))
    except service.SocialsError as exc:
        _raise_for(exc)
    return _save(db, post)


@router.post("/posts/{post_id}/publish-now", dependencies=[CAN_UPDATE])
async def publish_social_post_now(
    post_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """The approval guard runs first (409 on a stale approval). Wave 0 has no
    channel publishers, so a valid post answers 501."""
    post = _load(db, ctx, post_id)
    try:
        publish_post(db, post)
    except service.SocialsError as exc:
        _raise_for(exc)
    return _save(db, post)

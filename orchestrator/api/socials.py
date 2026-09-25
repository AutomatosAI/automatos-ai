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
409, PublishingUnavailable → 501, InvalidPost → 422, NotRenderable → 422,
RenderQuotaExceeded → 429, RendererUnavailable → 503.

Rendering (S1.1c): ``POST /posts/{id}/render`` checks the post, its template,
the month's render minutes (refused before anything reaches media-render),
storage and the renderer, then moves the post to ``rendering`` and answers 202;
the render runs in the background (``modules/socials/render.py``) and ends the
post in ``needs_approval`` or ``failed``. ``GET /posts/{id}/media/{file}``
streams a rendered file (the Deliverable's preview link), and ``GET /usage``
reads the render minutes used and the quota.

An approval binds to the content the approver saw (D6): the approve request
carries that version's ``content_hash``, and a post that changed before the
click answers 409.

Every write is a compare-and-set (``_commit_unchanged``): it commits only if the
post's row still has the status and ``content_hash`` the request loaded
(``service.claim_unchanged``). When another writer committed first, the request
rolls back, writes nothing and answers 409 with the current ``content_hash``.
An edit can therefore never slip under an approval, and an approval never lands
on copy nobody reviewed. ``review_log`` is reassigned whole, so this also keeps a
stale copy from erasing entries another writer committed.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Annotated, Any, Dict, List, NoReturn, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from core import media_render_quota as render_quota
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.core import DocumentTemplate
from core.models.socials import SOCIAL_POST_STATUSES, SocialPost
from core.models.workspaces import Workspace
from core.utils.background_tasks import launch_guarded
from modules.documents.brand_kit import get_brand_kit
from modules.documents.brand_logo import brand_kit_for_render
from modules.socials import media_store, render, service
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
MEDIA_NOT_FOUND = "Media not found"
MEDIA_STORAGE_UNAVAILABLE = "Media storage unavailable"
# A re-render replaces a file under the same name, so it is never cached as fresh.
MEDIA_CACHE_CONTROL = "private, no-cache"


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


def _raise_for(exc: Exception) -> NoReturn:
    if isinstance(exc, service.UnsourcedClaims):
        raise HTTPException(status_code=422, detail={"message": str(exc), "claims": exc.names})
    if isinstance(exc, service.StaleContent):
        raise HTTPException(status_code=409, detail={"message": str(exc), "content_hash": exc.current_hash})
    if isinstance(exc, PublishingUnavailable):
        raise HTTPException(status_code=501, detail=str(exc))
    if isinstance(exc, (service.IllegalTransition, service.NotPublishable)):
        raise HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, (service.InvalidPost, render.NotRenderable)):
        raise HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, render_quota.RenderQuotaExceeded):
        raise HTTPException(status_code=429, detail=str(exc))
    if isinstance(exc, render.RendererUnavailable):
        raise HTTPException(status_code=503, detail=str(exc))
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


def _commit_unchanged(
    db: Session, ctx: RequestContext, post: SocialPost, *, status: str, content_hash: str
) -> Dict[str, Any]:
    """Commit the request's change to ``post`` only if its row still has
    ``status`` and ``content_hash``, the version the request checked. When
    another writer committed first, roll back, write nothing and answer 409 with
    the post's current hash. Call it straight after the service mutation: a
    query in between would autoflush the change before the check."""
    post_id = post.id
    if not service.claim_unchanged(db, post, status=status, content_hash=content_hash):
        db.rollback()
        _raise_for(service.StaleContent(service.compute_content_hash(_load(db, ctx, post_id))))
    return _save(db, post)


def _workspace(db: Session, ctx: RequestContext) -> Workspace:
    workspace = db.get(Workspace, ctx.workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return workspace


def _render_template(db: Session, ctx: RequestContext, post: SocialPost) -> Any:
    """The post's template (its format and ``blocks``), from the caller's workspace only."""
    template = None
    if post.template_id is not None:
        template = (
            db.query(DocumentTemplate.id, DocumentTemplate.format, DocumentTemplate.blocks)
            .filter(DocumentTemplate.id == post.template_id, DocumentTemplate.workspace_id == ctx.workspace_id)
            .first()
        )
    if template is None:
        raise render.NotRenderable("this post has no template to render: choose a social template first")
    return template


def _render_brand_kit(settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The workspace brand kit from its ``settings``, render-ready (an uploaded
    logo inlined, which may read storage). It lives in the documents module,
    which modules/socials may not import, so the render bundle gets it here."""
    return brand_kit_for_render(get_brand_kit(settings))


def _launch_render(job: render.RenderJob) -> None:
    """The render runs in the background; its end is written by the task itself."""
    launch_guarded(
        render.run_render(job), subsystem="socials", operation="render", workspace_id=job.workspace_id
    )


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
    status, content_hash = post.status, post.content_hash
    changes = body.model_dump(exclude_unset=True, by_alias=True)
    if "title" in changes and changes["title"] is None:
        raise HTTPException(status_code=422, detail="title cannot be empty")
    _check_template(db, ctx, changes.get("template_id"))
    try:
        service.update_post(post, _actor(ctx), changes)
    except service.SocialsError as exc:
        _raise_for(exc)
    return _commit_unchanged(db, ctx, post, status=status, content_hash=content_hash)


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
    status, content_hash = post.status, post.content_hash
    try:
        service.submit(post, _actor(ctx))
    except service.SocialsError as exc:
        _raise_for(exc)
    return _commit_unchanged(db, ctx, post, status=status, content_hash=content_hash)


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
    return _commit_unchanged(db, ctx, post, status=service.NEEDS_APPROVAL, content_hash=body.content_hash)


@router.post("/posts/{post_id}/request-changes", dependencies=[CAN_REVIEW])
async def request_changes_social_post(
    post_id: UUID,
    body: RequestChangesRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    post = _load(db, ctx, post_id)
    status, content_hash = post.status, post.content_hash
    try:
        service.request_changes(post, _actor(ctx), body.comment)
    except service.SocialsError as exc:
        _raise_for(exc)
    return _commit_unchanged(db, ctx, post, status=status, content_hash=content_hash)


@router.post("/posts/{post_id}/reject", dependencies=[CAN_REVIEW])
async def reject_social_post(
    post_id: UUID,
    body: Optional[RejectRequest] = None,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    body = body or RejectRequest()
    post = _load(db, ctx, post_id)
    status, content_hash = post.status, post.content_hash
    try:
        service.reject(post, _actor(ctx), body.reason)
    except service.SocialsError as exc:
        _raise_for(exc)
    return _commit_unchanged(db, ctx, post, status=status, content_hash=content_hash)


@router.post("/posts/{post_id}/schedule", dependencies=[CAN_UPDATE])
async def schedule_social_post(
    post_id: UUID,
    body: ScheduleRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    post = _load(db, ctx, post_id)
    status, content_hash = post.status, post.content_hash
    try:
        service.schedule(post, _actor(ctx), body.scheduled_for, body.timezone)
    except service.SocialsError as exc:
        _raise_for(exc)
    return _commit_unchanged(db, ctx, post, status=status, content_hash=content_hash)


@router.post("/posts/{post_id}/unschedule", dependencies=[CAN_UPDATE])
async def unschedule_social_post(
    post_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    post = _load(db, ctx, post_id)
    status, content_hash = post.status, post.content_hash
    try:
        service.unschedule(post, _actor(ctx))
    except service.SocialsError as exc:
        _raise_for(exc)
    return _commit_unchanged(db, ctx, post, status=status, content_hash=content_hash)


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


# ---------------------------------------------------------------------------
# Rendering (S1.1c)
# ---------------------------------------------------------------------------


@router.post("/posts/{post_id}/render", status_code=202, dependencies=[CAN_UPDATE])
async def render_social_post(
    post_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Render the post in the background: 202 with it in ``rendering``.

    Refused, with nothing changed, when the post holds an approval or is already
    rendering (409), has no social template (422), the workspace has used its
    render minutes this month (429, before any call to media-render), or there
    is no storage or renderer to use (503). The render ends the post in
    ``needs_approval`` with the files in ``media``, or in ``failed`` with the
    report in ``review_log``.
    """
    post = _load(db, ctx, post_id)
    actor = _actor(ctx)
    status, content_hash = post.status, post.content_hash
    try:
        service.assert_can_render(post)
        template = _render_template(db, ctx, post)
        workspace = _workspace(db, ctx)
        brand_kit = await asyncio.to_thread(_render_brand_kit, workspace.settings)
        bundle = render.bundle_for(post, template, brand_kit, fallback_name=workspace.name or "")
        render_quota.enforce_render_quota(db, workspace)
        await render.ensure_renderer()
        service.start_render(post, actor)
    except (service.SocialsError, render_quota.RenderQuotaExceeded) as exc:
        _raise_for(exc)
    saved = _commit_unchanged(db, ctx, post, status=status, content_hash=content_hash)
    _launch_render(
        render.RenderJob(
            post_id=post.id,
            workspace_id=post.workspace_id,
            actor=actor,
            content_hash=content_hash,
            title=post.title,
            format=post.format,
            bundle=bundle,
        )
    )
    return saved


@router.get("/posts/{post_id}/media/{file_name}")
async def get_social_post_media(
    post_id: UUID,
    file_name: str,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """A rendered file of the caller's post, streamed from storage (D9)."""
    post = _load(db, ctx, post_id)
    if not media_store.valid_file_name(file_name):
        raise HTTPException(status_code=404, detail=MEDIA_NOT_FOUND)
    key = media_store.media_key(post.workspace_id, post.id, file_name)
    try:
        stored = await asyncio.to_thread(media_store.MediaStore().open, key)
    except Exception as exc:  # noqa: BLE001 — storage is down: say so, never a 500 trace
        logger.error("[Socials] opening %s failed: %s", key, exc, exc_info=True)
        raise HTTPException(status_code=503, detail=MEDIA_STORAGE_UNAVAILABLE) from exc
    if stored is None:
        raise HTTPException(status_code=404, detail=MEDIA_NOT_FOUND)
    return StreamingResponse(
        stored.body,
        media_type=stored.content_type,
        headers={
            "Content-Length": str(stored.content_length),
            "Content-Disposition": f'inline; filename="{file_name}"',
            "Cache-Control": MEDIA_CACHE_CONTROL,
        },
    )


@router.get("/usage")
async def get_socials_usage(
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """This month's render minutes used and the plan's quota (``null`` = no quota)."""
    reading = render_quota.render_quota(db, _workspace(db, ctx))
    return {"render_minutes": reading.to_dict()}

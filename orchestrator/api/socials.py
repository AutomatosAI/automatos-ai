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
``content_hash``), UnsourcedClaims → 422 (naming the claims), SourcesNotFound →
422 (naming each claim and why), NotPublishable → 409, PublishingUnavailable →
501, InvalidPost → 422, NotRenderable → 422, RenderQuotaExceeded → 429,
RendererUnavailable → 503, ReportNotFound → 404, ChartNotBindable → 422,
ReportUnreadable → 503, PostNotFound → 404.

Facts carry sources (D7, S1.4): a save refuses a source it adds or changes
unless it resolves in the caller's workspace (``modules/socials/sources.py``);
an approval resolves every source again, and a claim whose source is gone counts
as unsourced. ``GET /sources`` searches candidates per kind for the composer.

Rendering (S1.1c): ``POST /posts/{id}/render`` checks the post, its template,
the month's render minutes (refused before anything reaches media-render),
storage and the renderer, then moves the post to ``rendering`` and answers 202;
the render runs in the background (``modules/socials/render.py``) and ends the
post in ``needs_approval`` or ``failed``. ``GET /posts/{id}/media/{file}``
streams a rendered file (the Deliverable's preview link), and ``GET /usage``
reads the render minutes used and the quota.

Voice (S1.5, D11): a post is spoken by Kokoro unless its ``voice`` names a
voice toolkit the workspace has connected in Composio (Fish Audio, ElevenLabs);
a save or a render with a voice the workspace cannot speak with now answers
422 saying why. ``GET /voices`` lists the choices (Kokoro, the connected voice
toolkits, and the allowlisted ones to connect), and ``GET /voices/{toolkit}``
a toolkit's own voices (``modules/socials/recipes/voice.py``).

Footage (S1.8, D12, D13): a post may ask its template's slots for footage or a
still (``footage``: ``{slot: {"prompt"}}``); a save names only slots the
template has and lets a toolkit fill (422 otherwise). The render generates them
through the workspace's Composio generation toolkit, priced and checked against
the post's cap and the workspace's monthly media cap before anything is
submitted (``modules/socials/recipes/footage.py``); with no generation toolkit
connected, the slots play the template's own motion graphics. ``GET /footage``
lists what the slots can be filled with here, and this month's media spend
against the cap.

Music credit (S1.6): a CC BY track needs its credit wherever the video is
published. Every save appends to the post's copy the credit lines the music of
its media asks for (``modules/socials/credits.py``: each rendered file records
its music on its Deliverable), and a render appends its own music's line.

A chart bound to a report (S1.7, D7): ``GET /sources/reports/{id}/chart``
fills a chart template (the Infographic) from a report of the workspace: its
top rows, each figure a claim bound to the report, and the chip naming it
(``modules/socials/report_charts.py``). A render of a chart whose first figure
is bound to a report is refused (422) unless the chart shows that report's rows
as the report has them now, so every number on it comes from its source.

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

Agents draft through the same code (US-116, S4.1): creating, editing, submitting
and rendering a post are the flows ``create_post``, ``edit_post``,
``submit_post`` and ``render_post``, which the routes and the platform tools
(``modules/tools/discovery/handlers_socials.py``) both call. There is no flow
that approves, schedules or publishes for a tool to reach: a person approves in
the Socials tab (D6), and the platform publishes (Wave 3).
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

from config import config
from core import media_render_quota as render_quota
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.media_render_bundle import voice_script
from core.models.core import DocumentTemplate
from core.models.socials import SOCIAL_POST_STATUSES, SocialPost
from core.models.workspaces import Workspace
from core.social_templates import SocialTemplateError, slot_generatable, validate_social_blocks
from core.utils.background_tasks import launch_guarded
from modules.documents.brand_kit import get_brand_kit
from modules.documents.brand_fonts import brand_kit_for_media_render
from modules.socials import media_caps, media_store, render, service
from modules.socials import credits as post_credits
from modules.socials import report_charts
from modules.socials import sources as post_sources
from modules.socials.capabilities import media_capabilities
from modules.socials.publisher import PublishingUnavailable, publish_post
from modules.socials.recipes import footage as footage_recipes
from modules.socials.recipes import voice as voice_recipes
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
CAN_READ_SOURCES = Depends(require_workspace_permission("documents:read"))

POST_NOT_FOUND = "Post not found"
VOICE_QUERY_MAX_CHARS = 100
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
    # D11: None is Kokoro; {"toolkit", "voice_id", "name"} a connected voice toolkit.
    voice: Optional[Dict[str, Any]] = None
    # D12: {slot: {"prompt"}}, footage or stills from a connected generation toolkit.
    footage: Optional[Dict[str, Any]] = None


class UpdateSocialPostRequest(_Strict):
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    brief: Optional[str] = None
    post_copy: PostCopy = None
    format: Optional[str] = None
    template_id: Optional[UUID] = None
    variables: Optional[Dict[str, Any]] = None
    sources: Optional[Dict[str, Any]] = None
    media: Optional[Dict[str, Any]] = None
    voice: Optional[Dict[str, Any]] = None
    footage: Optional[Dict[str, Any]] = None


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
        raise HTTPException(
            status_code=422,
            detail={"message": str(exc), "claims": exc.names, "unresolved": exc.unresolved},
        )
    if isinstance(exc, post_sources.SourcesNotFound):
        raise HTTPException(status_code=422, detail={"message": str(exc), "unresolved": exc.unresolved})
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
    if isinstance(exc, report_charts.ReportNotFound):
        raise HTTPException(status_code=404, detail=f"Report not found: {exc}")
    if isinstance(exc, report_charts.ChartNotBindable):
        raise HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, report_charts.ReportUnreadable):
        raise HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, service.PostNotFound):
        raise HTTPException(status_code=404, detail=POST_NOT_FOUND)
    raise HTTPException(status_code=400, detail=str(exc))


def _load(db: Session, ctx: RequestContext, post_id: UUID) -> SocialPost:
    post = service.get_post(db, ctx.workspace_id, post_id)
    if post is None:
        raise HTTPException(status_code=404, detail=POST_NOT_FOUND)
    return post


def _check_template(db: Session, workspace_id: UUID, template_id: Optional[UUID]) -> None:
    """A template must be one of the workspace's own (never another workspace's)."""
    if template_id is None:
        return
    found = (
        db.query(DocumentTemplate.id)
        .filter(DocumentTemplate.id == template_id, DocumentTemplate.workspace_id == workspace_id)
        .first()
    )
    if found is None:
        raise service.InvalidPost("template_id is not a template in this workspace")


def _save(db: Session, post: SocialPost) -> Dict[str, Any]:
    db.commit()
    db.refresh(post)
    return post.to_dict()


def _commit_unchanged(db: Session, post: SocialPost, *, status: str, content_hash: str) -> Dict[str, Any]:
    """Commit the request's change to ``post`` only if its row still has
    ``status`` and ``content_hash``, the version the request checked. When
    another writer committed first, roll back, write nothing and raise
    :class:`service.StaleContent` with the post's current hash (409). Call it
    straight after the service mutation: a query in between would autoflush the
    change before the check."""
    post_id, workspace_id = post.id, post.workspace_id
    if not service.claim_unchanged(db, post, status=status, content_hash=content_hash):
        db.rollback()
        current = service.get_post(db, workspace_id, post_id)
        if current is None:
            raise service.PostNotFound()
        raise service.StaleContent(service.compute_content_hash(current))
    return _save(db, post)


def _workspace(db: Session, ctx: RequestContext) -> Workspace:
    workspace = db.get(Workspace, ctx.workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return workspace


def _render_template(db: Session, workspace_id: UUID, post: SocialPost) -> Any:
    """The post's template (its format and ``blocks``), from the workspace's own only."""
    template = None
    if post.template_id is not None:
        template = (
            db.query(DocumentTemplate.id, DocumentTemplate.format, DocumentTemplate.blocks)
            .filter(DocumentTemplate.id == post.template_id, DocumentTemplate.workspace_id == workspace_id)
            .first()
        )
    if template is None:
        raise render.NotRenderable("this post has no template to render: choose a social template first")
    return template


def _render_brand_kit(settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The workspace brand kit from its ``settings``, render-ready: the uploaded
    logo, logo mark and font files inlined (PRD-251 D5), which may read storage.
    It lives in the documents module, which modules/socials may not import, so
    the render bundle gets it here."""
    return brand_kit_for_media_render(get_brand_kit(settings))


async def _capabilities(db: Session, workspace_id: UUID):
    """The workspace's media capability registry (D16). It reads the database and
    may commit the session (a pending connection upgraded): call it before any
    change of the request's own is staged."""
    return await asyncio.to_thread(media_capabilities, db, workspace_id)


async def _check_voice(db: Session, workspace_id: UUID, voice: Any) -> None:
    """A voice toolkit a save names must be one the workspace can speak with now (D11)."""
    clean = service.validate_voice(voice)
    if clean is not None:
        voice_recipes.plan_for(clean, await _capabilities(db, workspace_id))


def _check_footage(db: Session, workspace_id: UUID, footage: Any, template_id: Optional[UUID]) -> None:
    """The footage a save asks for names only slots the post's template has and
    lets a generation toolkit fill (S1.8). With no template yet, the render checks."""
    clean = service.validate_footage(footage)
    if not clean or template_id is None:
        return
    row = (
        db.query(DocumentTemplate.blocks)
        .filter(DocumentTemplate.id == template_id, DocumentTemplate.workspace_id == workspace_id)
        .first()
    )
    blocks = row.blocks if row is not None and isinstance(row.blocks, dict) else {}
    slots = blocks.get("slots") if isinstance(blocks.get("slots"), dict) else {}
    for slot in clean:
        spec = slots.get(slot)
        if not isinstance(spec, dict):
            has = f"its slots are {', '.join(sorted(slots))}" if slots else "it has none"
            raise service.InvalidPost(f"footage.{slot}: the post's template has no slot {slot} ({has})")
        if not slot_generatable(spec):
            label = spec.get("label") or slot
            raise service.InvalidPost(f"footage.{slot}: {label} takes the workspace's own file, never generated footage")


def _credited(db: Session, workspace_id: UUID, changes: Dict[str, Any], post: Optional[SocialPost] = None) -> Dict[str, Any]:
    """``changes`` whose copy carries the credit lines the post's media asks for
    after this save (S1.6): the media the save sets, else the post's own."""
    media = changes["media"] if "media" in changes or post is None else post.media
    lines = post_credits.media_credits(db, workspace_id, media)
    if not lines:
        return changes
    copy_before = changes["copy"] if "copy" in changes or post is None else post.copy
    credited = service.with_credits(copy_before, lines)
    return changes if credited is copy_before else {**changes, "copy": credited}


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
# The post writes: one flow each, shared by the routes and the agent tools (US-116)
# ---------------------------------------------------------------------------
# Each is what its route does once the caller's post is loaded, so an agent's
# draft passes every check a person's does. They raise only the lifecycle's
# errors (service.SocialsError, and RenderQuotaExceeded from a render): the
# routes answer them through _raise_for, the tools with a refusal. ``agent``
# names the agent a tool acts for, and review_log records it (the service).


async def create_post(
    db: Session, workspace_id: UUID, actor: str, fields: Dict[str, Any], *, agent: Optional[str] = None
) -> SocialPost:
    """A new draft of ``fields`` (``CreateSocialPostRequest``'s, by alias), committed.

    The template must be the workspace's own, a voice toolkit one it can speak
    with now (D11) and footage only slots its template lets a toolkit fill
    (D12); every source must resolve in the workspace (D7). The copy carries
    the credit lines its media's music asks for (S1.6).
    """
    _check_template(db, workspace_id, fields.get("template_id"))
    await _check_voice(db, workspace_id, fields.get("voice"))
    _check_footage(db, workspace_id, fields.get("footage"), fields.get("template_id"))
    if fields.get("sources"):
        post_sources.require_resolved(db, workspace_id, fields["sources"])
    fields = _credited(db, workspace_id, fields)
    post = service.create_draft(db, workspace_id=workspace_id, created_by=actor, agent=agent, **fields)
    db.commit()
    db.refresh(post)
    return post


async def edit_post(
    db: Session, post: SocialPost, actor: str, changes: Dict[str, Any], *, agent: Optional[str] = None
) -> Dict[str, Any]:
    """Edit ``post`` with ``changes`` (``UpdateSocialPostRequest``'s set fields, by
    alias); the compare-and-set commit.

    A content change voids an approval (D6). A source the edit adds or changes
    must resolve in the workspace (D7); one it keeps as it was is checked again
    at approval. A voice toolkit it names must be one the workspace can speak
    with now (D11); the voice is a render setting, so changing it alone voids
    nothing. So is the footage (D12): a slot asked for with its prompt unchanged
    keeps what a render made for it. The copy keeps the credit lines its
    media's music asks for (S1.6): an edit that drops one gets it back.
    """
    status, content_hash = post.status, post.content_hash
    if "title" in changes and changes["title"] is None:
        raise service.InvalidPost("title cannot be empty")
    workspace_id = post.workspace_id
    _check_template(db, workspace_id, changes.get("template_id"))
    if "voice" in changes:
        await _check_voice(db, workspace_id, changes["voice"])
    if "footage" in changes:
        _check_footage(db, workspace_id, changes["footage"], changes.get("template_id", post.template_id))
    if changes.get("sources"):
        post_sources.require_resolved(db, workspace_id, changes["sources"], unchanged_from=post.sources)
    changes = _credited(db, workspace_id, changes, post)
    service.update_post(post, actor, changes, agent=agent)
    return _commit_unchanged(db, post, status=status, content_hash=content_hash)


def submit_post(db: Session, post: SocialPost, actor: str, *, note: Optional[str] = None) -> Dict[str, Any]:
    """draft or changes_requested → needs_approval, with the submitter's ``note``
    for the reviewer (an agent's); the compare-and-set commit."""
    status, content_hash = post.status, post.content_hash
    service.submit(post, actor, note=note)
    return _commit_unchanged(db, post, status=status, content_hash=content_hash)


async def render_post(db: Session, workspace: Workspace, post: SocialPost, actor: str) -> Dict[str, Any]:
    """Start rendering ``post`` in the background: it is ``rendering`` when this returns.

    Refused, with nothing changed, when the post holds an approval or is already
    rendering (IllegalTransition), has no social template (NotRenderable), is a
    chart bound to a report that no longer shows the report's rows or names it,
    names a voice toolkit the workspace cannot speak with now, the workspace has
    used its render minutes this month (RenderQuotaExceeded, before any call to
    media-render), or there is no storage or renderer to use
    (RendererUnavailable). The render ends the post in ``needs_approval`` with
    the files in ``media``, or in ``failed`` with the report in ``review_log``:
    a render whose footage would take the post or the workspace over its media
    cap submits nothing and fails saying why (D13).
    """
    status, content_hash = post.status, post.content_hash
    voice = post.voice
    service.assert_can_render(post)
    template = _render_template(db, workspace.id, post)
    brand_kit = await asyncio.to_thread(_render_brand_kit, workspace.settings)
    # D12: the footage the post asks for, planned now over the template's slots;
    # a slot no connected toolkit can make plays the template's motion graphics.
    caps = await _capabilities(db, workspace.id) if post.footage else None
    footage_plan = render.footage_plan_for(post, template, caps) if caps is not None else None
    bundle = render.bundle_for(
        post, template, brand_kit, fallback_name=workspace.name or "",
        footage_slots=footage_plan.shown if footage_plan is not None else (),
    )
    # S1.7 (D7): a chart bound to a report shows that report's rows as it has them now.
    await report_charts.check_bound_chart(
        db, workspace.id, render.composition_of(template), post.sources, bundle["variables"]
    )
    # D11: a voice toolkit speaks the script before the render; resolved now,
    # so a toolkit the workspace cannot use is refused with nothing changed.
    voice_plan = None
    if voice and voice_script(bundle):
        voice_plan = voice_recipes.plan_for(voice, caps or await _capabilities(db, workspace.id))
    render_quota.enforce_render_quota(db, workspace)
    await render.ensure_renderer()
    service.start_render(post, actor)
    saved = _commit_unchanged(db, post, status=status, content_hash=content_hash)
    _launch_render(
        render.RenderJob(
            post_id=post.id,
            workspace_id=post.workspace_id,
            actor=actor,
            content_hash=content_hash,
            title=post.title,
            format=post.format,
            bundle=bundle,
            voice=voice_plan,
            footage=footage_plan,
        )
    )
    return saved


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
    """Create a draft (``create_post``). Every source must resolve in the
    workspace (D7). The copy carries the credit lines its media's music asks
    for (S1.6)."""
    actor = _actor(ctx)
    try:
        post = await create_post(db, ctx.workspace_id, actor, body.model_dump(by_alias=True))
    except service.SocialsError as exc:
        _raise_for(exc)
    return post.to_dict()


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
    """Edit a post (``edit_post``). A content change voids an approval (D6). A
    source the edit adds or changes must resolve in the workspace (D7); one it
    keeps as it was is checked again at approval. A voice toolkit the edit names
    must be one the workspace can speak with now (D11); the voice is a render
    setting, so changing it alone voids nothing. So is the footage (D12): a slot
    the edit asks for with its prompt unchanged keeps what a render made for it.
    The copy keeps the credit lines its media's music asks for (S1.6): an edit
    that drops one gets it back."""
    post = _load(db, ctx, post_id)
    actor = _actor(ctx)
    try:
        return await edit_post(db, post, actor, body.model_dump(exclude_unset=True, by_alias=True))
    except service.SocialsError as exc:
        _raise_for(exc)


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
    actor = _actor(ctx)
    try:
        return submit_post(db, post, actor)
    except service.SocialsError as exc:
        _raise_for(exc)


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
    written. Every source is resolved again first (D7): a claim whose source is
    gone from the workspace counts as unsourced."""
    post = _load(db, ctx, post_id)
    unresolved = post_sources.unresolved(db, ctx.workspace_id, post.sources)
    try:
        service.approve(
            post,
            _actor(ctx),
            content_hash=body.content_hash,
            override_unsourced=body.override_unsourced,
            comment=body.comment,
            unresolved_sources=unresolved,
        )
        return _commit_unchanged(db, post, status=service.NEEDS_APPROVAL, content_hash=body.content_hash)
    except service.SocialsError as exc:
        _raise_for(exc)


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
        return _commit_unchanged(db, post, status=status, content_hash=content_hash)
    except service.SocialsError as exc:
        _raise_for(exc)


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
        return _commit_unchanged(db, post, status=status, content_hash=content_hash)
    except service.SocialsError as exc:
        _raise_for(exc)


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
        return _commit_unchanged(db, post, status=status, content_hash=content_hash)
    except service.SocialsError as exc:
        _raise_for(exc)


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
        return _commit_unchanged(db, post, status=status, content_hash=content_hash)
    except service.SocialsError as exc:
        _raise_for(exc)


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
    """Render the post in the background (``render_post``): 202 with it in ``rendering``.

    Refused, with nothing changed, when the post holds an approval or is already
    rendering (409), has no social template (422), is a chart bound to a report
    that no longer shows the report's rows or names it (422, saying which row
    differs; 503 when the report's file cannot be read), names a voice toolkit the
    workspace cannot speak with now (422, saying why), the workspace has used
    its render minutes this month (429, before any call to media-render), or
    there is no storage or renderer to use (503). The render ends the post in
    ``needs_approval`` with the files in ``media``, or in ``failed`` with the
    report in ``review_log``: a render whose footage would take the post or the
    workspace over its media cap submits nothing and fails saying why (D13).
    """
    post = _load(db, ctx, post_id)
    actor = _actor(ctx)
    workspace = _workspace(db, ctx)
    try:
        return await render_post(db, workspace, post, actor)
    except (service.SocialsError, render_quota.RenderQuotaExceeded) as exc:
        _raise_for(exc)


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


# ---------------------------------------------------------------------------
# Voice (S1.5)
# ---------------------------------------------------------------------------


@router.get("/voices")
async def list_social_voice_sources(
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """What a post can be spoken with (D11, D15): Kokoro, always; each voice
    toolkit the workspace can speak with (``available``); each allowlisted one it
    has not connected (``connect``: the Composio connect flow); a connected one
    it cannot use now (``unavailable``, with the reason)."""
    return voice_recipes.voice_sources(await _capabilities(db, ctx.workspace_id))


@router.get("/voices/{toolkit}", dependencies=[CAN_UPDATE])
async def list_social_toolkit_voices(
    toolkit: str,
    q: Optional[str] = Query(None, max_length=VOICE_QUERY_MAX_CHARS, description="Only voices whose name holds this"),
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """A connected voice toolkit's voices, through its allowlisted ``voices``
    action on the workspace's own connection: 422 when the workspace cannot use
    the toolkit, 502 when the toolkit does not answer."""
    caps = await _capabilities(db, ctx.workspace_id)
    try:
        voices = await voice_recipes.list_voices(
            db, ctx.workspace_id, toolkit, caps=caps, query=q, limit=config.SOCIALS_VOICE_LIST_LIMIT
        )
    except voice_recipes.VoiceUnavailable as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except voice_recipes.VoiceToolError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"toolkit": toolkit, "voices": voices}


# ---------------------------------------------------------------------------
# Footage (S1.8)
# ---------------------------------------------------------------------------


@router.get("/footage")
async def list_social_footage_sources(
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """What a post's slots can be filled with here (D12, D15, D16): per kind
    (footage, stills), the connected generation toolkit a render would use or why
    none can; each generation toolkit ``available``, to ``connect`` (the Composio
    connect flow) or ``unavailable`` and why; and this month's media spend
    against the workspace's monthly media cap (D13)."""
    caps = await _capabilities(db, ctx.workspace_id)
    spend = media_caps.media_spend(db, _workspace(db, ctx))
    return {**footage_recipes.footage_sources(caps), "spend": spend.to_dict()}


# ---------------------------------------------------------------------------
# Sources (S1.4)
# ---------------------------------------------------------------------------


@router.get("/sources", dependencies=[CAN_READ_SOURCES])
async def search_social_sources(
    kind: Optional[str] = Query(None, description="deliverable, report, document, url or metric; every kind when omitted"),
    q: Optional[str] = Query(None, max_length=post_sources.SEARCH_QUERY_MAX_CHARS),
    limit: int = Query(post_sources.SEARCH_DEFAULT_LIMIT, ge=1, le=post_sources.SEARCH_MAX_LIMIT),
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """What a claim can be bound to (D7), from the caller's workspace only.

    Deliverables, reports and documents whose title or summary contains ``q``;
    metrics whose name does, each with its latest figure and the report it was
    read from; for ``url``, ``q`` itself when it is an http(s) address. Newest
    first, at most ``limit`` per kind. A candidate's ``kind``, ``ref`` and
    ``as_of`` are the source to store.
    """
    if kind is not None and kind not in service.SOURCE_KINDS:
        raise HTTPException(status_code=422, detail=f"kind must be one of {list(service.SOURCE_KINDS)}")
    candidates = post_sources.search(db, ctx.workspace_id, kind=kind, q=q, limit=limit)
    return {"candidates": candidates, "total": len(candidates)}


def _chart_template(db: Session, ctx: RequestContext, template_id: UUID) -> Dict[str, Any]:
    """A social template of the caller's workspace, checked: the chart a report fills (S1.7)."""
    template = (
        db.query(DocumentTemplate.format, DocumentTemplate.blocks)
        .filter(DocumentTemplate.id == template_id, DocumentTemplate.workspace_id == ctx.workspace_id)
        .first()
    )
    blocks = render.composition_of(template)
    if blocks is None:
        raise HTTPException(status_code=422, detail="template_id is not a social template in this workspace")
    try:
        return validate_social_blocks(blocks, template.format)
    except SocialTemplateError as exc:
        raise HTTPException(status_code=422, detail=f"this template cannot be used: {exc}") from exc


@router.get("/sources/reports/{report_id}/chart", dependencies=[CAN_READ_SOURCES])
async def chart_social_source_report(
    report_id: str,
    template_id: UUID = Query(..., description="A chart template of this workspace (one with a data block)"),
    chart: Optional[str] = Query(None, description="bar, line or grid; the template's own when the figures fit it"),
    column: Optional[str] = Query(None, max_length=report_charts.COLUMN_MAX_CHARS, description="The figures' column, by its header"),
    part: Optional[str] = Query(None, description="table or metrics; the report's table when its file has one"),
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """A chart template filled from a report of the caller's workspace (S1.7, D7).

    The report's data table (the first markdown table in its file; its first
    column the labels, ``column`` or its first numeric column the figures) or
    its metrics. The top rows, as many as the template shows, exactly as the
    report wrote them. ``variables`` and ``sources`` are the post's own shapes:
    each figure shown is a claim bound to the report, every row variable is set
    (past the report's rows, to empty), and the chip names the report as it
    resolves. A render checks the chart still matches the report. 404 when the
    report is not in the workspace; 422 when its data cannot fill the chart (and
    why: no table, no numeric column, figures a bar or a line cannot compare).
    """
    blocks = _chart_template(db, ctx, template_id)
    try:
        binding = await report_charts.bind_report(
            db, ctx.workspace_id, report_id, blocks, chart=chart, column=column, part=part
        )
    except service.SocialsError as exc:
        _raise_for(exc)
    return binding.to_dict()

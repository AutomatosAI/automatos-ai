"""PRD-251B Wave 1, US-B102 — a thumbnail for each social template, rendered once.

A thumbnail is one PNG of the template's first size at half resolution: an image
template renders as it is; a video template renders a still at its first second
(the bundle's ``still`` option, which media-render takes beside ``preview``). It goes
through the same pieces as a post's preview render (``render._submit``, ``_wait``,
``_store_outputs`` with ``preview=True``, so no Deliverable is registered), is stored
under the template's id like a preview file, and its KEY is written to
``DocumentTemplate.thumbnail_url``; the gallery presigns it (``template_gallery``).

Rendered at most once per template: a template with a key is skipped, and a save
that changes the composition clears the key (``template_service``) so the next list
renders it again. It is a still, so it books no render minutes and holds nothing
against the quota; media-render's own admission bound still applies through
``submit_when_free``.

The backfill runs OFF the request: the list route is a plain ``def`` (no event
loop under it), so ``start_backfill`` gives the work its own thread and loop, one
run per workspace at a time. Without a configured renderer or storage nothing is
attempted and the gallery shows its placeholders.

The brand kit comes in as a callable: it lives in ``modules/documents``, which this
module may not import (the render bundle gets it from the API layer, as the post
render does).
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Set
from uuid import UUID

import sqlalchemy as sa

from config import config
from core.media_render_client import MediaRenderClient, new_http_client
from core.models.core import DocumentTemplate
from core.models.workspaces import Workspace
from core.social_templates import SOCIAL_VIDEO, root_duration
from modules.socials import preview, render
from modules.socials.media_store import MediaStore, media_key

logger = logging.getLogger(__name__)

ACTOR = "system:thumbnail"
STILL_AT_SECONDS = 1.0
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
BrandKitOf = Callable[[Optional[Dict[str, Any]]], Dict[str, Any]]
TEMPLATE_COLUMNS = (
    DocumentTemplate.id, DocumentTemplate.name, DocumentTemplate.format, DocumentTemplate.blocks,
    DocumentTemplate.sample_data, DocumentTemplate.thumbnail_url,
)

_in_flight: Set[UUID] = set()
_lock = threading.Lock()


def _stub_post(template: Any, workspace_id: UUID) -> SimpleNamespace:
    """What ``render.bundle_for`` reads from a post, for a template on its own: its
    sample data as the variables' values."""
    sample = template.sample_data if isinstance(template.sample_data, dict) else {}
    variables = {name: {"value": value} for name, value in sample.items()}
    return SimpleNamespace(id=template.id, workspace_id=workspace_id, variables=variables, title=template.name)


def still_moment(blocks: Any) -> float:
    """The second the still is taken at: 1.0, or half of a shorter composition."""
    duration = root_duration((blocks or {}).get("html") or "") if isinstance(blocks, dict) else None
    if duration and duration > 0:
        return round(min(STILL_AT_SECONDS, duration / 2), 2)
    return STILL_AT_SECONDS


def thumbnail_bundle(template: Any, workspace: Any, brand_kit_of: BrandKitOf) -> Dict[str, Any]:
    """The render bundle for the thumbnail: the first size at half resolution; a video
    as one still. A still has no sound, so the template's audio plan stays out of it:
    media-render refuses a still that carries audio (F237: a video template with a voice
    or music never got a thumbnail)."""
    half = preview.preview_template(template)
    bundle = render.bundle_for(_stub_post(template, workspace.id), half, brand_kit_of(workspace.settings), fallback_name=workspace.name or "")
    if template.format == SOCIAL_VIDEO:
        bundle = {**bundle, "still": {"at": [still_moment(template.blocks)]}}
    return {key: value for key, value in bundle.items() if key != "audio"}


def _job(template: Any, workspace_id: UUID, bundle: Dict[str, Any]) -> render.RenderJob:
    return render.RenderJob(
        post_id=template.id, workspace_id=workspace_id, actor=ACTOR, content_hash="", title=template.name,
        format="video" if template.format == SOCIAL_VIDEO else "image", bundle=bundle, preview=True,
    )


def first_image_key(job: render.RenderJob, media: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
    """The storage key of the first image the render produced."""
    for records in media.values():
        for record in records:
            name = str(record.get("name") or "")
            if name.lower().endswith(IMAGE_SUFFIXES):
                return media_key(job.workspace_id, job.post_id, name)
    return None


def _write_key(session_factory: Callable[[], Any], template_id: UUID, workspace_id: UUID, key: str) -> None:
    """The key onto the template, only while it still has none (a targeted UPDATE)."""
    db = session_factory()
    try:
        table = DocumentTemplate.__table__
        db.execute(
            sa.update(table)
            .where(table.c.id == template_id, table.c.workspace_id == workspace_id, table.c.thumbnail_url.is_(None))
            .values(thumbnail_url=key)
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _load(session_factory: Callable[[], Any], workspace_id: UUID, template_ids: Iterable[UUID]):
    """The workspace and the templates among ``template_ids`` that still lack a thumbnail."""
    db = session_factory()
    try:
        workspace = db.get(Workspace, workspace_id)
        if workspace is not None:
            db.expunge(workspace)
        rows = (
            db.query(*TEMPLATE_COLUMNS)
            .filter(
                DocumentTemplate.id.in_(list(template_ids)),
                DocumentTemplate.workspace_id == workspace_id,
                DocumentTemplate.thumbnail_url.is_(None),
            )
            .order_by(DocumentTemplate.name)
            .all()
        )
        templates = [
            SimpleNamespace(id=row.id, name=row.name, format=row.format, blocks=row.blocks, sample_data=row.sample_data)
            for row in rows
        ]
        return workspace, templates
    finally:
        db.close()


async def render_thumbnail(
    template: Any, workspace: Any, *, client: MediaRenderClient, store: MediaStore,
    session_factory: Callable[[], Any], brand_kit_of: BrandKitOf,
) -> Optional[str]:
    """One template's thumbnail: rendered, stored, its key written. ``None`` when it
    could not be made (logged; the next list tries again)."""
    try:
        bundle = thumbnail_bundle(template, workspace, brand_kit_of)
    except render.NotRenderable as exc:
        logger.info("[Socials] no thumbnail for template %s: %s", template.id, exc)
        return None
    job = _job(template, workspace.id, bundle)
    deadline = time.monotonic() + config.SOCIALS_RENDER_MAX_WAIT_SECONDS
    try:
        accepted = await render._submit(client, bundle, deadline)
        finished = await render._wait(client, job, accepted, deadline)
        media = await render._store_outputs(client, store, session_factory, job, finished)
    except render.RenderFailure as failure:
        logger.warning("[Socials] thumbnail of template %s failed: %s (%s)", template.id, failure.message, failure.code)
        return None
    key = first_image_key(job, media)
    if key is None:
        logger.warning("[Socials] thumbnail of template %s produced no image", template.id)
        return None
    await asyncio.to_thread(_write_key, session_factory, template.id, workspace.id, key)
    return key


async def ensure_thumbnails(
    workspace_id: UUID, template_ids: Iterable[UUID], *, brand_kit_of: BrandKitOf,
    client: Optional[MediaRenderClient] = None, store: Optional[MediaStore] = None,
    session_factory: Optional[Callable[[], Any]] = None,
) -> List[str]:
    """A thumbnail for each of ``template_ids`` that lacks one; the keys made.
    Nothing is attempted without a healthy renderer and storage."""
    factory = session_factory or render._default_session_factory()
    client = client or MediaRenderClient()
    store = store or MediaStore()
    try:
        await render.ensure_renderer(client, store)
    except render.RendererUnavailable as exc:
        logger.info("[Socials] thumbnails skipped for workspace %s: %s", workspace_id, exc)
        return []
    workspace, templates = await asyncio.to_thread(_load, factory, workspace_id, template_ids)
    if workspace is None:
        return []
    made: List[str] = []
    for template in templates:
        key = await render_thumbnail(
            template, workspace, client=client, store=store, session_factory=factory, brand_kit_of=brand_kit_of,
        )
        if key:
            made.append(key)
    return made


async def _backfill(workspace_id: UUID, ids: List[UUID], brand_kit_of: BrandKitOf) -> None:
    """``ensure_thumbnails`` on this thread's own event loop, with a media-render client of this
    loop, closed when it ends: the process's shared client belongs to the server's loop (F238)."""
    async with new_http_client() as http:
        await ensure_thumbnails(workspace_id, ids, brand_kit_of=brand_kit_of, client=MediaRenderClient(http))


def start_backfill(workspace_id: UUID, template_ids: Iterable[UUID], *, brand_kit_of: BrandKitOf) -> bool:
    """Run ``ensure_thumbnails`` on its own thread and event loop (the list route runs
    off the loop), one run per workspace at a time. ``True`` when a run started."""
    ids = list(template_ids)
    if not ids:
        return False
    with _lock:
        if workspace_id in _in_flight:
            return False
        _in_flight.add(workspace_id)

    def run() -> None:
        try:
            asyncio.run(_backfill(workspace_id, ids, brand_kit_of))
        except Exception:  # noqa: BLE001 — a crashed backfill is logged; the next list starts another
            logger.exception("[Socials] thumbnail backfill for workspace %s crashed", workspace_id)
        finally:
            with _lock:
                _in_flight.discard(workspace_id)

    threading.Thread(target=run, name=f"socials-thumbnails-{workspace_id}", daemon=True).start()
    return True

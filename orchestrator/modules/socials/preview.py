"""PRD-251 S2.2b (US-208): the composer's preview render.

A video's preview renders the post's first size at half resolution (a 1080×1920
template previews at 540×960), with the same timing and the same music and sound
effects. It is stored as the post's ``preview``: NOT in ``media``, NOT in the
content hash, and never a move of the post's status, so it approves nothing and
voids nothing. Its files are named ``preview-…`` next to the post's media and are
registered as no Deliverable.

It spends render minutes like any render: the seconds are held against the month's
quota before anything reaches media-render (``RenderQuotaExceeded`` otherwise,
P251W1-RVW-3's rule), and the rendered seconds are booked on the ``media`` lane.
It spends no media money: a preview plays the template's own motion graphics in
its footage slots and speaks with Kokoro, whatever toolkit the post chose.

``preview`` is ``{"status": "rendering" | "done" | "failed", "content_hash": the
version it rendered, "files": [...], "error", "at"}``. The composer shows it
stale once the post's content moves past that hash.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional

from config import config
from core.media_render_client import MediaRenderClient
from core.media_render_quota import release_render
from core.social_templates import parse_size
from modules.socials import render, service, spoken_fields
from modules.socials.media_store import MediaStore, media_route

logger = logging.getLogger(__name__)

RENDERING = "rendering"
DONE = "done"
FAILED = "failed"
ACTION_PREVIEW = "preview"
SCALE = 2  # half resolution
FILE_FACTS = ("aspect", "content_type", "duration", "width", "height", "bytes")


class PreviewInProgress(service.SocialsError):
    """A preview of this post is rendering already (409)."""

    def __init__(self) -> None:
        super().__init__("A preview of this post is rendering already: wait for it to finish.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def assert_can_preview(post: Any) -> None:
    """A post whose content can still be edited previews, one preview at a time."""
    if post.status not in service.EDITABLE_STATUSES:
        raise service.IllegalTransition(post.status, ACTION_PREVIEW)
    if (post.preview or {}).get("status") == RENDERING:
        raise PreviewInProgress()


def half_size(size: str) -> str:
    """``1080x1920`` → ``540x960``: each side halved, rounded down to an even number."""
    width, height = parse_size(size)
    return f"{(width // SCALE) & ~1}x{(height // SCALE) & ~1}"


def preview_template(template: Any) -> Any:
    """``template`` rendering only its first size, at half resolution."""
    blocks = dict(render.composition_of(template) or {})
    sizes = blocks.get("sizes") or []
    if sizes:
        blocks["sizes"] = [half_size(str(sizes[0]))]
    return SimpleNamespace(id=getattr(template, "id", None), format=template.format, blocks=blocks)


def start(post: Any) -> None:
    """Mark the post's preview rendering, for the version it has now."""
    post.preview = {"status": RENDERING, "content_hash": post.content_hash, "files": [], "error": None, "at": _now()}


def preview_files(job: render.RenderJob, media: Mapping[str, List[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    """The preview's files as the composer shows them, each linked by the post's media route."""
    files = []
    for aspect, records in media.items():
        for record in records:
            facts = {key: record[key] for key in FILE_FACTS if key in record}
            files.append({"name": record["name"], "url": media_route(job.post_id, record["name"]), **facts, "aspect": aspect})
    return files


def _finish(factory: Callable[[], Any], job: render.RenderJob, *, files=None, error: Optional[str] = None) -> None:
    """Write how the preview ended on the post: only its ``preview``, never its status or content."""
    db = factory()
    try:
        post = service.get_post(db, job.workspace_id, job.post_id)
        if post is None:
            db.rollback()
            return
        post.preview = {
            "status": FAILED if error else DONE, "content_hash": job.content_hash,
            "files": files or [], "error": error, "at": _now(),
        }
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


async def _render_files(job: render.RenderJob, client: MediaRenderClient, store: MediaStore, factory) -> Dict[str, Any]:
    deadline = time.monotonic() + config.SOCIALS_RENDER_MAX_WAIT_SECONDS
    accepted = await render._submit(client, job.bundle, deadline)
    finished = await render._wait(client, job, accepted, deadline)
    return await render._store_outputs(client, store, factory, job, finished)


async def _preview(job: render.RenderJob, client: MediaRenderClient, store: MediaStore, factory) -> bool:
    started = time.monotonic()
    budget = config.SOCIALS_RENDER_MAX_WAIT_SECONDS
    try:
        try:
            media = await asyncio.wait_for(_render_files(job, client, store, factory), timeout=budget)
        except asyncio.TimeoutError:
            raise render.RenderFailure("timed_out", f"The preview did not finish within {budget // 60} minutes.") from None
    except render.RenderFailure as refused:
        failure = spoken_fields.named(refused, job.spoken_fields)  # F377: the fields, not the line
        logger.warning("[Socials] preview of post %s failed: %s (%s)", job.post_id, failure.message, failure.code)
        await asyncio.to_thread(_finish, factory, job, error=failure.message)
        return False
    except Exception:
        logger.exception("[Socials] preview of post %s failed unexpectedly", job.post_id)
        await asyncio.to_thread(_finish, factory, job, error="The preview failed unexpectedly. Try again.")
        raise
    await asyncio.to_thread(_finish, factory, job, files=preview_files(job, media))
    await asyncio.to_thread(render._book, job, render.rendered_seconds(media), int((time.monotonic() - started) * 1000))
    return True


async def run_preview(
    job: render.RenderJob,
    *,
    client: Optional[MediaRenderClient] = None,
    store: Optional[MediaStore] = None,
    session_factory: Optional[Callable[[], Any]] = None,
) -> bool:
    """Render the preview in the background; ``True`` when its files are on the
    post. However it ends, the seconds it held against the quota are given back,
    after its own were booked."""
    factory = session_factory or render._default_session_factory()
    try:
        return await _preview(job, client or MediaRenderClient(), store or MediaStore(), factory)
    finally:
        await release_render(factory, job.reservation)

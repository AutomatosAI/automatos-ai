"""PRD-251 S1.1c: rendering a post.

``POST /api/socials/posts/{id}/render`` checks the post, its template, the
monthly quota (render_quota.py), storage and the renderer, moves the post to
``rendering`` and hands a :class:`RenderJob` to :func:`run_render`, which runs
in the background:

1. submit the bundle to media-render (``core/media_render_client.py``), which
   answers once the job is staged, spoken, mixed and checked; a full renderer
   is asked again after its ``Retry-After``;
2. poll the job every ``SOCIALS_RENDER_POLL_SECONDS`` until it ends;
3. fetch every output, taking its sha256 on the way, then copy each into our
   storage, ``social-media/{workspace}/{post}/{file}`` (D9), and register it as
   a Deliverable. Nothing is stored until every output has arrived whole;
4. finish the post with a compare-and-set: ``rendering`` → ``needs_approval``
   with the files as ``media`` (the content hash covers their digests), or
   ``rendering`` → ``failed`` with the report in ``review_log``;
5. book the rendered seconds on the ``media`` lane at $0 (US-103), the units
   the quota counts.

Steps 1-3 together get at most ``SOCIALS_RENDER_MAX_WAIT_SECONDS``, which stays
under the boot reaper's stale cutoff, so the reaper only ever fails a render no
live task owns. A post that moved on while it rendered (the reaper failed it)
is left as it is, and nothing is booked. The renderer assembles; it never
generates (D3).
The bundle is the template's composition (``blocks.html`` and ``blocks.css``),
the post's variable values and the template's audio plan; brand tokens, fonts
and media inputs join it with the social templates (S1.2).
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional
from uuid import UUID

from config import config
from core.llm.providers import MEDIA_RENDER_PROVIDER
from core.llm.usage_context import LANE_MEDIA, usage_scope
from core.llm.usage_tracker import UsageTracker
from core.media_render_client import (
    BUSY,
    JOB_DONE,
    JOB_TERMINAL,
    NOT_CONFIGURED,
    NOT_FOUND,
    MediaRenderClient,
    MediaRenderError,
    MediaRenderUnavailable,
)
from modules.socials import service
from modules.socials.media_store import MediaNameError, MediaStore, content_type_for, media_key, media_route

logger = logging.getLogger(__name__)

# The booking's model id: the renderer's engine (core/llm/providers.py MEDIA_RENDER_PROVIDER).
RENDER_MODEL_ID = "hyperframes"
EXECUTION_PREFIX = "social_post:"
DELIVERABLE_SOURCE_TYPE = "social_post"
# The report kept in review_log: the first findings, each trimmed.
MAX_REPORTED_FINDINGS = 20
# The keys of a check finding (services/media-render/media_render/check_report.py) and a voice finding.
FINDING_KEYS = ("section", "severity", "code", "message", "selector", "time", "fixHint", "source", "line")
FINDING_TEXT_CHARS = 300
DEFAULT_ASPECT = "original"

MEDIA_PROFILE_MESSAGE = (
    "Rendering needs the media profile: start the renderer with "
    "`docker compose --profile media up -d media-render`."
)
NOT_CONFIGURED_MESSAGE = "Rendering is not configured on this server (SOCIALS_RENDER_URL is empty)."
UNREACHABLE_MESSAGE = "The renderer cannot be reached right now. Try again in a few minutes."
STORAGE_MESSAGE = "Rendering needs object storage: the rendered files are kept there."


class RendererUnavailable(service.SocialsError):
    """No renderer (or no storage) to render with; nothing changed."""


class NotRenderable(service.SocialsError):
    """The post has nothing a renderer can render."""


class RenderFailure(Exception):
    """A render that ends the post in ``failed``, with what to tell the user."""

    def __init__(self, code: str, message: str, report: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.report = {"code": code, **dict(report or {})}


def _local_edition() -> bool:
    return (config.AUTH_EDITION or "").strip().lower() == "local"


def unavailable_message(exc: MediaRenderUnavailable) -> str:
    """What a user is told when there is no renderer to reach."""
    if _local_edition():
        return MEDIA_PROFILE_MESSAGE
    return NOT_CONFIGURED_MESSAGE if exc.code == NOT_CONFIGURED else UNREACHABLE_MESSAGE


async def ensure_renderer(client: Optional[MediaRenderClient] = None, store: Optional[MediaStore] = None) -> None:
    """:class:`RendererUnavailable` unless storage is set up and the renderer answers ``/health``."""
    if not (store or MediaStore()).configured():
        raise RendererUnavailable(STORAGE_MESSAGE)
    try:
        await (client or MediaRenderClient()).health()
    except MediaRenderUnavailable as exc:
        raise RendererUnavailable(unavailable_message(exc)) from exc
    except MediaRenderError as exc:
        raise RendererUnavailable(f"The renderer is not healthy: {exc}") from exc


# ── the bundle ──────────────────────────────────────────────────────────────
def composition_of(template: Any) -> Optional[Dict[str, Any]]:
    """A social template's ``blocks``, or ``None`` when it carries no composition."""
    blocks = getattr(template, "blocks", None) if template is not None else None
    if not isinstance(blocks, dict):
        return None
    html = blocks.get("html")
    return blocks if isinstance(html, str) and html.strip() else None


def bundle_for(post: Any, template: Any) -> Dict[str, Any]:
    """The render bundle media-render takes (``services/media-render/media_render/bundle.py``).

    :class:`NotRenderable` when the post has no template, or its template no composition.
    """
    blocks = composition_of(template)
    if blocks is None:
        raise NotRenderable("this post has no social template to render: choose a template with a composition")
    variables = {
        name: spec.get("value")
        for name, spec in (getattr(post, "variables", None) or {}).items()
        if isinstance(spec, dict) and spec.get("value") is not None
    }
    bundle: Dict[str, Any] = {
        "workspace_id": str(post.workspace_id),
        "reference": f"{EXECUTION_PREFIX}{post.id}",
        "composition": {"html": blocks["html"], "css": blocks.get("css") or ""},
        "variables": variables,
    }
    audio = blocks.get("audio_plan")
    if isinstance(audio, dict) and audio:
        bundle["audio"] = audio
    return bundle


@dataclass(frozen=True)
class RenderJob:
    """What the background render needs, captured when the post moved to ``rendering``."""

    post_id: UUID
    workspace_id: UUID
    actor: str
    # The post's hash at the start: the finish is a compare-and-set on it.
    content_hash: str
    title: str
    format: Optional[str]
    bundle: Mapping[str, Any]


# ── the report ──────────────────────────────────────────────────────────────
def _finding(raw: Any) -> Dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {"message": str(raw)}
    kept = {key: raw[key] for key in FINDING_KEYS if raw.get(key) is not None}
    return {key: (value[:FINDING_TEXT_CHARS] if isinstance(value, str) else value) for key, value in kept.items()}


def _report(*, findings: Any = (), report: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The part of the renderer's report worth keeping: the first findings, the check, the timings."""
    out: Dict[str, Any] = {}
    listed = [f for f in (findings or []) if f is not None]
    if listed:
        out["findings"] = [_finding(f) for f in listed[:MAX_REPORTED_FINDINGS]]
        out["findings_total"] = len(listed)
    source = report or {}
    for key in ("check", "timings"):
        if isinstance(source.get(key), dict):
            out[key] = dict(source[key])
    return out


def _failure_from(exc: MediaRenderError) -> RenderFailure:
    if isinstance(exc, MediaRenderUnavailable):
        return RenderFailure(exc.code, unavailable_message(exc))
    if exc.code == "check_failed":
        errors = sum(1 for f in exc.findings if isinstance(f, dict) and f.get("severity") == "error")
        first = next((f for f in exc.findings if isinstance(f, dict) and f.get("severity") == "error"), None)
        detail = f": {str(first.get('message')).rstrip('.')}" if first and first.get("message") else ""
        message = f"The composition failed its check with {errors or len(exc.findings)} error(s){detail}. Nothing was rendered."
        return RenderFailure(exc.code, message, _report(findings=exc.findings, report=exc.report))
    reason = str(exc).rstrip(".")
    return RenderFailure(exc.code, f"The renderer refused the render: {reason}.", _report(report=exc.report))


# ── the background render ───────────────────────────────────────────────────
async def _submit(client: MediaRenderClient, job: RenderJob, deadline: float) -> Dict[str, Any]:
    while True:
        try:
            return await client.submit(job.bundle)
        except MediaRenderError as exc:
            wait = exc.retry_after or config.SOCIALS_RENDER_POLL_SECONDS
            if exc.code == BUSY and time.monotonic() + wait < deadline:
                logger.info("[Socials] renderer busy; post %s asks again in %ss", job.post_id, wait)
                await asyncio.sleep(wait)
                continue
            raise _failure_from(exc) from exc


async def _wait(client: MediaRenderClient, job: RenderJob, record: Dict[str, Any], deadline: float) -> Dict[str, Any]:
    """Poll until the job ends; a lost poll is retried until the deadline."""
    job_id = str(record.get("id") or "")
    if not job_id:
        raise RenderFailure("bad_response", "The renderer accepted the render but gave no job id.")
    while record.get("status") not in JOB_TERMINAL:
        if time.monotonic() >= deadline:
            minutes = config.SOCIALS_RENDER_MAX_WAIT_SECONDS // 60
            raise RenderFailure("timed_out", f"The render did not finish within {minutes} minutes.")
        await asyncio.sleep(config.SOCIALS_RENDER_POLL_SECONDS)
        try:
            record = await client.job(job_id)
        except MediaRenderError as exc:
            if exc.code == NOT_FOUND or exc.status == 404:
                raise RenderFailure(NOT_FOUND, "The renderer lost the render (it restarted). Render again.") from exc
            logger.warning("[Socials] polling render %s for post %s failed: %s", job_id, job.post_id, exc)
    if record.get("status") != JOB_DONE:
        error = record.get("error") if isinstance(record.get("error"), dict) else {}
        code = str(error.get("code") or record.get("status") or "failed")
        message = str(error.get("message") or "the render failed").rstrip(".")
        report = record.get("report") if isinstance(record.get("report"), dict) else {}
        raise RenderFailure(code, f"The render failed: {message}.", _report(findings=report.get("findings"), report=report))
    return record


def _aspect_slug(aspect: str) -> str:
    return "".join(ch if ch.isalnum() else "x" for ch in aspect.lower()) or DEFAULT_ASPECT


def stored_file_name(job: RenderJob, output: Mapping[str, Any]) -> str:
    """``<format>-<aspect>.<ext>``, e.g. ``video-9x16.mp4``: stable, so a re-render replaces its file."""
    name = str(output.get("name") or "")
    ext = Path(name).suffix.lower() if Path(name).suffix else ".mp4"
    aspect = str(output.get("aspect") or DEFAULT_ASPECT)
    return f"{job.format or 'render'}-{_aspect_slug(aspect)}{ext}"


def _register(session_factory: Callable[[], Any], job: RenderJob, key: str, file_name: str, entry: Dict[str, Any]) -> str:
    from services.deliverable_service import DeliverableService, _infer_artifact_type

    db = session_factory()
    try:
        result = DeliverableService(db, job.workspace_id).register(
            file_path=key,
            title=f"{job.title} ({entry['aspect']})",
            source_type=DELIVERABLE_SOURCE_TYPE,
            source_id=str(job.post_id),
            artifact_type=_infer_artifact_type(file_name),
            storage_type="s3",
            file_type=Path(file_name).suffix.lstrip(".") or None,
            file_size_bytes=entry["bytes"],
            preview_url=media_route(job.post_id, file_name),
            preview_type="file",
            extra={"social_post_id": str(job.post_id), "sha256": entry["sha256"], "aspect": entry["aspect"]},
        )
    finally:
        db.close()
    if not result.get("success") or not result.get("deliverable_id"):
        raise RenderFailure("deliverable_failed", f"The rendered file could not be saved as a Deliverable: {result.get('error')}")
    return str(result["deliverable_id"])


async def _store_outputs(
    client: MediaRenderClient,
    store: MediaStore,
    session_factory: Callable[[], Any],
    job: RenderJob,
    record: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch every output, then store and register each; the post's ``media``.

    Every file is fetched (and checked) before any is stored, so a missing or
    empty output never leaves another behind in storage or in Deliverables.
    """
    outputs = [o for o in (record.get("outputs") or []) if isinstance(o, dict) and o.get("name")]
    if not outputs:
        raise RenderFailure("no_output", "The renderer finished but returned no file.")
    with tempfile.TemporaryDirectory(prefix="socials-render-") as scratch:
        fetched = []
        for output in outputs:
            file_name = stored_file_name(job, output)
            try:
                key = media_key(job.workspace_id, job.post_id, file_name)
            except MediaNameError as exc:
                raise RenderFailure("bad_output", f"The renderer returned a file the post cannot store: {exc}") from exc
            if any(name == file_name for _, name, _, _, _, _ in fetched):
                raise RenderFailure("bad_output", f"The renderer returned two files for {file_name}.")
            path = Path(scratch) / file_name
            try:
                size, digest = await client.download(str(record["id"]), str(output["name"]), path)
            except MediaRenderError as exc:
                raise RenderFailure(exc.code, f"The rendered file could not be fetched: {exc}") from exc
            if size <= 0:
                raise RenderFailure("empty_output", "The renderer returned an empty file.")
            fetched.append((output, file_name, key, path, size, digest))

        media: Dict[str, List[Dict[str, Any]]] = {}
        for output, file_name, key, path, size, digest in fetched:
            aspect = str(output.get("aspect") or DEFAULT_ASPECT)
            content_type = content_type_for(file_name)
            try:
                await asyncio.to_thread(store.put_file, key, path, content_type)
            except Exception as exc:  # noqa: BLE001 — any storage error fails the render, loudly
                logger.exception("[Socials] storing %s for post %s failed", key, job.post_id)
                raise RenderFailure("storage_failed", "The rendered file could not be stored.") from exc
            entry: Dict[str, Any] = {"aspect": aspect, "bytes": size, "sha256": digest}
            deliverable_id = await asyncio.to_thread(_register, session_factory, job, key, file_name, entry)
            file_record: Dict[str, Any] = {
                "deliverable_id": deliverable_id,
                "name": file_name,
                "sha256": digest,
                "bytes": size,
                "content_type": content_type,
            }
            for fact in ("duration", "width", "height"):
                if isinstance(output.get(fact), (int, float)) and not isinstance(output.get(fact), bool):
                    file_record[fact] = output[fact]
            media.setdefault(aspect, []).append(file_record)
    return media


def rendered_seconds(media: Mapping[str, List[Mapping[str, Any]]]) -> float:
    """The seconds a render delivered: what the quota counts."""
    return float(sum(r.get("duration") or 0 for records in media.values() for r in records))


def _summary(media: Mapping[str, List[Mapping[str, Any]]]) -> str:
    parts = []
    for aspect, records in media.items():
        for r in records:
            size = f"{r['width']}×{r['height']}" if r.get("width") and r.get("height") else aspect
            length = f"{r['duration']:g} s " if r.get("duration") else ""
            parts.append(f"{length}{size}".strip())
    return "Rendered " + ", ".join(parts) + "."


def _finish(
    session_factory: Callable[[], Any],
    job: RenderJob,
    *,
    media: Optional[Mapping[str, List[Dict[str, Any]]]] = None,
    report: Optional[Mapping[str, Any]] = None,
    failure: Optional[RenderFailure] = None,
) -> Optional[str]:
    """End the render on the post, a compare-and-set on ``rendering`` and the
    start hash: the status it ended in, or ``None`` when the post moved on
    (nothing is written then). Files the post cannot record fail it instead."""
    db = session_factory()
    try:
        post = service.get_post(db, job.workspace_id, job.post_id)
        if post is None or post.status != service.RENDERING or post.content_hash != job.content_hash:
            db.rollback()
            logger.warning("[Socials] post %s moved on while it rendered; the render is dropped", job.post_id)
            return None
        if failure is None:
            try:
                service.finish_render(post, job.actor, media or {}, summary=_summary(media or {}), report=report)
            except service.InvalidPost as exc:
                failure = RenderFailure("bad_output", f"The rendered files could not be recorded: {exc}")
        if failure is not None:
            service.fail_render(post, job.actor, failure.message, report=failure.report)
        if not service.claim_unchanged(db, post, status=service.RENDERING, content_hash=job.content_hash):
            db.rollback()
            logger.warning("[Socials] post %s changed as its render finished; the render is dropped", job.post_id)
            return None
        db.commit()
        return post.status
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _default_session_factory() -> Callable[[], Any]:
    from core.database.database import SessionLocal

    return SessionLocal


def _book(job: RenderJob, seconds: float, latency_ms: int) -> None:
    with usage_scope(
        request_type=LANE_MEDIA,
        execution_id=f"{EXECUTION_PREFIX}{job.post_id}",
        workspace_id=job.workspace_id,
    ):
        UsageTracker.track_media(
            provider=MEDIA_RENDER_PROVIDER, model_id=RENDER_MODEL_ID, units=seconds, usd=0.0, latency_ms=latency_ms
        )


async def run_render(
    job: RenderJob,
    *,
    client: Optional[MediaRenderClient] = None,
    store: Optional[MediaStore] = None,
    session_factory: Optional[Callable[[], Any]] = None,
) -> bool:
    """Render the post in the background; ``True`` when it reached needs_approval."""
    client = client or MediaRenderClient()
    store = store or MediaStore()
    factory = session_factory or _default_session_factory()
    started = time.monotonic()
    budget = config.SOCIALS_RENDER_MAX_WAIT_SECONDS
    deadline = started + budget

    async def render_and_store():
        accepted = await _submit(client, job, deadline)
        finished = await _wait(client, job, accepted, deadline)
        return finished, await _store_outputs(client, store, factory, job, finished)

    try:
        try:
            finished, media = await asyncio.wait_for(render_and_store(), timeout=budget)
        except asyncio.TimeoutError:
            raise RenderFailure("timed_out", f"The render did not finish within {budget // 60} minutes.") from None
    except RenderFailure as failure:
        logger.warning("[Socials] render of post %s failed: %s (%s)", job.post_id, failure.message, failure.code)
        await asyncio.to_thread(_finish, factory, job, failure=failure)
        return False
    except Exception:
        logger.exception("[Socials] render of post %s failed unexpectedly", job.post_id)
        failure = RenderFailure("internal_error", "The render failed unexpectedly. Try again.")
        await asyncio.to_thread(_finish, factory, job, failure=failure)
        raise
    report = _report(report=finished.get("report") if isinstance(finished.get("report"), dict) else None)
    ended = await asyncio.to_thread(_finish, factory, job, media=media, report=report)
    if ended != service.NEEDS_APPROVAL:
        return False
    # latency: the whole render, from submit to the files in storage.
    _book(job, rendered_seconds(media), int((time.monotonic() - started) * 1000))
    return True

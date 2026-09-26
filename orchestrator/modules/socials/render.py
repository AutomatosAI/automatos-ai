"""PRD-251 S1.1c: rendering a post.

``POST /api/socials/posts/{id}/render`` checks the post, its template, the
monthly quota (``core/media_render_quota.py``), storage and the renderer, moves
the post to ``rendering`` and hands a :class:`RenderJob` to :func:`run_render`,
which runs in the background:

0. when the post asks for footage (US-114, D12), generate it through the
   workspace's Composio generation toolkit first: priced and capped before any
   submit (D13), submitted and polled, each file copied into our storage and
   registered as a Deliverable before its slot is marked done, and booked on the
   media lane; a slot generated earlier for the same prompt is reused, and one
   no connected toolkit can make plays the template's own motion graphics
   (``modules/socials/recipes/footage.py``). Then, when the post chose a voice
   toolkit (US-111, D11), speak its script through the workspace's Composio
   connection, one call per line, each line copied into our storage as it
   returns, and make the bundle's lines name those files
   (``modules/socials/recipes/voice.py``); Kokoro needs nothing here;
1. submit the bundle to media-render (``core/media_render_client.py``), which
   answers once the job is staged, spoken, mixed and checked; a full renderer
   is asked again after its ``Retry-After``;
2. poll the job every ``SOCIALS_RENDER_POLL_SECONDS`` until it ends;
3. fetch every output, taking its sha256 on the way, then copy each into our
   storage, ``social-media/{workspace}/{post}/{file}`` (D9), and register it as
   a Deliverable, which records the music it mixed (S1.6). Nothing is stored
   until every output has arrived whole;
4. finish the post with a compare-and-set: ``rendering`` → ``needs_approval``
   with the files as ``media`` (the content hash covers their digests) and, when
   the music is a CC BY track, its credit line appended to the copy
   (``core/music_credit.py``), or ``rendering`` → ``failed`` with the report in
   ``review_log``. A report that asks for credit and gives no line fails it;
5. book the rendered seconds on the ``media`` lane at $0 (US-103), the units
   the quota counts.

Steps 0-3 together get at most ``SOCIALS_RENDER_MAX_WAIT_SECONDS``, which stays
under the boot reaper's stale cutoff, so the reaper only ever fails a render no
live task owns. A post that moved on while it rendered (the reaper failed it)
is left as it is, and nothing is booked. The renderer assembles; it never
generates (D3).
The bundle (``core/media_render_bundle.py``, S1.2) is the social template's
composition, checked against its contract (``core/social_templates.py``), the
post's variable values with the template's defaults, the audio plan, and the
workspace brand kit: tokens as ``--brand-*`` CSS variables, an uploaded logo
and font files inlined. The api layer hands the brand kit in: this module may
not import the documents module that owns it.

A ``social_image`` template renders as stills (US-107): one PNG for a card, one
per slide for a carousel, each stored and registered like a video, numbered
when there are several (``carousel-4x5-01.png``). A still spends no render
minutes: it has no duration.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from uuid import UUID

from config import config
from core.media_render_bundle import build_bundle, render_size, voice_script, with_slot_files, with_voice_files
from core.media_render_client import (
    BAD_RESPONSE,
    JOB_DONE,
    NOT_CONFIGURED,
    NOT_FOUND,
    TIMEOUT,
    MediaRenderClient,
    MediaRenderError,
    MediaRenderUnavailable,
)
from core.media_render_quota import book_render_seconds
from core.music_credit import MusicCredit, MusicCreditMissing, credit_for_render
from core.social_templates import SocialTemplateError, is_social_format, resolve_variables, validate_social_blocks
from modules.socials import service
from modules.socials.media_store import MediaNameError, MediaStore, content_type_for, media_key, media_route
from modules.socials.recipes import footage as footage_recipes
from modules.socials.recipes import voice as voice_recipes

logger = logging.getLogger(__name__)

EXECUTION_PREFIX = "social_post:"
DELIVERABLE_SOURCE_TYPE = "social_post"
# The report kept in review_log: the first findings, each trimmed.
MAX_REPORTED_FINDINGS = 20
# The keys of a check finding (services/media-render/media_render/check_report.py) and a voice finding.
FINDING_KEYS = ("section", "severity", "code", "message", "selector", "containerSelector", "time", "fixHint", "source", "line")
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
    """A social template's ``blocks``, or ``None`` when it is no social template with a composition."""
    if template is None or not is_social_format(getattr(template, "format", None)):
        return None
    blocks = getattr(template, "blocks", None)
    if not isinstance(blocks, dict):
        return None
    html = blocks.get("html")
    return blocks if isinstance(html, str) and html.strip() else None


def bundle_for(
    post: Any,
    template: Any,
    brand_kit: Optional[Mapping[str, Any]] = None,
    *,
    fallback_name: str = "",
    footage_slots: Sequence[str] = (),
) -> Dict[str, Any]:
    """The render bundle media-render takes (``services/media-render/media_render/bundle.py``).

    ``brand_kit`` is the workspace's, render-ready; ``fallback_name`` (the
    workspace's name) is the brand name when the kit has none. ``footage_slots``
    are the slots this render fills with footage (S1.8): they stay in the
    composition, and their files join the bundle once they are in our storage;
    every other slot plays the template's own motion graphics.
    :class:`NotRenderable` when the post has no social template, the template
    breaks its contract, or the post leaves a variable without a default empty.
    """
    blocks = composition_of(template)
    if blocks is None:
        raise NotRenderable("this post has no social template to render: choose a template with a composition")
    try:
        blocks = validate_social_blocks(blocks, template.format)
    except SocialTemplateError as exc:
        raise NotRenderable(f"this post's template cannot be rendered: {exc}") from exc
    supplied = {
        name: spec.get("value")
        for name, spec in (getattr(post, "variables", None) or {}).items()
        if isinstance(spec, dict)
    }
    resolved = resolve_variables(blocks["variables_schema"], supplied)
    if resolved.missing:
        raise NotRenderable(f"fill in {', '.join(resolved.missing)} before rendering")
    if resolved.invalid:
        raise NotRenderable("; ".join(resolved.invalid))
    return build_bundle(
        workspace_id=post.workspace_id,
        reference=f"{EXECUTION_PREFIX}{post.id}",
        blocks=blocks,
        values=resolved.values,
        brand_kit=brand_kit,
        fallback_name=fallback_name,
        keep_slots=footage_slots,
        fmt=template.format,
    )


def footage_plan_for(post: Any, template: Any, caps: Any) -> Optional[footage_recipes.FootagePlan]:
    """The footage the post asks for (US-114), planned over its template's slots
    with the workspace's media capabilities ``caps``; ``None`` when it asks for none."""
    if not getattr(post, "footage", None):
        return None
    blocks = composition_of(template)
    if blocks is None:
        raise NotRenderable("this post has no social template to render: choose a template with a composition")
    try:
        blocks = validate_social_blocks(blocks, template.format)
    except SocialTemplateError as exc:
        raise NotRenderable(f"this post's template cannot be rendered: {exc}") from exc
    width, height = render_size(blocks)
    return footage_recipes.plan_for(post.footage, blocks.get("slots"), caps, width=width, height=height)


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
    # D11: the voice toolkit the post chose, resolved as the render started; None is Kokoro.
    voice: Optional[voice_recipes.VoicePlan] = None
    # D12: the footage the post asks for, resolved as the render started; None asks for none.
    footage: Optional[footage_recipes.FootagePlan] = None


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
    for key in ("check", "timings", "music"):
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


def _poll_failure(exc: MediaRenderError) -> RenderFailure:
    if exc.code == TIMEOUT:
        minutes = config.SOCIALS_RENDER_MAX_WAIT_SECONDS // 60
        return RenderFailure("timed_out", f"The render did not finish within {minutes} minutes.")
    if exc.code == NOT_FOUND:
        return RenderFailure(NOT_FOUND, "The renderer lost the render (it restarted). Render again.")
    if exc.code == BAD_RESPONSE:
        return RenderFailure("bad_response", "The renderer accepted the render but gave no job id.")
    return _failure_from(exc)


# ── the background render ───────────────────────────────────────────────────
async def _footaged(job: RenderJob, store: MediaStore, session_factory: Callable[[], Any]) -> Mapping[str, Any]:
    """The bundle with the footage its post asks for (US-114): each shot generated
    now, or reused from an earlier render, a file in our storage that media-render
    reaches through a presigned link. A render asking for none: the bundle as it is."""
    plan = job.footage
    if plan is None or not plan.shown:
        return job.bundle
    try:
        made = await footage_recipes.generate(
            plan, workspace_id=job.workspace_id, post_id=job.post_id, title=job.title,
            session_factory=session_factory, store=store,
        )
    except footage_recipes.FootageRefused as exc:
        message = f"{str(exc).rstrip('.')}. Nothing was rendered."
        raise RenderFailure("footage_refused", message, {"footage": plan.report()}) from exc
    except footage_recipes.FootageError as exc:
        message = f"The footage could not be made: {str(exc).rstrip('.')}. Nothing was rendered."
        raise RenderFailure("footage_failed", message, {"footage": plan.report()}) from exc
    keys = {kept.path: media_key(job.workspace_id, job.post_id, kept.name) for kept in plan.kept}
    keys.update({clip.path: clip.key for clip in made.values()})
    ttl = config.SOCIALS_RENDER_MEDIA_URL_TTL_SECONDS
    try:
        links = {path: await asyncio.to_thread(store.presigned_get, key, ttl) for path, key in keys.items()}
    except Exception as exc:  # noqa: BLE001 — storage cannot link the footage: fail the render, loudly
        logger.exception("[Socials] linking the footage of post %s failed", job.post_id)
        raise RenderFailure("storage_failed", "The footage could not be handed to the renderer.") from exc
    return with_slot_files(job.bundle, links)


async def _voiced(
    job: RenderJob, bundle: Mapping[str, Any], store: MediaStore, session_factory: Callable[[], Any]
) -> Mapping[str, Any]:
    """The bundle, its script spoken by the post's voice toolkit when it chose one
    (US-111): each line now a file in our storage, reached by media-render through
    a presigned link. Kokoro speaks inside media-render: the bundle as it is."""
    lines = voice_script(bundle)
    if job.voice is None or not lines:
        return bundle
    try:
        spoken = await voice_recipes.speak(
            job.voice, workspace_id=job.workspace_id, post_id=job.post_id, lines=lines,
            session_factory=session_factory, store=store,
        )
    except voice_recipes.VoiceError as exc:
        raise RenderFailure("voice_failed", f"{str(exc).rstrip('.')}. Nothing was rendered.") from exc
    ttl = config.SOCIALS_RENDER_MEDIA_URL_TTL_SECONDS
    try:
        links = {
            line_id: (line.extension, await asyncio.to_thread(store.presigned_get, line.key, ttl))
            for line_id, line in spoken.items()
        }
    except Exception as exc:  # noqa: BLE001 — storage cannot link the lines: fail the render, loudly
        logger.exception("[Socials] linking the voice lines of post %s failed", job.post_id)
        raise RenderFailure("storage_failed", "The spoken lines could not be handed to the renderer.") from exc
    return with_voice_files(bundle, links)


async def _submit(client: MediaRenderClient, bundle: Mapping[str, Any], deadline: float) -> Dict[str, Any]:
    """Submit the bundle; a busy renderer is asked again while the deadline allows."""
    try:
        return await client.submit_when_free(bundle, deadline=deadline, poll_seconds=config.SOCIALS_RENDER_POLL_SECONDS)
    except MediaRenderError as exc:
        raise _failure_from(exc) from exc


async def _wait(client: MediaRenderClient, job: RenderJob, record: Dict[str, Any], deadline: float) -> Dict[str, Any]:
    """Poll until the job ends; a lost poll is retried until the deadline."""
    try:
        record = await client.wait_for(record, deadline=deadline, poll_seconds=config.SOCIALS_RENDER_POLL_SECONDS)
    except MediaRenderError as exc:
        logger.warning("[Socials] waiting on the render of post %s failed: %s", job.post_id, exc)
        raise _poll_failure(exc) from exc
    if record.get("status") != JOB_DONE:
        error = record.get("error") if isinstance(record.get("error"), dict) else {}
        code = str(error.get("code") or record.get("status") or "failed")
        message = str(error.get("message") or "the render failed").rstrip(".")
        report = record.get("report") if isinstance(record.get("report"), dict) else {}
        raise RenderFailure(code, f"The render failed: {message}.", _report(findings=report.get("findings"), report=report))
    return record


def _aspect_slug(aspect: str) -> str:
    return "".join(ch if ch.isalnum() else "x" for ch in aspect.lower()) or DEFAULT_ASPECT


def stored_file_name(job: RenderJob, output: Mapping[str, Any], *, several: bool = False) -> str:
    """``<format>-<aspect>.<ext>``, e.g. ``video-9x16.mp4``: stable, so a re-render replaces its file.

    When a render returns several files (a carousel's slides), each carries its
    ``index`` and the name its number: ``carousel-4x5-01.png``.
    """
    name = str(output.get("name") or "")
    ext = Path(name).suffix.lower() if Path(name).suffix else ".mp4"
    aspect = str(output.get("aspect") or DEFAULT_ASPECT)
    index = output.get("index")
    number = f"-{index:02d}" if several and isinstance(index, int) and not isinstance(index, bool) else ""
    return f"{job.format or 'render'}-{_aspect_slug(aspect)}{number}{ext}"


def _music_of(job: RenderJob, finished: Mapping[str, Any]) -> Optional[MusicCredit]:
    """The music the render mixed (S1.6). A CC BY track the report gives no credit
    line for, or a report that does not name the track the bundle asked for, fails the render."""
    try:
        return credit_for_render(job.bundle, finished.get("report"))
    except MusicCreditMissing as exc:
        logger.error("[Socials] the render of post %s: %s", job.post_id, exc)
        raise RenderFailure("music_credit_missing", f"The render's music needs its credit line: {exc}. Nothing was stored.") from exc


def _register(
    session_factory: Callable[[], Any],
    job: RenderJob,
    key: str,
    file_name: str,
    entry: Dict[str, Any],
    music: Optional[MusicCredit] = None,
) -> str:
    from services.deliverable_service import DeliverableService, _infer_artifact_type

    extra: Dict[str, Any] = {"social_post_id": str(job.post_id), "sha256": entry["sha256"], "aspect": entry["aspect"]}
    if music is not None:
        extra["music"] = music.extra()
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
            extra=extra,
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
    music: Optional[MusicCredit] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch every output, then store and register each; the post's ``media``.

    Every file is fetched (and checked) before any is stored, so a missing or
    empty output never leaves another behind in storage or in Deliverables.
    Each Deliverable records ``music``, the track the render mixed (S1.6).
    """
    outputs = [o for o in (record.get("outputs") or []) if isinstance(o, dict) and o.get("name")]
    if not outputs:
        raise RenderFailure("no_output", "The renderer finished but returned no file.")
    with tempfile.TemporaryDirectory(prefix="socials-render-") as scratch:
        fetched = []
        for output in outputs:
            file_name = stored_file_name(job, output, several=len(outputs) > 1)
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
            deliverable_id = await asyncio.to_thread(_register, session_factory, job, key, file_name, entry, music)
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


def _size(record: Mapping[str, Any], aspect: str) -> str:
    return f"{record['width']}×{record['height']}" if record.get("width") and record.get("height") else aspect


def _summary(media: Mapping[str, List[Mapping[str, Any]]]) -> str:
    parts = []
    for aspect, records in media.items():
        if len(records) > 1 and not any(r.get("duration") for r in records):
            parts.append(f"{len(records)} images at {_size(records[0], aspect)}")  # a carousel's slides
            continue
        for r in records:
            length = f"{r['duration']:g} s " if r.get("duration") else ""
            parts.append(f"{length}{_size(r, aspect)}".strip())
    return "Rendered " + ", ".join(parts) + "."


def _finish(
    session_factory: Callable[[], Any],
    job: RenderJob,
    *,
    media: Optional[Mapping[str, List[Dict[str, Any]]]] = None,
    report: Optional[Mapping[str, Any]] = None,
    failure: Optional[RenderFailure] = None,
    credits: Sequence[str] = (),
) -> Optional[str]:
    """End the render on the post, a compare-and-set on ``rendering`` and the
    start hash: the status it ended in, or ``None`` when the post moved on
    (nothing is written then). Files the post cannot record fail it instead.
    ``credits`` join the post's copy (the render's CC BY music, S1.6)."""
    db = session_factory()
    try:
        post = service.get_post(db, job.workspace_id, job.post_id)
        if post is None or post.status != service.RENDERING or post.content_hash != job.content_hash:
            db.rollback()
            logger.warning("[Socials] post %s moved on while it rendered; the render is dropped", job.post_id)
            return None
        if failure is None:
            try:
                service.finish_render(
                    post, job.actor, media or {}, summary=_summary(media or {}), report=report, credits=credits
                )
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
    book_render_seconds(
        workspace_id=job.workspace_id,
        execution_id=f"{EXECUTION_PREFIX}{job.post_id}",
        seconds=seconds,
        latency_ms=latency_ms,
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
        bundle = await _footaged(job, store, factory)
        bundle = await _voiced(job, bundle, store, factory)
        accepted = await _submit(client, bundle, deadline)
        finished = await _wait(client, job, accepted, deadline)
        music = _music_of(job, finished)
        return finished, music, await _store_outputs(client, store, factory, job, finished, music)

    try:
        try:
            finished, music, media = await asyncio.wait_for(render_and_store(), timeout=budget)
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
    if job.footage is not None and job.footage.report():
        report["footage"] = job.footage.report()
    credits = [music.line] if music is not None and music.line else []
    ended = await asyncio.to_thread(_finish, factory, job, media=media, report=report, credits=credits)
    if ended != service.NEEDS_APPROVAL:
        return False
    # latency: the whole render, from submit to the files in storage.
    _book(job, rendered_seconds(media), int((time.monotonic() - started) * 1000))
    return True

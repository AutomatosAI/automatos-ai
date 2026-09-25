"""The media-render HTTP service (aiohttp, as the workspace worker is).

    GET  /health                     open: the image's versions and the queue
    POST /render                     a composition bundle -> 202 job | 422 findings | 400 | 502 | 503
    GET  /render/{id}                the job: status, check report, timings, outputs
    GET  /render/{id}/output/{name}  the rendered file (Range requests work)
    POST /tts                        Kokoro lines -> durations and voiced segments (WAVs on request)

Every path but /health needs X-Internal-Token when one is configured, copied
from the worker (services/workspace-worker/main.py). Boot refuses production
without one (boot.token_problems).

POST /render answers once the job is staged, spoken, mixed and checked. A
composition `hyperframes check` refuses gets 422 with the findings and never
renders. A bundle with a ``preview`` gets snapshot frames and a short reel
instead of the full render (US-106), through the same check and the same
slots. A checked job waits for a render slot: two at once overall, one per
workspace, first come first served (lanes.py). Staging and the check have
their own, smaller lane, so the renders' limit is never exceeded by a check.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Mapping, Optional, Set

from aiohttp import web

from .bundle import parse_bundle
from .config import Settings
from .hyperframes import LOG_TAIL_CHARS
from .jobs import DONE, FAILED, QUEUED, REJECTED, RENDERING, Job, JobStore, delete_files, job_json, remove_working_files, reset_work_dir
from .lanes import Lane
from .music import Track, load_library
from .pipeline import CheckOutcome, Pipeline, PipelineError, RenderPipeline
from .tts import Speaker, line_report, parse_tts_request
from .validate import BundleError
from .versions import read_versions

logger = logging.getLogger(__name__)

TOKEN_HEADER = "X-Internal-Token"
PUBLIC_PATHS = frozenset({"/health"})


@dataclass
class ServiceState:
    settings: Settings
    store: JobStore
    pipeline: Pipeline
    speaker: Speaker
    library: Mapping[str, Track]
    check_lane: Lane
    render_lane: Lane
    versions: Mapping[str, str]
    tasks: Set["asyncio.Task[None]"] = field(default_factory=set)


STATE = web.AppKey("media_render_state", ServiceState)


def output_path(job_id: str, name: str) -> str:
    return f"/render/{job_id}/output/{name}"


def _token_matches(supplied: str, expected: str) -> bool:
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def _error(status: int, code: str, message: str, *, headers: Optional[Dict[str, str]] = None, **extra: Any) -> web.Response:
    return web.json_response({"error": code, "message": message, **extra}, status=status, headers=headers)


def _job_body(state: ServiceState, job: Job) -> Dict[str, Any]:
    body = job_json(job, queue_position=state.render_lane.position(job.id) if job.status == QUEUED else None)
    for output in body["outputs"]:
        output["path"] = output_path(job.id, output["name"])
    return body


def _fail(state: ServiceState, job_id: str, code: str, message: str, detail: Optional[str] = None) -> Job:
    error = {"code": code, "message": message}
    if detail:
        error["detail"] = detail[-LOG_TAIL_CHARS:]
    logger.warning("render %s failed: %s: %s", job_id, code, message)
    return state.store.finish(job_id, FAILED, error=error)


async def _prepare_and_check(state: ServiceState, job: Job) -> CheckOutcome:
    """Stage, speak, mix and check the job inside the check lane."""
    granted = state.check_lane.submit(job.id, job.workspace_id)
    try:
        await granted
    except asyncio.CancelledError:
        if granted.cancelled():
            state.check_lane.withdraw(job.id)
        else:
            state.check_lane.release(job.workspace_id)
        raise
    try:
        return await state.pipeline.prepare_and_check(job)
    finally:
        state.check_lane.release(job.workspace_id)


async def _render(state: ServiceState, job_id: str, workspace_id: str, granted: "asyncio.Future[None]", checked_at: float) -> None:
    try:
        await granted
    except asyncio.CancelledError:
        if granted.cancelled():
            state.render_lane.withdraw(job_id)
        else:
            state.render_lane.release(workspace_id)
        raise
    try:
        job = state.store.get(job_id)
        if job is None:
            return
        if job.status != RENDERING:
            job = state.store.update(job_id, status=RENDERING, started_at=state.store.now())
        result = await state.pipeline.render(job)
        timings = {
            **job.report.get("timings", {}),
            **result.timings,
            "queue_seconds": round(max(0.0, (job.started_at or checked_at) - checked_at), 3),
        }
        state.store.finish(job_id, DONE, outputs=result.outputs, report={**job.report, "timings": timings})
        timings_of = result.timings
        seconds = timings_of.get("render_seconds", timings_of.get("preview_seconds", timings_of.get("still_seconds", 0)))
        logger.info("render %s done for workspace %s in %.1f s", job_id, workspace_id, seconds)
    except PipelineError as exc:
        _fail(state, job_id, exc.code, str(exc), exc.detail)
    except Exception:
        logger.exception("render %s failed unexpectedly", job_id)
        _fail(state, job_id, "internal_error", "the render failed unexpectedly")
    finally:
        state.render_lane.release(workspace_id)
        current = state.store.get(job_id)
        if current is not None:
            remove_working_files(current, keep_outputs=current.status == DONE)


def _queue_render(state: ServiceState, job: Job, outcome: CheckOutcome) -> Job:
    granted = state.render_lane.submit(job.id, job.workspace_id)
    now = state.store.now()
    if granted.done():
        job = state.store.update(job.id, status=RENDERING, started_at=now, report=outcome.report)
    else:
        job = state.store.update(job.id, status=QUEUED, report=outcome.report)
    task = asyncio.create_task(_render(state, job.id, job.workspace_id, granted, now))
    state.tasks.add(task)
    task.add_done_callback(state.tasks.discard)
    return job


async def _read_json(request: web.Request) -> Any:
    try:
        return await request.json()
    except ValueError:
        raise web.HTTPBadRequest(
            text='{"error": "invalid_json", "message": "the body is not JSON"}', content_type="application/json"
        ) from None


async def health(request: web.Request) -> web.Response:
    state = request.app[STATE]
    queue = {"running": state.render_lane.running, "queued": len(state.render_lane.waiting())}
    return web.json_response({"status": "healthy", "service": "media-render", "versions": state.versions, "renders": queue})


async def post_render(request: web.Request) -> web.Response:
    state = request.app[STATE]
    payload = await _read_json(request)
    active = state.store.active_count()
    if active >= state.settings.max_active_jobs:
        retry = {"Retry-After": str(state.settings.busy_retry_after_seconds)}
        return _error(503, "busy", f"{active} renders are already in progress; try again shortly", headers=retry)
    try:
        bundle = parse_bundle(payload, state.settings, state.library)
    except BundleError as exc:
        return _error(400, "invalid_bundle", str(exc))
    job = state.store.create(bundle)
    logger.info("render %s accepted for workspace %s (%s)", job.id, bundle.workspace_id, bundle.reference or "-")
    try:
        outcome = await _prepare_and_check(state, job)
    except PipelineError as exc:
        remove_working_files(_fail(state, job.id, exc.code, str(exc), exc.detail), keep_outputs=False)
        return _error(exc.status, exc.code, str(exc), id=job.id)
    except Exception:
        logger.exception("render %s: preparing failed unexpectedly", job.id)
        remove_working_files(_fail(state, job.id, "internal_error", "preparing the render failed"), keep_outputs=False)
        return _error(500, "internal_error", "preparing the render failed", id=job.id)
    if not outcome.ok:
        job = state.store.finish(job.id, REJECTED, report=outcome.report)
        remove_working_files(job, keep_outputs=False)
        errors = sum(1 for finding in outcome.findings if finding.get("severity") == "error")
        logger.info("render %s refused: %d error(s) in its check", job.id, errors)
        message = f"the composition failed its check with {errors} error(s); nothing was rendered"
        return _error(422, "check_failed", message, id=job.id, findings=list(outcome.findings), report=dict(outcome.report))
    job = _queue_render(state, job, outcome)
    return web.json_response(_job_body(state, job), status=202, headers={"Location": f"/render/{job.id}"})


async def get_render(request: web.Request) -> web.Response:
    state = request.app[STATE]
    job = state.store.get(request.match_info["job_id"])
    if job is None:
        return _error(404, "not_found", "no such render; finished renders expire")
    return web.json_response(_job_body(state, job))


async def get_output(request: web.Request) -> web.StreamResponse:
    state = request.app[STATE]
    job = state.store.get(request.match_info["job_id"])
    name = request.match_info["name"]
    if job is None or name not in {output["name"] for output in job.outputs}:
        return _error(404, "not_found", "no such output")
    path = job.output_dir / name
    if job.status != DONE or not path.is_file():
        return _error(410, "gone", "the output is no longer held")
    return web.FileResponse(path)


async def post_tts(request: web.Request) -> web.Response:
    state = request.app[STATE]
    payload = await _read_json(request)
    try:
        spec = parse_tts_request(payload, state.settings)
    except BundleError as exc:
        return _error(400, "invalid_request", str(exc))
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="media-render-tts-") as scratch:
        try:
            spoken = await state.speaker.speak(spec.lines, Path(scratch), voice=spec.voice, speed=spec.speed, lang=spec.lang)
        except ValueError as exc:  # Kokoro's refusal: an unknown voice, a line with nothing to say
            return _error(400, "voice_refused", f"Kokoro could not speak the lines: {exc}")
        lines = [
            line_report(line_id, said, include_audio=spec.include_audio) for (line_id, _), said in zip(spec.lines, spoken)
        ]
    body = {
        "voice": spec.voice,
        "speed": spec.speed,
        "lang": spec.lang,
        "lines": lines,
        "timings": {"tts_seconds": round(time.monotonic() - started, 3)},
    }
    return web.json_response(body)


async def _sweep(state: ServiceState) -> None:
    while True:
        await asyncio.sleep(state.settings.sweep_interval_seconds)
        expired = state.store.expire(state.settings.job_ttl_seconds)
        if expired:
            await asyncio.to_thread(delete_files, expired)
            logger.info("expired %d finished render(s)", len(expired))


def _lifecycle(real_pipeline: Optional[RenderPipeline]) -> Callable[[web.Application], AsyncIterator[None]]:
    async def lifecycle(app: web.Application) -> AsyncIterator[None]:
        state = app[STATE]
        await asyncio.to_thread(reset_work_dir, Path(state.settings.work_dir))
        if real_pipeline is not None:
            await real_pipeline.start()
        sweeper = asyncio.create_task(_sweep(state))
        yield
        sweeper.cancel()
        for task in list(state.tasks):
            task.cancel()
        await asyncio.gather(sweeper, *state.tasks, return_exceptions=True)
        if real_pipeline is not None:
            await real_pipeline.close()

    return lifecycle


def create_app(
    settings: Settings,
    *,
    pipeline: Optional[Pipeline] = None,
    speaker: Optional[Speaker] = None,
    library: Optional[Mapping[str, Track]] = None,
    clock: Callable[[], float] = time.time,
) -> web.Application:
    """The app. Tests pass a stub pipeline or speaker; the service uses the real ones."""

    @web.middleware
    async def internal_token(request: web.Request, handler):
        if request.path in PUBLIC_PATHS:
            return await handler(request)
        if settings.internal_token and not _token_matches(
            request.headers.get(TOKEN_HEADER, ""), settings.internal_token
        ):
            return web.json_response({"error": "Unauthorized"}, status=401)
        return await handler(request)

    speaker = speaker or Speaker(settings)
    real_pipeline = RenderPipeline(settings, speaker) if pipeline is None else None
    state = ServiceState(
        settings=settings,
        store=JobStore(Path(settings.work_dir), clock),
        pipeline=pipeline or real_pipeline,
        speaker=speaker,
        library=load_library(settings.music_dir) if library is None else library,
        check_lane=Lane(settings.max_concurrent_checks, settings.max_checks_per_workspace),
        render_lane=Lane(settings.max_concurrent_renders, settings.max_renders_per_workspace),
        versions=read_versions(settings.versions_path),
    )
    app = web.Application(middlewares=[internal_token], client_max_size=settings.max_bundle_bytes)
    app[STATE] = state
    app.router.add_get("/health", health)
    app.router.add_post("/render", post_render)
    app.router.add_get("/render/{job_id}", get_render)
    app.router.add_get("/render/{job_id}/output/{name}", get_output)
    app.router.add_post("/tts", post_tts)
    app.cleanup_ctx.append(_lifecycle(real_pipeline))
    return app


def serve(settings: Settings) -> None:
    logger.info("media-render listening on %s:%d", settings.bind_host, settings.port)
    web.run_app(
        create_app(settings),
        host=settings.bind_host,
        port=settings.port,
        access_log=None,
        print=None,
    )

"""The voice-tts HTTP service (aiohttp, as media-render and the workspace worker are).

    GET  /healthz   open: 200 {"ready": bool}; ready once the model has loaded
    GET  /readyz    open: 200 once the model has loaded, 503 until then (a deploy's health check)
    GET  /voices    open: {"default", "voices": [{id, name, accent, gender}]}, English first
    POST /speak     Authorization: Bearer <VOICE_TTS_TOKEN>; {"text", "voice", "speed"} -> audio/mpeg

``POST /speak`` refuses before it speaks, with ``{"code", "detail"}``: 401 without the
token (compared in constant time), 422 ``tts_bad_request`` for a body it cannot speak
(an unknown voice among them), 413 ``tts_too_long``, 503 ``tts_warming`` while the
model loads, and 429 ``tts_busy`` when ``VOICE_TTS_MAX_CONCURRENT`` syntheses are
already running. It never waits for a slot: the caller falls back to the device's
own voice at once. The text is never logged.

Once the service is listening the model loads in a background thread (the PRD-262
US-053 pattern from media-render), so /healthz answers at once.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Set

from aiohttp import web

from .config import TOKEN_ENV, Settings
from .engine import Engine
from .request import RequestError, parse_speech
from .speaking import Speaker, SpeakError
from .voices import Catalogue, load_catalogue

logger = logging.getLogger(__name__)

AUTHORIZATION = "Authorization"
BEARER = "Bearer "
PUBLIC_PATHS = frozenset({"/healthz", "/readyz", "/voices"})
# A body of 500 characters is far below this; anything bigger is not a /speak request.
MAX_BODY_BYTES = 16 * 1024


@dataclass
class ServiceState:
    settings: Settings
    catalogue: Catalogue
    engine: Any
    speaker: Speaker
    active: int = 0
    tasks: Set["asyncio.Task[None]"] = field(default_factory=set)


STATE = web.AppKey("voice_tts_state", ServiceState)


def refusal(status: int, code: str, detail: str) -> web.Response:
    return web.json_response({"code": code, "detail": detail}, status=status)


def token_matches(header: str, expected: str) -> bool:
    """``Authorization: Bearer <token>`` against the configured token, in constant time."""
    supplied = header[len(BEARER):] if header.startswith(BEARER) else ""
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


async def healthz(request: web.Request) -> web.Response:
    return web.json_response({"ready": request.app[STATE].engine.loaded})


async def readyz(request: web.Request) -> web.Response:
    ready = request.app[STATE].engine.loaded
    return web.json_response({"ready": ready}, status=200 if ready else 503)


async def voices(request: web.Request) -> web.Response:
    return web.json_response(request.app[STATE].catalogue.body())


async def _read_json(request: web.Request) -> Any:
    try:
        return await request.json()
    except ValueError:
        raise RequestError(422, "tts_bad_request", "the body is not JSON") from None


def _admission(state: ServiceState) -> Optional[web.Response]:
    """503 while the model loads, 429 when every synthesis slot is taken; None when it may speak."""
    if not state.engine.loaded:
        return refusal(503, "tts_warming", "the voice model is still loading; try again in a few seconds")
    if state.active >= state.settings.max_concurrent:
        logger.info("speak refused: %d syntheses already running", state.active)
        return refusal(429, "tts_busy", "the voice is busy; try again shortly")
    return None


async def speak(request: web.Request) -> web.StreamResponse:
    state = request.app[STATE]
    try:
        spec = parse_speech(await _read_json(request), state.settings, state.catalogue)
    except RequestError as exc:
        return refusal(exc.status, exc.code, exc.detail)
    refused = _admission(state)
    if refused is not None:
        return refused
    # No await between the check above and this count: admission is one step.
    state.active += 1
    try:
        return await state.speaker.stream(request, spec)
    except SpeakError as exc:
        return refusal(exc.status, exc.code, exc.detail)
    finally:
        state.active -= 1


def _bearer_gate(settings: Settings) -> Callable[..., Any]:
    @web.middleware
    async def bearer(request: web.Request, handler):
        if request.path in PUBLIC_PATHS or not settings.token:
            return await handler(request)
        if not token_matches(request.headers.get(AUTHORIZATION, ""), settings.token):
            return refusal(401, "unauthorized", f"send Authorization: Bearer <{TOKEN_ENV}>")
        return await handler(request)

    return bearer


def create_app(
    settings: Settings, *, engine: Any = None, catalogue: Optional[Catalogue] = None, encoder: Any = None
) -> web.Application:
    """The app. Tests pass a stand-in engine, catalogue or encoder; the service uses the real ones."""
    engine = engine or Engine(settings)
    catalogue = catalogue or load_catalogue(settings.kokoro_voices_path, settings.default_voice)
    speaker = Speaker(settings, engine) if encoder is None else Speaker(settings, engine, encoder=encoder)
    app = web.Application(middlewares=[_bearer_gate(settings)], client_max_size=MAX_BODY_BYTES)
    app[STATE] = ServiceState(settings=settings, catalogue=catalogue, engine=engine, speaker=speaker)
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/readyz", readyz)
    app.router.add_get("/voices", voices)
    app.router.add_post("/speak", speak)
    app.on_cleanup.append(_cancel_tasks)
    return app


async def _cancel_tasks(app: web.Application) -> None:
    tasks = list(app[STATE].tasks)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def _warm(state: ServiceState) -> None:
    logger.info("kokoro: loading the voice model in the background")
    try:
        seconds = await asyncio.to_thread(state.engine.load)
    except Exception:
        logger.exception("kokoro: the voice model did not load; /speak answers tts_warming until a restart")
        return
    logger.info("kokoro: the voice model loaded in the background in %.2f s", seconds)


def warm_in_background(app: web.Application) -> "asyncio.Task[None]":
    """Start loading the model off the event loop; cleanup cancels the wait with the app's other tasks."""
    state = app[STATE]
    task = asyncio.create_task(_warm(state))
    state.tasks.add(task)
    task.add_done_callback(state.tasks.discard)
    return task


def serve(settings: Settings) -> None:
    app = create_app(settings)

    def listening(*_message: object) -> None:
        # web.run_app calls its ``print`` once every site is listening: the model starts loading then.
        logger.info("voice-tts listening on %s:%d", settings.bind_host, settings.port)
        warm_in_background(app)

    started = time.monotonic()
    web.run_app(app, host=settings.bind_host, port=settings.port, access_log=None, print=listening)
    logger.info("voice-tts stopped after %.0f s", time.monotonic() - started)

"""Auto's server voice: ``POST /api/voice/speech`` and ``GET /api/voice/voices``.

The phone (Automatos Go) and the web quick-ask speak Auto's replies through these, a
sentence at a time, in a natural voice from voice-tts (``services/voice-tts``) instead of
the device's own. Hybrid auth as every mobile route (a Clerk bearer, or a paired device's
token, with ``X-Workspace-ID``), and the permission ``POST /api/chat`` asks for
(``agents:execute``): a member who may chat may hear the answer; a viewer may not.

* ``POST /api/voice/speech`` ``{"text", "voice", "speed"}`` -> 200 ``audio/mpeg``, streamed
  through from voice-tts as it speaks, with ``Cache-Control: no-store`` and ``X-Voice-Id``.
  Every refusal is ``{"code", "detail"}`` (``modules/voice/speech.py`` lists them). Each
  workspace may have ``VOICE_TTS_RATE_PER_MINUTE`` sentences spoken a minute, counted in the
  shared Redis window (``RateLimitStore``); past it, 429 ``tts_busy``.
* ``GET /api/voice/voices`` -> voice-tts's voice list (cached), or 503 ``tts_off`` /
  ``tts_unavailable``.

The stream needs no database. FastAPI ends ``get_db`` only after the whole response has gone
out, so the permission check's transaction is ended (rolled back) before the stream starts,
which hands its pooled connection back: a listener would otherwise hold one for the whole
answer. ``get_db`` still closes the session, once, when the request ends. The body is capped
at ``MAX_BODY_BYTES`` on the wire (``main.PATH_BODY_LIMITS``), before it is read.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from api.widgets.rate_limit import RateLimitStore
from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from modules.voice import speech, speech_client
from modules.voice.speech import TTS_BAD_REQUEST, TTS_BUSY, TTS_TOO_LONG, SpeechRefused

logger = logging.getLogger(__name__)
router = APIRouter(tags=["voice"])

SPEECH_ROUTE = "/api/voice/speech"
SPEECH_PERMISSION = "agents:execute"
MP3 = "audio/mpeg"
NO_STORE = {"Cache-Control": "no-store"}
# A body of 500 characters is far below this; a bigger one is no speech request.
MAX_BODY_BYTES = 16 * 1024
RATE_WINDOW_SECONDS = 60
RATE_BUCKET_PREFIX = "voice_tts"
RATE_LIMITER_LABEL = "voice speech rate limiter"
# Module-level so a test can hand it a fake Redis (RateLimitStore's redis_factory seam).
_rate_store = RateLimitStore()


def _refused(refusal: SpeechRefused) -> JSONResponse:
    headers = dict(NO_STORE)
    if refusal.retry_after is not None:
        headers["Retry-After"] = str(refusal.retry_after)
    return JSONResponse(refusal.body(), status_code=refusal.status, headers=headers)


def _declared_length(request: Request) -> int:
    try:
        return int(request.headers.get("content-length") or 0)
    except ValueError:
        return 0


async def _body(request: Request) -> Any:
    if _declared_length(request) > MAX_BODY_BYTES or len(await request.body()) > MAX_BODY_BYTES:
        raise SpeechRefused(413, TTS_TOO_LONG, f"the body is over {MAX_BODY_BYTES} bytes")
    try:
        return await request.json()
    except ValueError:
        raise SpeechRefused(422, TTS_BAD_REQUEST, "the body is not JSON") from None


async def _within_rate(workspace_id: str) -> None:
    """429 ``tts_busy`` once the workspace has had its sentences this minute (0 = no limit)."""
    limit = config.VOICE_TTS_RATE_PER_MINUTE
    if limit <= 0:
        return
    allowed, _limit, _remaining, reset_seconds = await asyncio.to_thread(
        _rate_store.check, f"{RATE_BUCKET_PREFIX}:{workspace_id}", limit, RATE_WINDOW_SECONDS,
        label=RATE_LIMITER_LABEL,
    )
    if not allowed:
        logger.info("voice_tts_rate_limited workspace=%s limit=%d", workspace_id, limit)
        raise SpeechRefused(429, TTS_BUSY, f"This workspace has had {limit} sentences spoken this minute.",
                            retry_after=reset_seconds)


@router.post(SPEECH_ROUTE, dependencies=[Depends(require_workspace_permission(SPEECH_PERMISSION))])
async def speak(
    request: Request,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Response:
    """Auto's words as MP3, streamed from voice-tts; a ``{"code", "detail"}`` refusal otherwise."""
    # The permission check above was this request's last use of the database: end its transaction,
    # which gives the pooled connection back now; get_db closes the session after the stream.
    await asyncio.to_thread(db.rollback)
    workspace_id = str(ctx.workspace_id)
    try:
        speech.require_configured()
        spec = speech.parse_speech(await _body(request))
        await _within_rate(workspace_id)
        stream = await speech_client.open_speech(spec, workspace_id)
    except SpeechRefused as refusal:
        return _refused(refusal)
    headers = {**NO_STORE, speech_client.VOICE_HEADER: stream.voice}
    return StreamingResponse(speech_client.relay(stream, spec, workspace_id), media_type=MP3, headers=headers)


@router.get("/api/voice/voices", response_model=None,
            dependencies=[Depends(require_workspace_permission(SPEECH_PERMISSION))])
async def voices() -> Any:
    """voice-tts's voices, English first: ``{"default", "voices": [{id, name, accent, gender}]}``."""
    try:
        return await speech_client.fetch_voices()
    except SpeechRefused as refusal:
        return _refused(refusal)

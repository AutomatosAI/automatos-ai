"""The httpx side of Auto's server voice: ``POST /speak`` streamed through, ``GET /voices`` cached.

voice-tts (``services/voice-tts``) answers ``audio/mpeg`` as it speaks, a sentence at a time.
``open_speech`` waits only for its status and headers: a refusal there becomes this server's
own (``speech.SpeechRefused``), and a 200 hands its body to ``relay``, which passes each chunk
on as it arrives, never the whole answer at once. ``VOICE_TTS_TIMEOUT_SECONDS`` bounds the
whole answer, the stream included; ``VOICE_TTS_CONNECT_TIMEOUT_SECONDS`` is short, so a voice
that is down is refused at once and the client falls back to the device's voice.

voice-tts's refusals map to this server's codes: 413 stays ``tts_too_long``, 422 stays
``tts_bad_request`` (an unknown voice among them), 429 stays ``tts_busy``, and anything else
(503 ``tts_warming`` while its model loads, a refused token, an error) is ``tts_unavailable``.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from pydantic import BaseModel, ValidationError

from config import config
from modules.voice.speech import (
    TTS_BAD_REQUEST,
    TTS_BUSY,
    TTS_TOO_LONG,
    SpeechRefused,
    SpeechRequest,
    SpeechUsage,
    record_usage,
    require_configured,
    unavailable,
)

logger = logging.getLogger(__name__)

SPEAK_PATH = "/speak"
VOICES_PATH = "/voices"
VOICE_HEADER = "X-Voice-Id"
# How long GET /api/voice/voices answers from its copy of voice-tts's list.
VOICES_CACHE_SECONDS = 300
# The most of voice-tts's own refusal detail passed on to the client.
DETAIL_MAX_CHARS = 200
# voice-tts's refusals this server passes on with its own code; any other status is tts_unavailable.
PASSED_ON = {413: TTS_TOO_LONG, 422: TTS_BAD_REQUEST, 429: TTS_BUSY}


class VoiceListing(BaseModel):
    id: str
    name: str
    accent: str
    gender: str


class VoicesBody(BaseModel):
    """voice-tts's ``GET /voices``, checked before it is cached or passed on."""

    default: str
    voices: List[VoiceListing]


@dataclass(frozen=True)
class _Client:
    loop: asyncio.AbstractEventLoop
    client: httpx.AsyncClient


@dataclass(frozen=True)
class _CachedVoices:
    url: str
    fetched: float
    body: Dict[str, Any]


_shared: Optional[_Client] = None
_voices: Optional[_CachedVoices] = None


def new_http_client() -> httpx.AsyncClient:
    """An httpx client for voice-tts: its token as a bearer, its timeouts (a short connect and pool wait)."""
    connect = float(config.VOICE_TTS_CONNECT_TIMEOUT_SECONDS)
    timeout = httpx.Timeout(float(config.VOICE_TTS_TIMEOUT_SECONDS), connect=connect, pool=connect)
    headers = {"Authorization": f"Bearer {config.VOICE_TTS_TOKEN}"} if config.VOICE_TTS_TOKEN else {}
    return httpx.AsyncClient(timeout=timeout, headers=headers)


def _client() -> httpx.AsyncClient:
    """One client per event loop: a pooled connection belongs to the loop that opened it (F238)."""
    global _shared
    loop = asyncio.get_running_loop()
    if _shared is None or _shared.loop is not loop or _shared.client.is_closed:
        _shared = _Client(loop=loop, client=new_http_client())
    return _shared.client


def _url(path: str) -> str:
    return f"{config.VOICE_TTS_URL}{path}"


async def _refusal(response: httpx.Response, workspace_id: str) -> SpeechRefused:
    """voice-tts's refusal as this server's: its own code for 413, 422 and 429, else tts_unavailable."""
    try:
        await response.aread()
        body = response.json()
    except (httpx.HTTPError, ValueError):
        body = {}
    finally:
        await response.aclose()
    upstream = body.get("code") if isinstance(body, dict) else None
    logger.info("voice_tts_refused workspace=%s status=%d code=%s", workspace_id, response.status_code, upstream)
    code = PASSED_ON.get(response.status_code)
    if code is None:
        return unavailable()
    detail = body.get("detail") if isinstance(body, dict) else None
    detail = detail[:DETAIL_MAX_CHARS] if isinstance(detail, str) and detail else f"The voice refused it ({code})."
    return SpeechRefused(response.status_code, code, detail)


@dataclass(frozen=True)
class SpeechStream:
    """voice-tts's 200: the voice it speaks with, and its body still to be read."""

    voice: str
    response: httpx.Response
    started: float


async def open_speech(spec: SpeechRequest, workspace_id: str) -> SpeechStream:
    """Ask voice-tts to speak, up to its status and headers. Raises SpeechRefused."""
    require_configured()
    client = _client()
    payload = {"text": spec.text, "voice": spec.voice, "speed": spec.speed}
    started = time.monotonic()
    try:
        response = await client.send(client.build_request("POST", _url(SPEAK_PATH), json=payload), stream=True)
    except httpx.HTTPError as exc:
        logger.warning("voice_tts_unreachable workspace=%s error=%s", workspace_id, type(exc).__name__)
        raise unavailable() from exc
    if response.status_code != 200:
        raise await _refusal(response, workspace_id)
    return SpeechStream(voice=response.headers.get(VOICE_HEADER, spec.voice), response=response, started=started)


async def relay(stream: SpeechStream, spec: SpeechRequest, workspace_id: str) -> AsyncIterator[bytes]:
    """voice-tts's MP3, chunk by chunk as it arrives, until it ends or the answer's time is up.

    A stream that breaks or runs out of time simply ends: the audio already sent has played.
    One usage line when it is over, however it ended.
    """
    deadline = stream.started + float(config.VOICE_TTS_TIMEOUT_SECONDS)
    first_byte: Optional[float] = None
    sent = 0
    outcome = "ok"
    try:
        async for chunk in stream.response.aiter_raw():
            if first_byte is None:
                first_byte = time.monotonic()
            sent += len(chunk)
            yield chunk
            if time.monotonic() > deadline:
                outcome = "timed_out"
                break
    except httpx.HTTPError as exc:
        outcome = "broken"
        logger.warning("voice_tts_stream_broken workspace=%s after=%d bytes error=%s", workspace_id, sent,
                       type(exc).__name__)
    finally:
        await stream.response.aclose()
        finished = time.monotonic()
        record_usage(SpeechUsage(
            workspace_id=workspace_id, voice=stream.voice, chars=len(spec.text),
            first_byte_seconds=None if first_byte is None else first_byte - stream.started,
            total_seconds=finished - stream.started, audio_bytes=sent, outcome=outcome,
        ))


async def fetch_voices() -> Dict[str, Any]:
    """voice-tts's voice list, answered from a copy for VOICES_CACHE_SECONDS. Raises SpeechRefused."""
    global _voices
    require_configured()
    url = _url(VOICES_PATH)
    now = time.monotonic()
    if _voices is not None and _voices.url == url and now - _voices.fetched < VOICES_CACHE_SECONDS:
        return _voices.body
    try:
        response = await _client().get(url)
        response.raise_for_status()
        body = VoicesBody.model_validate(response.json()).model_dump()
    except (httpx.HTTPError, ValueError, ValidationError) as exc:
        logger.warning("voice_tts_voices_unavailable error=%s", type(exc).__name__)
        raise unavailable() from exc
    _voices = _CachedVoices(url=url, fetched=now, body=body)
    return body


def forget_voices() -> None:
    """Drop the cached voice list (tests, and a changed VOICE_TTS_URL is already a miss)."""
    global _voices
    _voices = None

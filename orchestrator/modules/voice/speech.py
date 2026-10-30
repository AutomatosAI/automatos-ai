"""Auto's server voice, the orchestrator's side: what a speech request may ask, and how it is refused.

``POST /api/voice/speech`` (``api/voice_speech.py``) turns a short text, one sentence of Auto's
reply as the phone or the web quick-ask speaks it, into MP3 from voice-tts
(``services/voice-tts``, through ``speech_client.py``). Clients fall back to the device's own
voice on ANY answer that is not audio, so every refusal is fast and says why, as
``{"code", "detail"}``:

==========================  ==============================================================
503 ``tts_off``             no voice on this server: ``VOICE_TTS_URL`` is not set
503 ``tts_unavailable``     voice-tts is down, still loading its model, or too slow
422 ``tts_bad_request``     the body is not ``{text, voice, speed}``, the text is empty, the
                            voice is not a Kokoro voice id, or the speed is not a number
413 ``tts_too_long``        the text is over ``MAX_TEXT_CHARS`` once trimmed
429 ``tts_busy``            voice-tts is speaking all it can, or the workspace has spoken
                            ``VOICE_TTS_RATE_PER_MINUTE`` sentences this minute
==========================  ==============================================================

The speed is clamped to 0.8-1.2, as voice-tts clamps it. Which voices exist is voice-tts's
answer (``GET /api/voice/voices``): an id it does not hold comes back as its 422. A request
that names no voice is spoken in ``VOICE_TTS_DEFAULT_VOICE`` only when voice-tts holds it
(``speech_client.default_voice``): the hosted API still carries a value of that name from the
voice service retired in #625, which may be no Kokoro voice at all.

No text is ever logged. ``record_usage`` writes one line per answer, and it is the one place
speech would be metered. It charges no credit, by decision (PRD-265 D-6, Gerard, 30 Oct 2026):
the voice runs on our own server and costs us no more per use than the device's own voice.
The answer it reads is charged as any chat turn is.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt, StrictStr, ValidationError

from config import config

logger = logging.getLogger(__name__)

# The contract voice-tts holds too (services/voice-tts/README.md).
MAX_TEXT_CHARS = 500
SPEED_RANGE = (0.8, 1.2)
DEFAULT_SPEED = 1.0
# Kokoro's voice ids: language and gender letters, then the name (af_heart, bm_george).
VOICE_ID = re.compile(r"^[a-z][fm]_[a-z0-9]{1,32}$")

TTS_OFF = "tts_off"
TTS_UNAVAILABLE = "tts_unavailable"
TTS_BAD_REQUEST = "tts_bad_request"
TTS_TOO_LONG = "tts_too_long"
TTS_BUSY = "tts_busy"

UNAVAILABLE_DETAIL = "Auto's voice is not available right now; the device's voice is used instead."


class SpeechRefused(Exception):
    """A speech request this server will not answer with audio: its status, code and detail."""

    def __init__(self, status: int, code: str, detail: str, *, retry_after: Optional[int] = None) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail
        self.retry_after = retry_after

    def body(self) -> Dict[str, str]:
        return {"code": self.code, "detail": self.detail}


def unavailable() -> SpeechRefused:
    return SpeechRefused(503, TTS_UNAVAILABLE, UNAVAILABLE_DETAIL)


def is_configured() -> bool:
    """Whether this server has a voice: ``VOICE_TTS_URL`` is set. Discovery's ``tts`` follows it."""
    return bool(config.VOICE_TTS_URL)


def require_configured() -> None:
    if not is_configured():
        raise SpeechRefused(503, TTS_OFF, "This server has no voice configured (VOICE_TTS_URL).")


class SpeechBody(BaseModel):
    """``POST /api/voice/speech``'s body, as sent; the checks that need more than a type follow.

    A field it does not know is ignored, so a newer client still gets audio from this server."""

    model_config = ConfigDict(extra="ignore")

    text: StrictStr
    voice: Optional[StrictStr] = None
    speed: Optional[Union[StrictInt, StrictFloat]] = None


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    # None: the request named no voice; the default is chosen against voice-tts's list.
    voice: Optional[str]
    speed: float


def _bad(detail: str) -> SpeechRefused:
    return SpeechRefused(422, TTS_BAD_REQUEST, detail)


def _speed(raw: Optional[float]) -> float:
    if raw is None:
        return DEFAULT_SPEED
    if not math.isfinite(raw):
        raise _bad("speed must be a finite number or null")
    low, high = SPEED_RANGE
    return min(high, max(low, float(raw)))


def _voice(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    if not VOICE_ID.match(raw):
        raise _bad("voice must be a voice id from GET /api/voice/voices, or null")
    return raw


def parse_speech(payload: Any) -> SpeechRequest:
    """The body as a SpeechRequest. Raises SpeechRefused (422 or 413)."""
    try:
        body = SpeechBody.model_validate(payload)
    except ValidationError:
        raise _bad('send {"text": string, "voice": string or null, "speed": number or null}') from None
    text = body.text.strip()
    if not text:
        raise _bad("text is empty")
    if len(text) > MAX_TEXT_CHARS:
        raise SpeechRefused(413, TTS_TOO_LONG, f"text is {len(text)} characters; at most {MAX_TEXT_CHARS} are spoken")
    return SpeechRequest(text=text, voice=_voice(body.voice), speed=_speed(body.speed))


@dataclass(frozen=True)
class SpeechUsage:
    """One answer, as the usage line records it. Never its text."""

    workspace_id: str
    voice: str
    chars: int
    first_byte_seconds: Optional[float]
    total_seconds: float
    audio_bytes: int
    outcome: str


def record_usage(usage: SpeechUsage) -> None:
    """One line per synthesis: the workspace, the voice, the characters and the latency.

    Speech is free (PRD-265 D-6). Were that ever to change, the charge would go here, once
    per answer, by ``usage.chars`` for ``usage.workspace_id``.
    """
    first_ms = round(usage.first_byte_seconds * 1000) if usage.first_byte_seconds is not None else -1
    logger.info(
        "voice_tts_speech workspace=%s voice=%s chars=%d first_byte_ms=%d total_ms=%d bytes=%d outcome=%s",
        usage.workspace_id, usage.voice, usage.chars, first_ms, round(usage.total_seconds * 1000),
        usage.audio_bytes, usage.outcome,
    )

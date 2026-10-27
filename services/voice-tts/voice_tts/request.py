"""``POST /speak``'s body, validated, and the text cut into the sentences spoken one by one.

The body is ``{"text": str, "voice": str | null, "speed": float | null}`` and nothing else:

- ``text``: 1 to ``max_chars`` characters once trimmed. Empty is a 422; longer is a 413;
- ``voice``: a voice this image holds (``GET /voices``), or null for the default. Any other is a 422;
- ``speed``: a number, clamped to 0.8-1.2; null is 1.0. Anything but a finite number is a 422.

A long text is spoken a sentence at a time, so its first sentence is heard while the
rest are still being spoken: Kokoro's own batching would make the listener wait for
the whole text.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Tuple

from .config import DEFAULT_SPEED, SPEED_RANGE, Settings
from .voices import Catalogue, Voice

FIELDS = frozenset({"text", "voice", "speed"})
# A sentence ends at . ! ? or …, or at one followed by a closing quote or bracket, then space.
SENTENCE_END = re.compile(r"(?:(?<=[.!?…])|(?<=[.!?…][\"'”’)\]]))\s+")
# A piece shorter than this is spoken with the next one: a lone "Sure." costs a pause, not a lead.
MIN_SENTENCE_CHARS = 16
# A period after one of these, after a single letter, or inside dotted initials ("U.S.") ends no sentence.
ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc", "inc", "ltd", "co", "no", "approx",
})


class RequestError(ValueError):
    """The body cannot be spoken: ``status`` and ``code`` are the refusal's."""

    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


def _bad(detail: str) -> RequestError:
    return RequestError(422, "tts_bad_request", detail)


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    voice: Voice
    speed: float


def _text(raw: Any, max_chars: int) -> str:
    if not isinstance(raw, str):
        raise _bad("text must be a string")
    text = raw.strip()
    if not text:
        raise _bad("text is empty")
    if len(text) > max_chars:
        raise RequestError(413, "tts_too_long", f"text is {len(text)} characters; at most {max_chars} are spoken")
    return text


def _voice(raw: Any, catalogue: Catalogue) -> Voice:
    if raw is None:
        return catalogue.get(catalogue.default)
    if not isinstance(raw, str):
        raise _bad("voice must be a voice id or null")
    try:
        return catalogue.get(raw)
    except KeyError:
        raise _bad(f"voice {raw!r} is not one of this server's voices (GET /voices)") from None


def clamp_speed(raw: Any) -> float:
    """The speed to speak at: null is the default, a number is held to the range."""
    if raw is None:
        return DEFAULT_SPEED
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
        raise _bad("speed must be a number or null")
    low, high = SPEED_RANGE
    return min(high, max(low, float(raw)))


def parse_speech(payload: Any, settings: Settings, catalogue: Catalogue) -> SpeechRequest:
    """The body as a SpeechRequest. Raises RequestError with the refusal's status and code."""
    if not isinstance(payload, dict):
        raise _bad("the body must be a JSON object")
    unknown = sorted(set(payload) - FIELDS)
    if unknown:
        raise _bad(f"unknown field(s): {', '.join(unknown)}")
    if "text" not in payload:
        raise _bad("text is required")
    return SpeechRequest(
        text=_text(payload["text"], settings.max_chars),
        voice=_voice(payload.get("voice"), catalogue),
        speed=clamp_speed(payload.get("speed")),
    )


def _ends_with_abbreviation(piece: str) -> bool:
    last = piece.split()[-1].strip("\"'”’()[]").rstrip(".").lower()
    return last in ABBREVIATIONS or len(last) <= 1 or "." in last


def sentences(text: str) -> Tuple[str, ...]:
    """The text cut after each sentence's end, in order, none empty.

    A piece that ends with an abbreviation ("Dr.") or is shorter than MIN_SENTENCE_CHARS
    joins the piece after it, so no pause falls inside a sentence.
    """
    pieces = [piece.strip() for piece in SENTENCE_END.split(text) if piece.strip()]
    spoken = []
    held = ""
    for piece in pieces:
        joined = f"{held} {piece}" if held else piece
        if _ends_with_abbreviation(joined) or len(joined) < MIN_SENTENCE_CHARS:
            held = joined
            continue
        spoken.append(joined)
        held = ""
    if held:
        spoken.append(held)
    return tuple(spoken)

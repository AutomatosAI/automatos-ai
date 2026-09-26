"""Speech for the service: POST /tts and the render's Kokoro lines (PRD-251 S1.5).

POST /tts speaks a script before it is composed, so scene timing can flex to
the voice: each line comes back with its duration and its voiced segments (the
words, by energy gaps), and with the WAV itself when asked. A render speaks its
Kokoro lines again from the same text and settings, so they time the same.

Kokoro is CPU-bound and loads a 310 MB model once. ``Speaker`` runs synthesis
off the event loop and one line at a time, whether the call came from /tts or
from a render job.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from . import validate
from .config import Settings
from .kokoro_tts import SpokenLine, synthesize_line
from .validate import BundleError

Synthesize = Callable[..., SpokenLine]


@dataclass(frozen=True)
class TtsRequest:
    lines: Tuple[Tuple[str, str], ...]
    voice: str
    speed: float
    lang: str
    include_audio: bool


def parse_tts_request(payload: Any, settings: Settings) -> TtsRequest:
    body = validate.mapping(payload, "the request")
    validate.keys(body, "the request", required=("lines",), optional=("voice", "speed", "lang", "include_audio"))
    lines: List[Tuple[str, str]] = []
    for i, entry in enumerate(validate.items(body["lines"], "lines", limit=settings.tts_max_lines)):
        where = f"lines[{i}]"
        validate.keys(validate.mapping(entry, where), where, required=("id", "text"))
        line_id = validate.line_id(entry["id"], f"{where}.id")
        if any(existing == line_id for existing, _ in lines):
            raise BundleError(f"{where}.id {line_id} is used twice")
        lines.append((line_id, validate.text(entry["text"], f"{where}.text", max_chars=settings.tts_max_chars)))
    if not lines:
        raise BundleError("lines is empty")
    include_audio = body.get("include_audio", False)
    if not isinstance(include_audio, bool):
        raise BundleError("include_audio must be true or false")
    return TtsRequest(
        lines=tuple(lines),
        voice=validate.voice_name(body.get("voice", settings.kokoro_voice), "voice"),
        speed=validate.speed(body.get("speed", settings.kokoro_speed), "speed"),
        lang=validate.language(body.get("lang", settings.kokoro_lang), "lang"),
        include_audio=include_audio,
    )


class Speaker:
    """Every Kokoro synthesis goes through here: one line at a time, off the event loop."""

    def __init__(self, settings: Settings, synthesize: Synthesize = synthesize_line) -> None:
        self._settings = settings
        self._synthesize = synthesize
        self._lock = asyncio.Lock()

    async def speak(
        self, lines: Sequence[Tuple[str, str]], out_dir: Path, *, voice: str, speed: float, lang: str
    ) -> List[SpokenLine]:
        """Speak each (id, text) line into ``out_dir/<id>.wav``."""
        async with self._lock:
            return await asyncio.to_thread(self._speak_all, list(lines), out_dir, voice, speed, lang)

    def _speak_all(
        self, lines: List[Tuple[str, str]], out_dir: Path, voice: str, speed: float, lang: str
    ) -> List[SpokenLine]:
        return [
            self._synthesize(text, out_dir / f"{line_id}.wav", self._settings, voice=voice, speed=speed, lang=lang)
            for line_id, text in lines
        ]


def line_report(line_id: str, spoken: SpokenLine, *, include_audio: bool = False) -> Mapping[str, Any]:
    report: Dict[str, Any] = {
        "id": line_id,
        "seconds": spoken.seconds,
        "sample_rate": spoken.sample_rate,
        "segments": [{"start": start, "end": end} for start, end in spoken.segments],
    }
    if include_audio:
        report["audio_base64"] = base64.b64encode(spoken.path.read_bytes()).decode("ascii")
    return report

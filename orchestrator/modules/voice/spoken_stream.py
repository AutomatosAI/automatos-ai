"""PRD-266 US-007a (§4.7, D32): a chat turn that is heard, not read — its text frames, speechified.

``POST /api/chat`` with ``spokenMode`` (the app's quick-ask voice) tells Auto the reply is
spoken (the streaming service appends ``SPOKEN_OUTPUT_CONTRACT``) and runs the turn's AI-SDK
frames through ``spoken_text_frames``. As ``spoken_style`` says, the prompt alone is not a
guarantee for something the user hears, so every text frame also passes through ``speechify``.

Text is buffered to whole speech units (``split_speech_unit``, the rule the Retell lane uses)
before it is sanitised, because markdown straddles deltas: ``**`` arrives as ``*`` then ``*``.
A frame that is not text (data, tool, finish, error) passes at once when no unit is open; while
one is open it waits for that unit, so markdown on both sides of it is still sanitised whole,
and it never overtakes the text sent before it. A unit with nothing speakable emits no frame.

Voice latency B-5: a narration (or retraction) frame says "the round that just streamed was
not the answer", and the client takes that text out of what it holds by exact match. The
client holds speechified text, so with ``format_narration`` given the round's open text is
spoken as the frame passes (the round ended: it is whole) and the frame goes out carrying
the speechified text that was sent for it. Without it the frame passes as any other.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Callable, Optional, Tuple

from config import config
from modules.voice.providers.retell import extract_agent_text
from modules.voice.spoken_style import speechify, split_speech_unit

logger = logging.getLogger(__name__)

# An AI-SDK data-stream text frame is ``0:"<json string>"``; every other prefix is not text.
TEXT_FRAME_PREFIX = "0:"
# A narration frame's prefix (StreamingHandler.narration_frame). Equal to
# consumers.chatbot.on_screen.NARRATION_FRAME, and a copy on purpose: importing that module here
# would load the chat service into a voice module the chat route imports. A test pins them equal.
NARRATION_FRAME = 'd:{"type": "narration"'
# Units are speechified apart, which strips the space between them; it is put back.
UNIT_SEPARATOR = " "

# Speech units in the order they were cut: (the raw text of the unit, what was sent for it).
# A unit that speechified to nothing is kept with "".
Units = Tuple[Tuple[str, str], ...]


def _speak(buffer: str, max_chars: int) -> Tuple[Units, str]:
    """The buffer's complete speech units, each with its speechified text ("" when it says
    nothing), and the open remainder."""
    units: Units = ()
    unit, rest = split_speech_unit(buffer, max_chars)
    while unit:
        units = units + ((unit, speechify(unit)),)
        unit, rest = split_speech_unit(rest, max_chars)
    return units, rest


def _said(units: Units) -> Tuple[str, ...]:
    """What the units say: an unspeakable unit is dropped."""
    return tuple(spoken for _, spoken in units if spoken)


def _text_frames(said: Tuple[str, ...], format_text: Callable[[str], str],
                 spoken_before: bool) -> Tuple[str, ...]:
    """One text frame per spoken unit, each after the turn's first led by a separator."""
    return tuple(
        format_text(unit if index == 0 and not spoken_before else UNIT_SEPARATOR + unit)
        for index, unit in enumerate(said)
    )


def _narration(chunk: str, wanted: bool) -> Optional[Tuple[str, bool]]:
    """A narration frame's text and whether it retracts; None for any other chunk, and for
    every chunk when narration frames are not ``wanted`` rewritten (no formatter)."""
    if not wanted or not chunk.startswith(NARRATION_FRAME):
        return None
    try:
        data = json.loads(chunk[len("d:"):]).get("data")
    except (ValueError, AttributeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("text"), str):
        return None
    return data["text"], bool(data.get("retracted"))


def _normalised(text: str) -> str:
    return " ".join(text.split())


def _sent_run(log: Units, target: str) -> Optional[Units]:
    """The latest contiguous run of sent units whose raw texts, joined and
    whitespace-normalised, are ``target``; None when no run is."""
    for end in range(len(log), 0, -1):
        for start in range(end - 1, -1, -1):
            joined = _normalised("".join(raw for raw, _ in log[start:end]))
            if joined == target:
                return log[start:end]
            if len(joined) > len(target):
                break
    return None


def _as_spoken(log: Units, text: str, max_chars: int) -> str:
    """The speechified text that was sent for a frame's raw ``text``: an exact substring of
    the turn's text frames when a run of sent units is that text. When none is (a unit that
    straddles two rounds), the text speechified unit by unit, as a best effort."""
    target = _normalised(text)
    run = _sent_run(log, target) if target else ()
    if run is None:
        logger.debug("spoken narration: no sent run for the frame (chars=%d, units=%d)", len(text), len(log))
        units, rest = _speak(text, max_chars)
        run = units + ((rest, speechify(rest)),)
    return UNIT_SEPARATOR.join(_said(run))


def _closed(open_text: str) -> Units:
    """A round's open text as its last unit: the round ended, so the text is whole."""
    return ((open_text, speechify(open_text)),) if open_text else ()


def _any_said(spoken_before: bool, units: Units) -> bool:
    return spoken_before or bool(_said(units))


def _mark(log: Units, marked: Tuple[str, bool], max_chars: int,
          format_narration: Callable[[str, bool], str]) -> Tuple[str, ...]:
    """The narration frame for ``marked`` carrying the text that was sent for it; no frame
    when nothing of it was sent."""
    spoken = _as_spoken(log, marked[0], max_chars)
    return (format_narration(spoken, marked[1]),) if spoken else ()


async def spoken_text_frames(chunks: AsyncIterator[str], format_text: Callable[[str], str],
                             format_narration: Optional[Callable[[str, bool], str]] = None) -> AsyncIterator[str]:
    """The turn's frames with its text speechified; ``format_text`` writes a text frame.

    Every non-text frame comes out in its order; only the text is regrouped into speech
    units, at most ``VOICE_LIVE_SPEECH_UNIT_MAX_CHARS`` long. ``format_narration(said,
    retracted)`` writes a narration frame: given, a narration frame closes its round's open
    text and carries the text that was sent for it (none sent: no frame); None, it passes
    unchanged like every other frame.
    """
    max_chars = config.VOICE_LIVE_SPEECH_UNIT_MAX_CHARS
    open_text, held, spoken_any, log = "", (), False, ()
    async for chunk in chunks:
        if not chunk.startswith(TEXT_FRAME_PREFIX):
            marked = _narration(chunk, format_narration is not None)
            if marked:
                # The round ended, so its open text is whole: it is spoken now, then the frames
                # it held, then the frame that names it.
                closed = _closed(open_text)
                log = log + closed
                text_frames = _text_frames(_said(closed), format_text, spoken_any)
                for frame in text_frames + held + _mark(log, marked, max_chars, format_narration):
                    yield frame
                open_text, held, spoken_any = "", (), _any_said(spoken_any, closed)
            elif open_text.strip():
                held = held + (chunk,)
            else:
                yield chunk
            continue
        units, open_text = _speak(open_text + extract_agent_text(chunk), max_chars)
        # The frames held behind an open unit leave once no unit is open.
        released = () if open_text.strip() else held
        for frame in _text_frames(_said(units), format_text, spoken_any) + released:
            yield frame
        spoken_any, log, held = _any_said(spoken_any, units), log + units, held[len(released):]
    for frame in _text_frames(_said(_closed(open_text)), format_text, spoken_any) + held:
        yield frame


__all__ = ["spoken_text_frames"]

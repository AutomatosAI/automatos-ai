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
"""

from __future__ import annotations

from typing import AsyncIterator, Callable, Tuple

from config import config
from modules.voice.providers.retell import extract_agent_text
from modules.voice.spoken_style import speechify, split_speech_unit

# An AI-SDK data-stream text frame is ``0:"<json string>"``; every other prefix is not text.
TEXT_FRAME_PREFIX = "0:"
# Units are speechified apart, which strips the space between them; it is put back.
UNIT_SEPARATOR = " "


def _speak(buffer: str, max_chars: int) -> Tuple[Tuple[str, ...], str]:
    """The buffer's complete speech units, speechified (an unspeakable unit is dropped),
    and the open remainder."""
    said: Tuple[str, ...] = ()
    unit, rest = split_speech_unit(buffer, max_chars)
    while unit:
        spoken = speechify(unit)
        said = said + ((spoken,) if spoken else ())
        unit, rest = split_speech_unit(rest, max_chars)
    return said, rest


def _text_frames(said: Tuple[str, ...], format_text: Callable[[str], str],
                 spoken_before: bool) -> Tuple[str, ...]:
    """One text frame per spoken unit, each after the turn's first led by a separator."""
    return tuple(
        format_text(unit if index == 0 and not spoken_before else UNIT_SEPARATOR + unit)
        for index, unit in enumerate(said)
    )


async def spoken_text_frames(chunks: AsyncIterator[str],
                             format_text: Callable[[str], str]) -> AsyncIterator[str]:
    """The turn's frames with its text speechified; ``format_text`` writes a text frame.

    Every non-text frame comes out unchanged and in its order; only the text is regrouped
    into speech units, at most ``VOICE_LIVE_SPEECH_UNIT_MAX_CHARS`` long.
    """
    max_chars = config.VOICE_LIVE_SPEECH_UNIT_MAX_CHARS
    open_text, held, spoken_any = "", (), False
    async for chunk in chunks:
        if not chunk.startswith(TEXT_FRAME_PREFIX):
            if open_text.strip():
                held = held + (chunk,)
            else:
                yield chunk
            continue
        said, open_text = _speak(open_text + extract_agent_text(chunk), max_chars)
        for frame in _text_frames(said, format_text, spoken_any):
            yield frame
        spoken_any = spoken_any or bool(said)
        if not open_text.strip():
            for frame in held:
                yield frame
            held = ()
    tail = speechify(open_text)
    for frame in _text_frames((tail,) if tail else (), format_text, spoken_any) + held:
        yield frame


__all__ = ["spoken_text_frames"]

"""FX-017 (night 12): one answer on the screen, never two and never none.

F186 marks each streamed round as it goes: a round that called tools is narration, and
a reply the loop nudged is retracted when its retry replaces it. The frontend takes the
frame's text out of the answer it has shown (``withoutNarration``: the last exact
occurrence). Night 12 found two faults (A108, A209, A622; eval M-terms-a turn 2):

* The frame carried the response's content, which is not what streamed: inline
  ``<think>`` tags are lifted out after the stream, and the owner-words rewrite runs per
  delta live and again on the whole text. A frame that matched nothing left both
  replies on the screen.
* A nudged reply was retracted as soon as its retry returned, before the loop decided
  which one stands. A blank retry keeps the original (``kept_if_blank``, the F205
  re-prompt), whose retraction had already gone out: the screen was left empty.

Now each turn keeps a ``Screen``: every streamed call records the exact text it put on
the screen (core/llm/screen_watch.py), and a frame carries that text. A retraction goes
out when a reply with something in it took the round's place. After a blank reply it is
held until the loop's answer is known: dropped when the original is that answer (the
saved answer is then what the screen shows), sent when anything else is. The
frontend's rule is unchanged.

P256-FIX-RVW-24: a round is retracted at most once per turn. A retry that only calls
tools never becomes the latest streamed round, so the answer after it would retract the
nudged draft again; when that answer repeats the draft (or contains it), the frontend's
rule takes the answer off the screen instead. The Screen records what it has retracted.

P256-FIX-RVW-43: a held round is settled by where the answer came from, not by equal
text. A nudged reply that opens with an apology is saved without it (a copy,
``nudges.without_the_apology``), so its text no longer equals the round's; when the F205
re-prompt after it came back blank, the round was retracted and the saved copy, already
streamed, sent no tail: the screen was empty. A held round now stands when the answer is
its response or a streamed copy whose words end the round's. A reply to a re-prompt
goes through the apology gate (core/llm/turn_order.py) after it streamed through here:
the gate reports what it let through and the copy it saved (``restated``), so the round
is that copy and its frames carry the text the owner saw.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, List, Optional, Sequence, Tuple

from core.llm.screen_watch import WATCHER

Stream = Callable[..., AsyncGenerator[Any, None]]
_SENTENCE_ENDS = (". ", "! ", "? ")  # RVW-43: where a copy without its opening sentences starts


@dataclass(frozen=True)
class Round:
    """One streamed call: its answer's text, what it streamed, whether it said nothing,
    and the response it returned."""

    content: str
    said: str
    blank: bool
    response: Any = field(compare=False)


def _words(text: str) -> str:
    return " ".join((text or "").split())


def _stands(answer: Any, held: Round) -> bool:
    """RVW-43: ``answer`` is ``held``'s own response, or a copy of it: streamed, and its
    words (whitespace normalised) are the round's last sentences or all of them."""
    if answer is held.response:
        return True
    words, whole = _words(getattr(answer, "content", None) or ""), _words(held.content)
    before = whole[: len(whole) - len(words)]
    copied = whole.endswith(words) and (not before or before.endswith(_SENTENCE_ENDS))
    return bool(words) and bool(getattr(answer, "streamed", False)) and copied


def _latest(rounds: Sequence[Round], content: str) -> Optional[Round]:
    return next((r for r in reversed(rounds) if r.content == content), None)


class Screen:
    """A turn's streamed rounds, in order, and the retractions held back."""

    def __init__(self) -> None:
        self._rounds: Tuple[Round, ...] = ()
        self._held: Tuple[Round, ...] = ()
        self._gone: Tuple[Round, ...] = ()  # RVW-24: retracted (sent or settled), never again

    def ended(self, response: Any, said: str) -> None:
        """A streamed call returned ``response`` after putting ``said`` on the screen."""
        content = getattr(response, "content", None) or ""
        blank = not getattr(response, "tool_calls", None) and not content.strip()
        self._rounds = (*self._rounds, Round(content, said, blank, response))

    def restated(self, response: Any, copy: Any, said: str) -> None:
        """RVW-43: the round that returned ``response`` put ``said`` on the screen (the
        apology gate held back its opening) and is saved as ``copy``."""
        content = getattr(copy, "content", None) or ""
        self._rounds = tuple(Round(content, said, r.blank, copy) if r.response is response else r
                             for r in self._rounds)

    def narration(self, content: str) -> str:
        """What streamed for the latest round that answered ``content``."""
        match = _latest(self._rounds, content)
        return match.said if match else content

    def retraction(self, content: str) -> Optional[str]:
        """What to take off the screen for the nudged round that answered ``content``, the
        round after it being its replacement; None while that replacement is blank (held)."""
        if not self._rounds:
            return content
        *earlier, latest = self._rounds
        match = _latest(earlier, content)
        if match is None:  # its replacement did not stream through here (a failover answer)
            match = _latest(self._rounds, content)
            return content if match is None else self._retract(match)
        if self._retracted(match):
            return None
        if latest.blank:
            self._held = (*(r for r in self._held if r is not match), match)
            return None
        self._held = tuple(r for r in self._held if r is not match)
        return self._retract(match)

    def settled(self, answer: Any) -> List[str]:
        """The held rounds the loop's ``answer`` replaced, to retract now. A held round
        the answer is (its response, or a copy of it) stands: nothing is retracted for it."""
        replaced = [r for r in self._held if not _stands(answer, r)]
        self._held = ()
        return [said for said in (self._retract(r) for r in replaced) if said is not None]

    def _retracted(self, match: Round) -> bool:
        return any(r is match for r in self._gone)

    def _retract(self, match: Round) -> Optional[str]:
        """``match``'s streamed text, the first time it is retracted; None after that."""
        if self._retracted(match):
            return None
        self._gone = (*self._gone, match)
        return match.said


def _screen() -> Optional[Screen]:
    watcher = WATCHER.get()
    return watcher if isinstance(watcher, Screen) else None


def narrated(content: str) -> str:
    """The text a narration frame carries for the round that answered ``content``."""
    screen = _screen()
    return screen.narration(content) if screen else content


def retracted(content: str) -> Optional[str]:
    """The text a retraction frame carries for the nudged round that answered ``content``;
    None when no frame goes out yet."""
    screen = _screen()
    return screen.retraction(content) if screen else content


def keeps_one_answer_on_screen(turn: Stream) -> Stream:
    """Wrap ``StreamingChatService._stream_response_with_agent_scoped``: the turn's
    streamed calls are recorded on its own ``Screen`` (a turn run inside another gives
    the outer turn's back when it ends)."""
    @functools.wraps(turn)
    async def wrapped(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        outer = WATCHER.get()
        WATCHER.set(Screen())
        try:
            async for chunk in turn(*args, **kwargs):
                yield chunk
        finally:
            WATCHER.set(outer)
    return wrapped


def settles_held_retractions(loop: Stream) -> Stream:
    """Wrap ``StreamingChatService._stream_tool_loop``: a retraction held back (``""``,
    never sent) goes out before the loop's answer only when that answer replaced it."""
    @functools.wraps(loop)
    async def wrapped(chat: Any, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        async for chunk in loop(chat, *args, **kwargs):
            if chunk == "":
                continue
            final = chunk.get("_final_response") if isinstance(chunk, dict) else None
            screen = _screen()
            if final is not None and screen is not None:
                for said in screen.settled(final):
                    yield chat.streaming_handler.narration_frame(said, retracted=True)
            yield chunk
    return wrapped


__all__ = ["Round", "Screen", "keeps_one_answer_on_screen", "narrated", "retracted", "settles_held_retractions"]

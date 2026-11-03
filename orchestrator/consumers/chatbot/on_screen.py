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

P256-FIX-N15-1: the loop takes an unbacked claim out of a streamed reply, with no retry; the copy
that answers never streamed (``screen_watch.corrected``). The reply's round is held like a nudged
one and retracted when the loop's answer is known; the copy is sent in its place.

PRD-268 US-C12c: a round that offers reply chips may be the answer, because its chips can end the run
on its own text with no model call (US-C12b). Its narration frame is held by the turn's streams when
it passes them, in stream order (the loop runs in a task of its own, ahead of its frames). When the
loop's answer is that round's own text (``turn_parts.is_its_round``), the frame never goes out: the
text stays on the screen as the answer, and the answer is marked streamed, so the turn sends no tail.
Once the run goes on past the round (the next text on the stream, any other answer, or none), the
frame goes out first, as it would have.
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Sequence, Tuple

from consumers.chatbot.turn_parts import as_streamed, is_its_round, offers_replies
from core.llm.screen_watch import WATCHER

Stream = Callable[..., AsyncGenerator[Any, None]]
# The AI-SDK frames' prefixes (consumers/chatbot/streaming.py): a text frame (the model's words) and
# a narration frame (StreamingHandler.narration_frame).
TEXT_FRAME = "0:"
NARRATION_FRAME = 'd:{"type": "narration"'
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
        # PRD-268 US-C12c: rounds that offered reply chips, their narration frame made (not yet
        # passed), then held by the stream it passed (not yet sent).
        self._offered: Tuple[Round, ...] = ()
        self._holding: Tuple[Round, ...] = ()

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

    def corrected(self, response: Any, copy: Any) -> None:
        """P256-FIX-N15-1: the round that returned ``response`` is answered by ``copy``, which did not
        stream: its retraction is held until the loop's answer is known (``settled``)."""
        match = next((r for r in self._rounds if r.response is response), None)
        if match is not None and not self._retracted(match):
            self._held = (*(r for r in self._held if r is not match), match)

    def narration(self, content: str) -> str:
        """What streamed for the latest round that answered ``content``. PRD-268 US-C12c: a round
        that offers reply chips is noted, so its frame is held when it passes (``hold``)."""
        match = _latest(self._rounds, content)
        if match is not None and offers_replies(match.response):
            self._offered = (*(r for r in self._offered if r is not match), match)
        return match.said if match else content

    def hold(self, said: str) -> bool:
        """PRD-268 US-C12c: whether the narration frame carrying ``said``, passing a stream now, is
        a noted round's (one that offered reply chips): it is then held, not sent."""
        match = next((r for r in reversed(self._offered) if r.said == said), None)
        if match is None:
            return False
        self._offered = tuple(r for r in self._offered if r is not match)
        self._holding = (*self._holding, match)
        return True

    def offered_released(self) -> List[str]:
        """PRD-268 US-C12c: every held narration, to send now (a retracted round's never): the run
        went on past its round, or ended with no answer."""
        held, self._holding = self._holding, ()
        return [r.said for r in held if not self._retracted(r)]

    def offered_settled(self, answer: Any) -> Tuple[List[str], Optional[Round]]:
        """PRD-268 US-C12c: the held narrations the loop's ``answer`` settles. Returns what to send
        now and the round the answer is, its own text already on the screen as the answer (the
        latest such round, the one the run ended on; None when it is none of them)."""
        its_round = next((r for r in reversed(self._holding)
                          if is_its_round(answer, r.content) and not self._retracted(r)), None)
        self._holding = tuple(r for r in self._holding if r is not its_round)
        return self.offered_released(), its_round

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
    the outer turn's back when it ends). PRD-268 US-C12c: the first reply's narration frame
    is held here when that reply offered reply chips; the loop settles it (``settles_held_retractions``)."""
    @functools.wraps(turn)
    async def wrapped(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        outer, screen = WATCHER.get(), Screen()
        WATCHER.set(screen)
        handler = getattr(args[0], "streaming_handler", None) if args else None
        try:
            async for chunk in turn(*args, **kwargs):
                for frame in _passing(handler, screen, chunk):
                    yield frame
            for said in (screen.offered_released() if handler is not None else ()):
                yield handler.narration_frame(said)
        finally:
            WATCHER.set(outer)
    return wrapped


def _narration_said(chunk: Any) -> Optional[str]:
    """The text a (not retracted) narration frame carries; None for any other chunk."""
    if not isinstance(chunk, str) or not chunk.startswith(NARRATION_FRAME):
        return None
    try:
        data = json.loads(chunk[len("d:"):]).get("data")
    except (ValueError, AttributeError):
        return None
    if not isinstance(data, dict) or data.get("retracted") or not isinstance(data.get("text"), str):
        return None
    return data["text"]


def _passing(handler: Any, screen: Screen, chunk: Any) -> List[Any]:
    """What goes out for a chunk passing the turn's stream: nothing for a narration frame it holds
    (PRD-268 US-C12c), any held narration before the next text, else the chunk as it came."""
    if handler is None:
        return [chunk]
    said = _narration_said(chunk)
    if said is not None and screen.hold(said):
        return []
    if isinstance(chunk, str) and chunk.startswith(TEXT_FRAME):
        return [*(handler.narration_frame(r) for r in screen.offered_released()), chunk]
    return [chunk]


def settles_held_retractions(loop: Stream) -> Stream:
    """Wrap ``StreamingChatService._stream_tool_loop``: a retraction held back (``""``,
    never sent) goes out before the loop's answer only when that answer replaced it.
    PRD-268 US-C12c: a narration frame held back (a round that offered reply chips, ``hold``)
    goes out before the next text or the answer; when the answer is that round's own text, it
    never goes out and the answer is marked streamed. A loop that ends with no answer holds
    nothing back."""
    @functools.wraps(loop)
    async def wrapped(chat: Any, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        handler = getattr(chat, "streaming_handler", None)
        async for chunk in loop(chat, *args, **kwargs):
            if chunk == "":
                continue
            for frame in _settling(handler, chunk):
                yield frame
        screen = _screen()
        for said in (screen.offered_released() if screen is not None and handler is not None else ()):
            yield handler.narration_frame(said)
    return wrapped


def _settling(handler: Any, chunk: Any) -> List[Any]:
    """What goes out for one of the loop's chunks: for its answer, the frames it settles and then
    the answer (``_answer_known``); for anything else, as ``_passing`` says."""
    final = chunk.get("_final_response") if isinstance(chunk, dict) else None
    screen = _screen()
    if screen is None:
        return [chunk]
    if final is None:
        return _passing(handler, screen, chunk)
    frames, settled = _answer_known(handler, screen, chunk, final)
    return [*frames, settled]


def _answer_known(handler: Any, screen: Screen, chunk: Dict[str, Any],
                  final: Any) -> Tuple[List[str], Dict[str, Any]]:
    """The frames the loop's answer ``final`` settles, in order (FX-017's held retractions, then
    US-C12c's held narrations), and the final ``chunk``. Its response is marked streamed when it is
    a held round's own text, which streamed as that round's and stays on the screen."""
    retractions = [handler.narration_frame(said, retracted=True) for said in screen.settled(final)]
    narrations, its_round = screen.offered_settled(final)
    marked = as_streamed(final) if its_round is not None else final
    if its_round is not None and marked is final:  # it cannot be marked: the round is narration, as before
        narrations = [*narrations, its_round.said]
    frames = [*retractions, *(handler.narration_frame(said) for said in narrations)]
    return frames, (chunk if marked is final else {**chunk, "_final_response": marked})


__all__ = ["Round", "Screen", "keeps_one_answer_on_screen", "narrated", "retracted", "settles_held_retractions"]

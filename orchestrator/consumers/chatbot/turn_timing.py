"""The time a chat turn took, kept on the turn's trace and in one log line.

``/api/chat`` recorded no time to first token (widgets and voice log their own). Two
wrappers on ``StreamingChatService`` measure it without touching the turn's own function:

- ``keeps_the_turns_timing`` (on the turn) notes when the turn starts and, for the turn,
  hands it a chat service whose assistant ``save_message`` adds ``first_token_ms`` to the
  trace it saves;
- ``records_the_first_token`` (on ``_stream_llm_call``) notes when the first frame of the
  turn's first model call came.

Lean turns are measured against it (consumers.chatbot.lean_turn, evals.lean_turn).

While ``CHAT_TURN_TIMING_ENABLED`` is true the same two wrappers also feed the turn's clock
(``core.monitoring.turn_clock``): where the time went before, between and after the model's
calls. The saved reply's trace gains ``timing``, the clock as it stood at the save.

- The turn adopts the clock its request began (``POST /api/chat``: the producer writes that
  clock's line when the turn ends, whatever its ending). Any other turn (Auto Live, a widget,
  a direct call) begins its own here and writes its line when it ends.
- While ``CHAT_TURN_TIMING_STAGES_ENABLED`` is true too, five of the service's steps
  (``STAGES``) are timed by shadowing their bound methods on the service for the turn. The
  service is built per turn; the shadows are taken off when the turn ends, however it ends.

Numbers and ids only: nothing here reads what was said or answered.
"""
from __future__ import annotations

import contextlib
import functools
import inspect
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Dict, Optional, Tuple

from core.monitoring import turn_clock
from core.monitoring.turn_clock import TurnClock

FIRST_TOKEN_KEY = "first_token_ms"
TIMING_KEY = "timing"

CALL = "call"      # awaited
FRAMES = "frames"  # iterated with ``async for``
# (stage, method on the streaming service, kind)
STAGES = (
    (turn_clock.STAGE_AGENT_CONTEXT, "_load_agent_context", CALL),
    (turn_clock.STAGE_TOOLS, "_get_tools", CALL),
    (turn_clock.STAGE_CONTEXT, "_prepare_messages", CALL),
    (turn_clock.STAGE_COMPOSIO, "_inject_composio_tools", CALL),
    (turn_clock.STAGE_KNOWLEDGE, "_retrieval_first", FRAMES),
)
# What ``_unstaged`` puts back: (method name, the instance had its own attribute, that attribute).
Restore = Tuple[Tuple[str, bool, Any], ...]

Stream = Callable[..., AsyncGenerator[Any, None]]


def start(chat: Any) -> None:
    """Note that ``chat``'s turn starts now."""
    chat._turn_started = time.perf_counter()
    chat._first_token_ms = None


async def first_token(chat: Any, stream: AsyncIterator[Any]) -> AsyncIterator[Any]:
    """``stream`` unchanged; the first frame other than the call's final response
    sets ``chat._first_token_ms`` (once a turn)."""
    async for item in stream:
        if isinstance(item, dict) and "_response" in item:
            turn_clock.mark(turn_clock.MARK_FIRST_RESPONSE)  # first wins: the turn's first call, whole
        elif getattr(chat, "_first_token_ms", None) is None:
            started = getattr(chat, "_turn_started", None)
            if started is not None:
                chat._first_token_ms = round((time.perf_counter() - started) * 1000)
                turn_clock.mark(turn_clock.MARK_FIRST_TOKEN)
        yield item


def with_first_token(trace: Optional[Dict[str, Any]], chat: Any) -> Optional[Dict[str, Any]]:
    """``trace`` with the turn's ``first_token_ms``, a new dict; unchanged when unmeasured."""
    measured = getattr(chat, "_first_token_ms", None)
    if measured is None:
        return trace
    return {**(trace or {}), FIRST_TOKEN_KEY: measured}


def with_timing(trace: Optional[Dict[str, Any]], clock: Optional[TurnClock]) -> Optional[Dict[str, Any]]:
    """``trace`` with the clock as it stands now under ``timing``, a new dict; unchanged with no clock."""
    if clock is None:
        return trace
    return {**(trace or {}), TIMING_KEY: clock.snapshot()}


def records_the_first_token(stream_llm_call: Stream) -> Stream:
    """Wrap ``StreamingChatService._stream_llm_call``: its frames pass through, timed. On the
    turn's clock: when the first call started, when the second did (a tool round happened),
    the model's time summed over the calls, and how many there were."""
    @functools.wraps(stream_llm_call)
    async def wrapped(chat: Any, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        if not turn_clock.mark(turn_clock.MARK_MODEL_START):
            turn_clock.mark(turn_clock.MARK_MODEL_AGAIN)  # first wins: the turn's second call only
        started = time.perf_counter()
        try:
            async for item in first_token(chat, stream_llm_call(chat, *args, **kwargs)):
                yield item
        finally:
            turn_clock.add(turn_clock.STAGE_MODEL, time.perf_counter() - started)
            turn_clock.count(turn_clock.COUNT_MODEL_CALLS)
    return wrapped


class _TimedChats:
    """The turn's chat service: the service's own, except that an assistant message's
    ``context_trace`` carries the turn's ``first_token_ms`` and, when the turn has a clock,
    its ``timing`` as it stands at the save."""

    def __init__(self, chats: Any, chat: Any) -> None:
        self._chats = chats
        self._chat = chat

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chats, name)

    def save_message(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("role") == "assistant" and "context_trace" in kwargs:
            turn_clock.mark(turn_clock.MARK_REPLY_SAVED)
            trace = with_timing(with_first_token(kwargs["context_trace"], self._chat), turn_clock.current())
            kwargs = {**kwargs, "context_trace": trace}
        return self._chats.save_message(*args, **kwargs)


def _timed_call(inner: Callable[..., Any], stage: str, clock: TurnClock) -> Callable[..., Any]:
    @functools.wraps(inner)
    async def timed(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return await inner(*args, **kwargs)
        finally:
            clock.add(stage, time.perf_counter() - started)
    return timed


def _timed_frames(inner: Callable[..., Any], stage: str, clock: TurnClock) -> Callable[..., Any]:
    @functools.wraps(inner)
    async def timed(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        started = time.perf_counter()
        try:
            # Closed here, in the task that closes this wrapper: Stop leaves no open generator inside.
            async with contextlib.aclosing(inner(*args, **kwargs)) as frames:
                async for frame in frames:
                    yield frame
        finally:
            clock.add(stage, time.perf_counter() - started)
    return timed


def _staged(chat: Any, clock: TurnClock) -> Restore:
    """Shadow the ``STAGES`` methods ``chat`` has with timed ones, on the instance; what to put
    back. A method the object does not have, or one of another kind (not ``async def``, or not
    an async generator for a ``frames`` stage), is left alone."""
    own = getattr(chat, "__dict__", None)
    if own is None:  # no instance attributes to shadow with
        return ()
    restore = []
    for stage, name, kind in STAGES:
        inner = getattr(chat, name, None)
        if kind == CALL and inspect.iscoroutinefunction(inner):
            timed = _timed_call(inner, stage, clock)
        elif kind == FRAMES and inspect.isasyncgenfunction(inner):
            timed = _timed_frames(inner, stage, clock)
        else:
            continue
        restore.append((name, name in own, own.get(name)))
        setattr(chat, name, timed)
    return tuple(restore)


def _unstaged(chat: Any, restore: Restore) -> None:
    """Take the shadows off: the instance's own attribute back when it had one, else none."""
    for name, had_its_own, its_own in restore:
        if had_its_own:
            setattr(chat, name, its_own)
        elif name in vars(chat):
            delattr(chat, name)


def _the_turns_clock() -> Tuple[TurnClock, bool]:
    """The clock this turn is timed on, and whether the turn began it. A clock its request
    began is adopted, once (``claim``): a clock left behind in a context is never a later
    turn's. Any other turn begins its own."""
    clock = turn_clock.current()
    if clock is not None and clock.origin == turn_clock.ORIGIN_REQUEST and clock.claim():
        return clock, False
    clock = turn_clock.begin(turn_clock.ORIGIN_TURN)
    clock.claim()
    return clock, True


@dataclass(frozen=True)
class _Timed:
    """One turn's hold on its clock: what ``_timing_ends`` needs to let go of it."""

    clock: TurnClock
    mine: bool  # this turn began the clock, so it writes its line and leaves it
    previous: Optional[TurnClock]
    restore: Restore


def _timing_begins(chat: Any, kwargs: Dict[str, Any]) -> Optional[_Timed]:
    """The turn's clock from here on; None while ``CHAT_TURN_TIMING_ENABLED`` is false."""
    if not turn_clock.enabled():
        return None
    previous = turn_clock.current()
    clock, mine = _the_turns_clock()
    clock.mark(turn_clock.MARK_TURN)
    clock.note(spoken=bool(kwargs.get("spoken_mode")))
    restore = _staged(chat, clock) if turn_clock.stages_enabled() else ()
    return _Timed(clock=clock, mine=mine, previous=previous, restore=restore)


def _timing_ends(chat: Any, timed: Optional[_Timed], chat_id: Any) -> None:
    """The shadows come off. A clock the turn began gets its line and is left; a request's
    clock is neither finished nor left here (the producer finishes it: services.chat_turns)."""
    if timed is None:
        return
    _unstaged(chat, timed.restore)
    # A generator finalised from another context finds another clock (or none) there: not ours to end.
    if timed.mine and turn_clock.current() is timed.clock:
        turn_clock.finish(chat_id)
        turn_clock.leave(timed.previous)


def keeps_the_turns_timing(turn: Stream) -> Stream:
    """Wrap ``StreamingChatService._stream_response_with_agent_scoped``: the turn's clock
    starts, and its saved reply's trace says when the first token came and where the turn's
    time went. The service's own chat service and methods are put back when the turn ends."""
    @functools.wraps(turn)
    async def wrapped(chat: Any, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        start(chat)
        timed = _timing_begins(chat, kwargs)
        chats = getattr(chat, "chat_service", None)
        if chats is not None:
            chat.chat_service = _TimedChats(chats, chat)
        try:
            async for chunk in turn(chat, *args, **kwargs):
                yield chunk
        finally:
            if chats is not None:
                chat.chat_service = chats
            _timing_ends(chat, timed, args[0] if args else kwargs.get("chat_id"))
    return wrapped

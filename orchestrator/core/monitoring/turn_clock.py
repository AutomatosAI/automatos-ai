"""One clock per chat turn: where the time went between the request and the reply.

A turn's clock is carried on a ``ContextVar``, so code at any layer can mark a moment
without being handed the clock, and a task or thread started inside the turn (the detached
producer, ``asyncio.to_thread``) writes to the same clock: the context is copied, the
``TurnClock`` object is shared.

- **Marks** are moments, in whole milliseconds since the clock started. The first mark of a
  name wins.
- **Stages** are durations, summed over their calls. **Counts** are integers.
- **Facts** say which turn it was: ``spoken``, ``ref``, ``request_id``, ``outcome``.

With no clock in the context every module-level function does nothing, so a caller never
checks first. ``CHAT_TURN_TIMING_ENABLED`` false means no clock is ever begun by ``arrives``;
whoever else begins one checks ``enabled()``.

One line per turn, ``chat_turn_timing``, is written by ``finish``: ids and numbers only,
never what was said or answered.
"""
from __future__ import annotations

import logging
import threading
import time
from contextvars import ContextVar
from typing import Any, Dict, Mapping, Optional

from core.monitoring.turn_ref import REQUEST_ID_PATTERN, clean_request_id

logger = logging.getLogger(__name__)

ORIGIN_REQUEST = "request"   # started by the route's first dependency: before auth
ORIGIN_TURN = "turn"         # started by the streaming service: Auto Live, widgets, direct calls
TRACE_VERSION = 1
TIMING_LOG = "chat_turn_timing"

# Marks, in the order a turn usually takes them.
MARK_AUTHORISED = "authorised"
MARK_SAVED = "saved"
MARK_HISTORY = "history"
MARK_ROUTED = "routed"
MARK_RESPONSE_STARTED = "response_started"
MARK_PRODUCER = "producer"
MARK_LOCKED = "locked"
MARK_TURN = "turn"
MARK_AGENT = "agent"
MARK_TURN_MODEL = "turn_model"
MARK_MODEL_START = "model_start"
MARK_FIRST_TOKEN = "first_token"
MARK_FIRST_RESPONSE = "first_response"
MARK_MODEL_AGAIN = "model_again"
MARK_FIRST_TEXT = "first_text"
MARK_REPLY_SAVED = "reply_saved"
MARK_DONE = "done"

STAGE_AGENT_CONTEXT = "agent_context"
STAGE_TOOLS = "tools"
STAGE_CONTEXT = "context"
STAGE_COMPOSIO = "composio"
STAGE_KNOWLEDGE = "knowledge"
STAGE_MODEL = "model"

COUNT_MODEL_CALLS = "model_calls"

# The facts a clock keeps (``note``); any other name is ignored, so a snapshot stays small
# and JSON-safe whatever a caller passes.
FACTS = ("spoken", "ref", "request_id", "outcome")
ABSENT = "-"

_now = time.perf_counter
_clock_var: ContextVar[Optional["TurnClock"]] = ContextVar("chat_turn_clock", default=None)


class TurnClock:
    """One turn's moments on a monotonic clock (time.perf_counter), in whole milliseconds.

    Safe to write from several threads: every write takes the clock's lock and replaces the
    dict it changes, so a snapshot is never a view of live state."""

    def __init__(self, origin: str, started: Optional[float] = None) -> None:
        self.origin = origin
        self._started = _now() if started is None else started
        self._lock = threading.Lock()
        self._marks: Dict[str, int] = {}
        self._stages: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._facts: Dict[str, Any] = {}
        self._claimed = False
        self._closed = False

    def mark(self, name: str) -> bool:
        """Note ``name`` as now. The first call for a name wins; True when this call set it."""
        ms = round((_now() - self._started) * 1000)
        with self._lock:
            if name in self._marks:
                return False
            self._marks = {**self._marks, name: ms}
            return True

    def add(self, stage: str, seconds: float) -> None:
        """Add ``seconds`` to a stage's time (summed over its calls)."""
        with self._lock:
            self._stages = {**self._stages, stage: self._stages.get(stage, 0.0) + seconds}

    def count(self, name: str) -> None:
        """A counter, +1."""
        with self._lock:
            self._counts = {**self._counts, name: self._counts.get(name, 0) + 1}

    def note(self, **facts: Any) -> None:
        """Keep the facts named in ``FACTS``; a None value and any other name are ignored."""
        kept = {name: value for name, value in facts.items() if name in FACTS and value is not None}
        if not kept:
            return
        with self._lock:
            self._facts = {**self._facts, **kept}

    def claim(self) -> bool:
        """True once: the one turn that may adopt this clock."""
        with self._lock:
            first, self._claimed = not self._claimed, True
            return first

    def close(self) -> bool:
        """True once: the caller that writes the line."""
        with self._lock:
            first, self._closed = not self._closed, True
            return first

    def elapsed_seconds(self) -> float:
        return _now() - self._started

    def snapshot(self) -> Dict[str, Any]:
        """The clock as plain, new, JSON-safe dicts: ``v``, ``origin``, the facts noted,
        ``marks`` and ``stages`` in whole milliseconds, ``counts``."""
        with self._lock:
            facts, marks, stages, counts = self._facts, self._marks, self._stages, self._counts
        return {
            "v": TRACE_VERSION,
            "origin": self.origin,
            **{name: facts[name] for name in FACTS if name in facts},
            "marks": dict(marks),
            "stages": {stage: round(seconds * 1000) for stage, seconds in stages.items()},
            "counts": dict(counts),
        }


def enabled() -> bool:
    """``CHAT_TURN_TIMING_ENABLED``: false means no turn is timed at all."""
    from config import config

    return bool(config.CHAT_TURN_TIMING_ENABLED)


def stages_enabled() -> bool:
    """Timing is on and ``CHAT_TURN_TIMING_STAGES_ENABLED`` is too (the per-stage method shadows)."""
    from config import config

    return enabled() and bool(config.CHAT_TURN_TIMING_STAGES_ENABLED)


def current() -> Optional[TurnClock]:
    return _clock_var.get()


def begin(origin: str) -> TurnClock:
    """A new clock for this task's context. Read ``current()`` first to ``leave`` it later."""
    clock = TurnClock(origin)
    _clock_var.set(clock)
    return clock


def leave(previous: Optional[TurnClock]) -> None:
    """Put ``previous`` back as the context's clock. A plain set, never a token reset: an
    async generator's ``finally`` may run in another context than the one that began."""
    _clock_var.set(previous)


def mark(name: str) -> bool:
    """Mark the context's clock. No clock: False, nothing happens."""
    clock = _clock_var.get()
    return clock.mark(name) if clock is not None else False


def add(stage: str, seconds: float) -> None:
    clock = _clock_var.get()
    if clock is not None:
        clock.add(stage, seconds)


def count(name: str) -> None:
    clock = _clock_var.get()
    if clock is not None:
        clock.count(name)


def note(**facts: Any) -> None:
    clock = _clock_var.get()
    if clock is not None:
        clock.note(**facts)


def elapsed_seconds() -> Optional[float]:
    """Seconds since the context's clock started; None with no clock."""
    clock = _clock_var.get()
    return clock.elapsed_seconds() if clock is not None else None


def _token(value: Any) -> str:
    """``value`` for the log line when it is a short id-shaped string, else ``-``: nothing a
    client typed reaches the line unchecked."""
    if isinstance(value, bool):
        return str(value).lower()
    text = value if isinstance(value, str) else str(value) if value is not None else ""
    return text if REQUEST_ID_PATTERN.fullmatch(text) else ABSENT


def _pairs(values: Any) -> str:
    if not isinstance(values, Mapping):
        return ""
    return " ".join(f"{name}={int(number)}" for name, number in values.items())


def timing_line(chat_id: Any, snapshot: Mapping[str, Any]) -> str:
    """The turn's one log line: who, then marks in time order, then `` | `` stages, then
    `` | `` counts. A snapshot with no stages or no counts has no empty segment. Pure."""
    marks = snapshot.get("marks")
    in_time_order = dict(sorted(marks.items(), key=lambda item: item[1])) if isinstance(marks, Mapping) else {}
    head = (
        f"{TIMING_LOG} chat={_token(chat_id)} origin={_token(snapshot.get('origin'))}"
        f" spoken={_token(snapshot.get('spoken'))} ref={_token(snapshot.get('ref'))}"
        f" outcome={_token(snapshot.get('outcome'))}"
    )
    segments = (
        " ".join(part for part in (head, _pairs(in_time_order)) if part),
        _pairs(snapshot.get("stages")),
        _pairs(snapshot.get("counts")),
    )
    return " | ".join(segment for segment in segments if segment)


def finish(chat_id: Any, outcome: Optional[str] = None) -> None:
    """The turn is over: mark ``done``, note the outcome and, the first time only for this
    clock, write its line. No clock: nothing."""
    clock = _clock_var.get()
    if clock is None:
        return
    clock.mark(MARK_DONE)
    clock.note(outcome=outcome)
    if clock.close():
        logger.info(timing_line(chat_id, clock.snapshot()))


async def arrives() -> None:
    """FastAPI dependency, first on a route: the request's clock starts here (before auth).
    With ``CHAT_TURN_TIMING_ENABLED`` false no clock is begun."""
    if not enabled():
        return
    # Imported here: that module reads config when it is imported.
    from core.monitoring.automatos_logging import request_id_var

    begin(ORIGIN_REQUEST).note(request_id=clean_request_id(request_id_var.get("")))


async def authorised() -> None:
    """FastAPI dependency, last on a route: auth and the permission check are behind us."""
    mark(MARK_AUTHORISED)

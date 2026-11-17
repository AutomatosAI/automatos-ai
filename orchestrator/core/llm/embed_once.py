"""A turn embeds each text once (voice latency D-3).

In one chat turn the same text can reach ``EmbeddingManager.generate_embedding`` from
several places: memory L3 and L2 under one gather, then the knowledge search and the
owner's-documents leg. Each was a provider call of its own. While a turn's scope is open
(``turn_scope``, opened by ``consumers.chatbot.turn_embeddings`` when ``TURN_EMBED_ONCE``
is on), ``embed_once`` answers a repeat from the turn's memo.

Rules the callers rely on:

* **No scope, or a closed one, is exactly a plain call.** Every caller outside a chat turn
  (ingestion, a mission, a post-turn memory write that copied the turn's context) embeds
  as it always did.
* The key is (model key, the exact text). The text is never trimmed or normalised: a
  different string is a different embedding.
* A finished vector is kept while the scope is open, at most ``MAX_KEPT`` of them, and can
  be read from any loop or thread that copied the turn's context. Every caller gets a
  list of its own.
* A call in flight is shared only by callers on the same event loop, and **lives exactly
  as long as someone is waiting for it**: when its last waiter is cancelled, the call is
  cancelled (Stop ends a turn's embedding, as before).
* A caller sharing someone else's call that failed makes one call of its own before
  failing, so a provider error never reaches a caller that would have succeeded alone.
* When the scope closes the memo is closed and emptied. A call still in flight then ends
  with its waiters and is not kept.

Counts only in the log line: never a text, never a vector.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

MAX_KEPT = 32  # finished vectors a turn keeps; one is up to 4,096 floats

Key = Tuple[str, str]                             # (model key, the exact text)
Place = Tuple[asyncio.AbstractEventLoop, Key]     # a call in flight belongs to one loop
Embed = Callable[[str], Awaitable[List[float]]]


@dataclass
class _Flight:
    """One provider call in flight and how many callers are waiting for it."""

    task: "asyncio.Future[List[float]]"
    waiters: int = 0


class _Memo:
    """One turn's embeddings. The lock guards the two maps and the counts; a flight's
    ``waiters`` is only ever touched on the flight's own loop."""

    def __init__(self) -> None:
        self.done: Dict[Key, Tuple[float, ...]] = {}
        self.flying: Dict[Place, _Flight] = {}
        self.made = 0
        self.reused = 0
        self.closed = False
        self.lock = threading.Lock()

    def close(self) -> None:
        """Closed and emptied. Calls in flight are not cancelled: they end with their waiters."""
        with self.lock:
            self.closed = True
            self.done.clear()
            self.flying.clear()


_memo: contextvars.ContextVar[Optional[_Memo]] = contextvars.ContextVar("embed_once_memo", default=None)


@contextmanager
def turn_scope() -> Iterator[None]:
    """One turn's memo, from here to the exit. On exit the memo itself is closed, so work
    the turn started that outlives it (it copied this context) embeds as if there were none."""
    memo = _Memo()
    token = _memo.set(memo)
    try:
        yield
    finally:
        memo.close()
        if memo.made or memo.reused:
            logger.info("[embed-once] made=%d reused=%d", memo.made, memo.reused)
        try:
            _memo.reset(token)
        except ValueError:
            # An async generator closed from another task cannot reset a token minted in
            # its own context. Only our own (closed) memo is cleared: whatever else that
            # task's context holds is another turn's.
            if _memo.get() is memo:
                _memo.set(None)


def _forget(memo: _Memo, place: Place, flight: _Flight) -> None:
    """Take ``flight`` off the board if it is still the one at ``place``."""
    with memo.lock:
        if memo.flying.get(place) is flight:
            del memo.flying[place]


def _landed(memo: _Memo, place: Place, flight: _Flight, task: "asyncio.Future[List[float]]") -> None:
    """The call ended. Its vector is kept when the scope is still open and there is room."""
    _forget(memo, place, flight)
    if task.cancelled():
        return
    if task.exception() is not None:  # read, so it is never "never retrieved"
        return
    vector = tuple(task.result())
    with memo.lock:
        if not memo.closed and len(memo.done) < MAX_KEPT:
            memo.done[place[1]] = vector


def _board(memo: _Memo, key: Key, embed: Embed) -> Tuple[Optional[Tuple[float, ...]], Optional[_Flight], bool]:
    """What the memo has for ``key``: (a kept vector, the flight to wait on, whether this
    caller started that flight). All None/False when the memo closed in the meantime."""
    place = (asyncio.get_running_loop(), key)
    with memo.lock:
        if memo.closed:
            return None, None, False
        kept = memo.done.get(key)
        if kept is not None:
            memo.reused += 1
            return kept, None, False
        flight = memo.flying.get(place)
        mine = flight is None
        if flight is None:
            flight = _Flight(asyncio.ensure_future(embed(key[1])))
            memo.flying[place] = flight
            memo.made += 1
        else:
            memo.reused += 1
        flight.waiters += 1
    if mine:
        flight.task.add_done_callback(functools.partial(_landed, memo, place, flight))
    return None, flight, mine


async def _wait(memo: _Memo, key: Key, flight: _Flight) -> List[float]:
    """The flight's vector, a list of this caller's own. A cancelled last waiter cancels the call."""
    try:
        return list(await asyncio.shield(flight.task))
    except asyncio.CancelledError:
        if flight.waiters == 1 and not flight.task.done():
            # Nobody else is waiting: stop it, as a cancelled caller always stopped its own
            # call. Off the board first, so a later caller starts a call instead of joining
            # one that is ending.
            _forget(memo, (asyncio.get_running_loop(), key), flight)
            flight.task.cancel()
        raise
    finally:
        flight.waiters -= 1


async def embed_once(model_key: str, text: str, embed: Embed) -> List[float]:
    """``embed(text)``, made once per open turn scope for one ``model_key`` and ``text``."""
    memo = _memo.get()
    if memo is None or memo.closed:
        return await embed(text)  # no turn, or the turn is over: a plain call
    key = (model_key, text)
    kept, flight, mine = _board(memo, key, embed)
    if kept is not None:
        return list(kept)
    if flight is None:
        return await embed(text)  # the scope closed between the two reads
    if mine:
        return await _wait(memo, key, flight)
    try:
        return await _wait(memo, key, flight)
    except Exception:  # noqa: BLE001 — someone else's call failed: this caller makes one of its own, which raises its own error
        logger.debug("[embed-once] a shared call failed; this caller makes its own", exc_info=True)
        return await embed(text)


def current() -> Optional[Any]:
    """The open turn's memo in this context, or None (tests and diagnostics)."""
    return _memo.get()


__all__ = ["MAX_KEPT", "current", "embed_once", "turn_scope"]

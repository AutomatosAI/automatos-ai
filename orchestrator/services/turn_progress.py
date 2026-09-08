"""PRD-238 S4 — progress lines from inside a tool call.

A long-running platform action (``platform_wait_for_task``) needs to tell the
person watching the chat what is happening *while it runs*, but handlers know
nothing about the stream. The chat turn registers an emitter under its
``turn_id`` here; the executor hands every handler its ``_turn_id``; a handler
that has something to say calls :func:`emit`. Unknown turns are a no-op, so
headless callers (board dispatcher, heartbeats, playbooks) are unaffected.

Process-local by design: a turn's stream and its tool calls live in the same
worker. Pure asyncio, no I/O.
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

Emitter = Callable[[str], Awaitable[None]]

_emitters: Dict[str, Emitter] = {}


def register(turn_id: str, emitter: Emitter) -> None:
    """Attach the chat turn's emitter; replaces a stale one for the same id."""
    if turn_id:
        _emitters[turn_id] = emitter


def unregister(turn_id: str) -> None:
    _emitters.pop(turn_id or "", None)


def is_registered(turn_id: Optional[str]) -> bool:
    return bool(turn_id) and turn_id in _emitters


async def emit(turn_id: Optional[str], text: str) -> bool:
    """Send one progress line to the turn's stream. True when something listened."""
    emitter = _emitters.get(turn_id or "")
    if emitter is None or not text:
        return False
    try:
        await emitter(text)
        return True
    except Exception:  # noqa: BLE001 — progress is an optimisation, never a failure
        logger.debug("[turn_progress] emitter failed for turn %s", turn_id, exc_info=True)
        return False


def active_turns() -> int:
    return len(_emitters)

"""The time a chat turn took to its model's first output, kept on the turn's trace.

``/api/chat`` recorded no time to first token (widgets and voice log their own). The
turn notes when it started; ``first_token`` passes the first model call's stream
through and notes when the first frame came; ``with_first_token`` adds it to the
assembly trace the turn saves, as ``first_token_ms``. Lean turns are measured
against it (consumers.chatbot.lean_turn).
"""
from __future__ import annotations

import time
from typing import Any, AsyncIterator, Dict, Optional

FIRST_TOKEN_KEY = "first_token_ms"


def start(chat: Any) -> None:
    """Note that ``chat``'s turn starts now."""
    chat._turn_started = time.perf_counter()
    chat._first_token_ms = None


async def first_token(chat: Any, stream: AsyncIterator[Any]) -> AsyncIterator[Any]:
    """``stream`` unchanged; the first frame other than the call's final response
    sets ``chat._first_token_ms`` (once a turn)."""
    async for item in stream:
        if getattr(chat, "_first_token_ms", None) is None and not (isinstance(item, dict) and "_response" in item):
            started = getattr(chat, "_turn_started", None)
            if started is not None:
                chat._first_token_ms = round((time.perf_counter() - started) * 1000)
        yield item


def with_first_token(trace: Optional[Dict[str, Any]], chat: Any) -> Optional[Dict[str, Any]]:
    """``trace`` with the turn's ``first_token_ms``, a new dict; unchanged when unmeasured."""
    measured = getattr(chat, "_first_token_ms", None)
    if measured is None:
        return trace
    return {**(trace or {}), FIRST_TOKEN_KEY: measured}

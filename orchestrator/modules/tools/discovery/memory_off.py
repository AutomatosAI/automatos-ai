"""Auto's memory tools say memory is off when it is, never that it is empty (7 Nov 2026).

With no embedding provider the durable store refuses to embed: a store fails and a search finds
nothing, because nothing could ever be stored. On a self-hosted install the tools answered "0
results" and Auto told its owner "my memory is empty". A read that finds nothing, or a store
that failed, now asks the store why (``core.llm.embedding_status.store_cannot_embed``); when it
has no provider the answer is ``MEMORY_OFF``, with the reason code. A read that found something
is never asked, and neither is a refusal of the call's own parameters.
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Dict

from sqlalchemy.orm import Session

from core.llm.embedding_status import MEMORY_OFF, NO_EMBEDDING_PROVIDER, store_cannot_embed

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]
NOT_SAVED = "Memory NOT saved."


def memory_off_answer(lead: str = "") -> Dict[str, Any]:
    """A memory tool's answer while memory is off on this install: a NEW dict each time."""
    error = f"{lead} {MEMORY_OFF}" if lead else MEMORY_OFF
    return {"success": False, "memory_off": True, "reason_code": NO_EMBEDDING_PROVIDER, "error": error}


async def memory_is_off(service: Any) -> bool:
    """True when ``service`` (the UnifiedMemoryService) is known to have no embedding provider."""
    return await store_cannot_embed(getattr(service, "_durable", None))


def off_when_nothing_is_found(count_key: str) -> Callable[[Handler], Handler]:
    """Wrap a memory read whose answer counts what it found in ``count_key``: an answer that found
    nothing, while the store has no embedding provider, becomes ``memory_off_answer()``."""
    def wrap(handler: Handler) -> Handler:
        @functools.wraps(handler)
        async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
            answer = await handler(db, workspace_id, params)
            if not (answer.get("success") and answer.get(count_key) == 0):
                return answer
            from modules.memory.unified_memory_service import get_unified_memory_service

            return memory_off_answer() if await memory_is_off(get_unified_memory_service()) else answer
        return wrapped
    return wrap


__all__ = ["NOT_SAVED", "memory_is_off", "memory_off_answer", "off_when_nothing_is_found"]

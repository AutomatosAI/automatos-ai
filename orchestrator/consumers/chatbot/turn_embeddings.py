"""A chat turn embeds each text once (voice latency D-3).

One wrapper on ``StreamingChatService._stream_response_with_agent_scoped``. While
``TURN_EMBED_ONCE`` is true the turn runs inside ``core.llm.embed_once.turn_scope()``:
memory L3 and L2, the knowledge search and the owner's-documents leg share one embedding
of the same text instead of asking the provider each. The scope closes with the turn, so
work the turn started that outlives it embeds as it always did.

While the switch is false the turn passes straight through and no scope is opened.
"""
from __future__ import annotations

import contextlib
import functools
from typing import Any, AsyncGenerator, Callable

from config import config
from core.llm import embed_once

Stream = Callable[..., AsyncGenerator[Any, None]]


def embeds_each_text_once(turn: Stream) -> Stream:
    """Wrap the turn: with ``TURN_EMBED_ONCE`` on, its frames pass through inside one
    embedding scope. The turn is closed here, in the task that closes this wrapper, so the
    scope never closes ahead of the turn it belongs to."""
    @functools.wraps(turn)
    async def wrapped(chat: Any, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        if not config.TURN_EMBED_ONCE:
            async with contextlib.aclosing(turn(chat, *args, **kwargs)) as chunks:
                async for chunk in chunks:
                    yield chunk
            return
        with embed_once.turn_scope():
            async with contextlib.aclosing(turn(chat, *args, **kwargs)) as chunks:
                async for chunk in chunks:
                    yield chunk
    return wrapped


__all__ = ["embeds_each_text_once"]

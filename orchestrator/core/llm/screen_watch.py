"""FX-017 (night 12): what each of a chat turn's streamed calls put on the screen.

A response's ``content`` is not always what streamed: inline ``<think>`` tags are lifted
out of it once the stream ends (core/llm/reasoning.py), and the owner-words rewrite
(F264) runs per delta live and again on the whole text. A turn that needs the exact
text the owner saw sets ``WATCHER``; every streamed call through ``in_owner_words``
then reports, once it returns, its response and the text it streamed (the deltas after
every live rewrite, joined). Nothing is reported when no watcher is set.

P256-FIX-RVW-43: a reply to a re-prompt passes through the apology gate after this
(core/llm/turn_order.py), which drops the apology it opens with from the stream and
saves a copy without it. ``restated`` tells the watcher what of that call reached the
screen and which copy answers for it.
"""
from __future__ import annotations

import functools
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, List, Optional, Protocol

Delta = Callable[[str, str], Awaitable[None]]


class Watcher(Protocol):
    """Told about each streamed call once it has returned."""

    def ended(self, response: Any, said: str) -> None:
        ...

    def restated(self, response: Any, copy: Any, said: str) -> None:
        ...


WATCHER: ContextVar[Optional[Watcher]] = ContextVar("fx017_screen_watcher", default=None)


def watched(call: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Wrap ``in_owner_words(stream, messages, tools, on_delta)``: the text deltas it
    delivers are joined and reported to the turn's watcher with the response."""
    @functools.wraps(call)
    async def wrapped(stream: Any, messages: Any, tools: Any, on_delta: Delta) -> Any:
        watcher = WATCHER.get()
        if watcher is None:
            return await call(stream, messages, tools, on_delta)
        heard: List[str] = []

        async def said(kind: str, text: str) -> None:
            if kind == "text":
                heard.append(text)
            await on_delta(kind, text)

        response = await call(stream, messages, tools, said)
        watcher.ended(response, "".join(heard))
        return response
    return wrapped


def restated(response: Any, copy: Any, said: str) -> None:
    """The call that returned ``response`` put ``said`` on the screen and is saved as ``copy``."""
    watcher = WATCHER.get()
    if watcher is not None:
        watcher.restated(response, copy, said)


__all__ = ["WATCHER", "Watcher", "restated", "watched"]

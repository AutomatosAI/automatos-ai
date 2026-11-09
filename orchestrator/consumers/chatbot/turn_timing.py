"""The time a chat turn took to its model's first output, kept on the turn's trace.

``/api/chat`` recorded no time to first token (widgets and voice log their own). Two
wrappers on ``StreamingChatService`` measure it without touching the turn's own function:

- ``keeps_the_turns_timing`` (on the turn) notes when the turn starts and, for the turn,
  hands it a chat service whose assistant ``save_message`` adds ``first_token_ms`` to the
  trace it saves;
- ``records_the_first_token`` (on ``_stream_llm_call``) notes when the first frame of the
  turn's first model call came.

Lean turns are measured against it (consumers.chatbot.lean_turn, evals.lean_turn).
"""
from __future__ import annotations

import functools
import time
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Dict, Optional

FIRST_TOKEN_KEY = "first_token_ms"

Stream = Callable[..., AsyncGenerator[Any, None]]


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


def records_the_first_token(stream_llm_call: Stream) -> Stream:
    """Wrap ``StreamingChatService._stream_llm_call``: its frames pass through, timed."""
    @functools.wraps(stream_llm_call)
    async def wrapped(chat: Any, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        async for item in first_token(chat, stream_llm_call(chat, *args, **kwargs)):
            yield item
    return wrapped


class _TimedChats:
    """The turn's chat service: the service's own, except that an assistant message's
    ``context_trace`` carries the turn's ``first_token_ms`` (and any other stamp the turn adds)."""

    def __init__(self, chats: Any, chat: Any) -> None:
        self._chats = chats
        self._chat = chat

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chats, name)

    def save_message(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("role") == "assistant" and "context_trace" in kwargs:
            trace = with_first_token(kwargs["context_trace"], self._chat)
            for stamp in getattr(self._chat, "_trace_stamps", ()):
                trace = stamp(trace)
            kwargs = {**kwargs, "context_trace": trace}
        return self._chats.save_message(*args, **kwargs)


def keeps_the_turns_timing(turn: Stream) -> Stream:
    """Wrap ``StreamingChatService._stream_response_with_agent_scoped``: the turn's clock
    starts, and its saved reply's trace says when the first token came. The service's own
    chat service is put back when the turn ends."""
    @functools.wraps(turn)
    async def wrapped(chat: Any, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        start(chat)
        chats = getattr(chat, "chat_service", None)
        if chats is not None:
            chat.chat_service = _TimedChats(chats, chat)
        try:
            async for chunk in turn(chat, *args, **kwargs):
                yield chunk
        finally:
            if chats is not None:
                chat.chat_service = chats
    return wrapped

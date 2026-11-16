"""The reply's id: chosen before a chat turn streams, said on its first frame, saved under.

A client that only streams a turn through, such as a product's server on a server key, has no
other way to learn the id its reply is saved under: the reply is saved after the stream's
``finish`` frame, and the route that lists a chat's messages takes members only. It needs the id to
report the reply (``POST /api/v1/reports``). So one wrapper on
``StreamingChatService._stream_response_with_agent_scoped``, for each turn:

* chooses an id before the turn's first frame (``uuid4``, one per turn, never reused);
* says it on the turn's ``chat-id`` frame as ``messageId``, beside ``chatId``;
* saves the turn's reply under it: the turn's assistant ``save_message`` that carries a
  ``context_trace`` (the save ``turn_timing`` times), and only the first such save.

Any other message the turn saves takes its own id. That includes the note a failed turn leaves
(``turn_notes``), which has no trace. So the id a failed turn said names no message, and reporting it
finds nothing (404).

The stream's readers read ``chatId`` and ignore any other field: the web app (``lib/chat``), the
phone apps' vendored parser (a zod object, which drops unknown keys) and the product servers'
translators. A person's turn is unchanged apart from the extra field. The service is built per
turn; its own stream handler and chat service are put back when the turn ends, however it ends.
"""
from __future__ import annotations

import functools
import uuid
from typing import Any, AsyncGenerator, Callable, Mapping, Optional

ASSISTANT_ROLE = "assistant"
REPLY_TRACE_FIELD = "context_trace"

Stream = Callable[..., AsyncGenerator[Any, None]]


class _Reply:
    """One turn's reply id, and whether the reply has been saved under it."""

    def __init__(self) -> None:
        self.id = uuid.uuid4()
        self.saved = False

    def take(self) -> uuid.UUID:
        """The id, for the one save that is the reply; after it, the id is spent."""
        self.saved = True
        return self.id


def is_the_reply(kwargs: Mapping[str, Any]) -> bool:
    """Whether a ``save_message`` call saves the turn's reply: an assistant message with its trace."""
    return kwargs.get("role") == ASSISTANT_ROLE and REPLY_TRACE_FIELD in kwargs


class _ReplyFrames:
    """The turn's stream handler: the service's own, except that its ``chat-id`` frame names the reply."""

    def __init__(self, handler: Any, reply: _Reply) -> None:
        self._handler = handler
        self._reply = reply

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handler, name)

    def format_aisdk_chat_id(self, chat_id: str, message_id: Optional[str] = None) -> str:
        return self._handler.format_aisdk_chat_id(chat_id, message_id=message_id or str(self._reply.id))


class _ReplyChats:
    """The turn's chat service: the service's own, except that the reply is saved under its id."""

    def __init__(self, chats: Any, reply: _Reply) -> None:
        self._chats = chats
        self._reply = reply

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chats, name)

    def save_message(self, *args: Any, **kwargs: Any) -> Any:
        if not self._reply.saved and "message_id" not in kwargs and is_the_reply(kwargs):
            kwargs = {**kwargs, "message_id": self._reply.take()}
        return self._chats.save_message(*args, **kwargs)


def says_its_replys_id(turn: Stream) -> Stream:
    """Wrap ``StreamingChatService._stream_response_with_agent_scoped``: the turn's ``chat-id`` frame
    says the id its reply is saved under. The service's own handler and chat service come back after."""
    @functools.wraps(turn)
    async def wrapped(chat: Any, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        reply = _Reply()
        handler = getattr(chat, "streaming_handler", None)
        chats = getattr(chat, "chat_service", None)
        if handler is not None:
            chat.streaming_handler = _ReplyFrames(handler, reply)
        if chats is not None:
            chat.chat_service = _ReplyChats(chats, reply)
        try:
            async for chunk in turn(chat, *args, **kwargs):
                yield chunk
        finally:
            if handler is not None:
                chat.streaming_handler = handler
            if chats is not None:
                chat.chat_service = chats
    return wrapped


__all__ = ["is_the_reply", "says_its_replys_id"]

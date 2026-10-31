"""A tool loop that fails after it started fails its turn, like any other failed turn.

``StreamingChatService._stream_tool_loop`` catches a failure of a later round (the model
call after the first tool round, the F187/F205 re-prompts) and hands its sentence back as
the turn's answer. The turn then streamed that sentence as text and saved it as an ordinary
reply: no ⚠️, no ``turn_error`` source, stored in memory as Auto's answer, and read by the
next turn as Auto's own words, so the request before it still looked open (23 Oct: a
research request whose tool rounds failed, then "Hi").

``a_failed_loop_fails_the_turn`` wraps the loop. The loop's sentence is a
``FailedTurnSentence`` (``turn_errors``): the wrapper passes the loop's last chunk on, so the
receipts of the calls that already ran still go out (``the_loop_writes_receipts``, outside
it), then raises ``TurnFailed`` when the turn asks for the next chunk. The turn's own
``except`` streams the error frame and saves the note, as for a failure anywhere else.
"""
from __future__ import annotations

import functools
from typing import Any, AsyncGenerator, Callable, Optional

from consumers.chatbot.turn_errors import FailedTurnSentence, TurnError, TurnFailed

Stream = Callable[..., AsyncGenerator[Any, None]]
FINAL_KEY = "_final_response"  # the loop's last chunk (StreamingChatService._stream_tool_loop)


def loop_failure(chunk: Any) -> Optional[TurnError]:
    """The failure a loop chunk hands back in place of an answer, else None."""
    final = chunk.get(FINAL_KEY) if isinstance(chunk, dict) else None
    said = getattr(final, "content", None)
    if not isinstance(said, FailedTurnSentence):
        return None
    return TurnError(said.code, str(said))


def a_failed_loop_fails_the_turn(loop: Stream) -> Stream:
    """Wrap ``StreamingChatService._stream_tool_loop``: a failure it handed back as its
    answer is raised as ``TurnFailed`` once that chunk has gone on."""
    @functools.wraps(loop)
    async def wrapped(chat: Any, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        stream = loop(chat, *args, **kwargs)
        try:
            async for chunk in stream:
                yield chunk
                failed = loop_failure(chunk)
                if failed is not None:
                    raise TurnFailed(failed)
        finally:
            await stream.aclose()  # the loop is closed here, not left for the garbage collector
    return wrapped


__all__ = ["a_failed_loop_fails_the_turn", "loop_failure"]

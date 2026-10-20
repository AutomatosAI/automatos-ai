"""The chat's cut of a tool's answer: its budget, and a log of what it keeps.

PRD-157 S3: a single tool result fed back into the chat's tool loop is cut to a token budget
(it replaced the former 6000/4000-char cuts), on a token boundary, keeping the head.

P256-FIX-N17-11 (F443, night 16): the November price letter quoted the October terms after three
document searches, and whether the November sheet had been cut from an answer could not be told:
the cut left no trace. Each cut is now logged at INFO ('[tool-cut] kept N of M tokens'). The cut
itself sits inside ``StreamingChatService._stream_tool_loop``, which the code-shape gate holds
closed, so the log rides ``_track_search_spiral``, which the loop calls once per result, just
before the cut, with the same result.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, TypeVar

from core.context_guard import count_tokens

logger = logging.getLogger(__name__)

TOOL_RESULT_TOKEN_BUDGET = 2000
CUT = "[tool-cut] %s: kept %d of %d tokens"

Method = TypeVar("Method", bound=Callable[..., Any])


def logs_what_the_cut_keeps(method: Method) -> Method:
    """Wrap a chat-loop method called with ``(self, result, tool_name, ...)`` for each tool result:
    log what the cut to ``TOOL_RESULT_TOKEN_BUDGET`` keeps when the answer is over it."""
    @functools.wraps(method)
    def wrapped(self: Any, result: Dict[str, Any], tool_name: str, *args: Any, **kwargs: Any) -> Any:
        tokens = answer_tokens(result)
        if tokens > TOOL_RESULT_TOKEN_BUDGET:
            logger.info(CUT, tool_name, TOOL_RESULT_TOKEN_BUDGET, tokens)
        return method(self, result, tool_name, *args, **kwargs)
    return wrapped  # type: ignore[return-value]


def answer_tokens(result: Any) -> int:
    """The tokens of the answer the loop cuts (``llm_context``, else the raw result as text); 0 when
    it is no text or too short to be over the budget, which is never counted."""
    if not isinstance(result, dict):
        return 0
    answer = result.get("llm_context", str(result.get("raw_result", "")))
    if not isinstance(answer, str) or len(answer.encode()) <= TOOL_RESULT_TOKEN_BUDGET:   # a token is at least a byte
        return 0
    return count_tokens(answer)


__all__ = ["CUT", "TOOL_RESULT_TOKEN_BUDGET", "answer_tokens", "logs_what_the_cut_keeps"]

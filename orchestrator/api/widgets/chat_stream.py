"""
Widget chat SSE stream
======================

Turns the chunks ``StreamingChatService.stream_response_with_agent`` yields
into the Server-Sent Events the widget SDK reads. Split out of
``api/widgets/chat.py`` so each piece stays small enough to read.

The streaming service yields AI-SDK formatted chunks:

- ``0:"text"\\n``        — text content
- ``d:{...}\\n``         — data events (tool-start, finish, etc.)
- ``{"_final_response": ...}`` — internal dict (skipped)

The widget SDK expects standard SSE:
``event: message\\ndata: {"content":"text","conversation_id":"..."}\\n\\n``

PRD-264 EUAI-02a (Art. 50(1) and 50(2)): the first event of every stream is
``ai_disclosure``, sent before any token, and every assistant ``message``
event carries ``"ai_generated": true``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Mapping, Optional

from core.security.log_safe import log_safe
from modules.policy.ai_disclosure import (
    DISCLOSURE_EVENT,
    AIDisclosure,
    disclosure_event_payload,
    mark_ai_generated,
)
from modules.tools.widget_callback import (
    WIDGET_OPEN_CALLBACK_FORM_NAME,
    WIDGET_SIGNAL_KEY,
    WIDGET_SIGNAL_OPEN_CALLBACK_FORM,
)

logger = logging.getLogger(__name__)

TEXT_PREFIX = "0:"
DATA_PREFIX = "d:"
_COUNT_KEYS = (
    "total", "dict", "non_str", "text", "data_event", "other",
    "yielded_message", "yielded_tool_start", "yielded_tool_end", "text_chars",
)


def sse(event: str, payload: Mapping[str, Any]) -> str:
    """One Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _decode_text(raw: str) -> str:
    """The text of a ``0:"..."`` chunk, JSON-unescaped when it is quoted."""
    if not (raw.startswith('"') and raw.endswith('"')):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw[1:-1]


@dataclass(frozen=True)
class WidgetStreamContext:
    """What the stream needs from the request that opened it."""

    chat_id: str
    agent_id: Optional[int]
    team: Optional[str]
    is_proactive: bool
    log_extra: str
    started_at: float
    disclosure: AIDisclosure


class WidgetStreamTranslator:
    """Turns one AI-SDK chunk into the SSE frames the widget SDK reads.

    Holds the per-stream state: event counts for the closing log line, and
    the callback-form tool's arguments (PRD-008-A.2), captured at tool-start
    so ``product_context`` survives even if the LLM omits it from the result.
    """

    def __init__(self, chat_id: str, log_extra: str) -> None:
        self.chat_id = chat_id
        self.log_extra = log_extra
        self.counts: dict[str, int] = {key: 0 for key in _COUNT_KEYS}
        self._callback_form_args: dict = {}
        self._callback_form_emitted = False

    def translate(self, chunk: Any) -> list[str]:
        """The SSE frames for one chunk; internal and control chunks give none."""
        self.counts["total"] += 1
        if isinstance(chunk, dict):
            self.counts["dict"] += 1
            return []
        if not isinstance(chunk, str):
            self.counts["non_str"] += 1
            return []
        if chunk.startswith(TEXT_PREFIX):
            return self._text(chunk)
        if chunk.startswith(DATA_PREFIX):
            return self._data(chunk)
        self.counts["other"] += 1
        return []

    def _text(self, chunk: str) -> list[str]:
        self.counts["text"] += 1
        text = _decode_text(chunk[len(TEXT_PREFIX):].strip())
        if not text:
            return []
        self.counts["text_chars"] += len(text)
        self.counts["yielded_message"] += 1
        payload = mark_ai_generated({"content": text, "conversation_id": self.chat_id})
        return [sse("message", payload)]

    def _data(self, chunk: str) -> list[str]:
        self.counts["data_event"] += 1
        try:
            data = json.loads(chunk[len(DATA_PREFIX):].strip())
        except (json.JSONDecodeError, ValueError):
            return []
        inner = data.get("data") if isinstance(data, dict) else None
        if not isinstance(inner, dict):
            inner = {}
        event_type = data.get("type", "") if isinstance(data, dict) else ""
        if event_type == "tool-start":
            return self._tool_start(inner)
        if event_type == "tool-end":
            self.counts["yielded_tool_end"] += 1
            return [sse("tool-end", {"tool": inner.get("toolName") or "", "result": inner.get("result")})]
        if event_type == "tool-data":
            return self._tool_data(inner)
        return []

    def _tool_start(self, inner: dict) -> list[str]:
        self.counts["yielded_tool_start"] += 1
        tool_name = inner.get("toolName") or ""
        if tool_name == WIDGET_OPEN_CALLBACK_FORM_NAME:
            self._callback_form_args = inner.get("input") or {}
        return [sse("tool-start", {"tool": tool_name, "arguments": inner.get("input", {})})]

    def _tool_data(self, inner: dict) -> list[str]:
        """PRD-008-A.2: bridge the callback-form signal to its SSE event, once.

        PRD-141: ``product_context`` comes from the LLM tool call (tool-data
        first, tool-start input second); generic surfaces hold no vertical keys.
        """
        if self._callback_form_emitted or inner.get(WIDGET_SIGNAL_KEY) != WIDGET_SIGNAL_OPEN_CALLBACK_FORM:
            return []
        product_context = inner.get("product_context") or self._callback_form_args.get("product_context")
        logger.info(
            "%s OPEN_CALLBACK_FORM emitted (product_context=%s)",
            self.log_extra, log_safe(product_context, 120),
        )
        self._callback_form_emitted = True
        return [sse("open-callback-form", {"conversation_id": self.chat_id, "product_context": product_context})]


def _log_first_chunk(ctx: WidgetStreamContext, chunk: Any, stream_start: float) -> None:
    logger.info(
        "%s STREAM_FIRST_CHUNK: after %.0fms (type=%s, preview=%s)",
        ctx.log_extra, (time.perf_counter() - stream_start) * 1000,
        type(chunk).__name__, log_safe(repr(chunk), 100),
    )


def _log_first_text(ctx: WidgetStreamContext, before: int, after: int, stream_start: float) -> None:
    if before == 0 and after > 0:
        logger.info(
            "%s STREAM_FIRST_TEXT_YIELD: after %.0fms",
            ctx.log_extra, (time.perf_counter() - stream_start) * 1000,
        )


def _log_stream_done(ctx: WidgetStreamContext, counts: Mapping[str, int], stream_start: float) -> None:
    elapsed_ms = (time.perf_counter() - stream_start) * 1000
    total_ms = (time.perf_counter() - ctx.started_at) * 1000
    produced = counts["yielded_message"] > 0 or counts["yielded_tool_start"] > 0
    (logger.info if produced else logger.warning)(
        "%s STREAM_DONE: chat_id=%s stream=%.0fms total=%.0fms counts=%s",
        ctx.log_extra, ctx.chat_id, elapsed_ms, total_ms, dict(counts),
    )
    if counts["total"] == 0:
        logger.warning(
            "%s STREAM_EMPTY: no chunks from streaming service. Agent=%s may be misconfigured or LLM call returned nothing.",
            ctx.log_extra, ctx.agent_id,
        )
    elif counts["yielded_message"] == 0:
        logger.warning(
            "%s STREAM_NO_TEXT: agent produced %d chunks but no text (text_count=%d, data_event=%d, dict=%d). "
            "Skill prompt may be missing or agent is tool-only.",
            ctx.log_extra, counts["total"], counts["text"], counts["data_event"], counts["dict"],
        )


async def widget_event_stream(
    ctx: WidgetStreamContext,
    open_chunks: Callable[[], AsyncIterator[Any]],
) -> AsyncGenerator[str, None]:
    """The widget chat's SSE stream: the AI notice, the answer, then ``done``.

    ``open_chunks`` starts the streaming service's generator; it is called
    inside the error guard so a failure to start still ends in ``error`` and
    ``done`` events.
    """
    stream_start = time.perf_counter()
    logger.info(
        "%s STREAM_START: chat_id=%s agent_id=%s team=%s proactive=%s",
        ctx.log_extra, ctx.chat_id, ctx.agent_id, ctx.team, ctx.is_proactive,
    )
    translator = WidgetStreamTranslator(ctx.chat_id, ctx.log_extra)
    yield sse(DISCLOSURE_EVENT, disclosure_event_payload(ctx.disclosure, ctx.chat_id))
    try:
        async for chunk in open_chunks():
            if translator.counts["total"] == 0:
                _log_first_chunk(ctx, chunk, stream_start)
            texts_before = translator.counts["yielded_message"]
            frames = translator.translate(chunk)
            _log_first_text(ctx, texts_before, translator.counts["yielded_message"], stream_start)
            for frame in frames:
                yield frame
    except Exception as exc:
        logger.exception(
            "%s STREAM_ERROR: chat_id=%s after %.0fms",
            ctx.log_extra, ctx.chat_id, (time.perf_counter() - stream_start) * 1000,
        )
        yield sse("error", {"message": str(exc)})
    finally:
        _log_stream_done(ctx, translator.counts, stream_start)
        yield sse("done", {"conversation_id": ctx.chat_id})

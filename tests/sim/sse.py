"""Parser for what ``POST /api/chat`` streams back.

The route answers with the AI SDK v4 data stream (response header
``x-vercel-ai-data-stream: v1``): one frame per line, ``<prefix>:<json>``.
The prefixes that matter to a tester:

    0  text delta            9  tool call {toolCallId, toolName, args}
    2  data (list)           a  tool result {toolCallId, result}
    3  error (string)        d  finish message {finishReason, usage}
    8  message annotation    e  finish step      g  reasoning delta

``tests/api/helpers.parse_sse_response`` reads ``0:``/``2:``/``d:``/``e:`` only;
this parser keeps every frame so a scenario can assert on effects (which tools
ran, with which arguments) and on how the turn ended.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

TEXT, DATA, ERROR, ANNOTATION = "0", "2", "3", "8"
TOOL_CALL, TOOL_RESULT, FINISH_MESSAGE, FINISH_STEP, START_STEP, REASONING = "9", "a", "d", "e", "f", "g"
TOOL_CALL_START, TOOL_CALL_DELTA = "b", "c"
KNOWN = {TEXT, DATA, ERROR, ANNOTATION, TOOL_CALL, TOOL_RESULT, FINISH_MESSAGE, FINISH_STEP,
         START_STEP, REASONING, TOOL_CALL_START, TOOL_CALL_DELTA}
CHAT_ID_KEYS = ("chatId", "chat_id", "sessionId", "session_id")


@dataclass(frozen=True)
class ChatTurn:
    text: str
    chat_id: str | None
    tool_calls: tuple[dict[str, Any], ...] = ()
    tool_results: tuple[dict[str, Any], ...] = ()
    errors: tuple[str, ...] = ()
    finish: dict[str, Any] | None = None
    reasoning: str = ""
    data: tuple[Any, ...] = ()
    frames: int = 0
    unparsed: int = 0

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(str(call.get("toolName") or call.get("name") or "?") for call in self.tool_calls)

    @property
    def usage(self) -> dict[str, Any] | None:
        return (self.finish or {}).get("usage")


@dataclass
class _Acc:
    text: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    data: list[Any] = field(default_factory=list)
    finish: dict[str, Any] | None = None
    chat_id: str | None = None
    frames: int = 0
    unparsed: int = 0


def find_chat_id(payload: Any) -> str | None:
    """A chat id hiding in a data frame or annotation, whatever it is called."""
    if isinstance(payload, dict):
        for key in CHAT_ID_KEYS:
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        for value in payload.values():
            found = find_chat_id(value)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = find_chat_id(item)
            if found:
                return found
    return None


def _absorb(acc: _Acc, prefix: str, payload: Any) -> None:
    if prefix == TEXT and isinstance(payload, str):
        acc.text.append(payload)
    elif prefix == REASONING and isinstance(payload, str):
        acc.reasoning.append(payload)
    elif prefix == ERROR:
        acc.errors.append(payload if isinstance(payload, str) else json.dumps(payload))
    elif prefix == TOOL_CALL and isinstance(payload, dict):
        acc.tool_calls.append(payload)
    elif prefix == TOOL_RESULT and isinstance(payload, dict):
        acc.tool_results.append(payload)
    elif prefix == FINISH_MESSAGE and isinstance(payload, dict):
        acc.finish = payload
    elif prefix in (DATA, ANNOTATION):
        acc.data.append(payload)
        acc.chat_id = acc.chat_id or find_chat_id(payload)
        if isinstance(payload, dict) and payload.get("type") == "error":
            acc.errors.append(json.dumps(payload))
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and item.get("type") == "error":
                    acc.errors.append(json.dumps(item))


def parse_data_stream(body: str, header_chat_id: str | None = None) -> ChatTurn:
    """Parse a whole stream body. Lines that are not ``<prefix>:<json>`` count as unparsed."""
    acc = _Acc(chat_id=header_chat_id or None)
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("data:"):  # tolerate a plain SSE envelope
            line = line[5:].strip()
        prefix, sep, rest = line.partition(":")
        if not sep or prefix not in KNOWN:
            acc.unparsed += 1
            continue
        try:
            payload = json.loads(rest)
        except json.JSONDecodeError:
            acc.unparsed += 1
            continue
        acc.frames += 1
        _absorb(acc, prefix, payload)
    return ChatTurn(
        text="".join(acc.text), chat_id=acc.chat_id, tool_calls=tuple(acc.tool_calls),
        tool_results=tuple(acc.tool_results), errors=tuple(acc.errors), finish=acc.finish,
        reasoning="".join(acc.reasoning), data=tuple(acc.data), frames=acc.frames, unparsed=acc.unparsed,
    )

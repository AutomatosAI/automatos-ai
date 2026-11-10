"""Stand-ins for the Anthropic client's streamed call (voice latency B).

Two ways to drive a real ``AnthropicProvider`` without a network:

* ``provider_on`` puts it on a stand-in client whose ``messages.stream`` returns a fake of the
  SDK's stream manager (``FakeStream`` iterates recorded events, then holds the accumulated
  message) and whose ``messages.create`` returns a whole reply.
* ``sdk_provider`` puts it on the real SDK client over a mock transport that answers a streamed
  request with recorded server-sent events and a whole one with the same reply as JSON.

``streamed_call`` and ``interrupted_call`` run one streamed call and give back what the listener
heard beside the reply (or the error).

``manager_on`` puts either provider inside a real ``LLMManager`` (B-3: the manager asks the
provider whether a call streams). ``sdk_provider_in_rounds`` answers a turn's model calls, one
recorded round (``recorded_round``) each, in order.
"""
from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import httpx
import pytest

from core.llm.clients.anthropic_client import AnthropicProvider
from core.llm.clients.anthropic_stream import AnthropicStreamInterrupted
from core.llm.clients.base import LLMConfig, LLMProvider


MODEL = "claude-haiku-5-5"
CLIENT_LOGGER = "core.llm.clients.anthropic_client"
STREAM_LOGGER = "core.llm.clients.anthropic_stream"
CACHED_PREFIX = "You are Auto. "
SYSTEM = CACHED_PREFIX + "Today is Tuesday."
MESSAGES = [{"role": "system", "content": SYSTEM, "cache_prefix": CACHED_PREFIX},
            {"role": "user", "content": "What is open?"}]
TOOLS = [{"type": "function", "function": {"name": "platform_list_tasks", "description": "List the tasks",
                                           "parameters": {"type": "object", "properties": {}}}}]
TOOL_INPUT = {"status": "open", "limit": 3}
CONFIG_MAX_TOKENS = 8000
CALL_BUDGET = 1024
WAIT_SECONDS = 5.0
FAILOVER_MODEL = "claude-sonnet-5"
DROPPED = "the connection dropped"
OVERLOADED = "{'type': 'error', 'error': {'type': 'overloaded_error'}, 'request_id': 'req_404abc'}"
USAGE_AS_BOOKED = {"prompt_tokens": 1010, "completion_tokens": 5, "total_tokens": 1015,
                   "cache_read_tokens": 900, "cache_write_tokens": 100}


def usage_counts():
    return NS(input_tokens=10, output_tokens=5, cache_read_input_tokens=900, cache_creation_input_tokens=100)


def text_of(text):
    """The SDK's text event, and (the same shape) a text block of a message."""
    return NS(type="text", text=text)


def tool_call(call_id, name="platform_list_tasks"):
    return NS(type="tool_use", id=call_id, name=name, input=dict(TOOL_INPUT))


def reply_message(*blocks, stop_reason="end_turn"):
    """A Messages API reply, in the shape the existing tests give ``messages.create``."""
    return NS(content=list(blocks), usage=usage_counts(), model=MODEL, stop_reason=stop_reason)


class FakeStream:
    """The SDK's ``MessageStream``: iterates recorded events, then holds the accumulated message.
    An exception among the events is raised there; a ``threading.Event`` is waited for there (a
    stream with nothing to read yet). ``read`` counts the events handed over."""

    def __init__(self, events, final=None, *, snapshot=None):
        self.events, self.final = list(events), final
        self.snapshot = final if snapshot is None else snapshot
        self.read, self.exited = 0, False

    def __iter__(self):
        for event in self.events:
            if isinstance(event, BaseException):
                raise event
            if isinstance(event, threading.Event):
                event.wait(WAIT_SECONDS)
                continue
            self.read += 1
            yield event

    def get_final_message(self):
        return self.final

    @property
    def current_message_snapshot(self):
        if isinstance(self.snapshot, BaseException):
            raise self.snapshot
        return self.snapshot


class FakeStreamManager:
    """What ``client.messages.stream(...)`` returns: entering it makes the request."""

    def __init__(self, stream, refusal=None):
        self.stream, self.refusal = stream, refusal

    def __enter__(self):
        if self.refusal is not None:
            raise self.refusal
        return self.stream

    def __exit__(self, *exc_info):
        self.stream.exited = True
        return False


class Refused(Exception):
    """An SDK refusal before the stream opens: a rate limit."""
    status_code = 429


class ProviderError(Exception):
    """A provider's own mid-stream error, as the SDK raises it (its class lives in ``anthropic``)."""
    __module__ = "anthropic"
    status_code = 200


def provider_on(stream=None, whole=None, *, refusal=None):
    """A real ``AnthropicProvider`` on a stand-in client (the pattern of
    ``tests/test_anthropic_sampling_params.py``): ``stream`` answers ``messages.stream``,
    ``whole`` answers ``messages.create``."""
    made = AnthropicProvider.__new__(AnthropicProvider)  # bypass _initialize_client
    made.config = LLMConfig(provider=LLMProvider.ANTHROPIC, model=MODEL, max_tokens=CONFIG_MAX_TOKENS,
                            temperature=0.7)
    made.client = MagicMock()
    made.client.messages.stream.return_value = FakeStreamManager(stream, refusal)
    made.client.messages.create.return_value = whole
    return made


def manager_on(provider, monkeypatch):
    """A real ``LLMManager`` around ``provider``, built the way
    ``tests/test_prd238_w2_two_channels.py`` builds it (no ``__init__``, so no settings read). The
    routing, the owner-words rewrite, the Screen report and the opening gate are the real ones;
    what is stubbed is the bookkeeping around a call: the usage row, the credit gate and the
    credit-outage note, which read the database."""
    from core.llm.manager import LLMManager

    made = object.__new__(LLMManager)
    made.provider, made.config = provider, provider.config
    for bookkeeping in ("_ensure_provider_initialized", "_track_usage", "_credit_gate", "_note_success"):
        monkeypatch.setattr(made, bookkeeping, lambda *args, **kwargs: None, raising=False)
    return made


def listener():
    heard = []

    async def on_delta(kind, text):
        heard.append((kind, text))

    return heard, on_delta


def streamed_call(provider, messages=MESSAGES, tools=None):
    """One streamed call: (the reply, what the listener heard in order)."""
    heard, on_delta = listener()
    return asyncio.run(provider.stream_response(messages, tools, on_delta=on_delta)), heard


def interrupted_call(provider):
    """One streamed call that breaks after it opened: (the error, what the listener heard)."""
    heard, on_delta = listener()
    with pytest.raises(AnthropicStreamInterrupted) as caught:
        asyncio.run(provider.stream_response(MESSAGES, None, on_delta=on_delta))
    return caught.value, heard


def lines_of(caplog, logger_name):
    return [record.getMessage() for record in caplog.records if record.name == logger_name]


# --------------------------------------------------------------------------- the real SDK over recorded events


SDK_INPUT = {"input_tokens": 10, "cache_read_input_tokens": 900, "cache_creation_input_tokens": 100}
SDK_OUTPUT_TOKENS = 25
SDK_TEXT = ("Let me ", "check that.")
SDK_TOOL_JSON = ('{"status": "op', 'en", "limit": 3}')
SDK_USAGE_AS_BOOKED = {"prompt_tokens": 1010, "completion_tokens": SDK_OUTPUT_TOKENS, "total_tokens": 1035,
                       "cache_read_tokens": 900, "cache_write_tokens": 100}
SDK_PARTIAL_USAGE = {"prompt_tokens": 1010, "completion_tokens": 1, "total_tokens": 1011,
                     "cache_read_tokens": 900, "cache_write_tokens": 100}

_MESSAGE_START = ("message_start", {"type": "message_start", "message": {
    "id": "msg_1", "type": "message", "role": "assistant", "model": MODEL, "content": [],
    "stop_reason": None, "stop_sequence": None, "usage": {**SDK_INPUT, "output_tokens": 1}}})
_TEXT_OPENS = ("content_block_start", {"type": "content_block_start", "index": 0,
                                       "content_block": {"type": "text", "text": ""}})
_PING = ("ping", {"type": "ping"})


def _text_delta(text):
    return ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                    "delta": {"type": "text_delta", "text": text}})


def _json_delta(fragment):
    return ("content_block_delta", {"type": "content_block_delta", "index": 1,
                                    "delta": {"type": "input_json_delta", "partial_json": fragment}})


def _block_stops(index):
    return ("content_block_stop", {"type": "content_block_stop", "index": index})


_TOOL_OPENS = ("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {
    "type": "tool_use", "id": "toolu_1", "name": "platform_list_tasks", "input": {}}})
_MESSAGE_ENDS = (
    ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                       "usage": {"output_tokens": SDK_OUTPUT_TOKENS}}),
    ("message_stop", {"type": "message_stop"}),
)
_ERROR = ("error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})

A_WHOLE_STREAM = (_MESSAGE_START, _TEXT_OPENS, _PING, _text_delta(SDK_TEXT[0]), _text_delta(SDK_TEXT[1]),
                  _block_stops(0), _TOOL_OPENS, _json_delta(SDK_TOOL_JSON[0]), _json_delta(SDK_TOOL_JSON[1]),
                  _block_stops(1), *_MESSAGE_ENDS)
AN_ERROR_AFTER_TEXT = (_MESSAGE_START, _TEXT_OPENS, _text_delta(SDK_TEXT[0]), _ERROR)
CUT_SHORT_AFTER_TEXT = (_MESSAGE_START, _TEXT_OPENS, _text_delta(SDK_TEXT[0]))
THE_SAME_REPLY_WHOLE = {
    "id": "msg_1", "type": "message", "role": "assistant", "model": MODEL,
    "content": [{"type": "text", "text": "".join(SDK_TEXT)},
                {"type": "tool_use", "id": "toolu_1", "name": "platform_list_tasks", "input": TOOL_INPUT}],
    "stop_reason": "tool_use", "stop_sequence": None, "usage": {**SDK_INPUT, "output_tokens": SDK_OUTPUT_TOKENS}}


def sse_bytes(events):
    """Recorded server-sent events as the bytes the Messages API streams."""
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode("utf-8")


def sdk_provider(events=A_WHOLE_STREAM):
    """A real ``AnthropicProvider`` on the real SDK client over a mock transport (the way
    ``tests/test_1178_request_log_guard.py`` builds it): a streamed request is answered with the
    recorded events, a whole one with the same reply as JSON. ``made.requests`` keeps the bodies."""
    import anthropic

    requests = []

    def answer(request):
        body = json.loads(request.content)
        requests.append(body)
        if body.get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse_bytes(events))
        return httpx.Response(200, json=THE_SAME_REPLY_WHOLE)

    made = AnthropicProvider.__new__(AnthropicProvider)  # bypass _initialize_client
    made.config = LLMConfig(provider=LLMProvider.ANTHROPIC, model=MODEL, max_tokens=CALL_BUDGET, temperature=0.7)
    made.client = anthropic.Anthropic(api_key="test-key", http_client=httpx.Client(
        transport=httpx.MockTransport(answer)), max_retries=0)
    made.requests = requests
    return made


# --------------------------------------------------------------------------- a turn's rounds, recorded


def _tool_events(index, tool):
    """One tool call's events at content block ``index``; ``tool`` is (id, name, input)."""
    call_id, name, tool_input = tool
    return (
        ("content_block_start", {"type": "content_block_start", "index": index, "content_block": {
            "type": "tool_use", "id": call_id, "name": name, "input": {}}}),
        ("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {
            "type": "input_json_delta", "partial_json": json.dumps(tool_input)}}),
        _block_stops(index),
    )


def recorded_round(*pieces, tool=None):
    """One model round as recorded server-sent events: its text in the ``pieces`` given, then
    (``tool``: id, name, input) one tool call. It ends ``tool_use`` with a call, else ``end_turn``."""
    text = (_TEXT_OPENS, *(_text_delta(piece) for piece in pieces), _block_stops(0)) if pieces else ()
    call = _tool_events(1 if pieces else 0, tool) if tool else ()
    ends = ("message_delta", {"type": "message_delta", "usage": {"output_tokens": SDK_OUTPUT_TOKENS}, "delta": {
        "stop_reason": "tool_use" if tool else "end_turn", "stop_sequence": None}})
    return (_MESSAGE_START, *text, *call, ends, ("message_stop", {"type": "message_stop"}))


NOT_SCRIPTED = {"type": "error", "error": {"type": "invalid_request_error", "message": "not scripted"}}


def sdk_provider_in_rounds(*rounds):
    """A real ``AnthropicProvider`` on the real SDK client over a mock transport: the n-th
    streamed request is answered with the n-th recorded round. A request past the script, or a
    whole (not streamed) one, is refused (400), so a turn that asks more than was scripted, or
    that does not stream, fails. ``made.requests`` keeps the bodies; ``made.unasked`` the rounds
    nobody asked for."""
    import anthropic

    unasked, requests = list(rounds), []

    def answer(request):
        body = json.loads(request.content)
        requests.append(body)
        if not body.get("stream") or not unasked:
            return httpx.Response(400, json=NOT_SCRIPTED)
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=sse_bytes(unasked.pop(0)))

    made = AnthropicProvider.__new__(AnthropicProvider)  # bypass _initialize_client
    made.config = LLMConfig(provider=LLMProvider.ANTHROPIC, model=MODEL, max_tokens=CALL_BUDGET, temperature=0.7)
    made.client = anthropic.Anthropic(api_key="test-key", http_client=httpx.Client(
        transport=httpx.MockTransport(answer)), max_retries=0)
    made.requests, made.unasked = requests, unasked
    return made

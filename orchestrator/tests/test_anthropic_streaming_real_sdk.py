"""Anthropic streaming (voice latency B-2) through the real SDK, over recorded server-sent events.

The fakes in ``tests/test_anthropic_streaming.py`` cannot catch a change of the SDK's shape; these
can. The client is the real ``anthropic.Anthropic`` on a mock transport
(``tests/helpers_anthropic_stream.sdk_provider``): a streamed request is answered with recorded
events (``message_start`` with input and cache usage, text deltas, a ``ping``, a ``tool_use`` block
in two JSON fragments, ``message_delta`` with the stop reason and the output count,
``message_stop``), a whole one with the same reply as JSON.

* Text arrives in its recorded pieces; the tool call's arguments are the joined fragments; the
  usage holds the recorded input, output and cache counts.
* The streamed reply equals the whole call's reply but for ``streamed``, and so do the two requests
  but for ``stream``.
* An ``error`` event after text, and a body that ends with no ``message_delta``, are each an
  interruption with the fixed sentence and the input and cache counts that had arrived.
* The SDK still has what the reader uses.

No database, no model and no socket. A mock transport's body is read whole before the SDK sees
it, which is fine here (the order of events is what is tested); a connection that drops mid-body
is covered with the fake stream.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json

import pytest

from config import config
from core.llm.clients import anthropic_stream
from core.llm.clients.anthropic_stream import INTERRUPTED, NO_STOP_REASON
from tests.helpers_anthropic_stream import (
    AN_ERROR_AFTER_TEXT, CUT_SHORT_AFTER_TEXT, MESSAGES, MODEL, SDK_PARTIAL_USAGE, SDK_TEXT, SDK_TOOL_JSON,
    SDK_USAGE_AS_BOOKED, TOOL_INPUT, TOOLS, interrupted_call, sdk_provider, streamed_call,
)


@pytest.fixture(autouse=True)
def _the_switch_lets_every_call_stream(monkeypatch):
    """The client offers its streamed call only when ``ANTHROPIC_STREAMING`` covers the call."""
    monkeypatch.setattr(config, "ANTHROPIC_STREAMING", "all")


def test_the_real_sdk_streams_text_and_tool_calls():
    provider = sdk_provider()

    response, heard = streamed_call(provider, tools=TOOLS)

    assert heard == [("text", SDK_TEXT[0]), ("text", SDK_TEXT[1])]
    assert response.content == "".join(SDK_TEXT) and response.streamed is True
    (call,) = response.tool_calls
    assert (call["id"], call["function"]["name"]) == ("toolu_1", "platform_list_tasks")
    assert json.loads(call["function"]["arguments"]) == json.loads("".join(SDK_TOOL_JSON)) == TOOL_INPUT
    assert (response.finish_reason, response.model) == ("tool_use", MODEL)
    assert response.usage == SDK_USAGE_AS_BOOKED
    assert [body.get("stream") for body in provider.requests] == [True]


def test_the_real_sdk_and_the_whole_call_agree():
    provider = sdk_provider()

    whole = asyncio.run(provider.generate_response(MESSAGES, TOOLS))
    streamed, _ = streamed_call(provider, tools=TOOLS)

    assert dataclasses.replace(streamed, streamed=False) == whole
    asked_whole, asked_streamed = provider.requests
    assert {**asked_streamed, "stream": None} == {**asked_whole, "stream": None}
    assert "stream" not in asked_whole or asked_whole["stream"] is False


def test_a_real_error_event_mid_stream_is_an_interruption():
    import anthropic

    provider = sdk_provider(AN_ERROR_AFTER_TEXT)

    error, heard = interrupted_call(provider)

    assert heard == [("text", SDK_TEXT[0])]
    assert str(error) == INTERRUPTED
    assert isinstance(error.__cause__, anthropic.APIError) and "overloaded_error" in str(error.__cause__)
    assert error.partial_response.content == SDK_TEXT[0]
    assert error.partial_response.usage == SDK_PARTIAL_USAGE
    assert [body.get("stream") for body in provider.requests] == [True]     # not asked again: text was heard


def test_a_real_stream_cut_short_is_an_interruption():
    error, heard = interrupted_call(sdk_provider(CUT_SHORT_AFTER_TEXT))

    assert heard == [("text", SDK_TEXT[0])]
    assert error.why == NO_STOP_REASON
    assert error.partial_response.content == SDK_TEXT[0]
    assert error.partial_response.usage == SDK_PARTIAL_USAGE


def test_the_sdk_has_what_the_reader_uses():
    from anthropic.lib.streaming import MessageStream, TextEvent

    assert hasattr(sdk_provider().client.messages, "stream")
    assert TextEvent.construct(type="text", text="x", snapshot="x").type == anthropic_stream.TEXT_EVENT
    assert callable(MessageStream.get_final_message)
    assert isinstance(MessageStream.current_message_snapshot, property)

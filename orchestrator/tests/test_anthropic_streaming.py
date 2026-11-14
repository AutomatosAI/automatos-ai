"""Anthropic streaming (voice latency B): the switch and the spoken-surface flag (B-1), and the
client's streamed call (B-2).

``ANTHROPIC_STREAMING`` chooses which turns' Anthropic calls hand their text over as it is
generated: ``off`` (the default), ``quick-ask``, ``spoken`` or ``all``. The client knows a turn's
surface from a context variable (``core.llm.request_scope``), set where the turn starts.

* The setting is ``off`` by default and is named in the config surface.
* Outside any turn the flag is ``""``. ``set_spoken_turn`` hands back what was there before.
* A turn sent with ``spoken_mode`` says ``quick-ask`` to its model calls, also from a task the turn
  creates (the service's model calls run in tasks). A turn that is read says ``""``.
* A spoken turn started inside Auto Live stays ``live``; a turn that is read inside it says ``""``.
* The flag is put back when the turn ends: exhausted, raised, or closed early. A service built
  without an agent factory is marked and put back the same way.
* Auto Live marks its turn ``live`` around its frame loop and puts the flag back in a ``finally``.

B-2, ``AnthropicProvider.stream_response`` and ``streams_now``:

* Text reaches the listener as it comes, and the reply is the one the whole call returns (text,
  tool calls, stop reason, usage with the cache counts, model) but for ``streamed``. The request
  is the whole call's request, with this call's budget.
* A refusal before the stream opens is the SDK's own error. A stream that breaks before any text
  is asked again whole, once; after text it fails the call with one fixed sentence that says
  nothing of the provider's. A reader nobody listens to any more stops and closes its stream.
* ``streams_now`` follows ``ANTHROPIC_STREAMING`` and the turn's surface; a headless run never
  streams. The client offers ``stream_response`` only to a call that streams: to any other it is
  None, so a caller that looks for it answers whole. The B-2 cases below run with the switch on
  ``all`` (``_the_switch_lets_every_call_stream``); a case about another value sets its own.

B-4, a failed call's row in the usage ledger (``LLMManager._call_failed``):

* A stream that broke after text is booked as an error with what it had spent: the input and
  cache tokens, and the output count so far (a lower bound). The ask's running cost counts them.
* Any other failed call books zeros, as before: a whole call that raised, a refusal before the
  stream opened, and a broken stream that left no readable snapshot.

The manager's choice between the two calls (B-3) is ``tests/test_anthropic_streaming_manager.py``,
and a whole spoken turn ``tests/test_anthropic_streaming_spoken_turn.py``.

The stand-ins are in ``tests/helpers_anthropic_stream.py``. The same call through the real SDK,
over recorded server-sent events, is ``tests/test_anthropic_streaming_real_sdk.py``.

No database, no model, no SDK call and no socket.
"""
from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import logging
import threading
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from config import config
from consumers.chatbot import spoken_model as sm
from core.llm import request_scope
from core.llm.clients.anthropic_client import AnthropicProvider
from core.llm.clients.anthropic_stream import INTERRUPTED, NO_STOP_REASON, AnthropicStreamInterrupted, read_stream
from core.llm.request_scope import LIVE, QUICK_ASK, set_spoken_turn, spoken_turn
from core.llm.usage_counts import UsageCounts
from tests.helpers_anthropic_stream import (
    CACHED_PREFIX, CALL_BUDGET, CLIENT_LOGGER, CONFIG_MAX_TOKENS, DROPPED, FAILOVER_MODEL, MESSAGES, MODEL,
    OVERLOADED, STREAM_LOGGER, SYSTEM, TOOL_INPUT, TOOLS, USAGE_AS_BOOKED, WAIT_SECONDS, FakeStream,
    FakeStreamManager, ProviderError, Refused, interrupted_call, lines_of, listener, provider_on, reply_message,
    streamed_call, text_of, tool_call, usage_counts as sdk_usage,
)

ORCH = Path(__file__).resolve().parents[1]
AUTO = 337
USER_PK = 7
CHAT_ID = "chat-1"
ASKED = "Read me what needs me today"
NO_SURFACE = ""
BROKE = "the turn broke"


def _chat(*, factory=True):
    """The service the wrapper is given: with a stand-in agent factory, or built without one."""
    return NS(agent_factory=NS(name="the service's own factory") if factory else None)


async def _surface_in_a_task():
    """The surface a model call sees: the service makes its calls in tasks the turn creates."""
    async def _model_call():
        return spoken_turn()

    return await asyncio.create_task(_model_call())


async def _stub_turn(chat, chat_id, messages, agent_id, user_id, **kwargs):
    """A stand-in turn of two frames: each is the surface a task created by the turn sees."""
    yield await _surface_in_a_task()
    yield await _surface_in_a_task()


async def _raising_turn(chat, chat_id, messages, agent_id, user_id, **kwargs):
    yield await _surface_in_a_task()
    raise RuntimeError(BROKE)


def _wrapped_turn(turn=_stub_turn, *, chat=None, **kwargs):
    """One turn through ``answers_on_the_turns_model``, as an async generator not yet started."""
    wrapped = sm.answers_on_the_turns_model(turn)
    messages = [{"role": "user", "content": ASKED}]
    return wrapped(chat or _chat(), CHAT_ID, messages, AUTO, USER_PK, **kwargs)


async def _surfaces(**kwargs):
    return [surface async for surface in _wrapped_turn(**kwargs)]


# --------------------------------------------------------------------------- the switch


def test_the_switch_is_off_by_default_and_named_in_the_config_surface(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_STREAMING", raising=False)
    source = (ORCH / "config.py").read_text(encoding="utf-8")
    surface = json.loads((ORCH / "reports" / "config-surface.json").read_text(encoding="utf-8"))

    assert 'ANTHROPIC_STREAMING: str = os.getenv("ANTHROPIC_STREAMING", "off").strip().lower()' in source
    assert isinstance(config.ANTHROPIC_STREAMING, str)
    assert "ANTHROPIC_STREAMING" in json.dumps(surface)


# --------------------------------------------------------------------------- the flag


def test_no_surface_outside_a_turn_and_set_hands_back_what_was_there():
    async def run():
        seen = [spoken_turn()]
        seen.append(set_spoken_turn(LIVE))
        seen.append(spoken_turn())
        seen.append(set_spoken_turn(None))
        seen.append(spoken_turn())
        return seen

    assert asyncio.run(run()) == [NO_SURFACE, NO_SURFACE, LIVE, LIVE, NO_SURFACE]
    assert (QUICK_ASK, LIVE) == ("quick-ask", "live")


def test_a_quick_ask_says_so_to_its_model_calls():
    assert asyncio.run(_surfaces(spoken_mode=True)) == [QUICK_ASK, QUICK_ASK]
    assert asyncio.run(_surfaces()) == [NO_SURFACE, NO_SURFACE]
    assert asyncio.run(_surfaces(spoken_mode=False)) == [NO_SURFACE, NO_SURFACE]


def test_an_auto_live_turn_stays_live():
    async def run(**kwargs):
        set_spoken_turn(LIVE)
        return await _surfaces(**kwargs)

    assert asyncio.run(run(spoken_mode=True)) == [LIVE, LIVE]
    assert asyncio.run(run()) == [NO_SURFACE, NO_SURFACE]


@pytest.mark.parametrize("outside", [NO_SURFACE, LIVE])
@pytest.mark.parametrize("spoken", [True, False])
def test_the_flag_is_put_back_when_the_turn_ends(outside, spoken):
    async def run():
        set_spoken_turn(outside)
        await _surfaces(spoken_mode=spoken)
        exhausted = spoken_turn()

        turn = _wrapped_turn(spoken_mode=spoken)
        await turn.__anext__()
        await turn.aclose()
        closed_early = spoken_turn()

        with pytest.raises(RuntimeError, match=BROKE):
            [surface async for surface in _wrapped_turn(_raising_turn, spoken_mode=spoken)]
        return exhausted, closed_early, spoken_turn()

    assert asyncio.run(run()) == (outside, outside, outside)


def test_a_service_without_a_factory_is_marked_and_put_back_too():
    async def run():
        set_spoken_turn(LIVE)
        chat = _chat(factory=False)
        surfaces = [surface async for surface in _wrapped_turn(chat=chat, spoken_mode=True)]
        typed = [surface async for surface in _wrapped_turn(chat=chat)]
        return surfaces, typed, spoken_turn(), chat.agent_factory

    assert asyncio.run(run()) == ([LIVE, LIVE], [NO_SURFACE, NO_SURFACE], LIVE, None)


def test_the_services_own_factory_is_back_after_a_marked_turn():
    async def run():
        chat = _chat()
        own = chat.agent_factory
        await _surfaces(chat=chat, spoken_mode=True)
        return chat.agent_factory is own

    assert asyncio.run(run()) is True


# --------------------------------------------------------------------------- Auto Live


def test_auto_live_marks_its_turn():
    from api import voice_retell

    source = inspect.getsource(voice_retell._agent_retell_stream_inner)
    marked = source.index("before = set_spoken_turn(LIVE)")
    frames = source.index("retell_response_frames(req.response_id, tap)")
    put_back = source.index("set_spoken_turn(before)")

    assert marked < frames < put_back
    assert source[frames:put_back].rstrip().endswith("finally:")
    assert voice_retell.set_spoken_turn is request_scope.set_spoken_turn
    assert voice_retell.LIVE == LIVE


# =========================================================================== B-2: the streamed call

@pytest.fixture(autouse=True)
def _the_switch_lets_every_call_stream(monkeypatch):
    """The client offers its streamed call only when ``ANTHROPIC_STREAMING`` covers the call; the
    code default is ``off``. A test about another value sets it itself, after this."""
    monkeypatch.setattr(config, "ANTHROPIC_STREAMING", "all")


def _text_only():
    return reply_message(text_of("You have three open tasks."))


def _text_then_a_tool():
    return reply_message(text_of("Let me check that."), tool_call("toolu_1"), stop_reason="tool_use")


def _two_tools_and_no_text():
    return reply_message(tool_call("toolu_1"), tool_call("toolu_2", "platform_list_people"), stop_reason="tool_use")


def _text_events(message):
    return [text_of(block.text) for block in message.content if block.type == "text"]


# --------------------------------------------------------------------------- text, reply, request


def test_text_reaches_the_listener_as_it_comes():
    stream = FakeStream([text_of("You have "), text_of("three open tasks.")], _text_only())
    provider = provider_on(stream)

    response, heard = streamed_call(provider)

    assert heard == [("text", "You have "), ("text", "three open tasks.")]
    assert response.content == "You have three open tasks."
    assert response.streamed is True
    assert stream.exited is True
    provider.client.messages.create.assert_not_called()


@pytest.mark.parametrize("final", [_text_only, _text_then_a_tool, _two_tools_and_no_text])
def test_the_streamed_reply_is_the_reply_the_whole_call_returns(final):
    message = final()
    whole = asyncio.run(provider_on(whole=message).generate_response(MESSAGES, TOOLS))

    streamed, heard = streamed_call(provider_on(FakeStream(_text_events(message), message)), tools=TOOLS)

    assert dataclasses.replace(streamed, streamed=False) == whole
    assert streamed.streamed is bool(heard) and whole.streamed is False
    assert (streamed.model, streamed.provider, streamed.finish_reason) == (MODEL, "anthropic", message.stop_reason)
    assert streamed.usage == USAGE_AS_BOOKED
    calls = [block for block in message.content if block.type == "tool_use"]
    assert [call["function"]["arguments"] for call in streamed.tool_calls or []] == [
        json.dumps(TOOL_INPUT) for _ in calls]
    assert [call["id"] for call in streamed.tool_calls or []] == [block.id for block in calls]


def test_a_round_of_tool_calls_only_is_not_marked_streamed():
    message = _two_tools_and_no_text()

    response, heard = streamed_call(provider_on(FakeStream([], message)), tools=TOOLS)

    assert heard == []
    assert response.streamed is False
    assert response.finish_reason == "tool_use"
    assert [call["function"]["name"] for call in response.tool_calls] == [
        "platform_list_tasks", "platform_list_people"]


def test_usage_carries_the_cache_counts():
    response, _ = streamed_call(provider_on(FakeStream([text_of("ok")], reply_message(text_of("ok")))))

    assert response.usage["prompt_tokens"] == 1010
    assert response.usage["completion_tokens"] == 5
    assert response.usage["cache_read_tokens"] == 900
    assert response.usage["cache_write_tokens"] == 100


def test_text_nobody_listens_to_is_read_and_not_marked_streamed():
    provider = provider_on(FakeStream([text_of("ok")], reply_message(text_of("ok"))))

    response = asyncio.run(provider.stream_response(MESSAGES, None, on_delta=None))

    assert (response.content, response.streamed) == ("ok", False)


def test_the_streamed_request_is_the_whole_calls_request():
    from core.llm.prompt_cache import count_cache_breakpoints

    message = _text_only()
    provider = provider_on(FakeStream(_text_events(message), message), whole=message)

    asyncio.run(provider.generate_response(MESSAGES, TOOLS))
    streamed_call(provider, tools=TOOLS)

    sent = provider.client.messages.stream.call_args.kwargs
    assert sent == provider.client.messages.create.call_args.kwargs
    assert provider.client.messages.stream.call_args.args == ()
    assert (sent["model"], sent["max_tokens"]) == (MODEL, CONFIG_MAX_TOKENS)
    assert [block["text"] for block in sent["system"]] == [CACHED_PREFIX, SYSTEM[len(CACHED_PREFIX):]]
    assert count_cache_breakpoints(sent["system"]) == 1 and "cache_control" in sent["system"][0]
    assert sent["messages"] == [{"role": "user", "content": "What is open?"}]
    assert [tool["name"] for tool in sent["tools"]] == ["platform_list_tasks"]
    assert "temperature" not in sent and "stream" not in sent


def test_a_streamed_request_sends_this_calls_budget():
    from core.llm.output_budget import call_budget

    provider = provider_on(FakeStream([text_of("ok")], reply_message(text_of("ok"))))

    async def run():
        with call_budget(CALL_BUDGET):
            await provider.stream_response(MESSAGES, None, on_delta=None)

    asyncio.run(run())

    assert provider.client.messages.stream.call_args.kwargs["max_tokens"] == CALL_BUDGET   # not the config's


def test_thinking_and_tool_json_events_are_not_text():
    events = [NS(type="message_start"), NS(type="thinking", thinking="The owner wants the list."),
              NS(type="input_json", partial_json='{"status"'),
              NS(type="content_block_delta", delta=NS(type="text_delta", text="raw")),
              NS(type="text", text=""), NS(type="message_delta"), NS(type="message_stop")]

    response, heard = streamed_call(provider_on(FakeStream(events, _text_then_a_tool())), tools=TOOLS)

    assert heard == []
    assert response.streamed is False
    assert response.reasoning is None


def test_a_call_with_no_key_fails_clearly_whole_and_streamed():
    provider = provider_on()
    provider.client = None

    for call in (provider.generate_response(MESSAGES), provider.stream_response(MESSAGES)):
        with pytest.raises(ValueError, match="Anthropic API key not configured"):
            asyncio.run(call)
    with pytest.raises(ValueError, match="Anthropic API key not configured"):
        provider.generate_response_sync(MESSAGES)


# --------------------------------------------------------------------------- refusals and breaks


def test_a_refusal_before_the_stream_opens_is_the_sdks_own_error():
    refusal = Refused("rate limited")
    stream = FakeStream([text_of("never read")], _text_only())
    provider = provider_on(stream, refusal=refusal)
    heard, on_delta = listener()

    with pytest.raises(Refused) as caught:
        asyncio.run(provider.stream_response(MESSAGES, None, on_delta=on_delta))

    assert caught.value is refusal
    assert heard == [] and stream.read == 0
    provider.client.messages.create.assert_not_called()


def test_a_stream_that_breaks_before_any_text_is_asked_again_whole(caplog):
    caplog.set_level(logging.INFO)
    stream = FakeStream([RuntimeError(DROPPED)], snapshot=reply_message(stop_reason=None))
    provider = provider_on(stream, whole=_text_only())

    response, heard = streamed_call(provider)

    assert heard == []
    assert (response.content, response.streamed) == ("You have three open tasks.", False)
    assert provider.client.messages.create.call_count == 1
    assert provider.client.messages.stream.call_count == 1
    (retry,) = lines_of(caplog, CLIENT_LOGGER)
    assert retry == (f"[anthropic] stream retried=true model={MODEL} why=RuntimeError "
                     "input=10 cache_read=900 cache_write=100")
    assert stream.exited is True


def test_a_stream_that_breaks_after_text_fails_the_call():
    stream = FakeStream([text_of("You have "), RuntimeError(DROPPED)],
                         snapshot=reply_message(text_of("You have "), stop_reason=None))
    provider = provider_on(stream, whole=_text_only())

    error, heard = interrupted_call(provider)

    assert heard == [("text", "You have ")]
    assert str(error) == INTERRUPTED and error.why == "RuntimeError"
    assert error.partial_response.content == "You have "
    assert error.partial_response.usage == USAGE_AS_BOOKED
    assert isinstance(error.__cause__, RuntimeError) and str(error.__cause__) == DROPPED
    provider.client.messages.create.assert_not_called()
    assert stream.exited is True


def test_a_stream_that_ends_with_no_stop_reason_is_interrupted():
    cut_short = reply_message(text_of("You have "), stop_reason=None)
    provider = provider_on(FakeStream([text_of("You have ")], cut_short))

    error, heard = interrupted_call(provider)

    assert heard == [("text", "You have ")]
    assert error.why == NO_STOP_REASON and error.__cause__ is None
    assert error.partial_response.content == "You have "
    provider.client.messages.create.assert_not_called()


def test_the_interruption_says_nothing_of_the_providers(caplog, monkeypatch):
    from consumers.chatbot.turn_errors import describe_turn_error
    from core.llm import failover
    from core.llm.manager import LLMManager

    caplog.set_level(logging.WARNING, logger=STREAM_LOGGER)
    monkeypatch.setattr(config, "LLM_FAILOVER_MODEL", FAILOVER_MODEL)
    cause = ProviderError(OVERLOADED)
    stream = FakeStream([text_of("You have "), cause], snapshot=reply_message(text_of("You have "), stop_reason=None))

    error, _ = interrupted_call(provider_on(stream))

    assert str(error) == INTERRUPTED
    said = describe_turn_error(error).message
    assert INTERRUPTED in said
    assert "{" not in said and "request_id" not in said and "req_" not in said
    assert LLMManager._is_retriable_model_error(object.__new__(LLMManager), error) is False
    assert failover.failover_for(error, MODEL) is None
    assert failover.failover_for(Refused("rate limited"), MODEL) == FAILOVER_MODEL   # the failover is set
    assert error.__cause__ is cause
    (line,) = lines_of(caplog, STREAM_LOGGER)
    assert line == (f"[anthropic] stream interrupted why=ProviderError detail={OVERLOADED} "
                    "input=10 cache_read=900 cache_write=100")


def test_an_error_that_could_quote_the_model_is_logged_by_its_name_only(caplog):
    """The SDK raises a plain ValueError that quotes the tool JSON the model wrote."""
    caplog.set_level(logging.DEBUG)
    quoted = 'Unable to parse tool parameter JSON from model. JSON: {"to": "sam@example.com"'
    stream = FakeStream([text_of("Sending. "), ValueError(quoted)],
                         snapshot=reply_message(text_of("Sending. "), stop_reason=None))

    error, _ = interrupted_call(provider_on(stream))

    (line,) = lines_of(caplog, STREAM_LOGGER)
    assert line == "[anthropic] stream interrupted why=ValueError detail=- input=10 cache_read=900 cache_write=100"
    assert "sam@example.com" not in caplog.text and "Sending." not in caplog.text
    assert str(error.__cause__) == quoted


@pytest.mark.parametrize("snapshot, to_response", [
    (AssertionError(), AnthropicProvider._llm_response),
    (reply_message(stop_reason=None), lambda message: (_ for _ in ()).throw(TypeError("not a message"))),
], ids=["the snapshot asserts", "the reply cannot be built"])
def test_a_bad_snapshot_never_masks_the_interruption(snapshot, to_response, caplog):
    caplog.set_level(logging.WARNING, logger=STREAM_LOGGER)
    stream = FakeStream([RuntimeError(DROPPED)], snapshot=snapshot)

    with pytest.raises(AnthropicStreamInterrupted) as caught:
        read_stream(lambda: FakeStreamManager(stream), lambda text: None, lambda: False, to_response)

    assert caught.value.partial_response is None
    assert str(caught.value) == INTERRUPTED
    assert lines_of(caplog, STREAM_LOGGER) == [
        "[anthropic] stream interrupted why=RuntimeError detail=- input=None cache_read=None cache_write=None"]
    assert stream.exited is True


# --------------------------------------------------------------------------- a reader nobody hears


async def _until(condition):
    async def poll():
        while not condition():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), WAIT_SECONDS)


def test_a_cancelled_reader_closes_its_stream():
    """The provider only: a real turn's Stop never cancels this coroutine, so this does not prove
    Stop end to end."""
    more = threading.Event()
    stream = FakeStream([text_of("You have "), more, text_of("three "), text_of("open tasks.")], _text_only())
    provider = provider_on(stream)

    async def run():
        first = asyncio.Event()

        async def on_delta(kind, text):
            first.set()

        call = asyncio.ensure_future(provider.stream_response(MESSAGES, None, on_delta=on_delta))
        await asyncio.wait_for(first.wait(), WAIT_SECONDS)
        call.cancel()
        outcomes = await asyncio.wait_for(asyncio.gather(call, return_exceptions=True), WAIT_SECONDS)
        while_the_reader_waits = (stream.exited, stream.read)
        more.set()
        await _until(lambda: stream.exited)
        return outcomes, while_the_reader_waits

    try:
        (outcome,), while_the_reader_waits = asyncio.run(run())
    finally:
        more.set()

    assert isinstance(outcome, asyncio.CancelledError)
    assert while_the_reader_waits == (False, 1)       # the call ended without waiting for its thread
    assert stream.exited is True and stream.read == 2   # one more event, then closed: not read to its end


def test_a_closed_loop_stops_the_reader():
    """``_emit`` sets the stop flag when the loop it hands text to is closed (an ended Auto Live call)."""
    stop = threading.Event()
    stream = FakeStream([text_of("You have "), text_of("three "), text_of("open tasks.")], _text_only())

    answer = read_stream(lambda: FakeStreamManager(stream), lambda text: stop.set(), stop.is_set,
                         AnthropicProvider._llm_response)

    assert answer is None
    assert stream.exited is True and stream.read == 2


def test_the_client_tells_its_reader_to_stop_when_the_loop_is_closed():
    source = inspect.getsource(AnthropicProvider._read_live)
    handed_over = source.index("loop.call_soon_threadsafe(handoff.put_nowait, text)")
    closed = source.index("except RuntimeError:")

    assert handed_over < closed < source.index("stop.set()")
    assert "await producer" in source.split("finally:")[-2]     # awaited on the normal path only
    assert source.split("finally:")[-1].strip() == "stop.set()"   # the way out never waits for the thread


# --------------------------------------------------------------------------- the switch, per surface

NO_SURFACES, ONLY_QUICK_ASK, BOTH_SPOKEN = (), (QUICK_ASK,), (QUICK_ASK, LIVE)
EVERY_TURN = (NO_SURFACE, QUICK_ASK, LIVE)


def _streams_now(surface, *, headless=False):
    async def run():
        set_spoken_turn(surface)
        if not headless:
            return provider_on().streams_now()
        with request_scope.headless_run():
            return provider_on().streams_now()

    return asyncio.run(run())


@pytest.mark.parametrize("mode, streams_on", [
    ("off", NO_SURFACES), ("quick-ask", ONLY_QUICK_ASK), ("spoken", BOTH_SPOKEN), ("all", EVERY_TURN),
    ("on", NO_SURFACES), ("", NO_SURFACES), ("QUICK-ASK", NO_SURFACES),
])
def test_streams_now_follows_the_switch(monkeypatch, mode, streams_on):
    monkeypatch.setattr(config, "ANTHROPIC_STREAMING", mode)

    assert {surface for surface in EVERY_TURN if _streams_now(surface)} == set(streams_on)
    assert [_streams_now(surface, headless=True) for surface in EVERY_TURN] == [False, False, False]


def _offered(surface, *, headless=False):
    """What a caller that looks for the client's ``stream_response`` finds, on this surface."""
    async def run():
        set_spoken_turn(surface)
        if not headless:
            return getattr(provider_on(), "stream_response", None)
        with request_scope.headless_run():
            return getattr(provider_on(), "stream_response", None)

    return asyncio.run(run())


@pytest.mark.parametrize("mode, streams_on", [
    ("off", NO_SURFACES), ("quick-ask", ONLY_QUICK_ASK), ("spoken", BOTH_SPOKEN), ("all", EVERY_TURN),
    ("on", NO_SURFACES), ("", NO_SURFACES),
])
def test_the_client_offers_its_stream_only_to_a_call_that_streams(monkeypatch, mode, streams_on):
    monkeypatch.setattr(config, "ANTHROPIC_STREAMING", mode)

    assert {surface for surface in EVERY_TURN if _offered(surface) is not None} == set(streams_on)
    assert [_offered(surface, headless=True) for surface in EVERY_TURN] == [None, None, None]


def test_with_the_switch_off_a_listened_call_opens_no_stream(monkeypatch):
    """The manager's choice (``live_stream``) looks for the client's ``stream_response``: with the
    switch off it finds none and asks whole, as before the stream existed."""
    monkeypatch.setattr(config, "ANTHROPIC_STREAMING", "off")
    message = _text_only()
    provider = provider_on(FakeStream(_text_events(message), message), whole=message)

    async def run():
        set_spoken_turn(QUICK_ASK)
        stream = getattr(provider, "stream_response", None)
        return stream, await provider.generate_response(MESSAGES, None)

    stream, response = asyncio.run(run())

    assert stream is None
    assert (response.content, response.streamed) == ("You have three open tasks.", False)
    provider.client.messages.create.assert_called_once()
    provider.client.messages.stream.assert_not_called()


def test_the_stream_the_client_offers_is_the_streamed_call(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_STREAMING", "quick-ask")
    stream = FakeStream([text_of("You have "), text_of("three open tasks.")], _text_only())
    provider = provider_on(stream)
    heard, on_delta = listener()

    async def run():
        set_spoken_turn(QUICK_ASK)
        return await provider.stream_response(MESSAGES, None, on_delta=on_delta)

    response = asyncio.run(run())

    assert heard == [("text", "You have "), ("text", "three open tasks.")]
    assert response.streamed is True
    provider.client.messages.create.assert_not_called()


def test_read_stream_needs_no_sdk():
    source = (ORCH / "core" / "llm" / "clients" / "anthropic_stream.py").read_text(encoding="utf-8")

    assert "import anthropic" not in source and "from anthropic" not in source


# =========================================================================== B-4: a failed call's row

OUTPUT_SO_FAR = 1
BOOKED_FOR_THE_BROKEN_STREAM = UsageCounts(input_tokens=1010, output_tokens=OUTPUT_SO_FAR, cache_read_tokens=900,
                                           cache_write_tokens=100)
NOTHING_BOOKED = UsageCounts()


def _cut_short_after_one_token():
    """What the SDK had accumulated when the stream broke: the input and cache counts of
    ``message_start`` and one output token (the real count only arrives at the end)."""
    spent_so_far = NS(**{**vars(sdk_usage()), "output_tokens": OUTPUT_SO_FAR})
    return NS(content=[text_of("You have ")], usage=spent_so_far, model=MODEL, stop_reason=None)


def _broken_after_text(snapshot):
    return provider_on(FakeStream([text_of("You have "), RuntimeError(DROPPED)], snapshot=snapshot))


def _a_whole_call_that_raises():
    provider = provider_on()
    provider.client.messages.create.side_effect = RuntimeError(DROPPED)
    return provider


def _refused_before_the_stream_opens():
    return provider_on(FakeStream([text_of("never read")], _text_only()), refusal=Refused("rate limited"))


def _failed_call(provider, monkeypatch, raised, *, listened=True):
    """One call that fails, through a real ``LLMManager`` (no ``__init__``, so no settings read)
    with its real ``_call_failed`` and ``_track_usage``: (the error, the rows it would write as
    (counts, status), the counts noted against the ask's running cost). What is stubbed reads the
    database: the usage row's write, the ask-cost note and the credit gate."""
    from core.llm.manager import LLMManager

    monkeypatch.setattr(config, "LLM_FAILOVER_MODEL", "")
    manager = object.__new__(LLMManager)
    manager.provider, manager.config = provider, provider.config
    manager._tracking_ctx, manager.service_name = {}, "chat"
    rows, noted = [], []
    for unread in ("_ensure_provider_initialized", "_credit_gate"):
        monkeypatch.setattr(manager, unread, lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(manager, "_record_usage", lambda counts, latency_ms, status: rows.append((counts, status)))
    monkeypatch.setattr(manager, "_note_ask_cost", noted.append)
    _heard, on_delta = listener()

    with pytest.raises(raised) as caught:
        asyncio.run(manager.generate_response(MESSAGES, None, on_delta=on_delta if listened else None))
    return caught.value, rows, noted


def test_a_broken_streams_tokens_are_on_its_error_row(monkeypatch):
    provider = _broken_after_text(_cut_short_after_one_token())

    error, rows, noted = _failed_call(provider, monkeypatch, AnthropicStreamInterrupted)

    assert str(error) == INTERRUPTED
    assert error.partial_response.usage["prompt_tokens"] == 1010
    assert error.partial_response.usage["completion_tokens"] == OUTPUT_SO_FAR
    assert rows == [(BOOKED_FOR_THE_BROKEN_STREAM, "error")]
    assert noted == [BOOKED_FOR_THE_BROKEN_STREAM]       # the ask's running cost counts them
    provider.client.messages.create.assert_not_called()    # not asked again: text was heard


@pytest.mark.parametrize("failing, listened, raised", [
    (_a_whole_call_that_raises, False, RuntimeError),
    (_refused_before_the_stream_opens, True, Refused),
    (lambda: _broken_after_text(AssertionError()), True, AnthropicStreamInterrupted),
], ids=["a whole call raises", "refused before the stream opens", "a broken stream with no snapshot"])
def test_any_other_failed_call_books_zeros_as_before(monkeypatch, failing, listened, raised):
    error, rows, noted = _failed_call(failing(), monkeypatch, raised, listened=listened)

    assert type(error) is raised
    assert getattr(error, "partial_response", None) is None
    assert rows == [(NOTHING_BOOKED, "error")]
    assert noted == [NOTHING_BOOKED]


def test_the_failure_path_books_what_the_error_carries():
    from core.llm.manager import LLMManager

    source = inspect.getsource(LLMManager._call_failed)

    assert 'self._track_usage(getattr(exc, "partial_response", None), start, status="error")' in source

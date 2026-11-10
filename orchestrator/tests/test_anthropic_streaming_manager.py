"""Anthropic streaming (voice latency B-3): the manager asks the provider whether a call streams.

``LLMManager.generate_response`` takes its stream from ``live_stream`` (``core/llm/clients/base.py``):
the provider's ``stream_response`` when the caller listens (``on_delta``) and the provider streams
this call (``streams_now``: ``ANTHROPIC_STREAMING`` and the turn's surface), else None.

* With the switch off a listened-to Anthropic call is today's call: ``messages.create``, no
  stream, the reply as the model wrote it (no owner-words rewrite) and no Screen round. This is
  the test that pins "off is today".
* Each surface streams only on its value: a quick ask on ``quick-ask``, ``spoken`` and ``all``;
  Auto Live on ``spoken`` and ``all``; a turn that is read on ``all`` only.
* A call that streams goes through ``in_owner_words``: heard and saved in the owner's words, and
  reported to the turn's Screen. A call nobody listens to and a headless run never stream.
* A provider with no ``streams_now`` streams whenever it can, as before; nobody listening, the
  provider is not even asked.

Every case drives the real manager method (its decorators included) over a real
``AnthropicProvider`` on a stand-in client (``tests/helpers_anthropic_stream.py``). A whole spoken
turn on the real SDK is ``tests/test_anthropic_streaming_spoken_turn.py``.

No database, no model, no SDK call and no socket.
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect
from types import SimpleNamespace as NS

import pytest

from config import config
from consumers.chatbot.on_screen import Screen
from core.llm import request_scope
from core.llm.clients.base import live_stream
from core.llm.request_scope import LIVE, QUICK_ASK, set_spoken_turn
from core.llm.screen_watch import WATCHER
from core.llm.usage_context import LANE_CHAT, usage_scope
from tests.helpers_anthropic_stream import (
    MESSAGES, FakeStream, listener, manager_on, provider_on, reply_message, text_of,
)

NO_SURFACE = ""
BOTH_SPOKEN, EVERY_TURN = (QUICK_ASK, LIVE), ("", QUICK_ASK, LIVE)

A_PLATFORM_NAME = "platform_list_social_posts"
OFFERED = [{"type": "function", "function": {"name": A_PLATFORM_NAME, "description": "List the social posts",
                                             "parameters": {"type": "object", "properties": {}}}}]
NAMED_IN_TWO = ("Your posts are under `platform_", "list_social_posts`.")
AS_THE_MODEL_WROTE_IT = "".join(NAMED_IN_TWO)
IN_THE_OWNERS_WORDS = "Your posts are under list social posts."


class _WatchedScreen(Screen):
    """A turn's real Screen that also keeps what each streamed call reported to it."""

    def __init__(self):
        super().__init__()
        self.reported = []

    def ended(self, response, said):
        self.reported.append(said)
        super().ended(response, said)


def _managed_call(monkeypatch, mode, surface, *, listened=True, headless=False):
    """One call through a real manager to a real ``AnthropicProvider`` on the stand-in client, in
    Auto's chat lane with the turn's Screen watching. The model's reply names a platform call."""
    monkeypatch.setattr(config, "ANTHROPIC_STREAMING", mode)
    message = reply_message(text_of(AS_THE_MODEL_WROTE_IT))
    provider = provider_on(FakeStream([text_of(piece) for piece in NAMED_IN_TWO], message), whole=message)
    provider.client.beta.messages.create.return_value = message
    manager, screen = manager_on(provider, monkeypatch), _WatchedScreen()
    heard, on_delta = listener()

    async def run():
        set_spoken_turn(surface)
        WATCHER.set(screen)
        with contextlib.ExitStack() as scopes:
            scopes.enter_context(usage_scope(request_type=LANE_CHAT, execution_id="chat:b3"))
            if headless:
                scopes.enter_context(request_scope.headless_run())
            return await manager.generate_response(MESSAGES, OFFERED, on_delta=on_delta if listened else None)

    return NS(response=asyncio.run(run()), heard=heard, screen=screen, client=provider.client)


def _answered_as_before(call):
    """Today's Anthropic call: nothing streamed, the reply as the model wrote it (no owner-words
    rewrite) and nothing reported to the Screen."""
    call.client.messages.stream.assert_not_called()
    assert call.heard == []
    assert call.response.streamed is False
    assert call.response.content == AS_THE_MODEL_WROTE_IT and A_PLATFORM_NAME in call.response.content
    assert call.screen.reported == []


def _answered_whole(call):
    _answered_as_before(call)
    call.client.messages.create.assert_called_once()
    call.client.beta.messages.create.assert_not_called()


def _streamed(call):
    """The streamed call, through ``in_owner_words``: heard and saved in the owner's words, and
    reported to the Screen as it was heard."""
    call.client.messages.stream.assert_called_once()
    call.client.messages.create.assert_not_called()
    assert {kind for kind, _text in call.heard} == {"text"}
    assert "".join(text for _kind, text in call.heard) == IN_THE_OWNERS_WORDS
    assert call.response.streamed is True
    assert call.response.content == IN_THE_OWNERS_WORDS
    assert call.screen.reported == [IN_THE_OWNERS_WORDS]


def test_with_the_switch_off_an_anthropic_call_answers_whole_and_unwatched(monkeypatch):
    """The test that pins "off is today": a listened-to quick ask, the Screen set, Auto's lane."""
    _answered_whole(_managed_call(monkeypatch, "off", QUICK_ASK))


@pytest.mark.parametrize("surface", EVERY_TURN)
def test_with_the_switch_off_no_turn_streams(monkeypatch, surface):
    _answered_whole(_managed_call(monkeypatch, "off", surface))


@pytest.mark.parametrize("mode", ["quick-ask", "spoken"])
def test_a_typed_turn_answers_whole_on_quick_ask_and_on_spoken(monkeypatch, mode):
    _answered_whole(_managed_call(monkeypatch, mode, NO_SURFACE))


def test_an_auto_live_turn_answers_whole_on_quick_ask(monkeypatch):
    _answered_whole(_managed_call(monkeypatch, "quick-ask", LIVE))


def test_a_quick_ask_streams_on_quick_ask(monkeypatch):
    _streamed(_managed_call(monkeypatch, "quick-ask", QUICK_ASK))


@pytest.mark.parametrize("surface", BOTH_SPOKEN)
def test_an_auto_live_turn_streams_on_spoken(monkeypatch, surface):
    _streamed(_managed_call(monkeypatch, "spoken", surface))


@pytest.mark.parametrize("surface", EVERY_TURN)
def test_every_listened_call_streams_on_all(monkeypatch, surface):
    _streamed(_managed_call(monkeypatch, "all", surface))


def test_a_call_nobody_listens_to_never_streams(monkeypatch):
    _answered_whole(_managed_call(monkeypatch, "all", QUICK_ASK, listened=False))


def test_a_headless_run_never_streams_though_it_is_listened_to(monkeypatch):
    """Its request is the beta one (context editing, the memory tool), asked whole."""
    call = _managed_call(monkeypatch, "all", QUICK_ASK, headless=True)

    _answered_as_before(call)
    call.client.beta.messages.create.assert_called_once()
    call.client.messages.create.assert_not_called()


class _AlwaysStreams:
    """A provider with a streamed call and no ``streams_now`` (the OpenAI-compatible client)."""

    async def stream_response(self, messages, tools, on_delta=None):
        return None


class _StreamsWhenAsked(_AlwaysStreams):
    def __init__(self, streams):
        self.streams, self.asked = streams, 0

    def streams_now(self):
        self.asked += 1
        return self.streams


def test_a_provider_without_streams_now_streams_as_before():
    _heard, on_delta = listener()
    provider = _AlwaysStreams()

    assert live_stream(provider, on_delta) == provider.stream_response
    assert live_stream(provider, None) is None
    assert live_stream(object(), on_delta) is None  # a provider with no streamed call answers whole


def test_a_provider_is_asked_whether_this_call_streams():
    _heard, on_delta = listener()
    streams, whole = _StreamsWhenAsked(True), _StreamsWhenAsked(False)

    assert live_stream(streams, on_delta) == streams.stream_response
    assert live_stream(whole, on_delta) is None
    assert (streams.asked, whole.asked) == (1, 1)
    assert live_stream(streams, None) is None and streams.asked == 1  # nobody listens: not even asked


def test_the_manager_takes_its_stream_from_live_stream():
    """Source pins, the way ``tests/test_f264_d_replies_to_the_owner_name_no_platform_calls.py``
    pins the owner-words line: the choice is ``live_stream``'s, and what it gives is streamed
    through ``in_owner_words`` as before."""
    from core.llm.manager import LLMManager

    source = inspect.getsource(LLMManager.generate_response)

    assert "stream = live_stream(self.provider, on_delta)" in source
    assert "response = await in_owner_words(stream, messages, tools, on_delta)" in source
    assert 'getattr(self.provider, "stream_response"' not in source

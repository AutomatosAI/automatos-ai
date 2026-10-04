"""F264 (night 8): a reply that streams never ends "could not finish this reply".

02:34: "Auto could not finish this reply: Server tool "openrouter:web_search" failed:
upstream returned an invalid response", after #913's retry. Auto's chat streams, and
on a streamed call the provider's failure arrives while the stream is read, not when
the call is made, so the retry never saw it.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import httpx
import openai
import pytest

SEARCH = {"type": "openrouter:web_search", "max_uses": 3, "max_results": 5}
FUNCTION = {"type": "function", "function": {"name": "platform_update_task", "parameters": {}}}
SEARCH_FAILED = 'Server tool "openrouter:web_search" failed: upstream returned an invalid response'
REPLY = ("I'm sorry, Gerard, but I can't include your words as the reason directly. Would you like me to send it "
         "back without a specific reason?")


def _error(text):
    return openai.APIError(text, request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
                           body={"message": text})


def _chunk(text):
    return NS(choices=[NS(delta=NS(content=text, tool_calls=None), finish_reason=None)], usage=None)


def _stream(*texts, fail=None):
    def chunks():
        yield NS(choices=[], usage=None)                           # the provider's opening chunk says nothing
        for text in texts:
            yield _chunk(text)
        if fail is not None:
            raise fail
    return chunks()


class _Completions:
    def __init__(self, *streams):
        self.streams, self.sent = list(streams), []

    def create(self, **kwargs):
        self.sent.append(kwargs)
        return self.streams.pop(0)


def _read(completions, tools=(FUNCTION, SEARCH)):
    from core.llm.clients.search_recovery import SearchRecoveringClient

    client = SearchRecoveringClient(NS(chat=NS(completions=completions)))
    stream = client.chat.completions.create(model="m", messages=[], tools=list(tools), tool_choice="auto",
                                            stream=True)
    return [c.choices[0].delta.content for c in stream if c.choices]


def test_a_search_that_fails_before_anything_was_said_is_sent_again_without_it():
    completions = _Completions(_stream(fail=_error(SEARCH_FAILED)), _stream(REPLY))
    assert _read(completions) == [REPLY]
    assert len(completions.sent) == 2
    assert completions.sent[1]["tools"] == [FUNCTION] and completions.sent[1]["stream"] is True


def test_a_search_that_fails_after_the_reply_started_ends_the_reply_there():
    """Sent again, the reply would be said twice: it ends as it stood."""
    completions = _Completions(_stream(REPLY[:60], REPLY[60:], fail=_error(SEARCH_FAILED)))
    assert "".join(_read(completions)) == REPLY
    assert len(completions.sent) == 1


def test_any_other_failure_inside_the_stream_is_the_callers():
    completions = _Completions(_stream("Hello", fail=_error("Rate limit exceeded")))
    with pytest.raises(openai.APIError):
        _read(completions)


def test_a_stream_that_carried_no_search_is_read_as_it_comes():
    completions = _Completions(_stream(fail=_error(SEARCH_FAILED)))
    with pytest.raises(openai.APIError):
        _read(completions, tools=(FUNCTION,))
    assert len(completions.sent) == 1

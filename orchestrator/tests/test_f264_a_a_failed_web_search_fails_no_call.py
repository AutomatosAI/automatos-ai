"""F264 (night 7b): a reply that finished never ends "could not finish this reply".

Most of Auto's replies ended "Auto could not finish this reply: the AI provider's web
search failed on its side". The provider's own search (openrouter:web_search, put on
every call while web access is on) failed on a later call in the turn, F205's
re-prompt that takes the platform's tool names out of a reply, and the provider's 502
failed that call and the whole turn: the tail was added and the tool names stayed.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import httpx
import openai
import pytest

SEARCH = {"type": "openrouter:web_search", "max_uses": 3, "max_results": 5}
FUNCTION = {"type": "function", "function": {"name": "platform_get_task", "parameters": {}}}
SEARCH_FAILED = ('Error code: 502 - {\'error\': {\'message\': \'Server tool "openrouter:web_search" failed: '
                 'upstream returned an invalid response\'}}')


class _Completions:
    def __init__(self, fail_with=None):
        self.sent = []
        self.fail_with = fail_with

    def create(self, **kwargs):
        self.sent.append(kwargs)
        if self.fail_with is not None and len(self.sent) == 1:
            raise self.fail_with
        return NS(choices=["ok"])


def _error(text):
    return openai.APIError(text, request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
                           body=None)


def _client(completions):
    from core.llm.clients.search_recovery import SearchRecoveringClient

    return SearchRecoveringClient(NS(chat=NS(completions=completions), models="the models"))


def test_a_call_whose_web_search_failed_is_sent_again_without_it():
    completions = _Completions(_error(SEARCH_FAILED))
    out = _client(completions).chat.completions.create(model="m", messages=[], tools=[FUNCTION, SEARCH],
                                                       tool_choice="auto")

    assert out.choices == ["ok"] and len(completions.sent) == 2
    assert completions.sent[1]["tools"] == [FUNCTION] and completions.sent[1]["tool_choice"] == "auto"


def test_a_call_that_carried_only_the_search_is_sent_again_with_no_tools():
    completions = _Completions(_error(SEARCH_FAILED))
    _client(completions).chat.completions.create(model="m", messages=[], tools=[SEARCH], tool_choice="auto")
    assert "tools" not in completions.sent[1] and "tool_choice" not in completions.sent[1]


def test_any_other_failure_is_the_callers():
    completions = _Completions(_error("Error code: 400 - bad request"))
    with pytest.raises(openai.APIError):
        _client(completions).chat.completions.create(model="m", messages=[], tools=[SEARCH])
    assert len(completions.sent) == 1


def test_a_search_failure_on_a_call_that_carried_no_search_is_not_retried():
    completions = _Completions(_error(SEARCH_FAILED))
    with pytest.raises(openai.APIError):
        _client(completions).chat.completions.create(model="m", messages=[], tools=[FUNCTION])
    assert len(completions.sent) == 1


def test_everything_else_passes_through_to_the_sdk_client():
    assert _client(_Completions()).models == "the models"


def test_every_openai_compatible_provider_gets_the_retry(monkeypatch):
    import core.llm.clients.openai_compatible_client as oc
    from core.llm.clients.search_recovery import SearchRecoveringClient
    from core.llm.clients.base import LLMConfig, LLMProvider

    monkeypatch.setattr(oc, "OpenAI", lambda **kwargs: NS(chat=NS(completions=_Completions()), kwargs=kwargs))
    provider = oc.OpenAICompatibleProvider(LLMConfig(provider=LLMProvider.OPENROUTER, model="m", api_key="k-test"))
    assert isinstance(provider.client, SearchRecoveringClient)

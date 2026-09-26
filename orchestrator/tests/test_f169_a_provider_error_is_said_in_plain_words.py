"""F169 (night 5, B10/B26) — an AI provider's error is said in plain words.

Auto's replies in chat 3b7e2824 (09:25:53Z) and chat 0ad4ba56 (12:44:01Z) ended
with the provider SDK's own text, "Error: Error code: 502 - {'error':
{'message': 'Server tool \\"openrouter:web_search\\" failed: upstream returned
an invalid response', …". A failed tool loop put ``f"Error: {loop_err}"`` in the
reply. It now says what failed and what the owner can do; the raw text stays in
the server log.
"""
from __future__ import annotations

import inspect

import httpx
import openai
import pytest

from consumers.chatbot import service
from consumers.chatbot.turn_errors import CODE_RATE_LIMITED, describe_turn_error

# Chat 3b7e2824, 09:25:53Z (the start of it, as the SDK raised it).
NIGHT_5 = ("Error code: 502 - {'error': {'message': 'Server tool \"openrouter:web_search\" failed: upstream "
           "returned an invalid response', 'code': 502, 'metadata': {'provider_name': None, 'previous_errors': "
           "[{'code': 502, 'message': 'Server tool \"openrouter:web_search\" failed: upstream returned an invalid "
           "response'}]}}}")


def _provider_error(status: int, message: str) -> openai.APIStatusError:
    """What the OpenAI SDK raises when the provider answers ``status``."""
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return openai.APIStatusError(message, response=httpx.Response(status, request=request), body=None)


def _raw_text_left(message: str) -> bool:
    return "Error code" in message or "{" in message or "upstream" in message


def test_night_5s_web_search_502_is_said_in_plain_words():
    err = describe_turn_error(_provider_error(502, NIGHT_5), agent_name="Auto")

    assert err.message == ("Auto could not finish this reply: the AI provider's web search failed on its side. "
                           "Nothing needs changing on your side; ask again in a minute.")
    assert err.code == "provider_failed"


@pytest.mark.parametrize("status, said", [
    (500, "had a problem on its side"),
    (503, "had a problem on its side"),
    (504, "took too long to answer"),
    (402, "out of credits"),
    (401, "rejected its API key"),
    (400, "refused the request"),
])
def test_every_provider_answer_is_said_without_its_raw_text(status, said):
    err = describe_turn_error(_provider_error(status, f"Error code: {status} - {{'error': {{'message': 'x'}}}}"),
                              agent_name="Auto")
    assert said in err.message and not _raw_text_left(err.message), err.message


def test_a_rate_limit_from_the_sdk_is_the_rate_limited_code():
    err = describe_turn_error(_provider_error(429, "Error code: 429 - {'error': {'message': 'slow down'}}"))
    assert err.code == CODE_RATE_LIMITED and not _raw_text_left(err.message)


def test_an_error_of_our_own_keeps_its_short_text():
    """PRD-239 S4's line for anything that is not a provider's answer is unchanged."""
    err = describe_turn_error(RuntimeError("boom"), agent_name="Auto")
    assert err.message == "Auto could not finish this reply: boom"


def test_a_failed_tool_loop_replies_through_describe_turn_error():
    """The reply a failed chat tool loop gives (StreamingChatService._stream_tool_loop)."""
    source = inspect.getsource(service.StreamingChatService._stream_tool_loop)
    assert 'content=f"Error: {loop_err}"' not in source
    assert "describe_turn_error(" in source and "content=err.message" in source

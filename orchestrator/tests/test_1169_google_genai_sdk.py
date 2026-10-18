"""#1169: the Google provider uses the Google Gen AI SDK, one client per key.

``google-generativeai`` is retired (support ended 30 Nov 2025), and two bugs came with it:

- every async call failed after Gemini had answered (``.get`` on a ``Candidate``);
- ``genai.configure`` held one key for the whole process, so with two workspaces' own
  keys a call could go out on the other workspace's key.

Gemini is faked at the HTTP layer (an httpx mock transport handed to the real SDK), so
these tests see what each request carries: its key, its turns, its system instruction.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from google.genai.errors import ClientError

import api.user_api_keys as uak
from core.llm.clients import google_client
from core.llm.clients.base import LLMConfig, LLMProvider

MODEL = "gemini-2.5-flash"


def _answer(finish: str = "STOP") -> dict:
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "Paris"}]}, "finishReason": finish}],
        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 1, "totalTokenCount": 13},
    }


@pytest.fixture
def gemini(monkeypatch):
    """The real SDK over a fake Gemini API. Yields the requests it saw and the answer to give."""
    from google.genai import types

    seen, reply = [], {"status": 200, "body": _answer()}

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(reply["status"], json={"models": [{"name": f"models/{MODEL}"}]})
        return httpx.Response(reply["status"], json=reply["body"])

    real = google_client.genai.Client

    def client(**kwargs):
        options = kwargs.pop("http_options", None) or types.HttpOptions()
        transport = httpx.Client(transport=httpx.MockTransport(handle))
        return real(**kwargs, http_options=options.model_copy(update={"httpx_client": transport}))

    monkeypatch.setattr(google_client.genai, "Client", client)
    monkeypatch.setattr(google_client.config, "GOOGLE_API_KEY", None, raising=False)
    yield seen, reply


def _provider(key: str, **overrides) -> google_client.GoogleProvider:
    return google_client.GoogleProvider(LLMConfig(provider=LLMProvider.GOOGLE, model=MODEL, api_key=key, **overrides))


def _body(request: httpx.Request) -> dict:
    return json.loads(request.content)


def test_an_async_call_returns_the_answer_its_usage_and_why_it_stopped(gemini):
    seen, _ = gemini
    response = asyncio.run(_provider("key-a").generate_response([{"role": "user", "content": "Capital of France?"}]))
    assert (response.content, response.provider, response.finish_reason) == ("Paris", "google", "stop")
    assert response.usage == {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13}
    assert seen[0].url.path.endswith(f"/models/{MODEL}:generateContent")


def test_a_truncated_answer_is_length_as_the_tool_loop_reads_it(gemini):
    _, reply = gemini
    reply["body"] = _answer("MAX_TOKENS")
    assert _provider("key-a").generate_response_sync([{"role": "user", "content": "Go on"}]).finish_reason == "length"


def test_a_withheld_answer_is_content_filter_not_an_empty_stop(gemini):
    _, reply = gemini
    reply["body"] = {"promptFeedback": {"blockReason": "SAFETY"}}
    blocked = _provider("key-a").generate_response_sync([{"role": "user", "content": "…"}])
    assert (blocked.content, blocked.finish_reason) == ("", "content_filter")
    reply["body"] = _answer("SAFETY")
    assert _provider("key-a").generate_response_sync([{"role": "user", "content": "…"}]).finish_reason == "content_filter"


def test_the_key_given_wins_over_the_environment(gemini, monkeypatch):
    seen, _ = gemini
    monkeypatch.setenv("GOOGLE_API_KEY", "key-from-env")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    _provider("key-a").generate_response_sync([{"role": "user", "content": "hi"}])
    assert seen[0].url.host == "generativelanguage.googleapis.com"
    assert seen[0].headers["x-goog-api-key"] == "key-a"


def test_each_workspaces_key_goes_out_on_its_own_calls(gemini):
    seen, _ = gemini
    first, second = _provider("key-workspace-a"), _provider("key-workspace-b")
    asyncio.run(first.generate_response([{"role": "user", "content": "hi"}]))
    second.generate_response_sync([{"role": "user", "content": "hi"}])
    asyncio.run(first.generate_response([{"role": "user", "content": "hi"}]))
    assert [r.headers["x-goog-api-key"] for r in seen] == ["key-workspace-a", "key-workspace-b", "key-workspace-a"]


def test_system_messages_are_the_instruction_and_turns_keep_their_roles(gemini):
    seen, _ = gemini
    _provider("key-a", temperature=0.2, max_tokens=256).generate_response_sync([
        {"role": "system", "content": "Answer in one word."},
        {"role": "user", "content": [{"type": "text", "text": "Capital of France?"}]},
        {"role": "assistant", "content": "Paris"},
        {"role": "tool", "content": "{\"ok\": true}"},
        {"role": "user", "content": "And Spain?"},
    ])
    body = _body(seen[0])
    assert body["systemInstruction"]["parts"] == [{"text": "Answer in one word."}]
    assert [(c["role"], c["parts"][0]["text"]) for c in body["contents"]] == [
        ("user", "Capital of France?"), ("model", "Paris"), ("user", "And Spain?")]
    assert body["generationConfig"] == {"temperature": 0.2, "maxOutputTokens": 256}


def test_a_request_of_system_messages_alone_is_sent_as_the_users_turn(gemini):
    seen, _ = gemini
    _provider("key-a").generate_response_sync([{"role": "system", "content": "Say hello."}])
    body = _body(seen[0])
    assert [(c["role"], c["parts"][0]["text"]) for c in body["contents"]] == [("user", "Say hello.")]
    assert "systemInstruction" not in body


def test_no_key_fails_the_call_with_the_setup_message(gemini):
    with pytest.raises(ValueError, match="Google API key not configured"):
        asyncio.run(_provider(None).generate_response([{"role": "user", "content": "hi"}]))


def test_the_key_check_lists_models_with_that_key_and_fails_on_a_refusal(gemini):
    seen, reply = gemini
    assert uak._check_google("google", "key-to-check", None) == uak.KEY_VALID
    assert seen[0].url.path.endswith("/models") and seen[0].headers["x-goog-api-key"] == "key-to-check"
    reply["status"] = 403
    with pytest.raises(ClientError):
        uak._check_google("google", "bad-key", None)

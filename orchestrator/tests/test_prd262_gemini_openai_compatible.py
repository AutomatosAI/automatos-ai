"""PRD-262 US-024: Gemini through its OpenAI-compatible endpoint.

The deprecated google-generativeai SDK (135 MB installed, with 112 MB of Google API
discovery documents) left the image. The ``google`` provider is now an
OpenAI-compatible registry entry (``GOOGLE_BASE_URL``), so the adapter that serves
OpenRouter, NVIDIA and DeepSeek serves Gemini too: a completion, a stream and tool
calls (the old GoogleProvider neither streamed nor sent tools). A saved key is checked
with the same models-list call as the other OpenAI-compatible providers.

Two shapes of Gemini's endpoint are pinned as well:
- a streamed tool call arrives whole, in one chunk, with no ``index``; two parallel
  calls stay two calls;
- a tool call carries a thought signature in ``extra_content``, which Gemini 3 needs
  back on the assistant message that repeats the call. The tool loops rebuild that
  message from id, type and function, so the adapter's client puts it back.

The real OpenAI SDK talks to an ``httpx.MockTransport``, and every test asserts the
URL, the headers and the body Gemini's endpoint receives. A streamed answer is a
``SyncByteStream`` body: ``httpx.Response(content=…)`` is read in advance and would
never stream. Nothing leaves the process.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import httpx
import openai
import pytest

from config import config
from core.llm import clients as llm_clients
from core.llm import manager as llm_manager
from core.llm import output_budget, workspace_keys
from core.llm import providers as reg
from core.llm.clients import openai_compatible_client as oc
from core.llm.clients.base import LLMConfig, LLMProvider
from core.llm.clients.openai_compatible_client import OpenAICompatibleProvider
from core.llm.clients.tool_call_extras import EXTRA_KEY, MAX_KEPT_CALLS, ToolCallExtras

GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL = "gemini-2.5-flash"
KEY = "AIza-test-key"
SIGNATURE = {"google": {"thought_signature": "c2lnLWE="}}
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "The weather in a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        "strict": True,
    },
}
ASK = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather in Lisbon?"}]


# --------------------------------------------------------------------------- the fake endpoint


class _Body(httpx.SyncByteStream):
    """A body that streams as a network response does (``content=`` is read at once)."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def __iter__(self):
        for start in range(0, len(self.data), 24):
            yield self.data[start:start + 24]


def _sse(*chunks: Dict[str, Any]) -> httpx.Response:
    events = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, stream=_Body(events.encode()), headers={"content-type": "text/event-stream"})


def _chunk(delta: Dict[str, Any], finish: Any = None, usage: Any = None) -> Dict[str, Any]:
    chunk = {"id": "chatcmpl-g", "object": "chat.completion.chunk", "created": 1, "model": MODEL,
             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return {**chunk, "usage": usage} if usage else chunk


def _completion(message: Dict[str, Any], finish: str = "stop") -> httpx.Response:
    return httpx.Response(200, json={
        "id": "chatcmpl-g", "object": "chat.completion", "created": 1, "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
    })


@pytest.fixture
def gemini(monkeypatch):
    """Gemini's endpoint: queue ``answers``; every request the SDK sends lands in ``sent``."""
    real = openai.OpenAI
    endpoint = SimpleNamespace(sent=[], answers=[])

    def handler(request: httpx.Request) -> httpx.Response:
        endpoint.sent.append(request)
        return endpoint.answers.pop(0)

    def client(**kwargs):
        return real(http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0, **kwargs)

    monkeypatch.setattr(oc, "OpenAI", client)
    monkeypatch.setattr(openai, "OpenAI", client)  # the saved-key check imports it when it runs
    return endpoint


def _provider(**overrides: Any) -> OpenAICompatibleProvider:
    cfg = {"provider": LLMProvider.GOOGLE, "model": MODEL, "api_key": KEY, "max_tokens": 256, **overrides}
    return OpenAICompatibleProvider(LLMConfig(**cfg))


def _body(request: httpx.Request) -> Dict[str, Any]:
    return json.loads(request.content)


def _assert_sent_to_gemini(request: httpx.Request, path: str, method: str = "POST") -> None:
    assert request.method == method
    assert str(request.url) == GEMINI + path
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert "x-title" not in request.headers and "http-referer" not in request.headers, "OpenRouter's only"


# --------------------------------------------------------------------------- the registry and the manager


def test_google_is_an_openai_compatible_provider_with_its_own_base_url():
    spec = reg.get_spec("google")
    assert spec.adapter == reg.ADAPTER_OPENAI_COMPATIBLE
    assert spec.base_url_key == "GOOGLE_BASE_URL" and spec.env_key == "GOOGLE_API_KEY"
    assert reg.base_url_for("google") == config.GOOGLE_BASE_URL == GEMINI
    assert reg.headers_for("google") == {}
    assert not hasattr(reg, "ADAPTER_GOOGLE")
    surface = json.loads((Path(__file__).resolve().parents[1] / "reports" / "config-surface.json").read_text())
    assert "GOOGLE_BASE_URL" in surface["settings"]


def test_the_google_sdk_client_is_gone_and_the_manager_builds_the_adapter():
    assert importlib.util.find_spec("core.llm.clients.google_client") is None
    assert not hasattr(llm_clients, "GoogleProvider") and "GoogleProvider" not in llm_clients.__all__
    built = llm_manager.LLMManager._create_provider(LLMConfig(provider=LLMProvider.GOOGLE, model=MODEL, api_key=KEY))
    assert isinstance(built, OpenAICompatibleProvider) and built.spec.slug == "google"


@pytest.fixture
def from_settings(monkeypatch):
    """A manager built from settings for google, with ``cred`` as the stored credential."""
    monkeypatch.setattr(output_budget, "manager_budgets", lambda *a, **k: (None, None))
    monkeypatch.setattr(workspace_keys, "get_platform_workspace_key", lambda provider: None)
    monkeypatch.setattr(llm_manager, "env_api_key", lambda provider: None)
    monkeypatch.setattr(llm_manager, "get_system_setting", lambda category, key, default=None: default)

    def build(cred: Dict[str, Any]) -> llm_manager.LLMManager:
        monkeypatch.setattr(llm_manager, "get_credential_data",
                            lambda provider, environment=None, service_name="orchestrator": dict(cred))
        return llm_manager.LLMManager(service_name="chatbot", provider="google", model=MODEL)
    return build


def test_the_stored_key_is_read_as_for_the_other_openai_compatible_providers(from_settings):
    assert from_settings({"api_key": KEY}).config.api_key == KEY
    assert from_settings({"api_token": KEY}).config.api_key == KEY, "the google branch read api_key only"
    assert from_settings({"api_key": KEY}).config.base_url is None
    assert from_settings({"api_key": KEY, "base_url": "https://proxy.example/v1"}).config.base_url == \
        "https://proxy.example/v1"


def test_a_settings_built_manager_calls_gemini(from_settings, gemini):
    manager = from_settings({"api_key": KEY})
    gemini.answers.append(_completion({"content": "Sunny."}))
    response = manager._create_provider(manager.config).generate_response_sync(ASK)
    _assert_sent_to_gemini(gemini.sent[0], "chat/completions")
    assert response.content == "Sunny." and response.provider == "google"


# --------------------------------------------------------------------------- a completion


def test_a_completion(gemini):
    gemini.answers.append(_completion({"content": "Sunny, 21 degrees."}))
    response = asyncio.run(_provider().generate_response(ASK))

    (request,) = gemini.sent
    _assert_sent_to_gemini(request, "chat/completions")
    body = _body(request)
    assert body["model"] == MODEL and body["messages"] == ASK
    assert body["max_tokens"] == 256 and body["temperature"] == 0.7
    assert not {"tools", "tool_choice", "stream", "usage"} & set(body), body
    assert response.content == "Sunny, 21 degrees."
    assert response.provider == "google" and response.model == MODEL and response.finish_reason == "stop"
    assert response.usage["prompt_tokens"] == 11 and response.usage["total_tokens"] == 14


# --------------------------------------------------------------------------- a stream


def test_a_stream(gemini):
    gemini.answers.append(_sse(
        _chunk({"role": "assistant", "content": "Sun"}),
        _chunk({"content": "ny."}),
        _chunk({}, finish="stop", usage={"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11}),
    ))
    deltas: List[tuple] = []

    async def on_delta(kind: str, text: str) -> None:
        deltas.append((kind, text))

    response = asyncio.run(_provider().stream_response(ASK, on_delta=on_delta))

    (request,) = gemini.sent
    _assert_sent_to_gemini(request, "chat/completions")
    body = _body(request)
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["model"] == MODEL and body["messages"] == ASK
    assert deltas == [("text", "Sun"), ("text", "ny.")]
    assert response.content == "Sunny." and response.streamed is True and response.provider == "google"
    assert response.usage["total_tokens"] == 11


# --------------------------------------------------------------------------- a tool call


def _repeated_by_the_tool_loop(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The next round's messages as the tool loop builds them: id, type and function only."""
    assistant = {"role": "assistant", "content": None, "tool_calls": [
        {"id": c["id"], "type": "function", "function": dict(c["function"])} for c in calls]}
    results = [{"role": "tool", "tool_call_id": c["id"], "name": c["function"]["name"], "content": "21C"}
               for c in calls]
    return [*ASK, assistant, *results]


def test_a_tool_call_and_its_thought_signature_on_the_next_round(gemini):
    call = {"id": "call_a", "type": "function",
            "function": {"name": "get_weather", "arguments": json.dumps({"city": "Lisbon"})}, EXTRA_KEY: SIGNATURE}
    gemini.answers += [_completion({"content": None, "tool_calls": [call]}, finish="tool_calls"),
                       _completion({"content": "21 degrees in Lisbon."})]
    provider = _provider()

    first = asyncio.run(provider.generate_response(ASK, tools=[WEATHER_TOOL]))

    body = _body(gemini.sent[0])
    assert body["tool_choice"] == "auto"
    assert body["tools"] == [{"type": "function", "function": {
        k: v for k, v in WEATHER_TOOL["function"].items() if k != "strict"}}], "strict is OpenAI's own"
    assert first.tool_calls == [{"id": "call_a", "type": "function", "function": call["function"]}]

    following = _repeated_by_the_tool_loop(first.tool_calls)
    second = asyncio.run(provider.generate_response(following, tools=[WEATHER_TOOL]))

    _assert_sent_to_gemini(gemini.sent[1], "chat/completions")
    sent = _body(gemini.sent[1])["messages"]
    assert sent[2]["tool_calls"] == [{**first.tool_calls[0], EXTRA_KEY: SIGNATURE}], "Gemini 3 needs it back"
    assert sent[3] == following[3]
    assert EXTRA_KEY not in following[2]["tool_calls"][0], "the caller's messages are never changed"
    assert second.content == "21 degrees in Lisbon."


def test_streamed_parallel_tool_calls_with_no_index_stay_two_calls(gemini):
    """Gemini's endpoint sends each streamed call whole, in one chunk, with no index,
    and finishes with "stop"; read as index 0 the two calls ran together."""
    weather = {"id": "call_a", "type": "function",
               "function": {"name": "get_weather", "arguments": json.dumps({"city": "Lisbon"})}}
    clock = {"id": "call_b", "type": "function",
             "function": {"name": "get_time", "arguments": json.dumps({"tz": "WET"})}}
    gemini.answers += [
        _sse(_chunk({"role": "assistant", "tool_calls": [{**weather, EXTRA_KEY: SIGNATURE}, clock]}, finish="stop",
                    usage={"prompt_tokens": 9, "completion_tokens": 8, "total_tokens": 17})),
        _sse(_chunk({"content": "Done."}, finish="stop")),
    ]
    provider = _provider()

    response = asyncio.run(provider.stream_response(ASK, tools=[WEATHER_TOOL]))

    assert response.tool_calls == [weather, clock]
    asyncio.run(provider.stream_response(_repeated_by_the_tool_loop(response.tool_calls), tools=[WEATHER_TOOL]))
    repeated = _body(gemini.sent[1])["messages"][2]["tool_calls"]
    assert repeated == [{**weather, EXTRA_KEY: SIGNATURE}, clock], "only the call that had one gets it"


# --------------------------------------------------------------------------- the saved-key check


def test_the_key_check_lists_models_on_gemini(gemini):
    import api.user_api_keys as uak

    assert "google" not in uak._LIVE_CHECKS and not hasattr(uak, "_check_google")
    assert uak._live_check_for("google") is uak._check_openai_compatible
    gemini.answers.append(httpx.Response(200, json={"object": "list", "data": [
        {"id": "models/gemini-2.5-flash", "object": "model", "created": 0, "owned_by": "google"}]}))

    result = asyncio.run(uak._validate_provider_key("google", KEY))

    (request,) = gemini.sent
    _assert_sent_to_gemini(request, "models", method="GET")
    assert request.content == b""
    assert result.valid is True and result.message == uak.KEY_VALID


def test_the_key_check_reports_a_key_gemini_refuses(gemini):
    import api.user_api_keys as uak

    gemini.answers.append(httpx.Response(400, json={"error": {
        "code": 400, "message": "API key not valid. Please pass a valid API key.", "status": "INVALID_ARGUMENT"}}))

    result = asyncio.run(uak._validate_provider_key("google", KEY))

    _assert_sent_to_gemini(gemini.sent[0], "models", method="GET")
    assert result.valid is False and "API key not valid" in result.message


# --------------------------------------------------------------------------- the pieces, alone


def test_the_extras_go_back_on_a_new_message_and_only_where_missing():
    extras = ToolCallExtras()
    assert extras.restored(ASK) is ASK, "nothing kept: the request is unchanged"
    extras.keep_calls([{"id": "call_a", EXTRA_KEY: SIGNATURE}, {"id": "call_b"}])
    own = {"google": {"thought_signature": "b3du"}}
    messages = [
        {"role": "user", "content": "hi", "tool_calls": [{"id": "call_a"}]},
        {"role": "assistant", "tool_calls": [{"id": "call_a"}, {"id": "call_b"}, {"id": "call_c"}]},
        {"role": "assistant", "tool_calls": [{"id": "call_a", EXTRA_KEY: own}]},
    ]
    restored = extras.restored(messages)
    assert restored[0] is messages[0], "only an assistant message repeats calls"
    assert restored[1]["tool_calls"] == [{"id": "call_a", EXTRA_KEY: SIGNATURE}, {"id": "call_b"}, {"id": "call_c"}]
    assert restored[2] is messages[2], "a call that carries its own extras keeps them"
    assert messages[1]["tool_calls"][0] == {"id": "call_a"}, "the caller's dicts are never changed"


def test_the_extras_kept_are_bounded():
    extras = ToolCallExtras()
    for n in range(MAX_KEPT_CALLS + 1):
        extras.keep(f"call_{n}", SIGNATURE)
    assert extras.kept("call_0") is None and extras.kept(f"call_{MAX_KEPT_CALLS}") == SIGNATURE


def test_indexed_fragments_still_assemble_as_openai_streams_them():
    assembler = oc._StreamAssembler()
    fragments = [
        {"index": 0, "id": "call_a", "type": "function", "function": {"name": "get_", "arguments": '{"ci'}},
        {"index": 1, "id": "call_b", "type": "function", "function": {"name": "get_time", "arguments": "{}"}},
        {"index": 0, "function": {"name": "weather", "arguments": 'ty": "Lisbon"}'}},
    ]
    for fragment in fragments:
        assembler._fold_tool_call(fragment)
    assert [assembler.tool_calls[i] for i in sorted(assembler.tool_calls)] == [
        {"id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Lisbon"}'}},
        {"id": "call_b", "type": "function", "function": {"name": "get_time", "arguments": "{}"}},
    ]


def test_unindexed_fragments_follow_their_id_or_the_last_call():
    assembler = oc._StreamAssembler()
    for fragment in [
        {"id": "call_a", "function": {"name": "get_weather", "arguments": '{"city":'}},
        {"function": {"arguments": ' "Lisbon"}'}},
        {"id": "call_b", "function": {"name": "get_time", "arguments": "{"}},
        {"id": "call_b", "function": {"arguments": "}"}},
    ]:
        assembler._fold_tool_call(fragment)
    assert [(c["id"], c["function"]["name"], c["function"]["arguments"]) for c in assembler.tool_calls.values()] == [
        ("call_a", "get_weather", '{"city": "Lisbon"}'),
        ("call_b", "get_time", "{}"),
    ]

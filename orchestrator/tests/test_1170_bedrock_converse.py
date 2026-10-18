"""#1170: the Bedrock provider speaks Converse, one request shape for every model.

Before, ``invoke_model`` got a hand-written body per family. A model the client didn't
know (Amazon Nova, the Titan models it listed itself) was sent a Claude body and failed;
a tool loop never sent its tool calls or results back; and a Bedrock API key went out
in a header AWS doesn't read, and only when it started ``bedrock-api-``.

These tests run botocore's real request pipeline and stop it just before it sends
(``before-send``), so each one sees the serialised Converse request and how it was
signed: SigV4 or bearer, with which key, to which region.
"""
from __future__ import annotations

import asyncio
import base64
import json

import pytest
from botocore.awsrequest import AWSResponse

from core.llm.clients.base import LLMConfig, LLMProvider
from core.llm.clients.bedrock_client import BedrockProvider

NOVA = "amazon.nova-pro-v1:0"
OPUS = "anthropic.claude-opus-5-5"
API_KEY = "ABSKQmVkcm9ja0FQSUtleS1leGFtcGxlLWtleQ=="   # the shape of a long-term Bedrock API key; not a real one


class _Raw:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def stream(self, **_kwargs):
        yield self._body


def _reply(text: str = "Paris", stop: str = "end_turn", **extra) -> dict:
    return {"output": {"message": {"role": "assistant", "content": [{"text": text}]}}, "stopReason": stop,
            "usage": {"inputTokens": 12, "outputTokens": 1, "totalTokens": 13}, **extra}


@pytest.fixture(autouse=True)
def _no_ambient_aws(monkeypatch, tmp_path):
    """No credentials, profile or bearer token from the machine running the tests."""
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE",
                 "AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "none"))


def _bedrock(model: str = NOVA, *, api_key: str = API_KEY, secret: str = None, region: str = "eu-west-1",
             reply: dict = None, status: int = 200, error_type: str = None, **config):
    """A provider whose requests are caught before they leave. Returns it and the requests seen."""
    provider = BedrockProvider(LLMConfig(provider=LLMProvider.AWS_BEDROCK, model=model, api_key=api_key,
                                         secret_key=secret, base_url=region, **config))
    seen = []

    def send(request, **_kwargs):
        seen.append(request)
        headers = {"content-type": "application/json", **({"x-amzn-ErrorType": error_type} if error_type else {})}
        body = reply if reply is not None else _reply()
        return AWSResponse(request.url, status, headers, _Raw(json.dumps(body).encode()))

    provider.client.meta.events.register("before-send.bedrock-runtime.Converse", send)
    return provider, seen


def _body(request) -> dict:
    return json.loads(request.body)


def test_a_model_the_old_client_didnt_know_gets_a_converse_request_and_its_answer(monkeypatch):
    provider, seen = _bedrock(NOVA, temperature=0.3, max_tokens=256)
    response = asyncio.run(provider.generate_response([
        {"role": "system", "content": "Answer in one word."},
        {"role": "user", "content": "Capital of France?"},
    ]))
    assert seen[0].url == f"https://bedrock-runtime.eu-west-1.amazonaws.com/model/{NOVA.replace(':', '%3A')}/converse"
    body = _body(seen[0])
    assert body["system"] == [{"text": "Answer in one word."}]
    assert body["messages"] == [{"role": "user", "content": [{"text": "Capital of France?"}]}]
    assert body["inferenceConfig"] == {"maxTokens": 256, "temperature": 0.3}
    assert (response.content, response.finish_reason, response.provider) == ("Paris", "stop", "aws_bedrock")
    assert response.usage == {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13}


def test_a_tool_loop_sends_its_calls_and_results_and_reads_the_next_call():
    reply = {"output": {"message": {"role": "assistant", "content": [
        {"reasoningContent": {"reasoningText": {"text": "Need the weather."}}},
        {"toolUse": {"toolUseId": "t2", "name": "weather", "input": {"city": "Madrid"}}}]}},
        "stopReason": "tool_use", "usage": {"inputTokens": 30, "outputTokens": 9, "totalTokens": 39}}
    provider, seen = _bedrock(OPUS, reply=reply)
    tools = [{"type": "function", "function": {"name": "weather", "description": "Weather for a city",
                                               "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                                               "strict": True}}]
    response = asyncio.run(provider.generate_response([
        {"role": "user", "content": "Weather in Paris, then Madrid?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "weather", "arguments": "{\"city\": \"Paris\"}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "18C, sunny"},
    ], tools=tools))
    body = _body(seen[0])
    assert body["messages"][1:] == [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "weather", "input": {"city": "Paris"}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "18C, sunny"}]}}]},
    ]
    assert body["toolConfig"]["tools"] == [{"toolSpec": {
        "name": "weather", "description": "Weather for a city",
        "inputSchema": {"json": {"type": "object", "properties": {"city": {"type": "string"}}}}}}]
    assert response.tool_calls == [{"id": "t2", "type": "function",
                                    "function": {"name": "weather", "arguments": "{\"city\": \"Madrid\"}"}}]
    assert (response.finish_reason, response.reasoning) == ("tool_calls", "Need the weather.")


def test_a_closing_call_without_tools_still_declares_the_tools_its_history_used():
    provider, seen = _bedrock(OPUS)
    provider.generate_response_sync([
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "Checking.", "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "weather", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": {"temp": 18}},
        {"role": "user", "content": "Now summarise."},
    ])
    body = _body(seen[0])
    assert [spec["toolSpec"]["name"] for spec in body["toolConfig"]["tools"]] == ["weather"]
    assert body["messages"][2] == {"role": "user", "content": [
        {"toolResult": {"toolUseId": "t1", "content": [{"text": "{\"temp\": 18}"}]}}, {"text": "Now summarise."}]}


def test_turns_alternate_a_later_system_message_is_a_user_turn_and_images_go_as_bytes():
    provider, seen = _bedrock(NOVA)
    png = base64.b64encode(b"\x89PNG fake").decode()
    provider.generate_response_sync([
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": [{"type": "text", "text": "What is this?"},
                                     {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png}}]},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{png}"}}]},
        {"role": "system", "content": "Answer now."},
    ])
    body = _body(seen[0])
    assert body["system"] == [{"text": "Be brief."}]
    [turn] = body["messages"]
    assert turn["role"] == "user"
    assert turn["content"] == [
        {"text": "What is this?"},
        {"image": {"format": "png", "source": {"bytes": png}}},
        {"image": {"format": "jpeg", "source": {"bytes": png}}},
        {"text": "Answer now."},
    ]


def test_a_newer_claude_gets_no_temperature_and_a_cut_off_answer_is_length():
    provider, seen = _bedrock(OPUS, reply=_reply(stop="max_tokens"), temperature=0.7)
    assert provider.generate_response_sync([{"role": "user", "content": "Go on"}]).finish_reason == "length"
    assert "temperature" not in _body(seen[0])["inferenceConfig"]


def test_a_bedrock_api_key_is_a_bearer_token_of_this_provider_alone():
    first, seen_first = _bedrock(api_key=API_KEY)
    second, seen_second = _bedrock(api_key="bedrock-api-key-c2hvcnQtdGVybQ==")
    first.generate_response_sync([{"role": "user", "content": "hi"}])
    second.generate_response_sync([{"role": "user", "content": "hi"}])
    first.generate_response_sync([{"role": "user", "content": "hi"}])
    assert [r.headers["Authorization"].decode() for r in seen_first + seen_second] == [
        f"Bearer {API_KEY}", f"Bearer {API_KEY}", "Bearer bedrock-api-key-c2hvcnQtdGVybQ=="]
    assert "X-Amz-Date" not in seen_first[0].headers   # not SigV4


def test_an_access_key_and_secret_are_signed_with_sigv4_for_the_saved_region():
    provider, seen = _bedrock(api_key="AKIAIOSFODNN7EXAMPLE", secret="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                              region="ap-southeast-2")
    provider.generate_response_sync([{"role": "user", "content": "hi"}])
    authorization = seen[0].headers["Authorization"].decode()
    assert authorization.startswith("AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/")
    assert "/ap-southeast-2/bedrock/aws4_request" in authorization


def test_a_model_served_only_by_an_inference_profile_says_which_ids_to_use():
    message = ("Invocation of model ID anthropic.claude-sonnet-4-5-20250929-v1:0 with on-demand throughput isn't "
               "supported. Retry your request with the ID or ARN of an inference profile that contains this model.")
    provider, _ = _bedrock("anthropic.claude-sonnet-4-5-20250929-v1:0", reply={"message": message}, status=400,
                           error_type="ValidationException")
    with pytest.raises(ValueError, match=r"us\.anthropic\.claude-sonnet-4-5-20250929-v1:0.*global\."):
        provider.generate_response_sync([{"role": "user", "content": "hi"}])


def test_any_other_bedrock_error_is_raised_as_it_came():
    from botocore.exceptions import ClientError

    provider, _ = _bedrock(reply={"message": "You don't have access to the model"}, status=403,
                           error_type="AccessDeniedException")
    with pytest.raises(ClientError, match="access to the model"):
        asyncio.run(provider.generate_response([{"role": "user", "content": "hi"}]))

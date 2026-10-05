"""#873 (daarthur): the Azure OpenAI provider used the legacy AzureOpenAI client on
api-version 2024-02-15-preview and sent max_tokens and temperature on every call,
which reasoning-model deployments (o-series, GPT-5) refuse.

The provider now calls Microsoft Foundry's v1 route (<resource>/openai/v1/) with
the plain OpenAI client, sends the budget as max_completion_tokens, and retries a
400 that names a sampling parameter once without it, remembering the refusal for
that deployment. These tests pin the request the client actually sends."""
import asyncio
from types import SimpleNamespace as NS

import pytest

from core.llm.clients import azure_client
from core.llm.clients.azure_v1 import QUIRKS, refused_params, v1_base_url
from core.llm.clients.base import LLMConfig, LLMProvider, accepts_sampling_params

RESOURCE = "https://contoso-uk.openai.azure.com"
V1 = f"{RESOURCE}/openai/v1/"
HI = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def _forget_deployments():
    QUIRKS.clear()
    yield
    QUIRKS.clear()


def _reply(model="gpt-4o-2024-11-20", tool_calls=None):
    message = NS(content="ok", tool_calls=tool_calls)
    usage = NS(prompt_tokens=3, completion_tokens=2, total_tokens=5, prompt_tokens_details=None)
    return NS(choices=[NS(message=message, finish_reason="stop")], usage=usage, model=model)


def _refusal(message, param, code="unsupported_value"):
    """The 400 Azure sends a reasoning deployment's temperature (a real SDK error)."""
    import httpx
    import openai

    request = httpx.Request("POST", f"{V1}chat/completions")
    body = {"message": message, "type": "invalid_request_error", "param": param, "code": code}
    return openai.BadRequestError(
        f"Error code: 400 - {{'error': {body}}}", response=httpx.Response(400, request=request), body=body,
    )


TEMPERATURE_REFUSED = (
    "Unsupported value: 'temperature' does not support 0.7 with this model. "
    "Only the default (1) value is supported."
)


class _Completions:
    """Records every create() call's kwargs; answers from a script of replies/errors."""

    def __init__(self, script):
        self.calls, self._script = [], list(script)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        step = self._script.pop(0) if self._script else _reply()
        if isinstance(step, Exception):
            raise step
        return step


def _provider(deployment="prod-chat", script=(), monkeypatch=None, endpoint=RESOURCE):
    """A real AzureProvider whose OpenAI client is a recorder."""
    built = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            built.update(kwargs)
            self.chat = NS(completions=_Completions(script))

    monkeypatch.setattr(azure_client, "OpenAI", _FakeOpenAI)
    config = LLMConfig(
        provider=LLMProvider.AZURE, model=deployment, temperature=0.7, max_tokens=1024,
        api_key="azure-key", base_url=endpoint,
    )
    provider = azure_client.AzureProvider(config)
    return provider, built


def _sent(provider):
    return provider.client.chat.completions.calls


# --------------------------------------------------------------------------- #
# The v1 route
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("saved", [
    RESOURCE,
    f"{RESOURCE}/",
    f"  {RESOURCE}/  ",
    f"{RESOURCE}/openai/v1",
    f"{RESOURCE}/openai/v1/",
    f"{RESOURCE}/openai/v1/chat/completions",
    f"{RESOURCE}/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview",
    f"{RESOURCE}?api-version=2024-02-15-preview",
    "contoso-uk.openai.azure.com",
])
def test_every_saved_endpoint_shape_reaches_the_v1_route(saved):
    assert v1_base_url(saved) == V1


def test_a_foundry_project_endpoint_is_cut_back_to_its_resource():
    saved = "https://contoso.services.ai.azure.com/api/projects/bank-poc"
    assert v1_base_url(saved) == "https://contoso.services.ai.azure.com/openai/v1/"


def test_a_gateway_path_before_openai_v1_is_kept():
    assert v1_base_url("https://apim.bank.example/azure/openai/v1") == "https://apim.bank.example/azure/openai/v1/"


@pytest.mark.parametrize("saved", ["", "   ", "https://"])
def test_an_endpoint_without_a_host_is_refused_with_a_clear_message(saved):
    with pytest.raises(ValueError, match="Azure endpoint"):
        v1_base_url(saved)


def test_the_client_is_the_plain_openai_client_on_v1_with_no_api_version(monkeypatch):
    saved = f"{RESOURCE}/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview"
    provider, built = _provider(endpoint=saved, monkeypatch=monkeypatch)
    assert built["base_url"] == V1
    assert built["api_key"] == "azure-key"
    assert "api_version" not in built and "azure_endpoint" not in built
    assert not hasattr(azure_client, "AzureOpenAI")


def test_the_env_endpoint_is_used_when_the_credential_has_none(monkeypatch):
    monkeypatch.setattr(azure_client.config, "AZURE_OPENAI_ENDPOINT", f"{RESOURCE}/openai/v1")
    provider, built = _provider(endpoint=None, monkeypatch=monkeypatch)
    assert built["base_url"] == V1


# --------------------------------------------------------------------------- #
# The request a chat deployment gets
# --------------------------------------------------------------------------- #


def test_a_chat_deployment_gets_max_completion_tokens_and_its_temperature(monkeypatch):
    provider, _ = _provider(monkeypatch=monkeypatch)
    out = asyncio.run(provider.generate_response(HI))

    [sent] = _sent(provider)
    assert sent["model"] == "prod-chat"
    assert sent["max_completion_tokens"] == 1024
    assert "max_tokens" not in sent
    assert sent["temperature"] == 0.7
    assert out.content == "ok" and out.provider == "azure"
    assert out.usage["total_tokens"] == 5


def test_this_calls_budget_crosses_the_thread_hop_into_max_completion_tokens(monkeypatch):
    from core.llm.output_budget import call_budget

    provider, _ = _provider(monkeypatch=monkeypatch)

    async def run():
        with call_budget(256):
            await provider.generate_response(HI)
    asyncio.run(run())
    assert _sent(provider)[0]["max_completion_tokens"] == 256  # not the config's 1,024


def test_tools_go_out_and_the_tool_calls_come_back(monkeypatch):
    call = NS(id="call_1", type="function", function=NS(name="lookup", arguments='{"q": "x"}'))
    provider, _ = _provider(script=[_reply(tool_calls=[call])], monkeypatch=monkeypatch)
    tool = {"name": "lookup", "parameters": {"type": "object", "properties": {}}}
    out = asyncio.run(provider.generate_response(HI, tools=[tool]))

    [sent] = _sent(provider)
    assert sent["tools"][0]["function"]["name"] == "lookup"
    assert sent["tool_choice"] == "auto"
    assert out.tool_calls == [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"q": "x"}'}}
    ]


# --------------------------------------------------------------------------- #
# A deployment that refuses temperature
# --------------------------------------------------------------------------- #


def test_a_refused_temperature_is_retried_once_without_it_then_remembered(monkeypatch):
    refusal = _refusal(TEMPERATURE_REFUSED, "temperature")
    provider, _ = _provider("gpt5-uk", script=[refusal, _reply("gpt-5.1-2025-11-13")], monkeypatch=monkeypatch)
    out = asyncio.run(provider.generate_response(HI))

    first, retry = _sent(provider)
    assert first["temperature"] == 0.7
    assert "temperature" not in retry
    assert retry["max_completion_tokens"] == 1024
    assert out.content == "ok"

    # The next call to the same deployment, from a new client, sends no temperature at all.
    again, _ = _provider("gpt5-uk", monkeypatch=monkeypatch)
    again.generate_response_sync(HI)
    [only] = _sent(again)
    assert "temperature" not in only


def test_the_memory_is_per_deployment(monkeypatch):
    refused, _ = _provider("gpt5-uk", script=[_refusal(TEMPERATURE_REFUSED, "temperature")], monkeypatch=monkeypatch)
    asyncio.run(refused.generate_response(HI))

    other, _ = _provider("prod-chat", monkeypatch=monkeypatch)
    asyncio.run(other.generate_response(HI))
    assert _sent(other)[0]["temperature"] == 0.7


def test_the_sync_path_retries_the_same_way(monkeypatch):
    provider, _ = _provider("gpt5-uk", script=[_refusal(TEMPERATURE_REFUSED, "temperature")], monkeypatch=monkeypatch)
    out = provider.generate_response_sync(HI)
    assert [("temperature" in c) for c in _sent(provider)] == [True, False]
    assert out.content == "ok"


def test_a_second_refusal_is_not_retried_again(monkeypatch):
    refusal = _refusal(TEMPERATURE_REFUSED, "temperature")
    still = _refusal("Unsupported parameter: 'max_completion_tokens' is not supported.", "max_completion_tokens")
    provider, _ = _provider("gpt5-uk", script=[refusal, still], monkeypatch=monkeypatch)
    with pytest.raises(Exception, match="max_completion_tokens"):
        asyncio.run(provider.generate_response(HI))
    assert len(_sent(provider)) == 2


def test_a_refused_max_completion_tokens_falls_back_to_max_tokens(monkeypatch):
    refusal = _refusal("Unsupported parameter: 'max_completion_tokens' is not supported with this model.",
                       "max_completion_tokens", code="unsupported_parameter")
    provider, _ = _provider("llama-eu", script=[refusal], monkeypatch=monkeypatch)
    asyncio.run(provider.generate_response(HI))

    first, retry = _sent(provider)
    assert "max_completion_tokens" in first
    assert retry["max_tokens"] == 1024 and "max_completion_tokens" not in retry
    assert retry["temperature"] == 0.7


def test_any_other_400_is_raised_untouched_and_nothing_is_remembered(monkeypatch):
    other = _refusal("The response was filtered due to the prompt triggering content management policy.",
                     None, code="content_filter")
    provider, _ = _provider("prod-chat", script=[other], monkeypatch=monkeypatch)
    with pytest.raises(Exception, match="content management"):
        asyncio.run(provider.generate_response(HI))
    assert len(_sent(provider)) == 1

    again, _ = _provider("prod-chat", monkeypatch=monkeypatch)
    asyncio.run(again.generate_response(HI))
    assert _sent(again)[0]["temperature"] == 0.7


def test_a_reasoning_model_named_in_the_response_is_remembered(monkeypatch):
    provider, _ = _provider("prod-reasoner", script=[_reply("o3-mini-2025-01-31")], monkeypatch=monkeypatch)
    asyncio.run(provider.generate_response(HI))
    assert _sent(provider)[0]["temperature"] == 0.7  # nothing known before the first reply

    again, _ = _provider("prod-reasoner", monkeypatch=monkeypatch)
    asyncio.run(again.generate_response(HI))
    assert "temperature" not in _sent(again)[0]


def test_a_deployment_named_after_a_reasoning_model_sends_no_temperature(monkeypatch):
    provider, _ = _provider("o4-mini", monkeypatch=monkeypatch)
    asyncio.run(provider.generate_response(HI))
    assert "temperature" not in _sent(provider)[0]


def test_refused_params_reads_only_parameters_that_were_sent():
    refusal = _refusal("Unsupported parameter: 'max_tokens' is not supported with this model. "
                       "Use 'max_completion_tokens' instead.", "max_tokens", code="unsupported_parameter")
    assert refused_params(refusal, ["model", "messages", "max_tokens", "temperature"]) == frozenset()
    assert refused_params(_refusal(TEMPERATURE_REFUSED, "temperature"), ["temperature"]) == {"temperature"}
    assert refused_params(ValueError("unsupported temperature"), ["temperature"]) == frozenset()


# --------------------------------------------------------------------------- #
# One rule for which models refuse sampling parameters
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", [
    "o1", "o1-mini", "o3", "o3-mini", "o3-mini-2025-01-31", "o4-mini", "o3-pro", "codex-mini",
    "gpt-5", "gpt-5-mini", "gpt-5-nano-2025-08-07", "gpt-5-codex", "openai/o3", "openai/gpt-5-mini",
])
def test_openai_reasoning_models_refuse_sampling_params(model):
    assert accepts_sampling_params(model) is False


@pytest.mark.parametrize("model", [
    "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-5-chat-latest", "gpt-5.1", "openai/gpt-4o",
    "prod-chat", "DeepSeek-R1", "Llama-4-Maverick-17B-128E-Instruct-FP8", None,
])
def test_chat_models_and_unknown_deployments_take_sampling_params(model):
    assert accepts_sampling_params(model) is True

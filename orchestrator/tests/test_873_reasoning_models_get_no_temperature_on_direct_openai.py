"""#873: the direct OpenAI client sent temperature and max_tokens to every model,
so an agent on o3 or gpt-5 got a 400 ("Unsupported parameter: 'max_tokens'").
It now shares the Azure client's rule (accepts_sampling_params): a reasoning
model gets no sampling parameters and its budget as max_completion_tokens; every
other model gets the request it got before.

The provider registry names the Azure provider for what it reaches, Microsoft
Foundry, and tells the user the model field is the deployment name."""
import asyncio
from types import SimpleNamespace as NS
from unittest.mock import MagicMock
from urllib.parse import urlsplit

import pytest

from core.llm import providers as registry
from core.llm.clients.base import LLMConfig, LLMProvider

HI = [{"role": "user", "content": "hi"}]


def _openai(model):
    from core.llm.clients.openai_client import OpenAIProvider

    provider = OpenAIProvider.__new__(OpenAIProvider)  # bypass _initialize_client
    provider.config = LLMConfig(provider=LLMProvider.OPENAI, model=model, temperature=0.7, max_tokens=1024, top_p=0.9)
    provider.client = MagicMock()
    usage = NS(prompt_tokens=1, completion_tokens=1, total_tokens=2, prompt_tokens_details=None)
    provider.client.chat.completions.create.return_value = NS(
        choices=[NS(message=NS(content="ok", tool_calls=None), finish_reason="stop")], usage=usage, model=model,
    )
    return provider


@pytest.mark.parametrize("model", ["o3", "o4-mini", "gpt-5-mini"])
def test_a_reasoning_model_gets_max_completion_tokens_and_no_sampling(model):
    provider = _openai(model)
    asyncio.run(provider.generate_response(HI))
    sent = provider.client.chat.completions.create.call_args.kwargs
    assert sent["max_completion_tokens"] == 1024
    assert not {"max_tokens", "temperature", "top_p"} & set(sent)


def test_a_chat_model_gets_the_request_it_always_got():
    provider = _openai("gpt-4o")
    out = provider.generate_response_sync(HI)
    sent = provider.client.chat.completions.create.call_args.kwargs
    assert sent["max_tokens"] == 1024 and sent["temperature"] == 0.7 and sent["top_p"] == 0.9
    assert "max_completion_tokens" not in sent
    assert out.content == "ok"


def test_the_azure_provider_is_named_for_microsoft_foundry():
    spec = registry.get_spec("azure")
    assert spec.label == "Azure OpenAI (Microsoft Foundry)"
    docs = urlsplit(spec.docs_url)
    assert docs.scheme == "https" and docs.hostname == "learn.microsoft.com"
    assert docs.path.startswith("/azure/foundry/")
    assert "deployment name" in spec.setup_note


def test_the_azure_slug_and_its_aliases_still_resolve():
    assert registry.normalize_slug("azure") == "azure"
    assert registry.normalize_slug("azure_openai") == "azure"
    assert registry.enum_for("azure") == LLMProvider.AZURE


def test_the_setup_note_reaches_the_ui():
    public = registry.to_public_dict(registry.get_spec("azure"))
    assert "deployment name" in public["setup_note"]
    assert registry.to_public_dict(registry.get_spec("openai"))["setup_note"] is None

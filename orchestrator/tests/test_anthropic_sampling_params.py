"""Claude Opus 4.7 and later, Sonnet 5, Fable and Mythos answer temperature,
top_p and top_k with a 400. The direct Anthropic route sent temperature on every
call, so an agent on one of those models could not get a reply.

PRD-256 US-008: Opus and Sonnet 4.6 moved to the first list. They take sampling
parameters, but the whole Claude 4.6+ family now goes out with one request shape
(no temperature), so switching Auto between them is one line."""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.llm.clients.base import accepts_sampling_params


@pytest.mark.parametrize("model", [
    "claude-opus-4-7", "claude-opus-4-8", "claude-opus-5", "claude-opus-5-5", "claude-sonnet-5",
    "claude-fable-5-1", "anthropic.claude-opus-4-8", "anthropic/claude-opus-4.8",
    "claude-sonnet-4-6", "claude-opus-4-6", "anthropic/claude-sonnet-4.6",
    "claude-haiku-5-5", "claude-sonnet-5-5", "anthropic/claude-haiku-5.5",
])
def test_models_that_reject_sampling_params(model):
    assert accepts_sampling_params(model) is False


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "anthropic/claude-haiku-4.5", "claude-sonnet-4-5", None])
def test_models_that_take_sampling_params(model):
    assert accepts_sampling_params(model) is True


def _provider(model):
    from core.llm.clients.anthropic_client import AnthropicProvider
    from core.llm.clients.base import LLMConfig, LLMProvider

    prov = AnthropicProvider.__new__(AnthropicProvider)  # bypass _initialize_client
    prov.config = LLMConfig(provider=LLMProvider.ANTHROPIC, model=model, max_tokens=1024, temperature=0.7)
    prov.client = MagicMock()
    prov.client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        model=model,
        stop_reason="end_turn",
    )
    return prov


def test_the_direct_route_leaves_temperature_out_where_the_model_rejects_it():
    prov = _provider("claude-opus-4-8")
    asyncio.run(prov.generate_response([{"role": "user", "content": "hi"}]))
    assert "temperature" not in prov.client.messages.create.call_args.kwargs


def test_the_direct_route_keeps_temperature_where_the_model_takes_it():
    prov = _provider("claude-haiku-4-5")
    asyncio.run(prov.generate_response([{"role": "user", "content": "hi"}]))
    assert prov.client.messages.create.call_args.kwargs["temperature"] == 0.7

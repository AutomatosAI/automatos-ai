"""PRD-256 US-008 — a Claude-ready LLM manager.

Switching Auto to a Claude 4.6+ model is one line: the request goes out without
sampling parameters on every client path (the OpenRouter adapter sent temperature on
every call), the governor prices the Claude arms of the model test at list price, and
a refused or rate-limited call fails the turn honestly: the failover model comes from
``LLM_FAILOVER_MODEL`` in config.py and is empty by default (Decision D5).

The OpenRouter tests run on a recorded transport: a fake SDK client that keeps the
request it was sent and answers with recorded stream chunks. No provider is called.
"""
from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from config import config  # noqa: E402
from core.llm import failover  # noqa: E402
from core.llm.clients.base import LLMConfig, LLMProvider, accepts_sampling_params  # noqa: E402
from core.llm.clients.openai_compatible_client import (  # noqa: E402
    OpenAICompatibleProvider,
    ProviderRateLimitError,
)
from core.llm.manager import LLMManager, estimate_cost_usd  # noqa: E402

SONNET_5 = "anthropic/claude-sonnet-5"
SAMPLING = ("temperature", "top_p", "top_k")
MESSAGES = [{"role": "system", "content": "You are Auto."}, {"role": "user", "content": "How many open tasks?"}]


# ── the recorded transport ──────────────────────────────────────────────────

def _chunk(*, content=None, reasoning=None, finish=None, usage=None, model=SONNET_5):
    delta = SimpleNamespace(model_dump=lambda: {"content": content, "reasoning": reasoning, "tool_calls": None})
    return SimpleNamespace(model=model, usage=usage, choices=[SimpleNamespace(delta=delta, finish_reason=finish)])


RECORDED_STREAM = [
    _chunk(reasoning="The board has "),
    _chunk(reasoning="three open cards."),
    _chunk(content="You have three open tasks."),
    _chunk(finish="stop"),
    SimpleNamespace(model=SONNET_5, choices=[],
                    usage=SimpleNamespace(prompt_tokens=1200, completion_tokens=40, total_tokens=1240)),
]


class _RecordedTransport:
    """``client.chat.completions.create``: keeps each request, answers from the recording."""

    def __init__(self, chunks):
        self.sent = []
        self.chunks = chunks
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.sent.append(kwargs)
        return iter(self.chunks)


def _openrouter(model=SONNET_5, *, top_p=1.0):
    from core.llm import providers as registry

    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)  # no SDK client is built
    provider.config = LLMConfig(provider=LLMProvider.OPENROUTER, model=model, max_tokens=1024,
                                temperature=0.7, top_p=top_p, api_key="k")
    provider.spec = registry.get_spec("openrouter")
    provider.client = _RecordedTransport(RECORDED_STREAM)
    return provider


@pytest.fixture(autouse=True)
def _no_web_tool(monkeypatch):
    monkeypatch.setattr(config, "WEB_ACCESS", False, raising=False)
    monkeypatch.setattr(config, "PROMPT_CACHE_TTL_1H", False, raising=False)


def _stream(provider):
    deltas = []

    async def on_delta(kind, text):
        deltas.append((kind, text))

    response = asyncio.run(provider.stream_response(MESSAGES, None, on_delta=on_delta))
    return response, deltas


# ── no sampling parameters for a Claude 4.6+ id, on every path ─────────────

@pytest.mark.parametrize("model", [
    "claude-sonnet-5", "anthropic/claude-sonnet-5", "claude-opus-5", "claude-sonnet-4-6",
    "anthropic/claude-opus-4.6", "anthropic.claude-opus-4-7", "claude-fable-5-1",
])
def test_the_one_rule_refuses_sampling_for_claude_4_6_and_later(model):
    assert accepts_sampling_params(model) is False


def test_a_sonnet_5_request_through_openrouter_carries_no_temperature():
    provider = _openrouter()
    _stream(provider)
    sent = provider.client.sent[0]
    assert sent["model"] == SONNET_5 and sent["stream"] is True
    assert not any(name in sent for name in SAMPLING)


def test_the_non_streaming_openrouter_path_carries_none_either():
    provider = _openrouter()
    kwargs = provider._base_kwargs(MESSAGES)
    assert not any(name in kwargs for name in SAMPLING) and kwargs["max_tokens"] == 1024


def test_a_model_that_takes_sampling_keeps_temperature_on_openrouter():
    provider = _openrouter("google/gemini-2.5-flash", top_p=0.9)
    kwargs = provider._base_kwargs(MESSAGES)
    assert kwargs["temperature"] == 0.7 and kwargs["top_p"] == 0.9


def test_an_older_claude_takes_temperature_but_never_top_p_beside_it():
    provider = _openrouter("anthropic/claude-haiku-4.5", top_p=1.0)
    kwargs = provider._base_kwargs(MESSAGES)
    assert kwargs["temperature"] == 0.7 and "top_p" not in kwargs


# ── reasoning passthrough: the request as the manager sends it, the deltas to PRD-238 ──

def test_claude_reasoning_through_openrouter_reaches_the_stream():
    provider = _openrouter()
    response, deltas = _stream(provider)
    sent = provider.client.sent[0]
    # The manager sets no reasoning field of its own: OpenRouter's default for the model
    # stands, and the request asks only for the usage and the cache it always asks for.
    assert "reasoning" not in sent and "reasoning" not in (sent.get("extra_body") or {})
    assert sent["extra_body"]["usage"] == {"include": True}
    assert deltas[:2] == [("reasoning", "The board has "), ("reasoning", "three open cards.")]
    assert ("text", "You have three open tasks.") in deltas
    assert response.reasoning == "The board has three open cards."
    assert response.content == "You have three open tasks." and response.model == SONNET_5


# ── the governor prices the Claude arms at list price ──────────────────────

@pytest.mark.parametrize("model, per_1k", [
    ("claude-sonnet-5", (0.002, 0.010)),
    (SONNET_5, (0.002, 0.010)),
    ("claude-opus-5", (0.005, 0.025)),
    ("claude-haiku-4-5", (0.001, 0.005)),
    ("anthropic/claude-haiku-4.5", (0.001, 0.005)),
])
def test_the_governor_prices_the_claude_arms(model, per_1k):
    assert estimate_cost_usd(model, 1000, 0) == pytest.approx(per_1k[0])
    assert estimate_cost_usd(model, 0, 1000) == pytest.approx(per_1k[1])


def test_haiku_4_5_is_not_priced_as_the_older_haiku_4():
    assert estimate_cost_usd("claude-haiku-4-5", 1000, 0) != pytest.approx(0.0008)


# ── the failover: from config, empty by default ────────────────────────────

class _Refusing:
    def __init__(self):
        self.calls = 0

    async def generate_response(self, messages, tools=None):
        self.calls += 1
        raise ProviderRateLimitError("OpenRouter rate limit reached for model 'anthropic/claude-sonnet-5'.")


@pytest.fixture
def quiet_manager(monkeypatch):
    for name in ("_track_usage", "_note_success", "_note_cut"):
        monkeypatch.setattr(LLMManager, name, lambda *a, **k: None)
    monkeypatch.setattr(LLMManager, "_call_budget", lambda self: None)


def _manager(provider):
    mgr = object.__new__(LLMManager)
    mgr.config = LLMConfig(provider=LLMProvider.OPENROUTER, model=SONNET_5, api_key="k")
    mgr.service_name = "orchestrator"
    mgr._tracking_ctx = {"workspace_id": None, "agent_id": 1, "execution_id": None,
                         "request_type": "chat", "is_byok": False, "trial": False}
    mgr.provider = provider
    return mgr


def test_the_failover_model_is_empty_by_default():
    assert config.LLM_FAILOVER_MODEL == ""
    assert failover.failover_model() is None


def test_a_rate_limited_call_fails_the_turn_honestly_and_asks_no_other_model(quiet_manager, monkeypatch):
    from consumers.chatbot.turn_errors import CODE_RATE_LIMITED, describe_turn_error

    monkeypatch.setattr(config, "LLM_FAILOVER_MODEL", "")
    refusing = _Refusing()
    created = []
    monkeypatch.setattr(LLMManager, "_create_provider", staticmethod(lambda cfg: created.append(cfg) or None))
    with pytest.raises(ProviderRateLimitError) as raised:
        asyncio.run(_manager(refusing).generate_response(MESSAGES))
    assert refusing.calls == 1 and created == []
    assert describe_turn_error(raised.value, agent_name="Auto").code == CODE_RATE_LIMITED


def test_a_failover_set_in_config_answers_once_and_names_its_model(quiet_manager, monkeypatch):
    from core.llm import output_budget

    monkeypatch.setattr(config, "LLM_FAILOVER_MODEL", "anthropic/claude-haiku-4.5")
    monkeypatch.setattr(output_budget, "manager_budgets", lambda *a, **k: (None, None))

    class _Answering:
        def __init__(self, cfg):
            self.cfg = cfg

        async def generate_response(self, messages, tools=None):
            return SimpleNamespace(content="Three.", model=self.cfg.model, usage=None)

    monkeypatch.setattr(LLMManager, "_create_provider", staticmethod(_Answering))
    response = asyncio.run(_manager(_Refusing()).generate_response(MESSAGES))
    assert response.model == "anthropic/claude-haiku-4.5" and response.content == "Three."


def test_with_no_failover_set_no_refusal_fails_over(monkeypatch):
    monkeypatch.setattr(config, "LLM_FAILOVER_MODEL", "")
    assert failover.failover_for(ProviderRateLimitError("rate limited"), SONNET_5) is None
    assert failover.failover_for(SimpleNamespace(status_code=429), SONNET_5) is None


def test_the_failover_is_asked_only_for_a_refusal(monkeypatch):
    monkeypatch.setattr(config, "LLM_FAILOVER_MODEL", "anthropic/claude-haiku-4.5")
    assert failover.failover_for(ProviderRateLimitError("429"), SONNET_5) == "anthropic/claude-haiku-4.5"
    assert failover.failover_for(ValueError("bad request"), SONNET_5) is None
    assert failover.failover_for(ProviderRateLimitError("429"), "anthropic/claude-haiku-4.5") is None  # never itself


def test_the_receipts_frame_carries_the_model_that_answered():
    from consumers.chatbot.receipts import model_of, receipts_frame
    from consumers.chatbot.streaming import get_streaming_handler

    frame = receipts_frame(get_streaming_handler(), [], model_of(SimpleNamespace(model=SONNET_5)))
    assert SONNET_5 in frame

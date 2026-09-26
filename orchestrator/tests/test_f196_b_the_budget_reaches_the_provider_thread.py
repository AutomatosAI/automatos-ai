"""F196, reopened after the refresh-6 build: the per-call budget reaches the
request, which is built on the executor thread.

Five minutes after the 09:26:02Z restart, 8 refusals read "You requested up to
8000 tokens, but can only afford 3972". They were the digest, whose budget is
1,024. The providers build their request inside _call, which ran through
loop.run_in_executor(None, _call), and that does not carry context variables
into the thread. So request_max_tokens found no per-call budget and sent the
config's 8,000. The log context (req/ws/agent) was empty on the same lines.
F196's own test called _base_kwargs in the coroutine, so it never crossed the
thread.

This drives the real OpenRouter-compatible provider, whose SDK client is swapped
for a stub that keeps what it was sent.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

from core.llm.clients.base import LLMConfig, LLMProvider

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
MESSAGES = [{"role": "user", "content": "Summarise the day."}]


class _Completions:
    def __init__(self):
        self.sent, self.seen_workspace = [], []

    def create(self, **kwargs):
        from core.monitoring.automatos_logging import workspace_id_var

        self.sent.append(kwargs)
        self.seen_workspace.append(workspace_id_var.get())
        if kwargs.get("stream"):
            return iter(())
        message = NS(content="ok", tool_calls=None, model_dump=lambda: {"content": "ok"})
        return NS(choices=[NS(message=message, finish_reason="stop")], model="google/gemini-2.5-flash", usage=None,
                  id="gen-1")


@pytest.fixture
def provider():
    from core.llm.clients.openai_compatible_client import OpenAICompatibleProvider

    made = OpenAICompatibleProvider(LLMConfig(provider=LLMProvider.OPENROUTER, model="google/gemini-2.5-flash",
                                              max_tokens=8000, api_key="k"))
    made.client = NS(chat=NS(completions=_Completions()))
    return made


def _under_the_digests_budget(call):
    from core.llm.output_budget import call_budget
    from core.monitoring.automatos_logging import workspace_id_var

    async def run():
        workspace_id_var.set(WS)
        with call_budget(1024):
            await call()
    asyncio.run(run())


def test_a_request_sends_this_calls_budget(provider):
    _under_the_digests_budget(lambda: provider.generate_response(MESSAGES))
    ((sent,),) = [provider.client.chat.completions.sent]
    assert sent["max_tokens"] == 1024                                     # was the config's 8,000


def test_a_streamed_request_sends_this_calls_budget(provider):
    _under_the_digests_budget(lambda: provider.stream_response(MESSAGES, None, on_delta=None))
    ((sent,),) = [provider.client.chat.completions.sent]
    assert sent["stream"] is True and sent["max_tokens"] == 1024


def test_the_provider_thread_keeps_the_log_context(provider):
    _under_the_digests_budget(lambda: provider.generate_response(MESSAGES))
    assert provider.client.chat.completions.seen_workspace == [WS]       # was "" (req= ws= agent=)


def test_every_provider_hops_to_its_thread_through_the_same_helper():
    """The same executor hop was at every provider call site; none is left."""
    import pathlib

    clients = pathlib.Path(__file__).resolve().parents[1] / "core" / "llm" / "clients"
    hops = {p.name: p.read_text(encoding="utf-8").count("run_in_executor(") for p in clients.glob("*.py")}
    assert {name: n for name, n in hops.items() if n and name != "base.py"} == {}

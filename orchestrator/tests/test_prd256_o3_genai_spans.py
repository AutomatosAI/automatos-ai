"""PRD-256 O3 (#847): every LLM call is one GenAI span; the PRD-185 seam emits spans.

On: an ``LLMManager`` call, async or sync, is a ``chat {model}`` CLIENT span under the
request that made it, with the provider, models, tokens, finish reason and cost; a tool
dispatch, a retrieval and a context assembly are spans under the current one. No prompt,
completion, query, tool argument or error message reaches a span. With no request
around them, none starts a trace (O4 gives background work its own). Off: nothing from
``opentelemetry`` is imported. ``LangfuseTracer`` and its settings are gone (GUARDRAILS B2).
Spans go to an in-memory exporter, through the real request-rooted sampler.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.observability import genai, otel, tracer as seam  # noqa: E402
from tests.helpers_otel import fresh_global_providers  # noqa: E402

SECRET = "SECRET-VALUE-7"
MODEL = "gpt-test-1"


# ── off ──────────────────────────────────────────────────────────────────────

def test_off_a_call_and_every_emit_import_nothing():
    script = (
        "import asyncio, sys\n"
        "from core.observability import genai, tracer\n"
        "@genai.traced_llm_call\n"
        "async def call(manager, messages):\n"
        "    return 'answer'\n"
        "assert asyncio.run(call(object(), [])) == 'answer'\n"
        "tracer.fire_tool_trace(tool_name='t', success=True, duration_ms=1)\n"
        "tracer.fire_retrieval_score(query='q', num_docs=0, top_score=0.0, status='empty')\n"
        "tracer.fire_assembly_trace(trace={'mode': 'chatbot'})\n"
        "assert type(tracer.get_tracer()).__name__ == 'NoOpTracer'\n"
        "loaded = [m for m in sys.modules if m.startswith('opentelemetry')]\n"
        "assert not loaded, loaded\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "OTEL_ENABLED"}
    done = subprocess.run([sys.executable, "-c", script], cwd=_ORCH, env=env, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]


def test_langfuse_and_its_settings_are_gone():
    from config import config

    assert not hasattr(seam, "LangfuseTracer")
    for name in ("TRACING_ENABLED", "TRACING_BACKEND", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        assert not hasattr(config, name), name


def test_the_attribute_names_are_the_semantic_conventions():
    from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as g

    for name in ("gen_ai.operation.name", "gen_ai.provider.name", "gen_ai.request.model", "gen_ai.response.model",
                 "gen_ai.response.finish_reasons", "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens",
                 "gen_ai.usage.cache_read.input_tokens", "gen_ai.usage.cache_creation.input_tokens",
                 "gen_ai.tool.name"):
        assert name in vars(g).values(), name


# ── on ───────────────────────────────────────────────────────────────────────

@pytest.fixture
def spans(monkeypatch):
    """Tracing on: a fresh global provider built as the app builds it (the request-rooted
    sampler, batched export) into an in-memory exporter; read with ``finished()``."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    monkeypatch.setattr(otel.config, "OTEL_ENABLED", True, raising=False)
    exporter = InMemorySpanExporter()
    provider = otel.build_provider(exporter, 1.0)

    def finished():
        provider.force_flush()
        return exporter.get_finished_spans()

    with fresh_global_providers():
        trace.set_tracer_provider(provider)
        seam.reset_tracer()
        yield finished
        seam.reset_tracer()
        provider.shutdown()


def _request():
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind

    return trace.get_tracer("t").start_as_current_span("POST /api/chat", kind=SpanKind.SERVER)


def _leaks(finished):
    found = []
    for span in finished:
        values = [*span.attributes.values(), span.status.description or ""]
        values += [v for event in span.events for v in (event.name, *event.attributes.values())]
        found += [(span.name, v) for v in values if SECRET in str(v)]
    return found


class _Provider:
    """A provider that answers with the secret, a tool call and the usage it reports."""

    def __init__(self, fail=False):
        self.fail = fail

    def _answer(self, messages):
        from core.llm.clients.base import LLMResponse

        if self.fail:
            raise RuntimeError(f"provider said no to {SECRET}")
        return LLMResponse(content=f"reply {SECRET}", model=f"{MODEL}-2026-10-01", finish_reason="tool_calls",
                           tool_calls=[{"id": "c1", "function": {"name": "search", "arguments": SECRET}}],
                           usage={"input_tokens": 120, "output_tokens": 30, "cache_read_tokens": 100,
                                  "cache_write_tokens": 5, "cost": 0.0123})

    async def generate_response(self, messages, tools=None):
        return self._answer(messages)

    def generate_response_sync(self, messages):
        return self._answer(messages)


def _manager(provider):
    from core.llm.clients.base import LLMConfig, LLMProvider
    from core.llm.manager import LLMManager

    manager = LLMManager(config=LLMConfig(provider=LLMProvider.AZURE, model=MODEL, api_key="test"),
                         agent_id=7, request_type="chat")
    manager.provider = provider
    return manager


PROMPT = [{"role": "user", "content": f"tell me about {SECRET}"}]


def test_an_llm_call_is_a_chat_span_under_the_request_with_tokens_and_cost(spans):
    manager = _manager(_Provider())
    with _request() as request:
        asyncio.run(manager.generate_response(PROMPT))
    [call] = [s for s in spans() if s.name.startswith("chat ")]
    assert call.name == f"chat {MODEL}" and call.kind.name == "CLIENT"
    assert call.parent.span_id == request.get_span_context().span_id
    assert dict(call.attributes) | {} == {
        "gen_ai.operation.name": "chat", "gen_ai.provider.name": "azure.ai.openai",
        "gen_ai.request.model": MODEL, "gen_ai.response.model": f"{MODEL}-2026-10-01",
        "gen_ai.response.finish_reasons": ("tool_calls",), "gen_ai.usage.input_tokens": 120,
        "gen_ai.usage.output_tokens": 30, "gen_ai.usage.cache_read.input_tokens": 100,
        "gen_ai.usage.cache_creation.input_tokens": 5, "automatos.llm.cost_usd": 0.0123,
        "automatos.llm.cost_source": "reported", "automatos.llm.tool_calls": 1, "automatos.agent_id": "7",
        "automatos.llm.request_type": "chat", "automatos.llm.byok": False, "automatos.llm.streamed": False,
    }
    assert _leaks(spans()) == []


def test_the_sync_path_is_the_same_span(spans):
    manager = _manager(_Provider())
    with _request():
        manager.generate_response_sync(PROMPT)
    [call] = [s for s in spans() if s.name.startswith("chat ")]
    assert call.attributes["gen_ai.usage.output_tokens"] == 30
    assert call.attributes["automatos.llm.cost_source"] == "reported"


def test_a_failed_call_is_an_error_span_without_its_message(spans):
    from opentelemetry.trace import StatusCode

    manager = _manager(_Provider(fail=True))
    with _request(), pytest.raises(RuntimeError):
        asyncio.run(manager.generate_response(PROMPT))
    [call] = [s for s in spans() if s.name.startswith("chat ")]
    assert call.status.status_code == StatusCode.ERROR
    assert call.attributes["error.type"] == "RuntimeError"
    assert call.events == () and _leaks(spans()) == []


def test_an_llm_call_outside_a_request_starts_no_trace(spans):
    asyncio.run(_manager(_Provider()).generate_response(PROMPT))
    assert spans() == ()


def test_a_tool_call_is_an_execute_tool_span_dated_by_its_duration(spans):
    from opentelemetry.trace import StatusCode

    with _request() as request:
        seam.fire_tool_trace(tool_name="search_knowledge", success=False, duration_ms=40, workspace_id="ws-1",
                             agent_id=7, error=f"failed on {SECRET}")
    [tool] = [s for s in spans() if s.name == "execute_tool search_knowledge"]
    assert tool.parent.span_id == request.get_span_context().span_id
    assert tool.attributes["gen_ai.tool.name"] == "search_knowledge"
    assert tool.attributes["automatos.tool.success"] is False and tool.attributes["error.type"] == "tool_error"
    assert tool.status.status_code == StatusCode.ERROR
    assert tool.end_time - tool.start_time == 40_000_000
    assert _leaks(spans()) == []


def test_a_retrieval_is_a_span_without_its_query(spans):
    with _request():
        seam.fire_retrieval_score(query=f"what about {SECRET}", num_docs=4, top_score=0.82, status=seam.STATUS_HIT,
                                  workspace_id="ws-1", metadata={"seam": "documents", "latency_ms": 12})
    [retrieval] = [s for s in spans() if s.name == "rag.retrieve"]
    assert retrieval.attributes["automatos.retrieval.num_docs"] == 4
    assert retrieval.attributes["automatos.retrieval.grounded"] is True
    assert retrieval.attributes["automatos.seam"] == "documents"
    assert retrieval.end_time - retrieval.start_time == 12_000_000
    assert _leaks(spans()) == []


def test_an_assembly_is_a_span_of_its_shape_not_its_content(spans):
    trace = {"mode": "chatbot", "model": MODEL, "budget_total": 1000, "token_estimate": 250, "token_budget": 900,
             "prep_ms": 3.5, "sections": [{"name": "memory", "rendered": SECRET}],
             "sections_included": ["identity", "memory"], "sections_trimmed": ["history"],
             "injected_memory_ids": ["m1", "m2"]}
    with _request():
        seam.fire_assembly_trace(trace=trace, workspace_id="ws-1", metadata={"agent_id": 7})
    [assembly] = [s for s in spans() if s.name == "context.assembly chatbot"]
    assert assembly.attributes["automatos.context.budget_fraction"] == 0.25
    assert assembly.attributes["automatos.context.sections_trimmed"] == ("history",)
    assert assembly.attributes["automatos.context.memory_count"] == 2
    assert assembly.attributes["automatos.agent_id"] == 7
    assert _leaks(spans()) == []


def test_an_emit_outside_a_request_records_nothing(spans):
    seam.fire_tool_trace(tool_name="search_knowledge", success=True, duration_ms=5)
    seam.fire_retrieval_score(query="q", num_docs=0, top_score=0.0, status=seam.STATUS_EMPTY)
    assert spans() == ()


def test_a_tracing_fault_never_fails_the_call(spans, monkeypatch):
    def broken(*_):
        raise RuntimeError("attributes")

    monkeypatch.setattr(genai, "response_attributes", broken)
    with _request():
        response = asyncio.run(_manager(_Provider()).generate_response(PROMPT))
    assert response.content.startswith("reply")
    assert [s.name for s in spans() if s.name.startswith("chat ")] == [f"chat {MODEL}"]

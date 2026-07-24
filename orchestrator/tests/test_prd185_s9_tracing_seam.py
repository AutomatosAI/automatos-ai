"""PRD-185 S9 — vendor-neutral tracing seam.

The two platform chokepoints (tool dispatch + RAG retrieval) must emit a live
trace/score so "was the tool call good / was retrieval grounded" becomes a
queryable number over real traffic. The seam is config-gated, default-OFF, and
never fails the caller.

Pure tests — no DB, network, ``langfuse``, or app import chain (per
``feedback-no-local-servers``). ``core/observability/tracer.py`` imports nothing
heavy at module load (langfuse + config are lazy), and the Langfuse client is
mocked at the boundary.
"""
import pathlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.observability.tracer import (
    Tracer,
    NoOpTracer,
    LangfuseTracer,
    get_tracer,
    reset_tracer,
    should_trace,
    fire_tool_trace,
    fire_retrieval_score,
    STATUS_HIT,
    STATUS_EMPTY,
    STATUS_ERROR,
)

ORCH = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _reset_tracer():
    """Isolate the memoized process-wide tracer between tests."""
    reset_tracer()
    yield
    reset_tracer()


# ── should_trace: the pure enable decision ────────────────────────────────────


def _cfg(**kw):
    base = dict(
        TRACING_ENABLED=True,
        TRACING_BACKEND="langfuse",
        LANGFUSE_PUBLIC_KEY="pk-1",
        LANGFUSE_SECRET_KEY="sk-1",
        LANGFUSE_HOST="https://cloud.langfuse.com",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_should_trace_true_when_enabled_with_backend_and_keys():
    assert should_trace(_cfg()) is True


def test_should_trace_false_when_disabled():
    # Default posture — the whole point of the flag. Off = no real tracer.
    assert should_trace(_cfg(TRACING_ENABLED=False)) is False


def test_should_trace_false_when_keys_missing():
    assert should_trace(_cfg(LANGFUSE_SECRET_KEY=None)) is False
    assert should_trace(_cfg(LANGFUSE_PUBLIC_KEY="")) is False


def test_should_trace_false_for_unknown_backend():
    assert should_trace(_cfg(TRACING_BACKEND="datadog")) is False
    assert should_trace(_cfg(TRACING_BACKEND=None)) is False


# ── get_tracer: default OFF is a memoized no-op ───────────────────────────────


def test_get_tracer_default_off_is_noop():
    # Real path, no patching: CI config has TRACING_ENABLED=false, so the real
    # _build_tracer returns NoOpTracer WITHOUT importing langfuse. This is the
    # default posture — zero overhead, zero data egress.
    reset_tracer()
    assert isinstance(get_tracer(), NoOpTracer)


def test_get_tracer_is_memoized_and_resettable():
    sentinel = NoOpTracer()
    with patch("core.observability.tracer._build_tracer", return_value=sentinel) as build:
        first = get_tracer()
        second = get_tracer()
    assert first is second is sentinel
    assert build.call_count == 1  # built once, then memoized

    reset_tracer()
    with patch("core.observability.tracer._build_tracer", return_value=NoOpTracer()) as build2:
        get_tracer()
    assert build2.call_count == 1  # reset forces a rebuild


def test_build_tracer_degrades_to_noop_when_langfuse_missing():
    # Enabled (should_trace True), but the optional SDK is not installed → no-op,
    # never a crash. sys.modules[langfuse]=None makes `import langfuse` raise.
    from core.observability import tracer as tmod

    with patch("core.observability.tracer.should_trace", return_value=True):
        with patch.dict("sys.modules", {"langfuse": None}):
            built = tmod._build_tracer()
    assert isinstance(built, NoOpTracer)


def test_noop_tracer_swallows_everything():
    t = NoOpTracer()
    assert t.trace_tool_call(tool_name="x", success=True, duration_ms=5) is None
    assert (
        t.score_retrieval(query="q", num_docs=0, top_score=0.0, status=STATUS_EMPTY) is None
    )


# ── fire_* helpers: delegate + guard ──────────────────────────────────────────


class _SpyTracer(Tracer):
    def __init__(self):
        self.tool_calls = []
        self.scores = []

    def trace_tool_call(self, **kw):
        self.tool_calls.append(kw)

    def score_retrieval(self, **kw):
        self.scores.append(kw)


def test_fire_tool_trace_delegates_with_args():
    spy = _SpyTracer()
    with patch("core.observability.tracer.get_tracer", return_value=spy):
        fire_tool_trace(
            tool_name="search_documents",
            success=True,
            duration_ms=42,
            workspace_id="ws-1",
            agent_id=7,
            error=None,
        )
    assert len(spy.tool_calls) == 1
    call = spy.tool_calls[0]
    assert call["tool_name"] == "search_documents"
    assert call["success"] is True
    assert call["duration_ms"] == 42
    assert call["workspace_id"] == "ws-1"
    assert call["agent_id"] == 7


def test_fire_retrieval_score_delegates_with_args():
    spy = _SpyTracer()
    with patch("core.observability.tracer.get_tracer", return_value=spy):
        fire_retrieval_score(
            query="how do agents work",
            num_docs=3,
            top_score=0.87,
            status=STATUS_HIT,
            workspace_id="ws-9",
        )
    assert len(spy.scores) == 1
    s = spy.scores[0]
    assert s["num_docs"] == 3
    assert s["top_score"] == 0.87
    assert s["status"] == STATUS_HIT
    assert s["workspace_id"] == "ws-9"


def test_fire_tool_trace_never_raises_on_tracer_fault():
    boom = MagicMock()
    boom.trace_tool_call.side_effect = RuntimeError("langfuse down")
    with patch("core.observability.tracer.get_tracer", return_value=boom):
        # Must not propagate — a tracing fault cannot break the tool call.
        fire_tool_trace(tool_name="x", success=False, duration_ms=1)


def test_fire_retrieval_score_never_raises_on_tracer_fault():
    boom = MagicMock()
    boom.score_retrieval.side_effect = RuntimeError("langfuse down")
    with patch("core.observability.tracer.get_tracer", return_value=boom):
        fire_retrieval_score(query="q", num_docs=0, top_score=0.0, status=STATUS_ERROR)


# ── LangfuseTracer: maps the seam onto the SDK (client mocked at the boundary) ─


def _mock_client_and_span():
    client = MagicMock()
    span = client.start_as_current_span.return_value.__enter__.return_value
    return client, span


def test_langfuse_tracer_traces_tool_success():
    client, span = _mock_client_and_span()
    LangfuseTracer(client).trace_tool_call(
        tool_name="search_knowledge", success=True, duration_ms=12, workspace_id="ws-1"
    )
    client.start_as_current_span.assert_called_once_with(name="tool:search_knowledge")
    span.update.assert_called_once()
    assert span.update.call_args.kwargs["level"] == "DEFAULT"
    span.score_trace.assert_any_call(name="tool_success", value=1.0)


def test_langfuse_tracer_traces_tool_failure_with_error_level():
    client, span = _mock_client_and_span()
    LangfuseTracer(client).trace_tool_call(
        tool_name="query_database", success=False, duration_ms=3, error="boom"
    )
    assert span.update.call_args.kwargs["level"] == "ERROR"
    span.score_trace.assert_any_call(name="tool_success", value=0.0)


def test_langfuse_tracer_scores_retrieval_hit():
    client, span = _mock_client_and_span()
    LangfuseTracer(client).score_retrieval(
        query="q", num_docs=4, top_score=0.9, status=STATUS_HIT
    )
    client.start_as_current_span.assert_called_once_with(name="rag:retrieve")
    span.score_trace.assert_any_call(name="retrieval_top_score", value=0.9)
    span.score_trace.assert_any_call(name="retrieval_grounded", value=1.0)


def test_langfuse_tracer_scores_retrieval_empty_as_ungrounded():
    client, span = _mock_client_and_span()
    LangfuseTracer(client).score_retrieval(
        query="q", num_docs=0, top_score=0.0, status=STATUS_EMPTY
    )
    span.score_trace.assert_any_call(name="retrieval_grounded", value=0.0)


def test_langfuse_score_falls_back_to_score_method():
    # SDK variant exposing `.score` but not `.score_trace` must still work.
    client = MagicMock()
    span = client.start_as_current_span.return_value.__enter__.return_value
    del span.score_trace  # force the fallback branch
    LangfuseTracer(client).score_retrieval(
        query="q", num_docs=1, top_score=0.5, status=STATUS_HIT
    )
    span.score.assert_any_call(name="retrieval_top_score", value=0.5)


# ── wiring guards: the emits are actually placed at the two chokepoints ────────
# Pure text guards (no heavy import), the same posture as the PRD-179 F070 /
# F056 source guards — they prove placement without pulling the app chain.


def _src(rel):
    return (ORCH / rel).read_text(encoding="utf-8")


def test_tool_dispatch_chokepoint_is_wired():
    src = _src("modules/tools/execution/unified_executor.py")
    assert "from core.observability.tracer import fire_tool_trace" in src
    assert "fire_tool_trace(" in src


def test_retrieval_chokepoint_is_wired():
    src = _src("modules/rag/service.py")
    assert "fire_retrieval_score(" in src
    # retrieve() must delegate to the wrapped impl so every path (incl. the
    # empty early-returns) is scored — not just the happy path.
    assert "_retrieve_impl(" in src
    assert "from core.observability.tracer import" in src

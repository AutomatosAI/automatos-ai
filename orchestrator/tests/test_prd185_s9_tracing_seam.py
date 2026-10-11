"""PRD-185 S9 — vendor-neutral tracing seam.

The two platform chokepoints (tool dispatch + RAG retrieval) must emit a live
trace/score so "was the tool call good / was retrieval grounded" becomes a
queryable number over real traffic. The seam is config-gated, default-OFF, and
never fails the caller.

Pure tests — no DB, network, or app import chain (per
``feedback-no-local-servers``). ``core/observability/tracer.py`` imports nothing
heavy at module load (``opentelemetry`` and config are lazy). The OpenTelemetry
tracer behind the seam (PRD-256 O3) is tested in ``test_prd256_o3_genai_spans.py``.
"""
import pathlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.observability.tracer import (
    Tracer,
    NoOpTracer,
    OtelTracer,
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


def test_should_trace_follows_otel_enabled():
    # PRD-256 O3: the seam emits through OpenTelemetry, so OTEL_ENABLED turns it on.
    assert should_trace(SimpleNamespace(OTEL_ENABLED=True)) is True


def test_should_trace_false_when_disabled():
    # Default posture — the whole point of the flag. Off = no real tracer.
    assert should_trace(SimpleNamespace(OTEL_ENABLED=False)) is False
    assert should_trace(SimpleNamespace()) is False


# ── get_tracer: default OFF is a memoized no-op ───────────────────────────────


def test_get_tracer_default_off_is_noop():
    # Real path, no patching: CI config has OTEL_ENABLED=false, so the real
    # _build_tracer returns NoOpTracer WITHOUT importing opentelemetry. This is the
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


def test_build_tracer_is_the_otel_tracer_when_tracing_is_on():
    from core.observability import tracer as tmod

    with patch("core.observability.tracer.should_trace", return_value=True):
        assert isinstance(tmod._build_tracer(), OtelTracer)


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
    boom.trace_tool_call.side_effect = RuntimeError("collector down")
    with patch("core.observability.tracer.get_tracer", return_value=boom):
        # Must not propagate — a tracing fault cannot break the tool call.
        fire_tool_trace(tool_name="x", success=False, duration_ms=1)


def test_fire_retrieval_score_never_raises_on_tracer_fault():
    boom = MagicMock()
    boom.score_retrieval.side_effect = RuntimeError("collector down")
    with patch("core.observability.tracer.get_tracer", return_value=boom):
        fire_retrieval_score(query="q", num_docs=0, top_score=0.0, status=STATUS_ERROR)


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

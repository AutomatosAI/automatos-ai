"""PRD-185 S9 — vendor-neutral tracing seam, on OpenTelemetry (PRD-256 O3).

Give the platform's chokepoints a *mouth* so "was the tool call good / was
retrieval grounded" becomes a **live, queryable number over real traffic**
instead of a synthetic one (the live complement to S10's offline recall@5/MRR).

Why a seam and not a tracing import at the call site:
- **The hooks are backend-agnostic.** The emit points stay put; the tracer behind
  them is :class:`OtelTracer`, whose spans go over OTLP to whatever collector the
  operator runs (Tempo, Jaeger, Datadog, Langfuse…). Choosing a backend is the
  collector's config, not a change here.
- **Default OFF** (``config.OTEL_ENABLED=false``): ``get_tracer()`` returns
  :class:`NoOpTracer` and nothing from ``opentelemetry`` is imported.
- **Never fails the caller.** Every emit is guarded; a tracing fault is logged
  and the tool call / retrieval returns normally. Mirrors the fire-and-forget
  telemetry posture in ``modules/tools/execution/telemetry.py``.
- **Private by default** (PRD-256 Principle 5): no query text, tool argument,
  error message or rendered context goes on a span, only names, counts and scores.

The emit points map 1:1 to the chokepoints:
- :func:`fire_tool_trace`   → tool dispatch (``unified_executor`` finally, beside telemetry)
- :func:`fire_retrieval_score` → RAG retrieval funnel (``RAGService.retrieve``) and the substrate searches
- :func:`fire_assembly_trace` → context assembly (``ContextService``, PRD-201 S1)
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Retrieval status vocabulary — the "empty-vs-error" signal the brief calls out.
STATUS_HIT = "hit"      # docs returned
STATUS_EMPTY = "empty"  # ran clean, returned nothing (not grounded)
STATUS_ERROR = "error"  # retrieval raised

TRACER_NAME = "automatos.observability"

ATTR_WORKSPACE_ID = "automatos.workspace_id"   # as core.observability.otel names it
ATTR_AGENT_ID = "automatos.agent_id"


# ── the seam ──────────────────────────────────────────────────────────────────


class Tracer:
    """Vendor-neutral trace/score surface. Two methods = the two chokepoints.

    Implementations must never raise to the caller; the :func:`fire_*` helpers
    guard anyway, but a well-behaved tracer swallows its own backend faults.
    """

    def trace_tool_call(
        self,
        *,
        tool_name: str,
        success: bool,
        duration_ms: int,
        workspace_id: Any = None,
        agent_id: Any = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError

    def score_retrieval(
        self,
        *,
        query: str,
        num_docs: int,
        top_score: float,
        status: str,
        workspace_id: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError

    def trace_assembly(
        self,
        *,
        trace: Dict[str, Any],
        workspace_id: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """PRD-201 S1: emit one context-assembly trace span.

        The *durable* per-turn record is written separately (JSONB on the turn
        row) so "what did Auto know?" is answerable offline even with tracing
        OFF — this method only mirrors its shape onto a live span when it is on.
        """
        raise NotImplementedError


class NoOpTracer(Tracer):
    """The default. Every method returns immediately — zero overhead, zero egress."""

    def trace_tool_call(self, **_: Any) -> None:
        return None

    def score_retrieval(self, **_: Any) -> None:
        return None

    def trace_assembly(self, **_: Any) -> None:
        return None


class OtelTracer(Tracer):
    """The seam on OpenTelemetry (PRD-256 O3): each emit is one span, a child of
    the span current where it fires (the request's, or the LLM call's).

    Only inside a trace: with no recording span around the emit (a Mission step,
    a heartbeat) nothing is recorded; O4 gives that work a trace of its own. The
    emits come after the work, so each span is back-dated by the duration it
    reports and ends when it fires.
    """

    def trace_tool_call(
        self,
        *,
        tool_name: str,
        success: bool,
        duration_ms: int,
        workspace_id: Any = None,
        agent_id: Any = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        attrs = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool_name,
            "automatos.tool.success": bool(success),
            **_context_attributes(workspace_id, agent_id, metadata),
        }
        # The error's class of failure only: its text can carry the tool's data.
        _emit(f"execute_tool {tool_name}", attrs, duration_ms, None if success else "tool_error")

    def score_retrieval(
        self,
        *,
        query: str,
        num_docs: int,
        top_score: float,
        status: str,
        workspace_id: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        attrs = {
            "automatos.retrieval.num_docs": int(num_docs or 0),
            "automatos.retrieval.top_score": float(top_score or 0.0),
            "automatos.retrieval.status": status,
            "automatos.retrieval.grounded": status == STATUS_HIT,
            **_context_attributes(workspace_id, None, metadata),
        }
        # The query is the owner's words: never on the span (Principle 5).
        _emit("rag.retrieve", attrs, (metadata or {}).get("latency_ms"),
              "retrieval_error" if status == STATUS_ERROR else None)

    def trace_assembly(
        self,
        *,
        trace: Dict[str, Any],
        workspace_id: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        shape = trace if isinstance(trace, dict) else {}
        total, estimate = shape.get("budget_total") or 0, shape.get("token_estimate") or 0
        attrs = {
            "automatos.context.mode": shape.get("mode"),
            "gen_ai.request.model": shape.get("model"),
            "automatos.context.budget_total": total,
            "automatos.context.token_estimate": estimate,
            "automatos.context.token_budget": shape.get("token_budget"),
            "automatos.context.budget_fraction": float(estimate) / float(total) if total else None,
            "automatos.context.section_count": len(shape.get("sections") or []),
            "automatos.context.sections_included": [str(n) for n in shape.get("sections_included") or []],
            "automatos.context.sections_trimmed": [str(n) for n in shape.get("sections_trimmed") or []],
            "automatos.context.memory_count": len(shape.get("injected_memory_ids") or []),
            **_context_attributes(workspace_id, None, metadata),
        }
        # Section names and sizes only, never the rendered content.
        _emit(f"context.assembly {shape.get('mode') or 'context'}", attrs, shape.get("prep_ms"))


def _context_attributes(workspace_id: Any, agent_id: Any, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Workspace, agent and the caller's scalar metadata, as ``automatos.*`` attributes."""
    attrs: Dict[str, Any] = {ATTR_WORKSPACE_ID: workspace_id, ATTR_AGENT_ID: agent_id}
    for key, value in (metadata or {}).items():
        if isinstance(value, (str, int, float, bool)):
            attrs[f"automatos.{key}"] = value
    return attrs


def _span_value(value: Any) -> Any:
    """A value OpenTelemetry takes as an attribute: scalars and string lists as they are, the rest as text."""
    if isinstance(value, (str, bool, int, float)) or (isinstance(value, list) and all(isinstance(v, str) for v in value)):
        return value
    return str(value)


def _emit(name: str, attributes: Dict[str, Any], duration_ms: Any = None, error_type: Optional[str] = None) -> None:
    """One finished span under the current one, ``duration_ms`` long and ending now."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    if not trace.get_current_span().is_recording():
        return
    end = time.time_ns()
    start = end - int(max(float(duration_ms or 0), 0.0) * 1_000_000)
    attrs = {key: _span_value(value) for key, value in attributes.items() if value is not None}
    span = trace.get_tracer(TRACER_NAME).start_span(name, start_time=start, attributes=attrs)
    if error_type:
        span.set_attribute("error.type", error_type)
        span.set_status(Status(StatusCode.ERROR))
    span.end(end_time=end)


# ── construction (config-gated, memoized) ─────────────────────────────────────

_TRACER: Optional[Tracer] = None
_LOCK = threading.Lock()


def should_trace(cfg: Any) -> bool:
    """Pure decision: is a real (non-noop) tracer warranted?

    True iff OpenTelemetry is on (``OTEL_ENABLED``, PRD-256). Kept pure (no
    imports, no side effects) so it is trivially unit-testable.
    """
    return bool(getattr(cfg, "OTEL_ENABLED", False))


def _build_tracer() -> Tracer:
    """Construct the tracer from config: :class:`OtelTracer` when tracing is on."""
    from config import config as cfg  # canonical singleton

    return OtelTracer() if should_trace(cfg) else NoOpTracer()


def get_tracer() -> Tracer:
    """Return the process-wide tracer (memoized). NoOpTracer when disabled."""
    global _TRACER
    if _TRACER is not None:
        return _TRACER
    with _LOCK:
        if _TRACER is None:
            _TRACER = _build_tracer()
    return _TRACER


def reset_tracer() -> None:
    """Drop the memoized tracer (config changed at runtime / test isolation)."""
    global _TRACER
    _TRACER = None


# ── fire-and-forget emit helpers (what the chokepoints call) ──────────────────


def fire_tool_trace(
    *,
    tool_name: str,
    success: bool,
    duration_ms: int,
    workspace_id: Any = None,
    agent_id: Any = None,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a tool-dispatch trace. Guarded — a tracing fault never fails the call."""
    try:
        get_tracer().trace_tool_call(
            tool_name=tool_name,
            success=success,
            duration_ms=duration_ms,
            workspace_id=workspace_id,
            agent_id=agent_id,
            error=error,
            metadata=metadata,
        )
    except Exception:
        logger.debug("[tracing] tool trace failed", exc_info=True)


def fire_retrieval_score(
    *,
    query: str,
    num_docs: int,
    top_score: float,
    status: str,
    workspace_id: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a retrieval grounding score. Guarded — never fails retrieval."""
    try:
        get_tracer().score_retrieval(
            query=query,
            num_docs=num_docs,
            top_score=top_score,
            status=status,
            workspace_id=workspace_id,
            metadata=metadata,
        )
    except Exception:
        logger.debug("[tracing] retrieval score failed", exc_info=True)


def fire_assembly_trace(
    *,
    trace: Dict[str, Any],
    workspace_id: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a context-assembly trace span (PRD-201 S1). Guarded — never fails a build.

    This mirrors the assembled trace's shape onto a live span when tracing is
    ON. It is *not* the durable record: the answerable per-turn/run row is the
    JSONB the assembler hands back on ``ContextResult.to_assembly_trace()``,
    persisted by the turn writer regardless of ``OTEL_ENABLED``.
    """
    try:
        get_tracer().trace_assembly(
            trace=trace,
            workspace_id=workspace_id,
            metadata=metadata,
        )
    except Exception:
        logger.debug("[tracing] assembly trace failed", exc_info=True)

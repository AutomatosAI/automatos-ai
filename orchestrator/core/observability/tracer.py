"""PRD-185 S9 — vendor-neutral tracing seam.

Give the platform's two chokepoints a *mouth* so "was the tool call good / was
retrieval grounded" becomes a **live, queryable number over real traffic**
instead of a synthetic one (the live complement to S10's offline recall@5/MRR).

Why a seam and not a Langfuse import at the call site:
- **The hooks are backend-agnostic.** Choosing a backend (Langfuse Cloud today)
  is a config swap, not a rewrite — the day data-residency demands self-host,
  only this file changes. That is the whole point of S9 (see the decision brief
  ``reports/PRD-185-S9-LANGFUSE-DECISION-BRIEF.md``).
- **Default OFF** (``config.TRACING_ENABLED=false``): ``get_tracer()`` returns
  :class:`NoOpTracer`, ``langfuse`` is never imported, zero overhead, zero data
  egress. You flip it on when you want to watch.
- **Never fails the caller.** Every emit is guarded; a tracing fault is logged
  and the tool call / retrieval returns normally. Mirrors the fire-and-forget
  telemetry posture in ``modules/tools/execution/telemetry.py``.
- **``langfuse`` is an OPTIONAL dependency**, imported lazily only when enabled.
  Enabled but not installed → we log once and degrade to no-op.

The two emit points map 1:1 to the two chokepoints:
- :func:`fire_tool_trace`   → tool dispatch (``unified_executor`` finally, beside telemetry)
- :func:`fire_retrieval_score` → RAG retrieval funnel (``RAGService.retrieve``)
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Retrieval status vocabulary — the "empty-vs-error" signal the brief calls out.
STATUS_HIT = "hit"      # docs returned
STATUS_EMPTY = "empty"  # ran clean, returned nothing (not grounded)
STATUS_ERROR = "error"  # retrieval raised

_SUPPORTED_BACKENDS = ("langfuse",)


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
        OFF — this method only mirrors that record onto the live Langfuse plane
        when it is enabled.
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


class LangfuseTracer(Tracer):
    """Langfuse-Cloud implementation of the seam.

    Targets the Langfuse Python SDK **v3** surface (context-manager spans +
    scores). Every backend call is best-effort and guarded: a wrong SDK method
    name or a network blip degrades to a logged no-op, never a crash. Because
    the OFF path never reaches here and CI keeps tracing disabled, the live
    emission path is exercised only when you enable it — **smoke-test on first
    enablement** (flip ``TRACING_ENABLED=true`` with keys set, run one tool call
    + one retrieval, confirm the trace lands in Langfuse).
    """

    def __init__(self, client: Any):
        self._client = client

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
        md: Dict[str, Any] = {
            "workspace_id": str(workspace_id) if workspace_id else None,
            "agent_id": agent_id,
            "duration_ms": duration_ms,
        }
        if error:
            md["error"] = error[:500]
        if metadata:
            md.update(metadata)
        # One span per tool call, scored 1.0 success / 0.0 failure — never carry
        # secret param *values*, only names/outcome (keys-only privacy posture).
        with self._client.start_as_current_span(name=f"tool:{tool_name}") as span:
            span.update(metadata=md, level="ERROR" if not success else "DEFAULT")
            self._score(span, "tool_success", 1.0 if success else 0.0)

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
        md: Dict[str, Any] = {
            "num_docs": num_docs,
            "status": status,
            "workspace_id": str(workspace_id) if workspace_id else None,
        }
        if metadata:
            md.update(metadata)
        with self._client.start_as_current_span(name="rag:retrieve") as span:
            span.update(input=(query or "")[:500], metadata=md)
            # Two scores: the raw top similarity, and a binary "was it grounded".
            self._score(span, "retrieval_top_score", float(top_score))
            self._score(span, "retrieval_grounded", 1.0 if status == STATUS_HIT else 0.0)

    def trace_assembly(
        self,
        *,
        trace: Dict[str, Any],
        workspace_id: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        md: Dict[str, Any] = {"workspace_id": str(workspace_id) if workspace_id else None}
        if metadata:
            md.update(metadata)
        # Fold the small trace shape onto the span metadata — section/token/trim
        # detail only, never rendered content (keys-only privacy posture).
        if isinstance(trace, dict):
            md.update({k: v for k, v in trace.items() if k != "sections"})
            md["section_count"] = len(trace.get("sections") or [])
        mode = (trace or {}).get("mode") if isinstance(trace, dict) else None
        with self._client.start_as_current_span(name=f"assembly:{mode or 'context'}") as span:
            span.update(metadata=md)
            # One score: budget honesty — assembled fraction of the ceiling.
            total = (trace or {}).get("budget_total") or 0
            est = (trace or {}).get("token_estimate") or 0
            if total:
                self._score(span, "assembly_budget_fraction", float(est) / float(total))

    @staticmethod
    def _score(span: Any, name: str, value: float) -> None:
        """Best-effort score attach across Langfuse SDK method-name variants."""
        for attr in ("score_trace", "score"):
            fn = getattr(span, attr, None)
            if callable(fn):
                fn(name=name, value=value)
                return
        logger.debug("[tracing] no score method on span for %s", name)


# ── construction (config-gated, memoized) ─────────────────────────────────────

_TRACER: Optional[Tracer] = None
_LOCK = threading.Lock()
_WARNED_MISSING_SDK = False


def should_trace(cfg: Any) -> bool:
    """Pure decision: is a real (non-noop) tracer warranted?

    True iff tracing is enabled, the backend is known, and credentials exist.
    Kept pure (no imports, no side effects) so it is trivially unit-testable
    without the ``langfuse`` package installed.
    """
    if not getattr(cfg, "TRACING_ENABLED", False):
        return False
    backend = (getattr(cfg, "TRACING_BACKEND", "") or "").lower()
    if backend not in _SUPPORTED_BACKENDS:
        return False
    return bool(getattr(cfg, "LANGFUSE_PUBLIC_KEY", None)) and bool(
        getattr(cfg, "LANGFUSE_SECRET_KEY", None)
    )


def _build_tracer() -> Tracer:
    """Construct the tracer from config. Any failure → NoOpTracer (fail-open)."""
    global _WARNED_MISSING_SDK
    try:
        from config import config as cfg  # canonical singleton (config.py:1177)
    except Exception:  # pragma: no cover — config is always importable in-app
        return NoOpTracer()

    if not should_trace(cfg):
        return NoOpTracer()

    try:
        from langfuse import Langfuse  # optional dep, imported only when enabled
    except Exception:
        if not _WARNED_MISSING_SDK:
            logger.warning(
                "[tracing] TRACING_ENABLED but 'langfuse' is not installed — "
                "tracing disabled (pip install langfuse). Degrading to no-op."
            )
            _WARNED_MISSING_SDK = True
        return NoOpTracer()

    try:
        client = Langfuse(
            public_key=cfg.LANGFUSE_PUBLIC_KEY,
            secret_key=cfg.LANGFUSE_SECRET_KEY,
            host=getattr(cfg, "LANGFUSE_HOST", None) or "https://cloud.langfuse.com",
        )
        logger.info("[tracing] Langfuse tracer active (host=%s)", cfg.LANGFUSE_HOST)
        return LangfuseTracer(client)
    except Exception:
        logger.warning("[tracing] Langfuse client init failed; degrading to no-op", exc_info=True)
        return NoOpTracer()


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

    This mirrors the assembled trace onto the live Langfuse plane when tracing
    is ON. It is *not* the durable record: the answerable per-turn/run row is
    the JSONB the assembler hands back on ``ContextResult.to_assembly_trace()``,
    persisted by the turn writer regardless of ``TRACING_ENABLED``.
    """
    try:
        get_tracer().trace_assembly(
            trace=trace,
            workspace_id=workspace_id,
            metadata=metadata,
        )
    except Exception:
        logger.debug("[tracing] assembly trace failed", exc_info=True)

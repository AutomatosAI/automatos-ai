"""Observability — vendor-neutral tracing seam (PRD-185 S9).

External distributed tracing lives here (kept distinct from ``core/monitoring``,
which holds internal metrics/logging/alerts). The seam is config-gated and
default-OFF; see ``tracer.py``.
"""
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

__all__ = [
    "Tracer",
    "NoOpTracer",
    "LangfuseTracer",
    "get_tracer",
    "reset_tracer",
    "should_trace",
    "fire_tool_trace",
    "fire_retrieval_score",
    "STATUS_HIT",
    "STATUS_EMPTY",
    "STATUS_ERROR",
]

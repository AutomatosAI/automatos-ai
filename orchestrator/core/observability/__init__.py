"""Observability — vendor-neutral tracing seam (PRD-185 S9).

External distributed tracing lives here (kept distinct from ``core/monitoring``,
which holds internal metrics/logging/alerts): the seam (``tracer.py``), the
OpenTelemetry setup (``otel.py``, PRD-256) and the LLM call spans (``genai.py``).
All config-gated and default-OFF.
"""
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

__all__ = [
    "Tracer",
    "NoOpTracer",
    "OtelTracer",
    "get_tracer",
    "reset_tracer",
    "should_trace",
    "fire_tool_trace",
    "fire_retrieval_score",
    "STATUS_HIT",
    "STATUS_EMPTY",
    "STATUS_ERROR",
]

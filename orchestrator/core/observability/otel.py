"""#847 / PRD-256 O1 — OpenTelemetry traces over OTLP, default off.

Distributed traces next to what PRD-73 already gives (Prometheus metrics, logs
with correlation IDs, the Loki pipeline) and the PRD-185 tracer seam, not
instead of them. This first phase is the foundation:

* a tracer provider per process, started by the app's lifespan
  (:func:`with_tracing`) and flushed when it ends;
* a server span for every HTTP request (FastAPI instrumentation), except the
  health probes and the metrics scrape;
* the request's own IDs on that span (:func:`annotate_request`):
  ``X-Request-ID`` and the workspace stay what they are, and become searchable
  next to the trace;
* export over OTLP/HTTP to whatever collector the operator runs; the collector
  chooses the backend (Tempo, Jaeger, a vendor). No backend SDK lives here;
* a parent-based ratio sampler (``OTEL_TRACES_SAMPLER_RATIO``).

The rules, from the issue:

* **Off by default.** With ``OTEL_ENABLED=false`` nothing from ``opentelemetry``
  is imported, as ``get_tracer()`` never imports ``langfuse`` when it is off.
* **Never fail the caller.** A tracing fault is logged; the request, or the
  boot, goes on without spans.
* **Private by default.** Spans carry IDs and HTTP metadata, never a request or
  response body.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Dict, Optional

from config import config

logger = logging.getLogger(__name__)

# Never traced: probes and the metrics scrape would drown every real request.
EXCLUDED_URLS = r"/health(/[a-z]+)?$,/metrics$"
# The ASGI layer's span per ``http.receive``/``http.send`` message: three or more per
# request that say nothing the server span doesn't.
EXCLUDED_ASGI_SPANS = ["receive", "send"]
# The OTLP/HTTP path for traces, appended to the collector's base address.
OTLP_TRACES_PATH = "/v1/traces"
# The IDs PRD-73 Phase 2 already carries, as span attributes.
ATTR_REQUEST_ID = "automatos.request_id"
ATTR_WORKSPACE_ID = "automatos.workspace_id"


class _State:
    """This process's provider, once :func:`start_tracing` has made one."""

    provider: Any = None


def tracing_enabled() -> bool:
    """True when the operator turned OpenTelemetry on (``OTEL_ENABLED``)."""
    return bool(getattr(config, "OTEL_ENABLED", False))


def otlp_headers(raw: Optional[str]) -> Dict[str, str]:
    """``OTEL_EXPORTER_OTLP_HEADERS`` (``key=value,key2=value2``, values URL-encoded,
    as the OpenTelemetry spec writes it) as a dict. A malformed pair is skipped."""
    from urllib.parse import unquote

    headers: Dict[str, str] = {}
    for pair in (raw or "").split(","):
        key, sep, value = pair.partition("=")
        if sep and key.strip() and value.strip():
            headers = {**headers, key.strip(): unquote(value.strip())}
    return headers


def _otlp_exporter() -> Any:
    """The OTLP/HTTP span exporter pointed at the configured collector."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    base = str(config.OTEL_EXPORTER_OTLP_ENDPOINT).rstrip("/")
    return OTLPSpanExporter(endpoint=base + OTLP_TRACES_PATH,
                            headers=otlp_headers(config.OTEL_EXPORTER_OTLP_HEADERS))


def build_provider(exporter: Any, ratio: float) -> Any:
    """A tracer provider: this service's resource, a parent-based ratio sampler,
    and batched export through ``exporter``. Nothing global is touched."""
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

    resource = Resource.create({
        "service.name": config.OTEL_SERVICE_NAME,
        "deployment.environment.name": str(getattr(config, "ENVIRONMENT", "") or "unknown"),
        "automatos.edition": str(getattr(config, "AUTH_EDITION", "") or "unknown"),
    })
    sampler = ParentBased(TraceIdRatioBased(min(max(float(ratio), 0.0), 1.0)))
    provider = TracerProvider(resource=resource, sampler=sampler)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


def start_tracing(exporter: Any = None) -> bool:
    """Install this process's provider as the global one; True when it did.
    ``exporter`` replaces the OTLP exporter (tests pass an in-memory one)."""
    if not tracing_enabled() or _State.provider is not None:
        return _State.provider is not None
    try:
        from opentelemetry import trace

        provider = build_provider(exporter or _otlp_exporter(), config.OTEL_TRACES_SAMPLER_RATIO)
        trace.set_tracer_provider(provider)
    except Exception:
        logger.exception("[otel] tracing not started; serving without traces")
        return False
    _State.provider = provider
    logger.info("[otel] tracing to %s (service %s, sampler ratio %s)", config.OTEL_EXPORTER_OTLP_ENDPOINT,
                config.OTEL_SERVICE_NAME, config.OTEL_TRACES_SAMPLER_RATIO)
    return True


def shutdown_tracing() -> None:
    """Flush what is batched and stop this process's provider."""
    provider, _State.provider = _State.provider, None
    if provider is None:
        return
    try:
        provider.shutdown()
    except Exception:
        logger.exception("[otel] tracing did not shut down cleanly")


def with_tracing(lifespan: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """The app's lifespan with this process's provider started around it.
    Unchanged when tracing is off, so nothing is imported then."""
    if not tracing_enabled():
        return lifespan

    @asynccontextmanager
    async def traced(app: Any) -> AsyncIterator[None]:
        start_tracing()
        try:
            async with lifespan(app):
                yield
        finally:
            shutdown_tracing()

    return traced


def instrument_app(app: Any, tracer_provider: Any = None) -> None:
    """Server spans for every request but the probes. Call once, right after the
    app is created: the instrumentation wraps the middleware stack Starlette
    builds on the first call, which is the lifespan's."""
    if not tracing_enabled():
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider, excluded_urls=EXCLUDED_URLS,
                                           exclude_spans=EXCLUDED_ASGI_SPANS)
    except Exception:
        logger.exception("[otel] FastAPI instrumentation failed; serving without server spans")


def annotate_request(request_id: str, workspace_id: str = "") -> None:
    """Put the request's own IDs on its server span, so a trace and the
    ``X-Request-ID`` in the logs and Loki find each other."""
    if not tracing_enabled():
        return
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if not span.is_recording():
            return
        if request_id:
            span.set_attribute(ATTR_REQUEST_ID, request_id)
        if workspace_id:
            span.set_attribute(ATTR_WORKSPACE_ID, workspace_id)
    except Exception:
        logger.exception("[otel] request IDs not set on the span")


__all__ = [
    "ATTR_REQUEST_ID", "ATTR_WORKSPACE_ID", "EXCLUDED_URLS", "annotate_request", "build_provider",
    "instrument_app", "otlp_headers", "shutdown_tracing", "start_tracing", "tracing_enabled", "with_tracing",
]

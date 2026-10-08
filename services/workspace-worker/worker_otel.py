"""PRD-256 O2b (#847): the workspace worker's half of one trace.

The platform's HTTP client sends ``traceparent`` with every call to the worker
(O2a). With ``OTEL_ENABLED`` on here too, each worker request is a server span
that continues that trace, so API → worker is one trace.

The same rules as the platform's tracing (``orchestrator/core/observability/otel.py``),
which this image can't import:

* **Off by default:** nothing from ``opentelemetry`` is imported, and the app is
  built as before.
* **Never fails the caller:** a tracing fault is logged; the request goes on.
* **Private:** the span carries the method, the route, the workspace and the
  status, never the URL, a query, a body or the internal token.
* **Not the probes:** ``/health`` and ``/metrics`` are never traced.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from worker_config import (otel_enabled, otel_endpoint, otel_headers, otel_sampler_ratio,
                           otel_service_name)

logger = logging.getLogger("workspace-worker")

UNTRACED_PATHS = frozenset({"/health", "/metrics"})
OTLP_TRACES_PATH = "/v1/traces"
TRACER_NAME = "automatos.workspace_worker"
ATTR_WORKSPACE_ID = "automatos.workspace_id"
DEFAULT_SAMPLER_RATIO = 1.0


class _State:
    """This process's provider, once :func:`start_tracing` has installed one."""

    provider: Any = None


def sampler_ratio(raw: Any) -> float:
    """The sampler ratio in [0, 1]; a value that isn't a number keeps every trace."""
    try:
        ratio = float(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning("[otel] OTEL_TRACES_SAMPLER_RATIO=%r is not a number; keeping every trace", raw)
        return DEFAULT_SAMPLER_RATIO
    return min(max(ratio, 0.0), 1.0)


def otlp_headers(raw: str) -> Dict[str, str]:
    """``key=value,key2=value2`` (URL-encoded values) as a dict; a malformed pair is skipped."""
    from urllib.parse import unquote

    headers: Dict[str, str] = {}
    for pair in (raw or "").split(","):
        key, sep, value = pair.partition("=")
        if sep and key.strip() and value.strip():
            headers = {**headers, key.strip(): unquote(value.strip())}
    return headers


def build_provider(exporter: Any) -> Any:
    """This service's provider: its resource, a parent-based ratio sampler, batched export."""
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

    provider = TracerProvider(resource=Resource.create({"service.name": otel_service_name()}),
                              sampler=ParentBased(TraceIdRatioBased(sampler_ratio(otel_sampler_ratio()))))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


def _otlp_exporter() -> Any:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter(endpoint=otel_endpoint().rstrip("/") + OTLP_TRACES_PATH,
                            headers=otlp_headers(otel_headers()))


def start_tracing(exporter: Any = None) -> bool:
    """Install this process's provider, once; True when it is the one in place."""
    if _State.provider is not None:
        return True
    try:
        from opentelemetry import trace

        provider = build_provider(exporter or _otlp_exporter())
        trace.set_tracer_provider(provider)
        if trace.get_tracer_provider() is not provider:
            logger.error("[otel] another tracer provider is already installed; serving without these traces")
            provider.shutdown()
            return False
    except Exception:
        logger.exception("[otel] tracing not started; serving without traces")
        return False
    _State.provider = provider
    logger.info("[otel] tracing to %s (service %s)", otel_endpoint(), otel_service_name())
    return True


async def flush_tracing(_app: Any = None) -> None:
    """Send what is batched when the app stops; the SDK shuts the provider down at exit."""
    if _State.provider is None:
        return
    try:
        _State.provider.force_flush()
    except Exception:
        logger.exception("[otel] batched spans not flushed")


def _route_of(request: Any) -> str:
    """The route template (``/workspaces/{workspace_id}/files``), never the URL."""
    resource = getattr(getattr(request.match_info, "route", None), "resource", None)
    return getattr(resource, "canonical", None) or "unmatched"


def _span_attributes(request: Any) -> Dict[str, Any]:
    attributes = {"http.method": request.method, "http.route": _route_of(request)}
    workspace_id = request.match_info.get("workspace_id")
    return {**attributes, ATTR_WORKSPACE_ID: workspace_id} if workspace_id else attributes


async def _traced(request: Any, handler: Any) -> Any:
    """Run ``handler`` inside a server span that continues the caller's trace."""
    from aiohttp import web
    from opentelemetry import propagate, trace
    from opentelemetry.trace import SpanKind

    tracer = trace.get_tracer(TRACER_NAME)
    name = f"{request.method} {_route_of(request)}"
    with tracer.start_as_current_span(name, context=propagate.extract(request.headers), kind=SpanKind.SERVER,
                                      attributes=_span_attributes(request)) as span:
        try:
            response = await handler(request)
        except web.HTTPException as exc:
            span.set_attribute("http.status_code", exc.status)
            raise
        span.set_attribute("http.status_code", response.status)
        return response


class TraceIdsFilter(logging.Filter):
    """PRD-256 O5: the current sampled span's ``trace_id`` and ``span_id`` on every log
    record, which log-relay ships to Loki; ``_trace_context`` is the console line's
    `` trace=<id>``, underscored so log-relay doesn't ship it twice."""

    def filter(self, record: logging.LogRecord) -> bool:
        from opentelemetry import trace

        context = trace.get_current_span().get_span_context()
        sampled = context.is_valid and context.trace_flags.sampled
        record.trace_id = format(context.trace_id, "032x") if sampled else ""
        record.span_id = format(context.span_id, "016x") if sampled else ""
        record._trace_context = f" trace={record.trace_id}" if sampled else ""
        return True


def _log_trace_ids() -> None:
    """Put the trace IDs on every handler's records (the console and log-relay)."""
    log_filter = TraceIdsFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(log_filter)


def attach(app: Any) -> None:
    """With ``OTEL_ENABLED``: this process's provider, a server span per request (the
    outermost middleware), the trace IDs in the logs (O5) and a flush when the app
    stops. Off: the app and the logs are untouched."""
    if not otel_enabled() or not start_tracing():
        return
    from aiohttp import web

    _log_trace_ids()

    @web.middleware
    async def tracing_middleware(request, handler):
        if request.path in UNTRACED_PATHS:
            return await handler(request)
        return await _traced(request, handler)

    app.middlewares.insert(0, tracing_middleware)
    app.on_cleanup.append(flush_tracing)


__all__ = ["ATTR_WORKSPACE_ID", "UNTRACED_PATHS", "TraceIdsFilter", "attach", "build_provider", "flush_tracing",
           "otlp_headers", "sampler_ratio", "start_tracing"]

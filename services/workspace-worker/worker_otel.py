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
from typing import Any, Dict, Tuple

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
    """This service's provider: its resource, a parent-based ratio sampler, batched export.
    There is no export-time rewrite as the API has (#1107): ``_traced`` is the only source
    of spans here and records an error as its type. Add an instrumentation, add the rewrite."""
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


def _route_of(scope: Any) -> Tuple[str, Dict[str, Any]]:
    """(the route template, its path parameters) for a request, never the URL.

    The router matches only after the middleware runs, so the span's name is read here
    the way the router will read it: the route that fully matches, or "unmatched" for a
    404 or a 405, as aiohttp's resource lookup gave it (PRD-262, row 36)."""
    from starlette.routing import Match

    for route in scope["app"].routes:
        match, child = route.matches(scope)
        if match == Match.FULL:
            return route.path, child.get("path_params", {})
    return "unmatched", {}


def _span_attributes(method: str, route: str, params: Dict[str, Any]) -> Dict[str, Any]:
    attributes = {"http.method": method, "http.route": route}
    workspace_id = params.get("workspace_id")
    return {**attributes, ATTR_WORKSPACE_ID: workspace_id} if workspace_id else attributes


def _failed(span: Any, exc: BaseException) -> None:
    """An error on the span as its type only: an exception's message and stack trace can
    name a workspace file or carry a URL, so they never leave on a span (#1107). The
    whole exception is in the worker's log, which carries the same trace_id (O5)."""
    from opentelemetry.trace import Status, StatusCode

    span.set_attribute("error.type", type(exc).__qualname__)
    span.set_status(Status(StatusCode.ERROR, type(exc).__qualname__))


class _StatusRecorder:
    """The ASGI ``send`` of a request, passed on, keeping the response's status."""

    def __init__(self, send: Any) -> None:
        self.send = send
        self.status: Any = None

    async def __call__(self, message: Dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            self.status = message["status"]
        await self.send(message)


async def _traced(app: Any, scope: Any, receive: Any, send: Any) -> None:
    """Run the request inside a server span that continues the caller's trace."""
    from opentelemetry import propagate, trace
    from opentelemetry.trace import SpanKind
    from starlette.datastructures import Headers

    route, params = _route_of(scope)
    recorder = _StatusRecorder(send)
    tracer = trace.get_tracer(TRACER_NAME)
    with tracer.start_as_current_span(f"{scope['method']} {route}", context=propagate.extract(Headers(scope=scope)),
                                      kind=SpanKind.SERVER, attributes=_span_attributes(scope["method"], route, params),
                                      record_exception=False, set_status_on_exception=False) as span:
        try:
            await app(scope, receive, recorder)
        except Exception as exc:
            _failed(span, exc)
            raise
        if recorder.status is not None:   # None: the client left before an answer started
            span.set_attribute("http.status_code", recorder.status)


class TracingMiddleware:
    """A server span per request, outermost; never for the probes (/health, /metrics).

    A 404 or 405 is an answer, not a failure: the router's own HTTPException becomes
    its response inside the app, so the span records the status. A handler that raises
    marks the span failed with the error's type, and the error goes on to the 500."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["path"] in UNTRACED_PATHS:
            await self.app(scope, receive, send)
            return
        await _traced(self.app, scope, receive, send)


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


def _flush_on_shutdown(app: Any) -> None:
    """Flush the batched spans when the app's lifespan ends (uvicorn's shutdown)."""
    from contextlib import asynccontextmanager

    lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def flushing_lifespan(lifespan_app: Any):
        async with lifespan(lifespan_app) as state:
            yield state
        await flush_tracing()

    app.router.lifespan_context = flushing_lifespan


def attach(app: Any) -> None:
    """With ``OTEL_ENABLED``: this process's provider, a server span per request (the
    outermost middleware), the trace IDs in the logs (O5) and a flush when the app
    stops. Off: the app and the logs are untouched."""
    if not otel_enabled() or not start_tracing():
        return
    _log_trace_ids()
    app.add_middleware(TracingMiddleware)
    _flush_on_shutdown(app)


__all__ = ["ATTR_WORKSPACE_ID", "UNTRACED_PATHS", "TraceIdsFilter", "TracingMiddleware", "attach", "build_provider",
           "flush_tracing", "otlp_headers", "sampler_ratio", "start_tracing"]

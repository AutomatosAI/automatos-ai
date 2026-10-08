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

O2a adds the server span's children (:func:`instrument_libraries`): a client span
for every SQL statement, Redis command, outbound HTTP call and AWS call. The HTTP
client spans carry ``traceparent`` to whoever is called, the workspace worker
first (whose own server span is O2b).

The rules, from the issue:

* **Off by default.** With ``OTEL_ENABLED=false`` nothing from ``opentelemetry``
  is imported, as ``get_tracer()`` never imports ``langfuse`` when it is off.
* **Never fail the caller.** A tracing fault is logged; the request, or the
  boot, goes on without spans.
* **Private by default.** Spans carry IDs and HTTP metadata, never a request or
  response body, and never a query-string value (:func:`_scrub_url_attributes`):
  an OAuth ``code`` or a ``token`` in a URL must not reach the collector.
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
# The server span's URL attributes, under the old and the stable HTTP conventions.
URL_ATTRIBUTES = ("http.url", "http.target", "url.full", "url.query")
# What a query value becomes on a span: the key stays, the value never leaves.
REDACTED = "REDACTED"
# When OTEL_TRACES_SAMPLER_RATIO is not a number: keep every trace, and say so.
DEFAULT_SAMPLER_RATIO = 1.0


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


def sampler_ratio(raw: Any) -> float:
    """``OTEL_TRACES_SAMPLER_RATIO`` as a ratio in [0, 1]; a value that isn't a number
    is logged and keeps every trace, so a typo can never stop tracing or a boot."""
    try:
        ratio = float(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning("[otel] OTEL_TRACES_SAMPLER_RATIO=%r is not a number; keeping every trace", raw)
        return DEFAULT_SAMPLER_RATIO
    return min(max(ratio, 0.0), 1.0)


def redact_query(url: str) -> str:
    """``url`` with every query value replaced by ``REDACTED``; keys, path and fragment kept."""
    head, sep, query = url.partition("?")
    if not sep:
        return url
    query, hash_sep, fragment = query.partition("#")
    pairs = [part.partition("=") for part in query.split("&") if part]
    scrubbed = "&".join(f"{key}={REDACTED}" if eq else key for key, eq, _ in pairs)
    return f"{head}?{scrubbed}{hash_sep}{fragment}"


def _scrub_url_attributes(span: Any, scope: Any = None) -> None:
    """The request hook of the server span and of every HTTP client span: no query value
    leaves on a URL attribute. The ASGI layer redacts only a few signature keys, not
    ``code`` or ``token``; the HTTP client redacts none (a ``?key=`` API key would go)."""
    attributes = getattr(span, "attributes", None) or {}
    for name in URL_ATTRIBUTES:
        value = attributes.get(name)
        if isinstance(value, str) and "?" in value:
            span.set_attribute(name, redact_query(value))
        elif name == "url.query" and isinstance(value, str) and value:
            span.set_attribute(name, redact_query("?" + value)[1:])


async def _scrub_url_attributes_async(span: Any, request: Any = None) -> None:
    """The async HTTP client's request hook: the same scrub."""
    _scrub_url_attributes(span, request)


def _instrument_sqlalchemy(engine: Any) -> None:
    """A span per statement on this process's engine; the SQL text, never its bound values."""
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

    if engine is None:
        from core.database.database import engine
    SQLAlchemyInstrumentor().instrument(engine=engine)


def _instrument_httpx(_: Any = None) -> None:
    """A span per outbound request, ``traceparent`` sent with it, no query value on its URL
    (an API key in a query string must not reach the collector)."""
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    HTTPXClientInstrumentor().instrument(request_hook=_scrub_url_attributes,
                                         async_request_hook=_scrub_url_attributes_async)


def _instrument_redis(_: Any = None) -> None:
    """A span per command, recorded as ``SET ? ?``: never a key or a value."""
    from opentelemetry.instrumentation.redis import RedisInstrumentor

    RedisInstrumentor().instrument()


def _instrument_botocore(_: Any = None) -> None:
    """A span per AWS call (S3, S3 Vectors, Bedrock): service and operation, no payload."""
    from opentelemetry.instrumentation.botocore import BotocoreInstrumentor

    BotocoreInstrumentor().instrument()


_LIBRARIES = (("sqlalchemy", _instrument_sqlalchemy), ("httpx", _instrument_httpx),
              ("redis", _instrument_redis), ("botocore", _instrument_botocore))


def instrument_libraries(engine: Any = None) -> None:
    """Client spans under each request: SQL, Redis, outbound HTTP, AWS. Once per process,
    at import; ``engine`` defaults to the app's own. One library failing never stops
    the others, or the app."""
    if not tracing_enabled():
        return
    for name, instrument in _LIBRARIES:
        try:
            instrument(engine)
        except Exception:
            logger.exception("[otel] %s not instrumented; its calls carry no spans", name)


def _otlp_exporter() -> Any:
    """The OTLP/HTTP span exporter pointed at the configured collector."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    base = str(config.OTEL_EXPORTER_OTLP_ENDPOINT).rstrip("/")
    return OTLPSpanExporter(endpoint=base + OTLP_TRACES_PATH,
                            headers=otlp_headers(config.OTEL_EXPORTER_OTLP_HEADERS))


def request_rooted_sampler(ratio: Any) -> Any:
    """The sampler for a new trace: kept by ratio, unless it would start at a library
    call. A SQL statement, Redis command, HTTP or AWS call with no parent is a
    background loop's (schedulers, the dispatcher, heartbeats poll all the time), and
    as a trace of its own it is noise: one root trace per query, tens of thousands
    an hour. Those are dropped; inside a request they are kept, as the request's
    children (parent-based). Background work gets its own spans in O4."""
    from opentelemetry.sdk.trace.sampling import Decision, ParentBased, Sampler, SamplingResult, TraceIdRatioBased
    from opentelemetry.trace import SpanKind

    class RequestRooted(Sampler):
        def __init__(self, keep: float) -> None:
            self._ratio = TraceIdRatioBased(keep)

        def should_sample(self, parent_context, trace_id, name, kind=None, attributes=None, links=None,
                          trace_state=None):
            if kind == SpanKind.CLIENT:
                return SamplingResult(Decision.DROP)
            return self._ratio.should_sample(parent_context, trace_id, name, kind, attributes, links, trace_state)

        def get_description(self) -> str:
            return f"RequestRooted({self._ratio.get_description()})"

    return ParentBased(RequestRooted(sampler_ratio(ratio)))


def build_provider(exporter: Any, ratio: float) -> Any:
    """A tracer provider: this service's resource, the request-rooted ratio sampler,
    and batched export through ``exporter``. Nothing global is touched."""
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({
        "service.name": config.OTEL_SERVICE_NAME,
        "deployment.environment.name": str(getattr(config, "ENVIRONMENT", "") or "unknown"),
        "automatos.edition": str(getattr(config, "AUTH_EDITION", "") or "unknown"),
    })
    provider = TracerProvider(resource=resource, sampler=request_rooted_sampler(ratio))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


def start_tracing(exporter: Any = None) -> bool:
    """Install this process's provider as the global one, once; True when it is in place.
    A later lifespan in the same process reuses it: the SDK refuses a second global
    provider, so replacing it would leave the app bound to a stopped one.
    ``exporter`` replaces the OTLP exporter (tests pass an in-memory one)."""
    if not tracing_enabled() or _State.provider is not None:
        return _State.provider is not None
    try:
        from opentelemetry import trace

        provider = build_provider(exporter or _otlp_exporter(), config.OTEL_TRACES_SAMPLER_RATIO)
        trace.set_tracer_provider(provider)
        if trace.get_tracer_provider() is not provider:
            logger.error("[otel] another tracer provider is already installed; serving without these traces")
            provider.shutdown()
            return False
    except Exception:
        logger.exception("[otel] tracing not started; serving without traces")
        return False
    _State.provider = provider
    logger.info("[otel] tracing to %s (service %s, sampler ratio %s)", config.OTEL_EXPORTER_OTLP_ENDPOINT,
                config.OTEL_SERVICE_NAME, config.OTEL_TRACES_SAMPLER_RATIO)
    return True


def flush_tracing() -> None:
    """Send what is batched when the lifespan ends. The provider stays installed for a
    later lifespan in this process; the SDK shuts it down itself when the process exits."""
    if _State.provider is None:
        return
    try:
        _State.provider.force_flush()
    except Exception:
        logger.exception("[otel] batched spans not flushed")


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
            flush_tracing()

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
                                           exclude_spans=EXCLUDED_ASGI_SPANS,
                                           server_request_hook=_scrub_url_attributes)
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
    "ATTR_REQUEST_ID", "ATTR_WORKSPACE_ID", "EXCLUDED_URLS", "REDACTED", "annotate_request", "build_provider",
    "flush_tracing", "instrument_app", "instrument_libraries", "otlp_headers", "redact_query",
    "request_rooted_sampler", "sampler_ratio", "start_tracing", "tracing_enabled", "with_tracing",
]

"""
Automatos Shared Metrics — Prometheus Exporter for All Services
===============================================================

Exposes a /metrics endpoint compatible with Prometheus scraping.
Works with FastAPI and with plain Starlette.

FastAPI usage:
    from automatos_metrics import add_fastapi_metrics
    app = FastAPI()
    add_fastapi_metrics(app, service="my-service")

Starlette usage (the workspace worker, PRD-262: it left aiohttp for the Starlette
and uvicorn its image already carries):
    from automatos_metrics import add_starlette_metrics
    app = Starlette(routes=[...])
    add_starlette_metrics(app, service="my-service")

Requires: prometheus_client

Requests are labelled by route template ("/optimize/{job_id}"), never by the raw
path, and every path no route matches shares one "unmatched" label; a method
outside the standard set is labelled "OTHER". A scanner's paths, a job id per
request or a forged Host header therefore add no new series (PRD-262 US-006).

services/shared holds the source; each service vendors a byte-identical copy,
because Railway builds every service from its own directory. A test in the
orchestrator suite (test_prd262_vendored_metrics.py) keeps the copies identical.
"""

import time
from typing import Optional

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Info,
        generate_latest,
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        REGISTRY,
    )
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


METRICS_PATH = "/metrics"
UNMATCHED = "unmatched"
OTHER_METHOD = "OTHER"
KNOWN_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})


def _method_label(method: str) -> str:
    """The request method, or OTHER: the client chooses it, so it must not mint series."""
    return method if method in KNOWN_METHODS else OTHER_METHOD


def _fastapi_route_label(scope) -> str:
    """The template of the route the router matched, or "unmatched".

    The router stores the matched route in the ASGI scope; the label never reads
    request.url, which a forged Host header can rewrite.
    """
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) and path else UNMATCHED


def _starlette_route_label(scope) -> str:
    """The template of the route that fully matches, or "unmatched" (404 and 405).

    Starlette routes after the middleware runs, so the label is matched here the way
    the router will match it; it never reads the URL or the Host header.
    """
    from starlette.routing import Match

    for route in scope["app"].routes:
        match, _ = route.matches(scope)
        if match == Match.FULL:
            return route.path
    return UNMATCHED


def _create_metrics(service: str, registry=None):
    """Create standard metrics for a service."""
    if not HAS_PROMETHEUS:
        return None

    reg = registry or REGISTRY
    prefix = service.replace("-", "_").replace(" ", "_")

    return {
        "requests_total": Counter(
            f"{prefix}_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
            registry=reg,
        ),
        "request_duration": Histogram(
            f"{prefix}_request_duration_seconds",
            "Request latency in seconds",
            ["method", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=reg,
        ),
        "in_progress": Gauge(
            f"{prefix}_requests_in_progress",
            "Requests currently being processed",
            registry=reg,
        ),
        "info": Info(
            f"{prefix}_build",
            "Service build info",
            registry=reg,
        ),
    }


# ---------------------------------------------------------------------------
# FastAPI integration
# ---------------------------------------------------------------------------

def add_fastapi_metrics(app, service: str = "unknown"):
    """Add Prometheus /metrics endpoint + request middleware to a FastAPI app.

    Args:
        app: FastAPI application instance
        service: Service name for metric prefixes
    """
    if not HAS_PROMETHEUS:
        import logging
        logging.getLogger(__name__).warning(
            "prometheus_client not installed — metrics disabled"
        )
        return

    from fastapi import Request, Response
    from starlette.middleware.base import BaseHTTPMiddleware

    metrics = _create_metrics(service)
    metrics["info"].info({"service": service})

    class PrometheusMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            if request.scope.get("path") == METRICS_PATH:
                return await call_next(request)

            method = _method_label(request.method)

            metrics["in_progress"].inc()
            start = time.monotonic()
            try:
                response = await call_next(request)
                status = str(response.status_code)
                return response
            except Exception:
                status = "500"
                raise
            finally:
                duration = time.monotonic() - start
                # Read after call_next: the router has stored the matched route by now.
                endpoint = _fastapi_route_label(request.scope)
                metrics["in_progress"].dec()
                metrics["requests_total"].labels(method, endpoint, status).inc()
                metrics["request_duration"].labels(method, endpoint).observe(duration)

    app.add_middleware(PrometheusMiddleware)

    @app.get(METRICS_PATH, include_in_schema=False)
    async def metrics_endpoint():
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST,
        )


# ---------------------------------------------------------------------------
# Starlette integration
# ---------------------------------------------------------------------------

class _StatusRecorder:
    """The ASGI ``send`` of a request, passed on, keeping the status it answered.

    "500" until a response starts: a request that raises first is a server error.
    """

    def __init__(self, send):
        self.send = send
        self.code = "500"

    async def __call__(self, message):
        if message["type"] == "http.response.start":
            self.code = str(message["status"])
        await self.send(message)


class _StarlettePrometheusMiddleware:
    """Counts, times and gauges every request but /metrics, by method and route template."""

    def __init__(self, app, metrics):
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] == METRICS_PATH:
            await self.app(scope, receive, send)
            return
        method = _method_label(scope["method"])
        endpoint = _starlette_route_label(scope)
        recorder = _StatusRecorder(send)
        self.metrics["in_progress"].inc()
        start = time.monotonic()
        try:
            await self.app(scope, receive, recorder)
        finally:
            duration = time.monotonic() - start
            self.metrics["in_progress"].dec()
            self.metrics["requests_total"].labels(method, endpoint, recorder.code).inc()
            self.metrics["request_duration"].labels(method, endpoint).observe(duration)


def add_starlette_metrics(app, service: str = "unknown"):
    """Add Prometheus /metrics endpoint + request middleware to a Starlette app.

    Args:
        app: starlette.applications.Starlette instance, before it serves
        service: Service name for metric prefixes
    """
    if not HAS_PROMETHEUS:
        import logging
        logging.getLogger(__name__).warning(
            "prometheus_client not installed — metrics disabled"
        )
        return

    from starlette.responses import Response

    metrics = _create_metrics(service)
    metrics["info"].info({"service": service})

    # Outermost of the middleware added so far.
    app.add_middleware(_StarlettePrometheusMiddleware, metrics=metrics)

    # prometheus_client's content type carries a charset ("text/plain;
    # version=...; charset=utf-8"); as a header it goes out exactly as the
    # client library names it (PRD-262 US-045).
    async def metrics_handler(request):
        return Response(
            content=generate_latest(REGISTRY),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )

    app.add_route(METRICS_PATH, metrics_handler, methods=["GET"])

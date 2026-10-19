"""
Automatos Shared Metrics — Prometheus Exporter for All Services
===============================================================

Exposes a /metrics endpoint compatible with Prometheus scraping.
Works with both FastAPI and aiohttp.

FastAPI usage:
    from automatos_metrics import add_fastapi_metrics
    app = FastAPI()
    add_fastapi_metrics(app, service="my-service")

aiohttp usage:
    from automatos_metrics import add_aiohttp_metrics
    app = web.Application()
    add_aiohttp_metrics(app, service="my-service")

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


def _aiohttp_route_label(request) -> str:
    """The canonical path of the resource aiohttp resolved, or "unmatched" (404 and 405)."""
    resource = request.match_info.route.resource
    return resource.canonical if resource is not None else UNMATCHED


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
# aiohttp integration
# ---------------------------------------------------------------------------

def add_aiohttp_metrics(app, service: str = "unknown"):
    """Add Prometheus /metrics endpoint + request middleware to an aiohttp app.

    Args:
        app: aiohttp.web.Application instance
        service: Service name for metric prefixes
    """
    if not HAS_PROMETHEUS:
        import logging
        logging.getLogger(__name__).warning(
            "prometheus_client not installed — metrics disabled"
        )
        return

    from aiohttp import web

    metrics = _create_metrics(service)
    metrics["info"].info({"service": service})

    @web.middleware
    async def prometheus_middleware(request, handler):
        if request.path == METRICS_PATH:
            return await handler(request)

        # aiohttp resolves the route before any middleware runs.
        method = _method_label(request.method)
        endpoint = _aiohttp_route_label(request)

        metrics["in_progress"].inc()
        start = time.monotonic()
        try:
            response = await handler(request)
            status = str(response.status)
            return response
        except Exception:
            status = "500"
            raise
        finally:
            duration = time.monotonic() - start
            metrics["in_progress"].dec()
            metrics["requests_total"].labels(method, endpoint, status).inc()
            metrics["request_duration"].labels(method, endpoint).observe(duration)

    # Prepend middleware
    app.middlewares.insert(0, prometheus_middleware)

    # prometheus_client's content type carries a charset ("text/plain;
    # version=...; charset=utf-8"). aiohttp refuses a charset in content_type
    # with a ValueError, so /metrics answered 500; as a header it goes out
    # exactly as the client library names it (PRD-262 US-045).
    async def metrics_handler(request):
        return web.Response(
            body=generate_latest(REGISTRY),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )

    app.router.add_get(METRICS_PATH, metrics_handler)

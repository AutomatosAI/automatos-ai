"""
Automatos Prometheus Metrics — FastAPI Integration

Exposes a /metrics endpoint with standard HTTP metrics plus custom
Automatos application metrics for Prometheus scraping.

Usage:
    from automatos_metrics import setup_metrics
    setup_metrics(app)  # FastAPI app

    # Then use custom metrics anywhere:
    from automatos_metrics import AGENT_HEARTBEATS, REQUEST_DURATION
    AGENT_HEARTBEATS.labels(agent_id="cto", status="success").inc()
"""

import os
import time
from typing import Optional

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

SERVICE_NAME = os.environ.get("SERVICE_NAME", "automatos-backend")

# ─────────────────────────────────────────────
# Standard HTTP Metrics
# ─────────────────────────────────────────────
if PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter(
        "automatos_http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status_code"],
    )

    REQUEST_DURATION = Histogram(
        "automatos_http_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "endpoint"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )

    REQUESTS_IN_PROGRESS = Gauge(
        "automatos_http_requests_in_progress",
        "Number of HTTP requests currently being processed",
        ["method"],
    )

    # ─────────────────────────────────────────────
    # Agent Metrics
    # ─────────────────────────────────────────────
    AGENT_HEARTBEATS = Counter(
        "automatos_agent_heartbeat_total",
        "Total agent heartbeat checks",
        ["agent_id", "status"],
    )

    AGENT_HEARTBEAT_DURATION = Histogram(
        "automatos_agent_heartbeat_duration_seconds",
        "Agent heartbeat check latency",
        ["agent_id"],
    )

    AGENT_TOKEN_USAGE = Counter(
        "automatos_agent_token_usage_total",
        "Total tokens consumed by agents",
        ["agent_id", "model", "direction"],
    )

    ACTIVE_AGENTS = Gauge(
        "automatos_active_agents",
        "Number of currently active agents",
    )

    # ─────────────────────────────────────────────
    # Worker Metrics
    # ─────────────────────────────────────────────
    WORKER_ACTIVE_TASKS = Gauge(
        "automatos_worker_active_tasks",
        "Number of tasks currently being processed",
    )

    WORKER_TASK_DURATION = Histogram(
        "automatos_worker_task_duration_seconds",
        "Task execution duration",
        ["task_type"],
        buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0],
    )

    WORKER_QUEUE_DEPTH = Gauge(
        "automatos_worker_queue_depth",
        "Number of tasks waiting in queue",
    )

    WORKER_TASK_TOTAL = Counter(
        "automatos_worker_task_total",
        "Total tasks processed",
        ["task_type", "status"],
    )

    WORKER_ERRORS = Counter(
        "automatos_worker_errors_total",
        "Total worker errors",
        ["error_type"],
    )

    # ─────────────────────────────────────────────
    # Chat / LLM Metrics
    # ─────────────────────────────────────────────
    LLM_REQUEST_DURATION = Histogram(
        "automatos_llm_request_duration_seconds",
        "LLM API call latency",
        ["model", "provider"],
        buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    )

    LLM_TOKEN_USAGE = Counter(
        "automatos_llm_tokens_total",
        "Total LLM tokens consumed",
        ["model", "provider", "direction"],
    )

    # ─────────────────────────────────────────────
    # Service Info
    # ─────────────────────────────────────────────
    SERVICE_INFO = Info(
        "automatos_service",
        "Service build information",
    )


def setup_metrics(app, service_name: Optional[str] = None):
    """Add Prometheus /metrics endpoint and request instrumentation to FastAPI.

    Args:
        app: FastAPI application instance
        service_name: Override SERVICE_NAME env var
    """
    if not PROMETHEUS_AVAILABLE:
        import logging
        logging.getLogger(__name__).warning(
            "prometheus_client not installed — /metrics endpoint disabled. "
            "Install with: pip install prometheus-client"
        )
        return

    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.middleware.base import BaseHTTPMiddleware

    svc = service_name or SERVICE_NAME

    # Set service info
    SERVICE_INFO.info({
        "service": svc,
        "environment": os.environ.get("ENVIRONMENT", "unknown"),
    })

    # Metrics endpoint
    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint():
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST,
        )

    # Request instrumentation middleware
    class PrometheusMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            method = request.method
            # Normalize path to avoid cardinality explosion
            path = _normalize_path(request.url.path)

            REQUESTS_IN_PROGRESS.labels(method=method).inc()
            start = time.perf_counter()

            try:
                response = await call_next(request)
                duration = time.perf_counter() - start

                REQUEST_COUNT.labels(
                    method=method,
                    endpoint=path,
                    status_code=response.status_code,
                ).inc()
                REQUEST_DURATION.labels(
                    method=method,
                    endpoint=path,
                ).observe(duration)

                return response
            except Exception:
                REQUEST_COUNT.labels(
                    method=method,
                    endpoint=path,
                    status_code=500,
                ).inc()
                raise
            finally:
                REQUESTS_IN_PROGRESS.labels(method=method).dec()

    app.add_middleware(PrometheusMiddleware)


def _normalize_path(path: str) -> str:
    """Replace dynamic path segments with placeholders to keep cardinality low."""
    import re
    # Replace UUIDs
    path = re.sub(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        "{id}",
        path,
    )
    # Replace numeric IDs
    path = re.sub(r"/\d+(?=/|$)", "/{id}", path)
    return path

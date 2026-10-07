"""PRD-256 O1 (#847): OpenTelemetry traces over OTLP, default off.

Off: nothing from ``opentelemetry`` is imported, and the app's lifespan is the
one it was. On: a server span per request (not the probes) carrying the
request's own IDs, sampled by ratio, exported in batches, and a tracing fault
never fails a request or the boot. Spans go to an in-memory exporter here.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.observability import otel  # noqa: E402

WS = "00000000-0000-0000-0000-0000000000c1"


# ── off ──────────────────────────────────────────────────────────────────────

def test_off_imports_nothing_from_opentelemetry():
    """The whole off path, in a fresh interpreter: no ``opentelemetry`` module loads."""
    script = (
        "import sys\n"
        "from fastapi import FastAPI\n"
        "from core.observability import otel\n"
        "app = FastAPI(lifespan=otel.with_tracing(None))\n"
        "otel.instrument_app(app)\n"
        "assert otel.start_tracing() is False\n"
        "otel.annotate_request('req-1', 'ws-1')\n"
        "otel.shutdown_tracing()\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('opentelemetry'))\n"
        "assert not loaded, loaded\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "OTEL_ENABLED"}
    done = subprocess.run([sys.executable, "-c", script], cwd=_ORCH, env=env, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]


def test_off_with_redis_loaded_starts_no_tracing_at_all():
    """redis-py 8 imports OpenTelemetry's *metrics* SDK on its own when the package is
    installed (``redis/observability/providers.py``, an optional import). It sets no
    provider and starts nothing. Off must still mean: no trace SDK, no exporter, no
    instrumentation, no provider, no export thread."""
    script = (
        "import sys, threading\n"
        "import redis\n"
        "from fastapi import FastAPI\n"
        "from core.observability import otel\n"
        "app = FastAPI(lifespan=otel.with_tracing(None))\n"
        "otel.instrument_app(app)\n"
        "otel.start_tracing()\n"
        "otel.annotate_request('req-1', 'ws-1')\n"
        "bad = [m for m in sys.modules if m.startswith(('opentelemetry.sdk.trace', 'opentelemetry.exporter',\n"
        "       'opentelemetry.instrumentation'))]\n"
        "assert not bad, bad\n"
        "from opentelemetry import trace\n"
        "assert type(trace.get_tracer_provider()).__name__ == 'ProxyTracerProvider'\n"
        "assert [t.name for t in threading.enumerate()] == ['MainThread']\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "OTEL_ENABLED"}
    done = subprocess.run([sys.executable, "-c", script], cwd=_ORCH, env=env, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]


def test_off_leaves_the_lifespan_as_it_was(monkeypatch):
    monkeypatch.setattr(otel.config, "OTEL_ENABLED", False, raising=False)

    async def lifespan(app):
        yield

    assert otel.with_tracing(lifespan) is lifespan


def test_main_wires_the_lifespan_and_the_instrumentation():
    """main.py: the provider around the lifespan, the server spans on the app, the IDs per request."""
    tree = ast.parse((_ORCH / "main.py").read_text(encoding="utf-8"))
    calls = {node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert {"with_tracing", "instrument_app", "annotate_request"} <= calls


# ── settings ─────────────────────────────────────────────────────────────────

def test_otlp_headers_follow_the_spec_format():
    assert otel.otlp_headers("Authorization=Basic%20abc%3D,x-tenant = acme") == {
        "Authorization": "Basic abc=", "x-tenant": "acme"}
    assert otel.otlp_headers("") == {} and otel.otlp_headers(None) == {}
    assert otel.otlp_headers("novalue,=x,k=") == {}


# ── on ───────────────────────────────────────────────────────────────────────

@pytest.fixture
def traced(monkeypatch):
    """Tracing on, an in-memory exporter, and an app shaped like main.py's request-ID middleware."""
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    monkeypatch.setattr(otel.config, "OTEL_ENABLED", True, raising=False)
    monkeypatch.setattr(otel.config, "OTEL_SERVICE_NAME", "automatos-api-test", raising=False)

    def make(ratio=1.0, exporter=None):
        exporter = exporter or InMemorySpanExporter()
        provider = otel.build_provider(exporter, ratio)
        app = FastAPI()

        @app.middleware("http")
        async def ids(request: Request, call_next):
            otel.annotate_request(request.headers.get("X-Request-ID", ""), request.headers.get("X-Workspace-ID", ""))
            return await call_next(request)

        @app.get("/api/things/{thing_id}")
        def thing(thing_id: int):
            return {"id": thing_id}

        @app.get("/health")
        def health():
            return {"status": "healthy"}

        @app.get("/health/ready")
        def ready():
            return {"status": "ready"}

        otel.instrument_app(app, tracer_provider=provider)
        return TestClient(app), provider, exporter

    return make


def _server_spans(provider, exporter):
    from opentelemetry.trace import SpanKind

    provider.force_flush()
    return [s for s in exporter.get_finished_spans() if s.kind == SpanKind.SERVER]


def test_a_request_is_a_server_span_carrying_its_ids(traced):
    client, provider, exporter = traced()
    r = client.get("/api/things/7", headers={"X-Request-ID": "req-abc", "X-Workspace-ID": WS})
    assert r.status_code == 200
    spans = _server_spans(provider, exporter)
    assert [s.name for s in spans] == ["GET /api/things/{thing_id}"]
    assert len(exporter.get_finished_spans()) == 1   # no span per ASGI send/receive message
    attrs = spans[0].attributes
    assert attrs[otel.ATTR_REQUEST_ID] == "req-abc" and attrs[otel.ATTR_WORKSPACE_ID] == WS
    assert attrs["http.route"] == "/api/things/{thing_id}"
    assert spans[0].resource.attributes["service.name"] == "automatos-api-test"


def test_the_probes_are_never_traced(traced):
    client, provider, exporter = traced()
    assert client.get("/health").status_code == 200 and client.get("/health/ready").status_code == 200
    assert _server_spans(provider, exporter) == []


def test_a_zero_ratio_keeps_no_new_trace(traced):
    client, provider, exporter = traced(ratio=0.0)
    assert client.get("/api/things/1").status_code == 200
    assert _server_spans(provider, exporter) == []


def test_a_sampled_caller_is_followed_whatever_the_ratio(traced):
    client, provider, exporter = traced(ratio=0.0)
    parent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    assert client.get("/api/things/1", headers={"traceparent": parent}).status_code == 200
    spans = _server_spans(provider, exporter)
    assert len(spans) == 1 and format(spans[0].context.trace_id, "032x") == "0af7651916cd43dd8448eb211c80319c"


def test_an_exporter_that_fails_never_fails_the_request(traced):
    from opentelemetry.sdk.trace.export import SpanExporter

    class Broken(SpanExporter):
        def export(self, spans):
            raise ConnectionError("collector down")

        def shutdown(self):
            return None

    client, provider, _ = traced(exporter=Broken())
    assert client.get("/api/things/3").status_code == 200
    provider.force_flush()   # the batch processor logs the failure; nothing reaches the caller


# ── the process's provider ───────────────────────────────────────────────────

@pytest.fixture
def fresh_global_provider():
    """The SDK allows one global provider per process; give each test its own."""
    from opentelemetry import trace
    from opentelemetry.util._once import Once

    saved = (trace._TRACER_PROVIDER, trace._TRACER_PROVIDER_SET_ONCE)
    trace._TRACER_PROVIDER, trace._TRACER_PROVIDER_SET_ONCE = None, Once()
    otel._State.provider = None
    yield
    otel.shutdown_tracing()
    trace._TRACER_PROVIDER, trace._TRACER_PROVIDER_SET_ONCE = saved


def test_the_lifespan_starts_the_provider_and_flushes_it_at_the_end(monkeypatch, fresh_global_provider):
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    monkeypatch.setattr(otel.config, "OTEL_ENABLED", True, raising=False)
    exporter = InMemorySpanExporter()
    monkeypatch.setattr(otel, "_otlp_exporter", lambda: exporter)
    seen = {}

    @asynccontextmanager
    async def lifespan(app):
        seen["provider_during"] = otel._State.provider
        yield

    app = FastAPI(lifespan=otel.with_tracing(lifespan))
    with TestClient(app):
        assert trace.get_tracer_provider() is seen["provider_during"]
        with trace.get_tracer("t").start_as_current_span("work"):
            pass
    assert seen["provider_during"] is not None and otel._State.provider is None
    assert [s.name for s in exporter.get_finished_spans()] == ["work"]   # flushed at shutdown


def test_a_provider_that_cannot_start_never_stops_the_boot(monkeypatch, fresh_global_provider):
    monkeypatch.setattr(otel.config, "OTEL_ENABLED", True, raising=False)

    def broken():
        raise ValueError("bad endpoint")

    monkeypatch.setattr(otel, "_otlp_exporter", broken)
    assert otel.start_tracing() is False and otel._State.provider is None


def test_the_otlp_exporter_targets_the_collectors_traces_path(monkeypatch):
    from opentelemetry.exporter.otlp.proto.http import trace_exporter

    seen = {}
    monkeypatch.setattr(trace_exporter, "OTLPSpanExporter", lambda **kwargs: seen.update(kwargs) or "exporter")
    monkeypatch.setattr(otel.config, "OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318/", raising=False)
    monkeypatch.setattr(otel.config, "OTEL_EXPORTER_OTLP_HEADERS", "Authorization=Bearer%20t", raising=False)
    assert otel._otlp_exporter() == "exporter"
    assert seen == {"endpoint": "http://collector:4318/v1/traces", "headers": {"Authorization": "Bearer t"}}

"""PRD-256 O2b (#847): the workspace worker continues the API's trace.

The platform's HTTP client sends ``traceparent`` with every call to the worker
(O2a). With ``OTEL_ENABLED`` on in the worker, each request is a server span
that continues that trace: one trace from the API's request, through its client
span, into the worker. Off: nothing is imported and the app is as it was.
Spans go to an in-memory exporter.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_TESTS = Path(__file__).resolve().parent
_ORCH = _TESTS.parent
for path in (_TESTS, _ORCH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from helpers_workspace_worker import WORKER_DIR, load_worker_main, worker_server  # noqa: E402

WS = "00000000-0000-0000-0000-0000000000c1"
CALLER = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


def _worker():
    return SimpleNamespace(_worker_id="w-test", _active_tasks={}, concurrency=1, _redis=None)


# ── off ──────────────────────────────────────────────────────────────────────

def test_off_imports_nothing_and_leaves_the_app_as_it_was():
    script = (
        "import sys\n"
        "from aiohttp import web\n"
        "import worker_otel\n"
        "app = web.Application()\n"
        "before = (list(app.middlewares), list(app.on_cleanup))\n"
        "worker_otel.attach(app)\n"
        "assert (list(app.middlewares), list(app.on_cleanup)) == before\n"
        "loaded = [m for m in sys.modules if m.startswith('opentelemetry')]\n"
        "assert not loaded, loaded\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "OTEL_ENABLED"}
    done = subprocess.run([sys.executable, "-c", script], cwd=WORKER_DIR, env=env, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]


# ── on ───────────────────────────────────────────────────────────────────────

@pytest.fixture
def traced_worker(monkeypatch, tmp_path):
    """The worker's modules with tracing on, under a fresh global provider that
    records to memory. Yields ``(worker_http, exporter)``."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.util._once import Once

    load_worker_main(monkeypatch, tmp_path)
    import worker_http
    import worker_otel

    monkeypatch.setenv("OTEL_ENABLED", "true")
    saved = (trace._TRACER_PROVIDER, trace._TRACER_PROVIDER_SET_ONCE)
    trace._TRACER_PROVIDER, trace._TRACER_PROVIDER_SET_ONCE = None, Once()
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    monkeypatch.setattr(worker_otel._State, "provider", provider)   # this process's, already installed
    yield worker_http, exporter
    provider.shutdown()
    trace._TRACER_PROVIDER, trace._TRACER_PROVIDER_SET_ONCE = saved


def _server_spans(exporter):
    from opentelemetry.trace import SpanKind

    return [s for s in exporter.get_finished_spans() if s.kind == SpanKind.SERVER]


def test_a_worker_request_continues_the_callers_trace(traced_worker):
    from aiohttp.test_utils import TestClient, TestServer

    worker_http, exporter = traced_worker

    async def run():
        async with TestClient(TestServer(worker_http.build_app(_worker()))) as client:
            files = await client.get(f"/workspaces/{WS}/files?path=.", headers={"traceparent": CALLER})
            assert files.status == 200
            assert (await client.get("/health")).status == 200

    asyncio.run(run())
    spans = _server_spans(exporter)
    assert [s.name for s in spans] == ["GET /workspaces/{workspace_id}/files"]   # /health is never traced
    span = spans[0]
    assert format(span.context.trace_id, "032x") == "0af7651916cd43dd8448eb211c80319c"
    assert format(span.parent.span_id, "016x") == "b7ad6b7169203331"
    assert dict(span.attributes) == {"http.method": "GET", "http.route": "/workspaces/{workspace_id}/files",
                                     "automatos.workspace_id": WS, "http.status_code": 200}


def test_the_api_and_the_worker_are_one_trace(traced_worker, monkeypatch, tmp_path):
    """The API's request span → its HTTP client span (O2a) → the worker's server span."""
    import httpx
    import sqlalchemy
    from opentelemetry import trace
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.trace import SpanKind

    from core.observability import otel

    _, exporter = traced_worker
    monkeypatch.setattr(otel.config, "OTEL_ENABLED", True, raising=False)
    monkeypatch.setattr(otel, "_LIBRARIES", (("httpx", otel._instrument_httpx),))
    otel.instrument_libraries(engine=sqlalchemy.create_engine("sqlite://"))

    async def run():
        async with worker_server(monkeypatch, tmp_path) as (base, _):
            with trace.get_tracer("api").start_as_current_span("GET /api/workspaces/{workspace_id}/files",
                                                               kind=SpanKind.SERVER) as api:
                async with httpx.AsyncClient() as client:
                    assert (await client.get(f"{base}/workspaces/{WS}/files?path=.")).status_code == 200
            return api

    try:
        api = asyncio.run(run())
    finally:
        HTTPXClientInstrumentor().uninstrument()
    by_kind = {s.kind: s for s in exporter.get_finished_spans() if s.name != "GET /api/workspaces/{workspace_id}/files"}
    client, worker = by_kind[SpanKind.CLIENT], by_kind[SpanKind.SERVER]
    trace_id = api.get_span_context().trace_id
    assert client.context.trace_id == worker.context.trace_id == trace_id
    assert client.parent.span_id == api.get_span_context().span_id
    assert worker.parent.span_id == client.context.span_id
    assert worker.name == "GET /workspaces/{workspace_id}/files"


def test_a_handler_error_is_recorded_on_the_span_and_still_answered(traced_worker):
    from aiohttp.test_utils import TestClient, TestServer

    worker_http, exporter = traced_worker

    async def run():
        async with TestClient(TestServer(worker_http.build_app(_worker()))) as client:
            assert (await client.get("/no/such/route")).status == 404

    asyncio.run(run())
    spans = _server_spans(exporter)
    assert [(s.name, s.attributes.get("http.status_code")) for s in spans] == [("GET unmatched", 404)]


def test_the_sampler_ratio_never_raises(traced_worker):
    import worker_otel

    assert [worker_otel.sampler_ratio(v) for v in ("0.5", "x", None, "3", "-2")] == [0.5, 1.0, 1.0, 1.0, 0.0]
    assert worker_otel.otlp_headers("Authorization=Basic%20a%3D,bad") == {"Authorization": "Basic a="}

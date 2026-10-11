"""PRD-256 O2a (#847): the server span's children — SQL, Redis, outbound HTTP, AWS.

On: one client span per SQL statement, Redis command, HTTP request and AWS call,
under the request that made it; ``traceparent`` goes out with every HTTP request
(the workspace worker continues the trace in O2b); no query value, Redis
argument or SQL bound value reaches the collector. Off: nothing is instrumented.
Spans go to an in-memory exporter; each test undoes the instrumentation.
"""
from __future__ import annotations

import ast
import asyncio
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.observability import otel  # noqa: E402
from tests.helpers_otel import fresh_global_providers  # noqa: E402

SECRET = "SECRET-VALUE-7"


# ── off ──────────────────────────────────────────────────────────────────────

def test_off_instruments_no_library_and_imports_no_instrumentation():
    script = (
        "import sys\n"
        "from core.observability import otel\n"
        "otel.instrument_libraries(engine=object())\n"
        "bad = [m for m in sys.modules if m.startswith(('opentelemetry.instrumentation', 'opentelemetry.sdk.trace'))]\n"
        "assert not bad, bad\n"
        "import httpx\n"
        "assert not hasattr(httpx.HTTPTransport.handle_request, '__wrapped__')\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "OTEL_ENABLED"}
    done = subprocess.run([sys.executable, "-c", script], cwd=_ORCH, env=env, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]


def test_main_instruments_the_libraries_and_lets_a_browser_send_traceparent():
    tree = ast.parse((_ORCH / "main.py").read_text(encoding="utf-8"))
    calls = {node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "instrument_libraries" in calls
    source = (_ORCH / "main.py").read_text(encoding="utf-8")
    assert '"traceparent", "tracestate"' in source


# ── on ───────────────────────────────────────────────────────────────────────

@pytest.fixture
def spans(monkeypatch):
    """Tracing on with a fresh global provider and an in-memory exporter; the libraries
    instrumented against a throwaway SQLite engine, and uninstrumented afterwards."""
    import sqlalchemy
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    monkeypatch.setattr(otel.config, "OTEL_ENABLED", True, raising=False)
    with fresh_global_providers():
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        engine = sqlalchemy.create_engine("sqlite://")
        otel.instrument_libraries(engine=engine)
        yield exporter, engine
        _uninstrument_all()
        provider.shutdown()


def _uninstrument_all():
    from opentelemetry.instrumentation.botocore import BotocoreInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

    for instrumentor in (SQLAlchemyInstrumentor(), HTTPXClientInstrumentor(), RedisInstrumentor(),
                         BotocoreInstrumentor()):
        instrumentor.uninstrument()


@pytest.fixture
def http_server():
    """A real local HTTP server (the instrumentation wraps the real transport) that
    records the ``traceparent`` it was sent."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            seen.append(self.headers.get("traceparent"))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", seen
    server.shutdown()
    server.server_close()


def _leaks(finished):
    return [(s.name, k) for s in finished for k, v in s.attributes.items() if SECRET in str(v)]


def test_an_outbound_request_is_a_child_span_that_carries_traceparent(spans, http_server):
    import httpx
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind

    exporter, _ = spans
    base, seen = http_server
    with trace.get_tracer("t").start_as_current_span("GET /api/agents/") as parent:
        assert httpx.get(f"{base}/v1/models?key={SECRET}&page=2").status_code == 200

        async def call():
            async with httpx.AsyncClient() as client:
                return await client.get(f"{base}/v1/models?key={SECRET}")

        assert asyncio.run(call()).status_code == 200
    clients = [s for s in exporter.get_finished_spans() if s.kind == SpanKind.CLIENT]
    trace_id = format(parent.get_span_context().trace_id, "032x")
    assert len(clients) == 2 and all(s.parent.span_id == parent.get_span_context().span_id for s in clients)
    assert len(seen) == 2 and all(tp and tp.split("-")[1] == trace_id for tp in seen)   # the callee continues it
    assert clients[0].attributes["http.url"].endswith("/v1/models?key=REDACTED&page=REDACTED")
    assert _leaks(exporter.get_finished_spans()) == []


def test_a_sql_statement_is_a_span_without_its_bound_values(spans):
    import sqlalchemy

    exporter, engine = spans
    with engine.connect() as conn:
        assert conn.execute(sqlalchemy.text("select :value"), {"value": SECRET}).scalar() == SECRET
    names = [s.name for s in exporter.get_finished_spans()]
    assert "select" in names
    assert _leaks(exporter.get_finished_spans()) == []


def test_a_redis_command_is_a_span_with_no_key_or_value(spans):
    import redis
    from redis.backoff import NoBackoff
    from redis.retry import Retry

    exporter, _ = spans
    with pytest.raises(redis.exceptions.ConnectionError):
        # No retries: the closed port fails at once (redis-py 8 retries with backoff by default).
        redis.Redis(port=1, socket_connect_timeout=0.2, retry=Retry(NoBackoff(), 0)).set(f"session:{SECRET}", SECRET)
    command = [s for s in exporter.get_finished_spans() if s.name == "SET"]
    assert command and command[0].attributes["db.statement"] == "SET ? ?"
    assert _leaks(exporter.get_finished_spans()) == []


def test_an_aws_call_is_a_span_with_no_payload(spans):
    import boto3
    from botocore.stub import Stubber

    exporter, _ = spans
    s3 = boto3.client("s3", region_name="us-east-1", aws_access_key_id="test", aws_secret_access_key="test")
    with Stubber(s3) as stub:
        stub.add_response("put_object", {}, {"Bucket": "b", "Key": f"k/{SECRET}", "Body": SECRET.encode()})
        s3.put_object(Bucket="b", Key=f"k/{SECRET}", Body=SECRET.encode())
    call = [s for s in exporter.get_finished_spans() if s.name == "S3.PutObject"]
    assert call and call[0].attributes["rpc.method"] == "PutObject"
    assert _leaks(exporter.get_finished_spans()) == []


def test_one_library_failing_never_stops_the_others(monkeypatch, spans):
    exporter, _ = spans
    _uninstrument_all()
    calls = []

    def broken(_):
        raise ImportError("not installed")

    monkeypatch.setattr(otel, "_LIBRARIES", (("sqlalchemy", broken), ("redis", lambda _: calls.append("redis"))))
    otel.instrument_libraries(engine=object())
    assert calls == ["redis"]


def test_a_background_query_starts_no_trace_but_a_requests_query_is_its_child(monkeypatch):
    """The provider's sampler (``request_rooted_sampler``): background loops poll SQL and
    Redis all the time, and each statement used to be a root trace of its own (tens of
    thousands an hour, seen live). A library call with no parent starts nothing; inside a
    request it is the request's child; any other new trace is kept by ratio."""
    import sqlalchemy
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import SpanKind

    exporter = InMemorySpanExporter()
    provider = otel.build_provider(exporter, 1.0)
    engine = sqlalchemy.create_engine("sqlite://")
    SQLAlchemyInstrumentor().instrument(engine=engine, tracer_provider=provider)
    try:
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("select 1"))              # a background loop's query
        provider.force_flush()
        assert exporter.get_finished_spans() == ()

        tracer = provider.get_tracer("t")
        with tracer.start_as_current_span("GET /api/agents/", kind=SpanKind.SERVER) as request:
            with engine.connect() as conn:
                conn.execute(sqlalchemy.text("select 2"))
        with tracer.start_as_current_span("mission.run"):             # a root of its own kind (O4)
            pass
        provider.force_flush()
        spans = exporter.get_finished_spans()
        children = [s for s in spans if s.parent and s.parent.span_id == request.get_span_context().span_id]
        assert {s.name for s in spans} >= {"GET /api/agents/", "mission.run"} and children
        assert all(s.kind == SpanKind.CLIENT for s in children)
    finally:
        SQLAlchemyInstrumentor().uninstrument()
        provider.shutdown()

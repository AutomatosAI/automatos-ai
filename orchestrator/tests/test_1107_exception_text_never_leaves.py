"""#1107 (PRD-256 Principle 5): an error leaves a span as its type, never its text.

The FastAPI and httpx instrumentations record an unhandled exception's message and stack
trace on the span and copy the message into its status; SQLAlchemy puts the driver's
error in the status (Postgres's DETAIL carries row values). Any of that can hold a URL
with a token, a customer's data or a file name. ``build_provider`` exports through
``type_only_exporter``: exception events keep ``exception.type`` (and ``exception.escaped``),
an ERROR status names only the type, and a span it can't rewrite is dropped, never sent
as it was. Spans go to an in-memory exporter, through the app's own provider build.
"""
from __future__ import annotations

import os
import sys
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

SECRET = "SECRET_VALUE_7"


@pytest.fixture
def provider(monkeypatch):
    """The app's provider build (request-rooted sampler, the type-only exporter) into memory."""
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    monkeypatch.setattr(otel.config, "OTEL_ENABLED", True, raising=False)
    memory = InMemorySpanExporter()
    built = otel.build_provider(memory, 1.0)

    def finished():
        built.force_flush()
        return memory.get_finished_spans()

    yield built, finished
    built.shutdown()


def _texts(span):
    """Everything on an exported span that could carry text."""
    values = [span.name, span.status.description or "", *map(str, span.attributes.values())]
    for event in span.events:
        values += [event.name, *map(str, (event.attributes or {}).values())]
    return values


def _leaks(spans):
    return [(span.name, value) for span in spans for value in _texts(span) if SECRET in value]


def test_an_unhandled_request_error_leaves_its_type_not_its_message(provider):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from opentelemetry.trace import SpanKind, StatusCode

    built, finished = provider
    app = FastAPI()

    @app.get("/boom")
    def boom():
        raise RuntimeError(f"calling http://workspace-worker:8081/files?token={SECRET} failed")

    otel.instrument_app(app, tracer_provider=built)
    assert TestClient(app, raise_server_exceptions=False).get("/boom").status_code == 500
    [server] = [s for s in finished() if s.kind == SpanKind.SERVER]
    assert server.status.status_code == StatusCode.ERROR
    assert server.status.description in (None, "RuntimeError")
    [exception] = [e for e in server.events if e.name == "exception"]
    assert exception.attributes["exception.type"] == "RuntimeError"
    assert set(exception.attributes) <= {"exception.type", "exception.escaped"}
    assert _leaks(finished()) == []


def test_a_database_error_leaves_no_driver_text(provider):
    import sqlalchemy
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.trace import SpanKind, StatusCode

    built, finished = provider
    engine = sqlalchemy.create_engine("sqlite://")
    SQLAlchemyInstrumentor().instrument(engine=engine, tracer_provider=built)
    try:
        with built.get_tracer("t").start_as_current_span("GET /api/agents/", kind=SpanKind.SERVER):
            with engine.connect() as conn, pytest.raises(sqlalchemy.exc.OperationalError):
                conn.execute(sqlalchemy.text(f"select * from {SECRET}"))   # "no such table: SECRET_VALUE_7"
    finally:
        SQLAlchemyInstrumentor().uninstrument()
    [query] = [s for s in finished() if s.kind == SpanKind.CLIENT and s.name == "select"]
    # The driver's text ("no such table: SECRET_VALUE_7") was the status; db.statement is
    # the SQL itself, which the app writes with bound parameters, so it is not checked here.
    assert query.status.status_code == StatusCode.ERROR and query.status.description is None
    assert query.events == ()


def test_a_postgres_detail_never_leaves_in_a_status(provider):
    from opentelemetry.trace import Status, StatusCode

    built, finished = provider
    with built.get_tracer("t").start_as_current_span("POST /api/users", record_exception=False) as span:
        span.set_status(Status(StatusCode.ERROR, 'duplicate key value violates unique constraint "users_email_key"\n'
                                                 f"DETAIL:  Key (email)=({SECRET}@bank.example) already exists."))
    [exported] = finished()
    assert exported.status.status_code == StatusCode.ERROR and exported.status.description is None
    assert _leaks(finished()) == []


def test_a_status_naming_the_spans_error_type_and_other_events_are_kept(provider):
    from opentelemetry.trace import Status, StatusCode

    built, finished = provider
    tracer = built.get_tracer("t")
    with tracer.start_as_current_span("invoke_agent Auto") as span:   # as genai.py marks a failed run
        span.set_attribute("error.type", "agent_error")
        span.set_status(Status(StatusCode.ERROR, "agent_error"))
        span.add_event("retry", {"attempt": 2})
    with tracer.start_as_current_span("chat gpt-test") as span:
        span.set_attribute("gen_ai.usage.input_tokens", 120)
    agent, chat = finished()
    assert agent.status.description == "agent_error"
    assert [(e.name, dict(e.attributes)) for e in agent.events] == [("retry", {"attempt": 2})]
    assert chat.status.status_code == StatusCode.UNSET and chat.attributes["gen_ai.usage.input_tokens"] == 120


def test_a_status_that_only_looks_like_a_type_never_leaves(provider):
    from opentelemetry.trace import Status, StatusCode

    built, finished = provider
    with built.get_tracer("t").start_as_current_span("GET /files", record_exception=False) as span:
        span.set_status(Status(StatusCode.ERROR, f"{SECRET}.xlsx"))
    [exported] = finished()
    assert exported.status.status_code == StatusCode.ERROR and exported.status.description is None


def test_an_exception_event_keeps_only_its_type_whatever_the_caller_added(provider):
    built, finished = provider
    with built.get_tracer("t").start_as_current_span("POST /api/chat", record_exception=False) as span:
        span.record_exception(ValueError(SECRET), attributes={"detail": SECRET, "user": f"{SECRET}@bank.example"})
    [exception] = finished()[0].events
    assert set(exception.attributes) == {"exception.type", "exception.escaped"}
    assert _leaks(finished()) == []


def test_a_span_that_cannot_be_rewritten_is_dropped_not_sent(provider, monkeypatch):
    built, finished = provider

    def broken(span):
        raise ValueError("rewrite failed")

    monkeypatch.setattr(otel, "error_type_only", broken)
    with built.get_tracer("t").start_as_current_span("GET /boom"):
        pass
    assert finished() == ()

"""#1306: FastAPI's own tracing is off; instrument_app's spans are the only ones.

FastAPI 0.142+ traces requests itself, on by default whenever a global tracer provider
exists, which ``otel.start_tracing`` sets up. Its server span carries ``url.query`` in
clear (our redaction is a request hook on instrument_app's spans only), and it adds a
span each for dependencies, endpoint and serialization. With instrument_app in place
FastAPI skips its own, but an app instrument_app doesn't wrap, or a boot where it failed,
would send them. The app is built with ``otel.FASTAPI_TELEMETRY``, which turns them off.
"""
from __future__ import annotations

import ast
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from core.observability import otel
from tests.helpers_otel import fresh_global_providers

_ORCH = Path(__file__).resolve().parents[1]
MARKER = "MARKER_QUERY_7"


def _dependency() -> int:
    return 1


def _spans_of_a_request(**app_options):
    """A request to an app instrument_app doesn't wrap, with tracing on (a global
    provider, as start_tracing sets one): every span exported."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    memory = InMemorySpanExporter()
    with fresh_global_providers():
        provider = otel.build_provider(memory, 1.0)
        trace.set_tracer_provider(provider)
        app = FastAPI(**app_options)

        @app.get("/items")
        def items(value: int = Depends(_dependency)):
            return {"ok": value}

        assert TestClient(app).get(f"/items?token={MARKER}&page=2").status_code == 200
        provider.force_flush()
        spans = memory.get_finished_spans()
        provider.shutdown()
    return spans


def test_with_fastapi_telemetry_off_an_unwrapped_app_sends_no_span():
    assert _spans_of_a_request(telemetry=otel.FASTAPI_TELEMETRY) == ()


def test_fastapis_own_tracing_is_what_the_setting_turns_off():
    """The control: FastAPI's defaults trace the request, query and all. If this starts
    failing, FastAPI changed its defaults or dropped its tracing: check the setting still
    says what it should, rather than letting the test above pass for the wrong reason."""
    spans = _spans_of_a_request()
    assert spans, "FastAPI no longer traces by default: re-check otel.FASTAPI_TELEMETRY"
    assert {span.instrumentation_scope.name for span in spans} == {"fastapi"}
    assert any(MARKER in str(span.attributes.get("url.query", "")) for span in spans)


def test_main_builds_the_app_with_fastapi_telemetry_off():
    tree = ast.parse((_ORCH / "main.py").read_text(encoding="utf-8"))
    builds = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
              and getattr(node.func, "id", None) == "FastAPI"]
    assert len(builds) == 1
    keywords = {keyword.arg: keyword.value for keyword in builds[0].keywords}
    assert getattr(keywords.get("telemetry"), "id", None) == "FASTAPI_TELEMETRY"

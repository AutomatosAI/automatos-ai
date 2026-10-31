"""PRD-264 FR-17 — a checked file's hash never reaches a metric label, a log line or a trace.

The public detection route carries the SHA-256 of a visitor's file in its path
(``/api/public/provenance/{sha256}``). The EUAI-10 security review (M1) found every
checked hash became its own Prometheus label: unbounded series, readable on the
unauthenticated ``/metrics``. Proven here, for each place that copies a request path
out of the process:

* the request-metrics normaliser collapses a 64-hex segment, so two hashes are one label;
* the log relay's ``path`` field and the console's ``http_path`` record field;
* uvicorn's access line (``DigestPathFilter``, installed by ``setup_logging``);
* the JSON-500 converter's log line;
* the OpenTelemetry URL attributes (the end-to-end span test is in
  ``test_prd256_o1_otel_foundation``).
"""
from __future__ import annotations

import contextvars
import logging
import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

ORCH = Path(__file__).resolve().parents[1]
FIRST = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
SECOND = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
ROUTE = "/api/public/provenance"
REDACTED_PATH = f"{ROUTE}/{{sha256}}"


# --------------------------------------------------------------------------- the rule


def test_a_64_hex_segment_collapses_and_nothing_else_does():
    from core.monitoring.path_digests import redact_digest_segments

    assert redact_digest_segments(f"{ROUTE}/{FIRST}") == REDACTED_PATH
    assert redact_digest_segments(f"{ROUTE}/{FIRST.upper()}") == REDACTED_PATH
    assert redact_digest_segments(f"/a/{FIRST}/b?x=1") == "/a/{sha256}/b?x=1"
    assert redact_digest_segments(f"https://h/a/{FIRST}#f") == "https://h/a/{sha256}#f"
    for kept in (f"{ROUTE}/{FIRST[1:]}", f"{ROUTE}/{FIRST}0", f"{ROUTE}/x{FIRST}", "/api/agents/42", ""):
        assert redact_digest_segments(kept) == kept, kept


# --------------------------------------------------------------------------- metrics (review M1)


def test_two_different_hashes_are_one_metrics_label():
    from core.monitoring.automatos_metrics import _normalize_path

    labels = {_normalize_path(f"{ROUTE}/{digest}") for digest in (FIRST, SECOND)}

    assert labels == {REDACTED_PATH}


def test_the_normaliser_still_collapses_ids():
    from core.monitoring.automatos_metrics import _normalize_path

    assert _normalize_path("/api/agents/42/runs/7") == "/api/agents/{id}/runs/{id}"
    assert _normalize_path("/api/w/123e4567-e89b-12d3-a456-426614174000") == "/api/w/{id}"
    # A hash of digits only is still a hash, not two ids.
    assert _normalize_path(f"{ROUTE}/{'1' * 64}") == REDACTED_PATH


# --------------------------------------------------------------------------- logs


def _in_request_context(path, fn):
    """Run ``fn`` with the request-context path set as main.py's middleware sets it."""
    from core.monitoring.automatos_logging import http_path_var

    def _run():
        http_path_var.set(path)
        return fn()

    return contextvars.copy_context().run(_run)


def _record(msg="a line", args=None):
    return logging.LogRecord("prd264.test", logging.ERROR, __file__, 1, msg, args, None)


def test_the_log_relay_path_field_holds_no_hash():
    from core.monitoring.automatos_logging import LogRelayHandler

    # Built without __init__: no flusher thread and no registration, only the formatting.
    handler = LogRelayHandler.__new__(LogRelayHandler)
    handler.formatter = None
    handler.service = "prd264-test"

    entry = _in_request_context(f"{ROUTE}/{FIRST}", lambda: handler._format_entry(_record()))

    assert entry["context"]["path"] == REDACTED_PATH
    assert FIRST not in repr(entry)


def test_the_console_context_field_holds_no_hash():
    from core.utils.logging_adapter import ContextFilter

    record = _record()
    _in_request_context(f"{ROUTE}/{FIRST}", lambda: ContextFilter().filter(record))

    assert record.http_path == REDACTED_PATH


def test_the_access_line_keeps_everything_but_the_hash():
    from core.monitoring.path_digests import DigestPathFilter

    record = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
                               ("203.0.113.5:4242", "GET", f"{ROUTE}/{FIRST}", "1.1", 200), None)

    assert DigestPathFilter().filter(record) is True
    assert record.getMessage() == f'203.0.113.5:4242 - "GET {REDACTED_PATH} HTTP/1.1" 200'


def test_setup_logging_puts_the_filter_on_the_access_logger():
    """In a fresh interpreter, as the app boots: setup_logging guards uvicorn.access, once."""
    script = (
        "import logging\n"
        "from core.monitoring.automatos_logging import setup_logging\n"
        "from core.monitoring.path_digests import DigestPathFilter\n"
        "setup_logging(service='prd264-test', enable_relay=False)\n"
        "setup_logging(service='prd264-test', enable_relay=False)\n"
        "guards = [f for f in logging.getLogger('uvicorn.access').filters if isinstance(f, DigestPathFilter)]\n"
        "assert len(guards) == 1, guards\n"
    )
    done = subprocess.run([sys.executable, "-c", script], cwd=ORCH, env=dict(os.environ),
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]


def test_an_unhandled_error_s_log_line_holds_no_hash(caplog):
    from core.observability.error_response import json_500_for_unhandled_errors

    app = FastAPI()
    app.middleware("http")(json_500_for_unhandled_errors)

    @app.get(ROUTE + "/{sha256}")
    def boom(sha256: str):
        raise RuntimeError("lookup failed")

    with caplog.at_level(logging.ERROR, logger="core.observability.error_response"):
        response = TestClient(app, raise_server_exceptions=False).get(f"{ROUTE}/{FIRST}")

    assert response.status_code == 500
    (line,) = [r.getMessage() for r in caplog.records if r.name == "core.observability.error_response"]
    assert REDACTED_PATH in line and FIRST not in line


# --------------------------------------------------------------------------- traces


class _Span:
    def __init__(self, attributes):
        self.attributes = dict(attributes)

    def set_attribute(self, name, value):
        self.attributes = {**self.attributes, name: value}


def test_no_url_attribute_leaves_with_a_hash_or_a_query_value():
    from core.observability import otel

    span = _Span({
        "http.target": f"{ROUTE}/{FIRST}?code=abc",
        "http.url": f"https://api.example.test{ROUTE}/{FIRST}",
        "url.path": f"{ROUTE}/{FIRST}",
        "url.query": "code=abc",
        "http.route": ROUTE + "/{sha256}",
    })

    otel._scrub_url_attributes(span)

    assert span.attributes == {
        "http.target": f"{REDACTED_PATH}?code=REDACTED",
        "http.url": f"https://api.example.test{REDACTED_PATH}",
        "url.path": REDACTED_PATH,
        "url.query": "code=REDACTED",
        "http.route": ROUTE + "/{sha256}",
    }

"""PRD-282 US-302 (FR-10): the workspace worker's logs in the API's shape.

* The worker reads ``LOG_FORMAT`` and ``LOG_LEVEL`` in its own config module
  (``worker_config``), as the text in the environment, and its ``setup_logging`` checks
  them with ``log_settings``: a value it cannot use stops start-up, naming the setting.
* With ``json`` each line is one JSON record with the API's fields and the worker's own
  service name, its version from ``AUTOMATOS_VERSION`` (``dev`` without it, as the API's).
* The formatter and the settings check are the platform's modules, copied byte for byte:
  the worker's image holds only its own folder, so it cannot import them.
* ``RequestLogContextMiddleware`` holds a request's id (the platform's ``X-Request-ID``),
  workspace, method and path for every line it writes, the access line included.
* The platform sends its request's id with every call to the worker.

Each ``setup_logging`` case runs in a fresh interpreter in the worker's folder, as the
worker boots. The text line is pinned by test_prd256_o5_logs_metrics.py.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKER_DIR = ROOT / "services" / "workspace-worker"
PLATFORM_MONITORING = ROOT / "orchestrator" / "core" / "monitoring"
VENDORED = ("json_log_format.py", "log_settings.py")
VERSION = "v9.8.7-prd282"
PLANTED = "planted-282-value"
WS = "00000000-0000-0000-0000-000000000282"
ALWAYS = {"time", "level", "logger", "message", "service", "version"}
REQUEST = {"request_id": "req-282", "workspace_id": WS, "method": "GET", "path": f"/workspaces/{WS}/files"}

BOOT = f"""
import asyncio, logging
from automatos_logging import setup_logging
from worker_log_context import RequestLogContextMiddleware
root = setup_logging(service="workspace-worker", enable_relay=False)
log = logging.getLogger("worker.test")

async def app(scope, receive, send):
    log.info("inside %s", "a request", extra={{"user_id": "u-282", "duration_ms": 7}})
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("failed")
    logging.getLogger("uvicorn.access").info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:1", "GET",
                                             "/workspaces/{WS}/files?path=a.txt", "1.1", 200)

scope = {{"type": "http", "method": "GET", "path": "/workspaces/{WS}/files", "query_string": b"path=a.txt",
          "headers": [(b"x-request-id", b"req-282")]}}
asyncio.run(RequestLogContextMiddleware(app)(scope, None, None))
log.info("outside")
"""


def _env(**settings: str) -> dict:
    kept = {k: v for k, v in os.environ.items()
            if k not in {"LOG_FORMAT", "LOG_LEVEL", "AUTOMATOS_VERSION", "OTEL_ENABLED"}}
    return {**kept, **settings}


def _boot(script: str, **settings: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-W", "ignore", "-c", script], cwd=WORKER_DIR, env=_env(**settings),
                          capture_output=True, text=True, timeout=120)


def _worker_module(monkeypatch, name: str):
    monkeypatch.syspath_prepend(str(WORKER_DIR))
    return importlib.import_module(name)


# ── the copies ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", VENDORED)
def test_the_worker_holds_the_platforms_formatter_and_check_byte_for_byte(name):
    assert (WORKER_DIR / name).read_bytes() == (PLATFORM_MONITORING / name).read_bytes(), (
        f"services/workspace-worker/{name} must stay a copy of orchestrator/core/monitoring/{name}")


# ── the settings, in the worker's own config module ──────────────────────────

def test_the_worker_config_reads_both_settings_as_set(monkeypatch):
    worker_config = _worker_module(monkeypatch, "worker_config")
    monkeypatch.setenv("LOG_FORMAT", " JSON ")
    monkeypatch.setenv("LOG_LEVEL", "warning")
    assert (worker_config.log_format_setting(), worker_config.log_level_setting()) == (" JSON ", "warning")
    monkeypatch.delenv("LOG_FORMAT")
    monkeypatch.delenv("LOG_LEVEL")
    assert (worker_config.log_format_setting(), worker_config.log_level_setting()) == ("", "")


def test_the_worker_names_the_same_version_as_the_api(monkeypatch):
    from config import UNVERSIONED_BUILD

    worker_config = _worker_module(monkeypatch, "worker_config")
    monkeypatch.delenv("AUTOMATOS_VERSION", raising=False)
    assert worker_config.automatos_version() == worker_config.UNVERSIONED_BUILD == UNVERSIONED_BUILD
    monkeypatch.setenv("AUTOMATOS_VERSION", f" {VERSION} ")
    assert worker_config.automatos_version() == VERSION


@pytest.mark.parametrize(("name", "value"), [("LOG_FORMAT", PLANTED), ("LOG_LEVEL", PLANTED),
                                             ("LOG_LEVEL", "NOTSET"), ("LOG_LEVEL", "10")])
def test_a_value_the_worker_cannot_use_stops_start_up_naming_the_setting(name, value):
    done = _boot("from automatos_logging import setup_logging\nsetup_logging(service='w', enable_relay=False)\n",
                 **{name: value})
    assert done.returncode != 0
    assert f"LogSettingRefused: {name} cannot be used" in done.stderr
    assert PLANTED not in done.stderr


@pytest.mark.parametrize(("value", "level"), [(" Warning ", "WARNING"), (" debug\t", "DEBUG")])
def test_log_level_sets_the_workers_level_read_like_the_apis(value, level):
    script = ("import logging\nfrom automatos_logging import setup_logging\n"
              "root = setup_logging(service='w', enable_relay=False)\n"
              f"assert logging.getLevelName(root.level) == {level!r}, root.level\n"
              f"assert logging.getLevelName(root.handlers[0].level) == {level!r}\n")
    done = _boot(script, LOG_LEVEL=value)
    assert done.returncode == 0, done.stderr[-2000:]


@pytest.mark.parametrize("value", ["", "text", " TEXT "])
def test_text_is_the_line_it_was(value):
    script = ("import logging\nimport automatos_logging\nfrom automatos_logging import setup_logging\n"
              "root = setup_logging(service='w', enable_relay=False)\n"
              "assert root.handlers[0].formatter._fmt == automatos_logging.TEXT_FORMAT\n"
              "assert automatos_logging.TEXT_FORMAT == "
              "'%(asctime)s [%(name)s] %(levelname)s:%(_trace_context)s %(message)s'\n"
              "record = logging.LogRecord('t', logging.INFO, 'f', 1, 'hello', (), None)\n"
              "line = root.handlers[0].format(record)\n"
              "assert line.endswith('[t] INFO: hello') and not line.startswith('{'), line\n")
    done = _boot(script, LOG_FORMAT=value)
    assert done.returncode == 0, done.stderr[-2000:]


# ── json ─────────────────────────────────────────────────────────────────────

def _json_lines() -> list[dict]:
    done = _boot(BOOT, LOG_FORMAT="json", AUTOMATOS_VERSION=VERSION)
    assert done.returncode == 0, done.stderr[-2000:]
    return [json.loads(line) for line in done.stderr.splitlines()]


def test_with_json_each_worker_line_is_one_record_with_the_apis_fields():
    from core.monitoring.json_log_format import JSON_LOG_FIELDS

    inside, failed, access, outside = _json_lines()
    for line in (inside, failed, access, outside):
        assert set(line) <= set(JSON_LOG_FIELDS), sorted(set(line) - set(JSON_LOG_FIELDS))
        assert (line["service"], line["version"]) == ("workspace-worker", VERSION)
    assert {k: v for k, v in inside.items() if k != "time"} == {
        "level": "INFO", "logger": "worker.test", "message": "inside a request",
        "service": "workspace-worker", "version": VERSION, **REQUEST}
    assert set(outside) == ALWAYS and outside["message"] == "outside"


def test_an_exception_is_one_worker_record_and_the_access_line_carries_the_request():
    _, failed, access, _ = _json_lines()
    assert failed["message"] == "failed" and failed["request_id"] == "req-282"
    assert failed["exception"].startswith("Traceback") and "ValueError: boom" in failed["exception"]
    assert access["logger"] == "uvicorn.access" and {k: access[k] for k in REQUEST} == REQUEST
    assert "path=a.txt" not in access["path"]


# ── the request's fields ─────────────────────────────────────────────────────

def _fields_seen(module, scope: dict) -> tuple[dict, dict]:
    """(the fields a request's app sees, the fields in the same task once it has ended)."""
    seen: dict = {}

    async def app(scope, receive, send):
        seen.update(module.request_log_fields())

    async def serve() -> dict:
        await module.RequestLogContextMiddleware(app)(scope, None, None)
        return module.request_log_fields()

    return seen, asyncio.run(serve())


def _scope(path: str, headers: list) -> dict:
    return {"type": "http", "method": "POST", "path": path, "query_string": b"q=1", "headers": headers}


def test_the_middleware_holds_the_request_and_puts_it_back(monkeypatch):
    context = _worker_module(monkeypatch, "worker_log_context")
    seen, after = _fields_seen(context, _scope(f"/workspaces/{WS}/exec", [(b"X-Request-ID", b"req 282\n<x>")]))
    assert seen == {"request_id": "req282x", "workspace_id": WS, "method": "POST", "path": f"/workspaces/{WS}/exec"}
    assert after == {"request_id": "", "workspace_id": "", "method": "", "path": ""}
    seen, _ = _fields_seen(context, _scope("/health", []))
    assert seen == {"request_id": "", "workspace_id": "", "method": "POST", "path": "/health"}
    long_id = ("a" * (context.MAX_REQUEST_ID_LEN + 9)).encode()
    seen, _ = _fields_seen(context, _scope("/health", [(b"x-request-id", long_id)]))
    assert seen["request_id"] == "a" * context.MAX_REQUEST_ID_LEN


def test_anything_but_http_passes_through_untouched(monkeypatch):
    context = _worker_module(monkeypatch, "worker_log_context")
    seen, _ = _fields_seen(context, {"type": "lifespan"})
    assert seen == {"request_id": "", "workspace_id": "", "method": "", "path": ""}


def test_the_trace_ids_are_on_a_line_once_tracing_has_started(monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider

    context = _worker_module(monkeypatch, "worker_log_context")
    worker_otel = context.worker_otel
    monkeypatch.setattr(worker_otel._State, "provider", None)
    tracer = TracerProvider().get_tracer("prd282")
    with tracer.start_as_current_span("GET /workspaces/{workspace_id}/files") as span:
        assert "trace_id" not in context.request_log_fields(), "tracing off: OpenTelemetry is not asked"
        monkeypatch.setattr(worker_otel._State, "provider", object())
        fields = context.request_log_fields()
    ids = span.get_span_context()
    assert (fields["trace_id"], fields["span_id"]) == (f"{ids.trace_id:032x}", f"{ids.span_id:016x}")
    assert worker_otel.current_trace_ids() == ("", "")


def test_the_workers_app_holds_every_requests_fields(monkeypatch, tmp_path):
    from helpers_workspace_worker import load_worker_main

    monkeypatch.setenv("WORKER_INTERNAL_TOKEN", "")
    monkeypatch.delenv("OTEL_ENABLED", raising=False)
    load_worker_main(monkeypatch, tmp_path)
    import worker_http
    from worker_log_context import RequestLogContextMiddleware

    worker = SimpleNamespace(_worker_id="w-test", _active_tasks={}, concurrency=1, _redis=None)
    app = worker_http.build_app(worker)
    assert [m.cls for m in app.user_middleware] == [RequestLogContextMiddleware, worker_http.InternalAuthMiddleware]


# ── the platform's half ──────────────────────────────────────────────────────

def test_the_platform_sends_its_request_id_to_the_worker(monkeypatch):
    from core import workspace_client as wc
    from core.monitoring.automatos_logging import request_id_var

    monkeypatch.setattr(wc, "_client", None)
    client = wc._get_client()
    try:
        assert client.event_hooks["request"] == [wc._send_request_id]
    finally:
        asyncio.run(client.aclose())

    def sent(request_id: str) -> str | None:
        request = httpx.Request("GET", "http://worker.invalid/health")
        token = request_id_var.set(request_id)
        try:
            asyncio.run(wc._send_request_id(request))
        finally:
            request_id_var.reset(token)
        return request.headers.get("X-Request-ID")

    assert sent("req-282") == "req-282"
    assert sent("req 282\r\n<x>") == "req282x"
    assert sent("") is None

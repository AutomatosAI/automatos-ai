"""PRD-282 US-301 (FR-10, FR-11): with ``LOG_FORMAT=json`` every line is one JSON record.

Each case runs in a fresh interpreter, as the API boots: uvicorn configures its own
loggers first (its ``LOGGING_CONFIG``), then main.py's ``setup_logging`` and
``install_request_context_logging`` run. A request's context is set as
``RequestContextMiddleware`` sets it.

* Every line, the app's and uvicorn's (its access and error loggers), parses as one JSON
  object whose keys are ``JSON_LOG_FIELDS`` and nothing else: no ``extra``.
* The fields: the time, level, logger, message, service, version (``config.AUTOMATOS_VERSION``),
  and the request's id, workspace, method and path when it has them.
* An exception is one record, its traceback in ``exception``.
* What the filters keep out of a text line stays out of a JSON one: a request library's
  INFO line, a request path in its warning, a file hash in a path, a sign-in query.
* With ``LOG_FORMAT`` unset, a line is as before, and uvicorn's access line ends with the
  request's id. The log-relay handler is the same in both formats.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from core.monitoring.json_log_format import JSON_LOG_FIELDS

ORCH = Path(__file__).resolve().parents[1]
MARKER = "MARKER282SECRET"
HASH = "ab" * 32
VERSION = "v9.8.7-prd282"
ALWAYS = {"time", "level", "logger", "message", "service", "version"}
TIME = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$")

BOOT = """
import contextvars, logging, logging.config
import uvicorn.config
logging.config.dictConfig(uvicorn.config.LOGGING_CONFIG)  # uvicorn, before it imports the app
from core.monitoring.automatos_logging import (http_method_var, http_path_var, request_id_var,
                                               setup_logging, workspace_id_var)
from core.utils.logging_adapter import install_request_context_logging
setup_logging(service='automatos-backend', enable_relay=False)
install_request_context_logging()
"""

REQUEST = f"""
MARKER, HASH = {MARKER!r}, {HASH!r}
ACCESS = '%s - "%s %s HTTP/%s" %d'

def request():
    request_id_var.set('req-282'); workspace_id_var.set('ws-282'); http_method_var.set('GET')
    http_path_var.set('/api/public/provenance/' + HASH)
    app = logging.getLogger('prd282.app')
    app.info('hello %s', 'world')
    try:
        raise ValueError('boom')
    except ValueError:
        app.exception('it failed')
    logging.getLogger('httpx').info('HTTP Request: GET https://api.example/bot%s/x', MARKER)
    logging.getLogger('urllib3.connectionpool').warning("Retrying after connection broken by '%r': %s",
                                                        'Refused', '/bot' + MARKER + '/getUpdates')
    access = logging.getLogger('uvicorn.access')
    access.info(ACCESS, '203.0.113.5:4242', 'GET', '/api/public/provenance/' + HASH, '1.1', 200)
    access.info(ACCESS, '203.0.113.5:4242', 'GET', '/api/v1/auth/oidc/callback?code=' + MARKER + '&state=s', '1.1', 302)
    logging.getLogger('uvicorn.error').info('Application startup complete.')

contextvars.copy_context().run(request)
logging.getLogger('prd282.boot').warning('no request here', extra={{'request_id': 'FORGED', 'leak': 'LEAKED'}})
logging.getLogger('uvicorn.access').info(ACCESS, '203.0.113.5:4242', 'GET', '/health', '1.1', 200)
"""


def _run(script: str, **settings) -> subprocess.CompletedProcess:
    env = {**{k: v for k, v in os.environ.items() if k not in ("LOG_FORMAT", "LOG_LEVEL", "OTEL_ENABLED")},
           "AUTOMATOS_VERSION": VERSION, **settings}
    done = subprocess.run([sys.executable, "-W", "ignore", "-c", script], cwd=ORCH, env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-3000:]
    return done


def _json_lines(done: subprocess.CompletedProcess) -> list:
    lines = [line for line in (done.stdout + done.stderr).splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def _one(records: list, message: str) -> dict:
    found = [record for record in records if record["message"] == message]
    assert len(found) == 1, (message, records)
    return found[0]


def test_every_line_is_one_json_object_with_the_fixed_fields_only():
    done = _run(BOOT + REQUEST, LOG_FORMAT="json")
    records = _json_lines(done)
    assert len(records) == 8, records   # the exception is one line; the httpx INFO line is dropped
    for record in records:
        assert isinstance(record, dict)
        assert ALWAYS <= set(record) <= set(JSON_LOG_FIELDS), record
        assert TIME.match(record["time"]), record["time"]
        assert (record["service"], record["version"]) == ("automatos-backend", VERSION)
    assert "LEAKED" not in done.stderr + done.stdout and "FORGED" not in done.stderr + done.stdout


def test_an_app_line_carries_the_request():
    hello = _one(_json_lines(_run(BOOT + REQUEST, LOG_FORMAT="json")), "hello world")
    assert hello == {
        "time": hello["time"], "level": "INFO", "logger": "prd282.app", "message": "hello world",
        "service": "automatos-backend", "version": VERSION, "request_id": "req-282", "workspace_id": "ws-282",
        "method": "GET", "path": "/api/public/provenance/{sha256}",
    }


def test_an_exception_is_one_record_with_its_traceback_in_a_field():
    failed = _one(_json_lines(_run(BOOT + REQUEST, LOG_FORMAT="json")), "it failed")
    assert failed["level"] == "ERROR" and failed["request_id"] == "req-282"
    assert failed["exception"].startswith("Traceback (most recent call last):")
    assert failed["exception"].endswith("ValueError: boom")


def test_uvicorns_access_and_error_lines_are_the_same_record():
    records = _json_lines(_run(BOOT + REQUEST, LOG_FORMAT="json"))
    access = [record for record in records if record["logger"] == "uvicorn.access"]
    assert [record.get("request_id") for record in access] == ["req-282", "req-282", None]
    assert access[0]["message"] == '203.0.113.5:4242 - "GET /api/public/provenance/{sha256} HTTP/1.1" 200'
    assert access[1]["message"] == '203.0.113.5:4242 - "GET /api/v1/auth/oidc/callback?{redacted} HTTP/1.1" 302'
    started = _one(records, "Application startup complete.")
    assert started["logger"] == "uvicorn.error" and started["level"] == "INFO"


def test_what_the_filters_keep_out_of_a_text_line_stays_out_of_a_json_line():
    done = _run(BOOT + REQUEST, LOG_FORMAT="json")
    output = done.stdout + done.stderr
    assert MARKER not in output and HASH not in output
    retry = _one(_json_lines(done), "Retrying after connection broken by ''Refused'': /REDACTED")
    assert retry["logger"] == "urllib3.connectionpool" and retry["level"] == "WARNING"


def test_a_line_outside_a_request_has_no_request_fields():
    boot = _one(_json_lines(_run(BOOT + REQUEST, LOG_FORMAT="json")), "no request here")
    assert set(boot) == ALWAYS


def test_text_lines_are_as_before_and_the_access_line_ends_with_the_request_id():
    done = _run(BOOT + REQUEST)
    assert not any(line.startswith("{") for line in (done.stdout + done.stderr).splitlines())
    assert " - prd282.app - INFO - [req=req-282 ws=ws-282 agent=] - hello world" in done.stderr
    access = done.stdout.splitlines()
    assert access == [
        'INFO:     203.0.113.5:4242 - "GET /api/public/provenance/{sha256} HTTP/1.1" 200 OK [req=req-282]',
        'INFO:     203.0.113.5:4242 - "GET /api/v1/auth/oidc/callback?{redacted} HTTP/1.1" 302 Found [req=req-282]',
        'INFO:     203.0.113.5:4242 - "GET /health HTTP/1.1" 200 OK',
    ]
    assert MARKER not in done.stdout + done.stderr and HASH not in done.stdout + done.stderr


def test_at_warning_uvicorns_access_and_start_up_lines_are_dropped_too():
    records = _json_lines(_run(BOOT + REQUEST, LOG_FORMAT="json", LOG_LEVEL="warning"))
    assert {record["logger"] for record in records} == {"prd282.app", "urllib3.connectionpool", "prd282.boot"}
    assert {record["level"] for record in records} == {"WARNING", "ERROR"}
    assert not [record for record in records if record["logger"].startswith("uvicorn")]


UNFIT = """
logging.getLogger('prd282.app').info('%d items in %s', 'many')
logging.getLogger('prd282.app').info('after')
"""


def test_a_message_that_cannot_be_built_is_still_one_record_that_says_so():
    done = _run(BOOT + UNFIT, LOG_FORMAT="json")
    unfit, after = _json_lines(done)
    assert unfit["message"] == "log line could not be formatted: TypeError"
    assert (unfit["logger"], unfit["level"]) == ("prd282.app", "INFO") and after["message"] == "after"
    assert "--- Logging error ---" not in done.stderr


RELAY = """
import logging
from core.monitoring.automatos_logging import LogRelayHandler, setup_logging
from core.monitoring.json_log_format import JsonLineFormatter
from core.utils.logging_adapter import install_request_context_logging
root = setup_logging(service='automatos-backend', enable_relay=True, relay_url='http://127.0.0.1:9/push')
install_request_context_logging()
relay = next(h for h in root.handlers if isinstance(h, LogRelayHandler))
console = next(h for h in root.handlers if not isinstance(h, LogRelayHandler))
assert isinstance(console.formatter, JsonLineFormatter), console.formatter
assert type(relay.formatter) is logging.Formatter and '[req=%(request_id)s' in relay.formatter._fmt
record = logging.LogRecord('prd282.app', logging.INFO, 'prd282_app.py', 1, 'shipped', (), None)
for f in relay.filters:
    f.filter(record)
entry = relay._format_entry(record)
assert entry['service'] == 'automatos-backend' and entry['message'].endswith('] - shipped'), entry
"""


def test_the_relay_handler_is_unchanged_in_json():
    _run(RELAY, LOG_FORMAT="json")

"""PRD-282 US-301: with ``LOG_FORMAT=json``, every line the web server's processes write is JSON.

With more than one server process, uvicorn's supervising process writes lines of its own and
never imports the app. With ``json`` the start script points uvicorn at
``core/monitoring/server_log_config.json`` (``UVICORN_LOG_CONFIG``), whose formatter
(``server_log_config``) imports neither ``config`` nor ``main``. Each server process then
puts the app's own formatter on uvicorn's handlers (``format_server_lines``).

* The start script sets ``UVICORN_LOG_CONFIG`` for a server command with ``json`` only. With
  text, unset or an operator's own configuration, nothing changes.
* The real server, two processes, ``LOG_FORMAT=json``: every line on its stdout and stderr,
  the parent's and the children's, from start to stop, parses as one JSON object.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ORCH = Path(__file__).resolve().parents[1]
CONFIG_FILE = ORCH / "core" / "monitoring" / "server_log_config.json"
ENTRYPOINT = ORCH / "docker-entrypoint.sh"
START_DEADLINE = 90.0
STOP_DEADLINE = 60.0

APP = """
from core.monitoring.automatos_logging import setup_logging
from core.utils.logging_adapter import install_request_context_logging

setup_logging(service="automatos-backend", enable_relay=False)
install_request_context_logging()


async def app(scope, receive, send):
    if scope["type"] == "lifespan":
        while True:
            message = await receive()
            await send({"type": message["type"] + ".complete"})
            if message["type"] == "lifespan.shutdown":
                return
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
    await send({"type": "http.response.body", "body": b"ok"})
"""


# ── the module and the start script ──────────────────────────────────────────

def test_the_parents_formatter_imports_neither_config_nor_main():
    script = ("import sys, logging.config, json\n"
              f"logging.config.dictConfig(json.load(open({str(CONFIG_FILE)!r})))\n"
              "assert 'config' not in sys.modules and 'main' not in sys.modules, sorted(sys.modules)\n")
    done = subprocess.run([sys.executable, "-c", script], cwd=ORCH, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]


def test_its_service_and_version_are_the_apis():
    from config import UNVERSIONED_BUILD
    from core.monitoring import server_log_config

    tree = ast.parse((ORCH / "main.py").read_text(encoding="utf-8"))
    services = {kw.value.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "setup_logging" for kw in node.keywords if kw.arg == "service"}
    assert services == {server_log_config.SERVICE}
    assert server_log_config.UNVERSIONED_BUILD == UNVERSIONED_BUILD


def _entrypoint_starts_uvicorn_with(tmp_path: Path, **settings: str) -> str:
    """The UVICORN_LOG_CONFIG a server command gets from the start script, with stub binaries."""
    bin_dir, calls = tmp_path / "bin", tmp_path / "calls.log"
    bin_dir.mkdir()
    stub = bin_dir / "uvicorn"
    stub.write_text('#!/bin/bash\necho "${UVICORN_LOG_CONFIG-<unset>}" >> "$CALL_LOG"\n', encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    env = {"PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin", "CALL_LOG": str(calls),
           "AUTOMATOS_SKIP_MIGRATIONS": "true", "WEB_CONCURRENCY": "2", **settings}
    done = subprocess.run(["bash", str(ENTRYPOINT), "uvicorn", "main:app"], cwd=ORCH, env=env,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    (started_with,) = calls.read_text(encoding="utf-8").splitlines()
    return started_with


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize(("settings", "expected"), [
    ({"LOG_FORMAT": "json"}, str(CONFIG_FILE)),
    ({"LOG_FORMAT": " JSON "}, str(CONFIG_FILE)),
    ({}, "<unset>"),
    ({"LOG_FORMAT": "text"}, "<unset>"),
    ({"LOG_FORMAT": "logfmt"}, "<unset>"),
    ({"LOG_FORMAT": "json", "UVICORN_LOG_CONFIG": "/etc/own.json"}, "/etc/own.json"),
])
def test_the_start_script_selects_it_for_json_only(tmp_path, settings, expected):
    assert _entrypoint_starts_uvicorn_with(tmp_path, **settings) == expected


# ── the real server ──────────────────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _answered(port: int) -> bool:
    deadline = time.monotonic() + START_DEADLINE
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/ping", timeout=2).status_code == 200:
                return True
        except httpx.TransportError:
            time.sleep(0.2)
    return False


def _stopped(server: subprocess.Popen) -> tuple[str, str]:
    server.send_signal(signal.SIGTERM)
    try:
        return server.communicate(timeout=STOP_DEADLINE)
    except subprocess.TimeoutExpired:
        server.kill()
        _, err = server.communicate()
        raise AssertionError(f"the server did not stop within {STOP_DEADLINE}s: {err[-2000:]}") from None


def test_every_line_two_server_processes_write_is_one_json_object(tmp_path):
    (tmp_path / "prd282_app.py").write_text(APP, encoding="utf-8")
    port = _free_port()
    env = {**{k: v for k, v in os.environ.items() if k not in ("LOG_LEVEL", "OTEL_ENABLED", "WEB_CONCURRENCY")},
           "LOG_FORMAT": "json", "UVICORN_LOG_CONFIG": str(CONFIG_FILE), "PYTHONWARNINGS": "ignore",
           "PYTHONUNBUFFERED": "1"}
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "prd282_app:app", "--app-dir", str(tmp_path), "--workers", "2",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=ORCH, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert _answered(port), "the server never answered"
    finally:
        out, err = _stopped(server)
    lines = [line for line in (out + err).splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    assert all(isinstance(record, dict) and record["service"] == "automatos-backend" for record in records)
    messages = [record["message"] for record in records]
    assert any(m.startswith("Started parent process") for m in messages), messages
    assert any(m.startswith("Stopping parent process") for m in messages), messages
    assert sum(m.startswith("Started server process") for m in messages) == 2, messages
    assert any(r["logger"] == "uvicorn.access" and "/ping" in r["message"] for r in records), messages

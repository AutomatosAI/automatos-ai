"""PRD-262, row 36 of the 19 Oct size breakdown: the worker serves through Starlette on uvicorn.

aiohttp and its six helper packages (12.2 MB) left the worker image; Starlette and
uvicorn were already there for mcp. The routes, their methods, the internal-token
check and the trace (PRD-256 O2b) keep their own suites (test_prd256_o2b_worker_*.py,
F173, F178, F333, the binary pieces). This one pins what aiohttp did without being
asked, so the move changes no answer:

- a JSON body is read up to aiohttp's 1 MiB ``client_max_size``; one byte more is the
  handler's 400 "Invalid JSON body", as before;
- a JSON answer is ``json.dumps`` sent as ``application/json; charset=utf-8``;
- GET routes answer HEAD, and a trailing slash is a 404, never a redirect;
- a download is an attachment named after the file, whatever its name;
- Prometheus's middleware sits outside the token check, so /metrics needs the token
  and counts the 401s;
- the server holds idle connections 75 s, aiohttp's keep-alive, never takes the
  worker's signals, and stops when the worker stops.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import signal
import socket
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from prometheus_client import CollectorRegistry
from prometheus_client.parser import text_string_to_metric_families

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from helpers_workspace_worker import load_worker_main  # noqa: E402

WS = "00000000-0000-0000-0000-0000000000c1"
SHARED_METRICS = _TESTS.parents[1] / "services" / "shared" / "automatos_metrics.py"
TOKEN = "s3cret"


def _worker(port: int = 0) -> SimpleNamespace:
    return SimpleNamespace(_worker_id="w-test", _active_tasks={}, concurrency=1, _redis=None,
                           health_port=port, _running=True)


@pytest.fixture
def worker_http(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKER_INTERNAL_TOKEN", "")
    load_worker_main(monkeypatch, tmp_path)
    import worker_http

    return worker_http


def _client(worker_http):
    from starlette.testclient import TestClient

    return TestClient(worker_http.build_app(_worker()), follow_redirects=False)


# --------------------------------------------------------------------------- bodies


def _exec_body(size: int) -> bytes:
    """A JSON body of exactly ``size`` bytes for /exec, with an empty command."""
    head, tail = b'{"command": "", "pad": "', b'"}'
    return head + b"x" * (size - len(head) - len(tail)) + tail


def test_a_body_at_aiohttps_limit_is_read_and_one_byte_more_is_the_handlers_400(worker_http):
    limit = worker_http.MAX_BODY_BYTES
    assert limit == 1024 ** 2, "aiohttp's client_max_size"
    headers = {"Content-Type": "application/json"}
    with _client(worker_http) as client:
        at_limit = client.post(f"/workspaces/{WS}/exec", content=_exec_body(limit), headers=headers)
        over = client.post(f"/workspaces/{WS}/exec", content=_exec_body(limit + 1), headers=headers)
    assert (at_limit.status_code, at_limit.json()) == (400, {"error": "command is required"})  # parsed
    assert (over.status_code, over.json()) == (400, {"error": "Invalid JSON body"})


def test_a_body_that_is_not_json_is_the_handlers_400(worker_http):
    with _client(worker_http) as client:
        answer = client.post(f"/workspaces/{WS}/canvas/session/message", content=b"\xff not json")
    assert (answer.status_code, answer.json()) == (400, {"error": "Invalid JSON body"})


# --------------------------------------------------------------------------- answers


def test_a_json_answer_is_aiohttps_json_response_byte_for_byte(worker_http, tmp_path):
    with _client(worker_http) as client:
        health = client.get("/health")
    assert health.headers["content-type"] == "application/json; charset=utf-8"
    expected = {"status": "healthy", "worker_id": "w-test", "active_tasks": 0, "concurrency": 1,
                "volume_path": str(tmp_path)}
    assert health.content == json.dumps(expected).encode()


def test_get_routes_answer_head_and_a_trailing_slash_is_not_redirected(worker_http):
    with _client(worker_http) as client:
        head = client.head("/health")
        slashed = client.get("/health/")
        listed = client.head(f"/workspaces/{WS}/files")
    assert head.status_code == 200 and head.content == b""
    assert slashed.status_code == 404 and "location" not in slashed.headers
    assert listed.status_code == 200


@pytest.mark.parametrize(("name", "disposition"), [
    ("report.txt", 'attachment; filename="report.txt"'),
    ("été 1.txt", "attachment; filename*=utf-8''%C3%A9t%C3%A9%201.txt"),
])
def test_a_download_is_an_attachment_named_after_the_file(worker_http, tmp_path, name, disposition):
    (tmp_path / WS).mkdir(parents=True, exist_ok=True)
    (tmp_path / WS / name).write_text("hello", encoding="utf-8")
    with _client(worker_http) as client:
        answer = client.get(f"/workspaces/{WS}/files/download", params={"path": name})
    assert answer.status_code == 200 and answer.content == b"hello"
    assert answer.headers["content-disposition"] == disposition
    assert answer.headers["content-type"] == "text/plain"


# --------------------------------------------------------------------------- metrics


def _private_metrics():
    """services/shared/automatos_metrics.py with a registry of its own."""
    spec = importlib.util.spec_from_file_location(f"prd262_worker_metrics_{uuid.uuid4().hex}", SHARED_METRICS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REGISTRY = CollectorRegistry()
    return module


def test_metrics_sit_outside_the_token_check(monkeypatch, tmp_path):
    from starlette.testclient import TestClient

    load_worker_main(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "automatos_metrics", _private_metrics())
    monkeypatch.setenv("WORKER_INTERNAL_TOKEN", TOKEN)
    import worker_http

    with TestClient(worker_http.build_app(_worker())) as client:
        assert client.get(f"/workspaces/{WS}/files").status_code == 401
        assert client.get("/metrics").status_code == 401, "the scrape needs the token, as under aiohttp"
        scrape = client.get("/metrics", headers={"X-Internal-Token": TOKEN})
    assert scrape.status_code == 200
    samples = {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in text_string_to_metric_families(scrape.text)
        for sample in family.samples
    }
    refused = ("workspace_worker_requests_total",
               (("endpoint", "/workspaces/{workspace_id}/files"), ("method", "GET"), ("status", "401")))
    assert samples.get(refused) == 1.0, "metrics run before the token check, so the 401 is counted"


# --------------------------------------------------------------------------- the server


def test_the_server_keeps_aiohttps_keep_alive_and_leaves_the_signals_to_the_worker(worker_http):
    server = worker_http.make_server(_worker(8081))
    assert isinstance(server, worker_http.EmbeddedServer)
    assert server.config.timeout_keep_alive == 75, "longer than httpx's 5 s pool expiry, as aiohttp's 75 s was"
    assert server.config.timeout_graceful_shutdown == 60 and server.config.log_config is None
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    with server.capture_signals():
        assert {sig: signal.getsignal(sig) for sig in before} == before


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_serve_answers_on_its_port_and_stops_when_the_worker_stops(worker_http, monkeypatch):
    monkeypatch.setenv("WORKER_BIND_HOST", "127.0.0.1")
    worker = _worker(_free_port())

    async def run() -> int:
        serving = asyncio.create_task(worker_http.serve(worker))
        status = 0
        async with httpx.AsyncClient() as client:
            for _ in range(100):
                try:
                    status = (await client.get(f"http://127.0.0.1:{worker.health_port}/health")).status_code
                    break
                except httpx.TransportError:
                    await asyncio.sleep(0.05)
        worker._running = False
        await asyncio.wait_for(serving, timeout=15)
        return status

    assert asyncio.run(run()) == 200

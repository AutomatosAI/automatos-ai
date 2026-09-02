"""The host loop against a fake backend: preflight, pairing, claim, events, result."""
from __future__ import annotations

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from automatos_cli_host.config import HostConfig
from automatos_cli_host.host import Host, HostRefused

from conftest import FAKE_CLAUDE


class FakeBackend:
    """The S1a contract, in-process. Records everything the host sends."""

    def __init__(self, workdir: Path, edition="local", enabled=True):
        self.workdir = workdir
        self.edition = edition
        self.enabled = enabled
        self.calls = []
        self.claimed = False
        self.result = None
        self.events = []
        self.token = "tok-" + uuid.uuid4().hex
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _json(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _body(self):
                n = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(n) or b"{}") if n else {}

            def do_GET(self):
                backend.calls.append(("GET", self.path))
                if self.path == "/health":
                    return self._json(200, {"status": "healthy", "edition": backend.edition,
                                            "cli_runtime_enabled": backend.enabled})
                return self._json(404, {"detail": "nope"})

            def do_POST(self):
                body = self._body()
                backend.calls.append(("POST", self.path, body, self.headers.get("X-CLI-Host-Token")))
                if self.path == "/api/v1/cli-hosts/pair":
                    if body.get("code") != "ABCD-2345":
                        return self._json(401, {"detail": "invalid or expired pairing code"})
                    return self._json(200, {"host_id": "h1", "workspace_id": "w1", "token": backend.token})
                if self.headers.get("X-CLI-Host-Token") != backend.token:
                    return self._json(401, {"detail": "invalid or missing CLI host token"})
                if self.path == "/api/v1/cli-hosts/h1/heartbeat":
                    return self._json(200, {"reattached": [], "stale": [], "server_time": "t"})
                if self.path == "/api/v1/cli-hosts/h1/claim":
                    if backend.claimed:
                        return self._json(200, {"tasks": []})
                    backend.claimed = True
                    return self._json(200, {"tasks": [{
                        "task_id": 42, "attempt": 1, "session_id": str(uuid.uuid4()), "agent_name": "Dwight",
                        "title": "Say hi", "prompt": "OBJECTIVE: hello", "cwd": str(backend.workdir),
                        "model": None, "allowed_tools": [], "provider": "claude", "workspace_id": "w1",
                    }]})
                if self.path == "/api/v1/cli-hosts/h1/tasks/42/events":
                    backend.events.extend(body.get("events") or [])
                    return self._json(200, {"status": "in_progress", "lease_renewed": True, "control": []})
                if self.path == "/api/v1/cli-hosts/h1/tasks/42/result":
                    backend.result = body
                    return self._json(200, {"applied": True, "status": "done"})
                return self._json(404, {"detail": f"unknown {self.path}"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _cfg(short_tmp, url, **over):
    base = dict(url=url, state_dir=short_tmp / "state", allow_dirs=[short_tmp / "ws"], pair_code="ABCD-2345",
                name="test-host", once=True, claude_binary=str(FAKE_CLAUDE), use_worktrees=False,
                poll_seconds=1.0, heartbeat_seconds=1.0, event_flush_seconds=0.5, session_timeout_seconds=120)
    base.update(over)
    return HostConfig(**base)


def test_once_cycle_pairs_claims_runs_and_reports(short_tmp, fake_home, env_clean):
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    backend = FakeBackend(workdir)
    try:
        host = Host(_cfg(short_tmp, backend.url))
        host.prepare()
        assert host.run_forever() == 0
    finally:
        backend.close()

    ident = json.loads((short_tmp / "state" / "host.json").read_text())
    assert ident["host_id"] == "h1" and ident["token"] == backend.token
    assert oct((short_tmp / "state" / "host.json").stat().st_mode & 0o777) == "0o600"
    assert backend.result is not None, "no result was posted"
    assert backend.result["status"] == "success" and backend.result["attempt"] == 1
    assert "Done. Wrote hello.txt" in backend.result["result_text"]
    assert backend.result["files_touched"] == [str((workdir / "hello.txt").resolve())]
    assert backend.result["usage"]["total_tokens"] == 150
    names = [e["event"] for e in backend.events]
    assert "SessionStart" in names and "Stop" in names
    heartbeat_calls = [c for c in backend.calls if c[1].endswith("/heartbeat")]
    assert heartbeat_calls and heartbeat_calls[0][3] == backend.token
    assert heartbeat_calls[0][2]["capabilities"]["claude"]["version"].startswith("9.9.9")
    # Pairing sent no secret: only the code and capabilities.
    pair_call = next(c for c in backend.calls if c[1].endswith("/pair"))
    assert set(pair_call[2]) == {"code", "name", "capabilities"}
    assert (short_tmp / "state" / "sessions.json").read_text().strip() == "{}"  # table cleared


def test_host_refuses_a_saas_backend_and_a_disabled_one(short_tmp, fake_home, env_clean):
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    saas = FakeBackend(workdir, edition="saas")
    try:
        with pytest.raises(HostRefused):
            Host(_cfg(short_tmp, saas.url)).prepare()
    finally:
        saas.close()
    off = FakeBackend(workdir, enabled=False)
    try:
        with pytest.raises(HostRefused):
            Host(_cfg(short_tmp, off.url)).prepare()
    finally:
        off.close()


def test_host_refuses_to_run_unpaired_or_without_directories(short_tmp, fake_home, env_clean):
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    backend = FakeBackend(workdir)
    try:
        with pytest.raises(HostRefused):
            Host(_cfg(short_tmp, backend.url, pair_code=None)).prepare()
        with pytest.raises(HostRefused):  # a fresh state dir: nothing registered yet
            Host(_cfg(short_tmp, backend.url, allow_dirs=[], state_dir=short_tmp / "state-empty")).prepare()
    finally:
        backend.close()

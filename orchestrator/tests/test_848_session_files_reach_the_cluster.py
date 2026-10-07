"""#848: a session's files reach the workspace when the host shares no folder with it.

On a cluster the operator's CLI host and the backend share no folder, so the host
uploads what its session left in the ticket's deliverables folder and the result
names the files; they become the ticket's deliverables like a shared folder's.
Pure suite: the router on a bare app, the service with its edges spied. The
claim → upload → result round trip against Postgres is in
``test_848_session_files_reach_the_cluster_realdb.py``.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import api.cli_hosts as routes  # noqa: E402
from core.database.database import get_db  # noqa: E402
from services import cli_host_service as svc  # noqa: E402
from services import session_uploads as uploads  # noqa: E402

WS = uuid4()
HOST_ID = uuid4()
TOKEN = "good-token"
MB = 1024 * 1024


def _host():
    return SimpleNamespace(id=HOST_ID, workspace_id=WS, name="laptop", status="paired")


def _task(status="in_progress", task_id=42):
    return SimpleNamespace(id=task_id, workspace_id=WS, status=status, runtime_ref={"host_id": str(HOST_ID)})


def _client(monkeypatch, *, enabled=True):
    monkeypatch.setattr(routes.config, "CLI_RUNTIME_ENABLED", enabled, raising=False)
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_db] = lambda: SimpleNamespace(name="dummy-session")
    monkeypatch.setattr(routes.svc, "resolve_host_by_token",
                        lambda db, token: _host() if token == TOKEN else None, raising=True)
    return TestClient(app)


def _upload_on(monkeypatch, *, enabled=True, total_mb=200):
    monkeypatch.setattr(uploads.config, "CLI_SESSION_FILE_UPLOAD", enabled, raising=False)
    monkeypatch.setattr(uploads.config, "CLI_SESSION_UPLOAD_MAX_TOTAL_MB", total_mb, raising=False)


async def _chunks(*parts: bytes):
    for part in parts:
        yield part


# ── the path the host names ──────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("report.md", "report.md"),
    ("charts/q3.png", "charts/q3.png"),
    ("a b/c-d_e.v2.pdf", "a b/c-d_e.v2.pdf"),
])
def test_a_path_inside_the_deliverables_folder_is_kept(raw, expected):
    assert uploads.safe_upload_path(raw) == expected


@pytest.mark.parametrize("raw", [
    "", None, "../secret", "a/../../b", "/etc/passwd", "C:/Windows/x", "C:x", "a\\b", "a//b", "./a",
    "a/./b", "a/", "line\nbreak", "x" * (uploads.MAX_UPLOAD_PATH_CHARS + 1),
])
def test_a_path_that_could_leave_the_folder_is_refused(raw):
    assert uploads.safe_upload_path(raw) is None


def test_an_upload_lands_in_the_tickets_sessions_folder():
    assert uploads.ticket_upload_target(42, "charts/q3.png") == "sessions/42/charts/q3.png"


# ── the claim ────────────────────────────────────────────────────────────────

def test_the_claim_asks_for_uploads_only_when_the_instance_takes_them(monkeypatch):
    _upload_on(monkeypatch, enabled=False)
    assert uploads.upload_claim_fields()["enabled"] is False
    _upload_on(monkeypatch, enabled=True, total_mb=200)
    assert uploads.upload_claim_fields() == {"enabled": True, "max_file_bytes": 50 * MB, "max_total_bytes": 200 * MB}


def test_the_claim_payload_carries_the_upload_fields(monkeypatch):
    _upload_on(monkeypatch, enabled=True)
    task = SimpleNamespace(id=7, workspace_id=WS, assigned_agent_id=None, title="t", review_mode=None,
                           attachment_ids=None)
    ref = {"provider": "claude", "model": None, "cwd": None, "session_id": "s", "attempt": 1}
    monkeypatch.setattr(svc, "_session_system_prompt", lambda agent: "")
    payload = svc._claim_payload(task, None, {}, ref, "prompt", "tok", ("edits", False))
    assert payload["upload"] == {"enabled": True, "max_file_bytes": 50 * MB, "max_total_bytes": 200 * MB}


# ── storing one file ─────────────────────────────────────────────────────────

def test_the_body_is_refused_the_moment_it_passes_the_limit():
    with pytest.raises(uploads.SessionUploadRefused) as err:
        asyncio.run(uploads.read_capped(_chunks(b"x" * 6, b"x" * 6), 10))
    assert err.value.status_code == 413
    assert asyncio.run(uploads.read_capped(_chunks(b"ab", b"cd"), 10)) == b"abcd"


def _store(monkeypatch, task, *, reserve=True, written=None, path="out/report.md", body=b"# hi\n",
           reserved=None):
    """Store one file with the edges spied: ``reserved`` collects every change to the run's total."""
    _upload_on(monkeypatch)
    monkeypatch.setattr(svc, "_owned_task", lambda db, host, task_id: task)
    ledger = reserved if reserved is not None else []
    monkeypatch.setattr(uploads, "_reserve", lambda db, task_id, size, limit: ledger.append(size) or reserve)
    calls = []

    class FakeWorkspace:
        def __init__(self, workspace_id):
            self.workspace_id = workspace_id

        async def write_binary(self, target, pieces):
            data = b"".join([piece async for piece in pieces])
            calls.append((self.workspace_id, target, data))
            if isinstance(written, Exception):
                raise written
            return written or {"success": True}

    import core.workspace_client as wc
    monkeypatch.setattr(wc, "WorkspaceClient", FakeWorkspace)
    out = asyncio.run(uploads.store_session_file(None, _host(), task.id, path, _chunks(body)))
    return out, calls


def test_a_file_is_written_into_the_tickets_folder_through_the_worker(monkeypatch):
    out, calls = _store(monkeypatch, _task())
    assert out == {"path": "out/report.md", "workspace_path": "sessions/42/out/report.md", "size": 5}
    assert calls == [(str(WS), "sessions/42/out/report.md", b"# hi\n")]


@pytest.mark.parametrize("kwargs,status", [
    ({"path": "../escape.md"}, 400),
    ({"reserve": False}, 413),
    ({"written": {"success": False, "error": "disk full"}}, 502),
])
def test_a_file_the_backend_cannot_take_is_refused_with_the_reason(monkeypatch, kwargs, status):
    with pytest.raises(uploads.SessionUploadRefused) as err:
        _store(monkeypatch, _task(), **kwargs)
    assert err.value.status_code == status


def test_a_write_the_workspace_refuses_gives_its_bytes_back(monkeypatch):
    """The host retries a 5xx: each try must not count the same file against the run again."""
    reserved = []
    with pytest.raises(uploads.SessionUploadRefused):
        _store(monkeypatch, _task(), written={"success": False, "error": "disk full"}, reserved=reserved)
    assert reserved == [5, -5]


def test_a_write_that_raises_gives_its_bytes_back_and_still_raises(monkeypatch):
    reserved = []
    with pytest.raises(ConnectionError):
        _store(monkeypatch, _task(), written=ConnectionError("worker gone"), reserved=reserved)
    assert reserved == [5, -5]


def test_a_stored_file_keeps_its_bytes(monkeypatch):
    reserved = []
    _store(monkeypatch, _task(), reserved=reserved)
    assert reserved == [5]


def test_a_ticket_that_is_no_longer_running_takes_no_files(monkeypatch):
    with pytest.raises(uploads.SessionUploadRefused) as err:
        _store(monkeypatch, _task(status="done"))
    assert err.value.status_code == 409


def test_an_instance_that_shares_a_folder_takes_no_uploads(monkeypatch):
    _upload_on(monkeypatch, enabled=False)
    with pytest.raises(uploads.SessionUploadRefused) as err:
        asyncio.run(uploads.store_session_file(None, _host(), 42, "a.md", _chunks(b"x")))
    assert err.value.status_code == 404


# ── the result ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("host_path,expected", [
    ("/Users/me/deliverables/sessions/42/report.md", "sessions/42/report.md"),       # the deliverables folder
    ("/Users/me/.automatos/cli-host/sessions/42/charts/q3.png", "sessions/42/charts/q3.png"),  # the host's own
    ("C:\\Users\\me\\deliverables\\sessions\\42\\report.md", "sessions/42/report.md"),  # a Windows host
    ("/Users/me/deliverables/sessions/420/report.md", None),                         # another ticket
    ("/Users/me/deliverables/sessions/42/../43/x.md", None),
    ("/Users/me/repo/app.py", None),
])
def test_a_result_path_under_the_tickets_sessions_folder_maps_to_where_the_upload_landed(
        monkeypatch, host_path, expected):
    _upload_on(monkeypatch)
    assert uploads.uploaded_rel(_task(), host_path) == expected


def test_nothing_maps_while_the_folder_is_shared(monkeypatch):
    _upload_on(monkeypatch, enabled=False)
    assert uploads.uploaded_rel(_task(), "/Users/me/deliverables/sessions/42/report.md") is None


def test_an_uploaded_file_on_the_volume_is_a_session_output(monkeypatch, tmp_path):
    """The result names the file by the host's path; it registers once it is on the volume."""
    _upload_on(monkeypatch)
    monkeypatch.setattr(svc.config, "WORKSPACE_VOLUME_PATH", str(tmp_path), raising=False)
    monkeypatch.setattr(svc.config, "AUTOMATOS_WORKSPACE_DIR", "", raising=False)
    monkeypatch.setattr(svc.config, "LOCAL_PROJECTS_DIR", "", raising=False)
    landed = tmp_path / str(WS) / "sessions" / "42" / "report.md"
    landed.parent.mkdir(parents=True)
    landed.write_text("# Q3\n")
    host = "/Users/me/deliverables/sessions/42"
    outputs = svc._session_outputs(_task(), [f"{host}/report.md", f"{host}/never-landed.md", "/Users/me/x.md"])
    assert [(o.rel, o.size) for o in outputs] == [("sessions/42/report.md", 5)]
    _upload_on(monkeypatch, enabled=False)
    assert svc._session_outputs(_task(), [f"{host}/report.md"]) == []


# ── the routes ───────────────────────────────────────────────────────────────

def _h(token=TOKEN):
    return {routes.HOST_TOKEN_HEADER: token}


def test_the_upload_route_is_404_while_session_mode_is_off(monkeypatch):
    c = _client(monkeypatch, enabled=False)
    assert c.put(f"/api/v1/cli-hosts/{HOST_ID}/tasks/42/files?path=a.md", content=b"x", headers=_h()).status_code == 404


def test_the_upload_route_needs_the_hosts_token(monkeypatch):
    c = _client(monkeypatch)
    assert c.put(f"/api/v1/cli-hosts/{HOST_ID}/tasks/42/files?path=a.md", content=b"x",
                 headers=_h("bad")).status_code == 401


def test_the_upload_route_streams_the_body_to_the_service(monkeypatch):
    c = _client(monkeypatch)
    seen = {}

    async def _store(db, host, task_id, path, chunks):
        seen.update(task_id=task_id, path=path, body=b"".join([c async for c in chunks]))
        return {"path": path, "workspace_path": f"sessions/{task_id}/{path}", "size": 3}

    monkeypatch.setattr(routes.uploads, "store_session_file", _store)
    r = c.put(f"/api/v1/cli-hosts/{HOST_ID}/tasks/42/files", params={"path": "out/a.md"}, content=b"abc",
              headers=_h())
    assert r.status_code == 200 and r.json()["workspace_path"] == "sessions/42/out/a.md"
    assert seen == {"task_id": 42, "path": "out/a.md", "body": b"abc"}


@pytest.mark.parametrize("raised,status", [
    (uploads.SessionUploadRefused(413, "too big"), 413),
    (LookupError("task 42 not found"), 404),
    (PermissionError("not claimed by this host"), 403),
])
def test_the_upload_route_answers_a_refusal_with_its_status(monkeypatch, raised, status):
    c = _client(monkeypatch)

    async def _store(*a, **k):
        raise raised

    monkeypatch.setattr(routes.uploads, "store_session_file", _store)
    r = c.put(f"/api/v1/cli-hosts/{HOST_ID}/tasks/42/files?path=a.md", content=b"x", headers=_h())
    assert r.status_code == status


def test_pairing_is_rate_limited_for_the_instance(monkeypatch):
    c = _client(monkeypatch)
    calls = []

    async def _limit(workspace_id, operation, *a, **k):
        calls.append((workspace_id, operation))
        raise HTTPException(status_code=429, detail="Rate limit exceeded for cli_host_pair.")

    monkeypatch.setattr(routes, "check_rate_limit", _limit)
    monkeypatch.setattr(routes.svc, "pair_host", lambda *a, **k: pytest.fail("a limited attempt never pairs"))
    assert c.post("/api/v1/cli-hosts/pair", json={"code": "ABCD-EFGH"}).status_code == 429
    assert calls == [(routes.PAIR_RATE_LIMIT_SCOPE, "cli_host_pair")]


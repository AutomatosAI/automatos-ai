"""#848 against real Postgres (``@integration``): claim → upload → result → deliverable.

A host that shares no folder with the backend uploads its session's file; the
worker's write is stood in for by a write onto the test's volume (the API pod
mounts the same volume read-only in the chart). The result names the file, and
it is registered as the ticket's deliverable. The run's total is one key of
``runtime_ref`` updated in one statement, refused past the cap. Skips cleanly
when no Postgres is reachable (CI runs it).
"""
from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from core.database.database import get_database_url  # noqa: E402
from core.models.core import BoardTask  # noqa: E402
from services import cli_host_service as svc  # noqa: E402
from services import session_uploads as uploads  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            for tbl in ("agents", "board_tasks", "cli_hosts", "deliverables"):
                c.execute(text(f"SELECT 1 FROM {tbl} LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"#848 suite needs a reachable Postgres with the session-mode schema: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture(autouse=True)
def _quiet_side_effects(monkeypatch):
    """Keep the test on the upload contract: no approval or report fan-out."""
    import api.board_tasks as bt

    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(bt, "_board_task_blocked_pending_approval", lambda *a, **k: False, raising=True)
    monkeypatch.setattr(bt, "_dispatch_task_complete", _noop, raising=True)
    monkeypatch.setattr(bt, "_dispatch_task_failed", _noop, raising=True)
    monkeypatch.setattr(bt, "_auto_create_task_report", _noop, raising=True)


@pytest.fixture
def cluster(engine, new_session, tmp_path, monkeypatch):
    """A workspace with one cli agent, uploads on, the volume at ``tmp_path`` and
    the worker's write landing there. Yields ``(ws_id, agent_id)``; sweeps after."""
    from config import config as cfg
    import core.workspace_client as wc

    monkeypatch.setattr(cfg, "CLI_SESSION_FILE_UPLOAD", True, raising=False)
    monkeypatch.setattr(cfg, "CLI_SESSION_UPLOAD_MAX_TOTAL_MB", 1, raising=False)
    monkeypatch.setattr(cfg, "WORKSPACE_VOLUME_PATH", str(tmp_path), raising=False)
    monkeypatch.setattr(cfg, "AUTOMATOS_WORKSPACE_DIR", "", raising=False)

    class VolumeWorkspace:
        def __init__(self, workspace_id):
            self.root = Path(tmp_path) / workspace_id

        async def write_binary(self, target, pieces):
            path = self.root / target
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"".join([piece async for piece in pieces]))
            return {"success": True}

    monkeypatch.setattr(wc, "WorkspaceClient", VolumeWorkspace)
    ws_id = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"),
              {"id": ws_id, "n": "848-cluster"})
    agent_id = s.execute(
        text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
             "VALUES ('CLI-848', 'custom', CAST(:w AS uuid), 'active', CAST(:c AS json)) RETURNING id"),
        {"w": ws_id, "c": '{"runtime": "cli", "provider": "claude", "model": "sonnet"}'},
    ).fetchone()[0]
    s.commit()
    s.close()
    yield ws_id, agent_id
    s = new_session.sweep()
    for table in ("deliverables", "llm_usage", "board_tasks", "cli_hosts", "agents"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws_id})  # noqa: S608
    s.commit()
    s.close()


def _claimed(s, ws_id, agent_id):
    task_id = s.execute(
        text("INSERT INTO board_tasks (workspace_id, title, status, assigned_agent_id, priority) "
             "VALUES (CAST(:w AS uuid), 'write the report', 'assigned', :a, 'medium') RETURNING id"),
        {"w": ws_id, "a": agent_id},
    ).fetchone()[0]
    s.commit()
    host, _token = svc.pair_host(s, svc.create_pairing_code(s, ws_id, "cluster-host")[1], "cluster-host", {})
    ticket = svc.claim_for_host(s, host, 1)["tasks"][0]
    assert ticket["task_id"] == task_id and ticket["upload"]["enabled"] is True
    return host, ticket


async def _body(data: bytes):
    yield data


def test_an_uploaded_file_becomes_the_tickets_deliverable(cluster, new_session):
    ws_id, agent_id = cluster
    s = new_session()
    host, ticket = _claimed(s, ws_id, agent_id)
    task_id = ticket["task_id"]

    stored = asyncio.run(uploads.store_session_file(s, host, task_id, "report.md", _body(b"# Q3 report\n")))
    assert stored["workspace_path"] == f"sessions/{task_id}/report.md"
    out = asyncio.run(svc.apply_result(s, host, task_id, {
        "attempt": ticket["attempt"], "status": "success", "result_text": "wrote report.md",
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "files_touched": [f"/Users/me/deliverables/sessions/{task_id}/report.md"],
        "uploaded_files": ["report.md"],
    }))
    assert out["applied"] is True
    row = s.query(BoardTask).get(task_id)
    assert [d["file_path"] for d in row.runtime_ref["deliverables"]] == [f"sessions/{task_id}/report.md"]
    found = s.execute(
        text("SELECT source_type, source_id, file_size_bytes FROM deliverables "
             "WHERE workspace_id = CAST(:w AS uuid) AND file_path = :p AND deleted_at IS NULL"),
        {"w": ws_id, "p": f"sessions/{task_id}/report.md"},
    ).fetchone()
    assert found is not None and tuple(found) == ("task", str(task_id), 12)
    s.close()


def test_a_run_cannot_upload_past_its_total(cluster, new_session):
    ws_id, agent_id = cluster
    s = new_session()
    host, ticket = _claimed(s, ws_id, agent_id)
    task_id = ticket["task_id"]
    half = b"x" * (600 * 1024)

    asyncio.run(uploads.store_session_file(s, host, task_id, "a.bin", _body(half)))
    with pytest.raises(uploads.SessionUploadRefused) as err:
        asyncio.run(uploads.store_session_file(s, host, task_id, "b.bin", _body(half)))
    assert err.value.status_code == 413
    s.expire_all()
    assert s.query(BoardTask).get(task_id).runtime_ref[uploads.UPLOADED_BYTES_KEY] == len(half)
    s.close()


def test_the_runs_total_survives_the_hosts_event_flush(cluster, new_session):
    """The total is one key in one statement: an events batch, which writes the whole
    ``runtime_ref``, keeps it as long as it read the row after the upload."""
    ws_id, agent_id = cluster
    s = new_session()
    host, ticket = _claimed(s, ws_id, agent_id)
    task_id = ticket["task_id"]
    asyncio.run(uploads.store_session_file(s, host, task_id, "a.md", _body(b"abc")))
    s.expire_all()
    asyncio.run(svc.record_events(s, host, task_id, [{"event": "PreToolUse", "tool_name": "Write"}]))
    s.expire_all()
    ref = s.query(BoardTask).get(task_id).runtime_ref
    assert ref[uploads.UPLOADED_BYTES_KEY] == 3 and ref["host_id"] == str(host.id)
    s.close()

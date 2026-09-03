"""PRD-234 S1a — the CLI host contract against real Postgres (``@integration``).

What the pure suites cannot exercise: the claim SQL really excludes/includes
``cli`` tickets by the agent's JSON configuration, pairing round-trips through
the ``cli_hosts`` table, a host's claim pre-assigns a session id in
``runtime_ref``, events renew the lease, and a result is applied exactly once.
Skips cleanly when no Postgres is reachable (CI runs it). PRD-158 lesson: seed
``workspaces`` FIRST for every FK'd table.
"""
from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from sqlalchemy import create_engine, text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from core.database.database import get_database_url  # noqa: E402
from core.models.core import BoardTask  # noqa: E402
from services import board_dispatcher as bd  # noqa: E402
from services import cli_host_service as svc  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            for tbl in ("agents", "board_tasks", "cli_hosts"):
                c.execute(text(f"SELECT 1 FROM {tbl} LIMIT 1"))
            c.execute(text("SELECT runtime_ref FROM board_tasks LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cli-host suite needs a reachable Postgres with the S1a schema: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def seeded(engine, new_session):
    """A workspace with one API agent and one cli agent. Yields ids; sweeps on teardown."""
    ws_id = str(uuid.uuid4())
    s = new_session()
    s.execute(
        text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"),
        {"id": ws_id, "n": "prd234-s1a"},
    )
    s.commit()
    api_agent = s.execute(
        text(
            "INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
            "VALUES (:n, 'custom', CAST(:w AS uuid), 'active', CAST(:c AS json)) RETURNING id"
        ),
        {"n": "API-AGENT", "w": ws_id, "c": '{"runtime": "api"}'},
    ).fetchone()[0]
    cli_agent = s.execute(
        text(
            "INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
            "VALUES (:n, 'custom', CAST(:w AS uuid), 'active', CAST(:c AS json)) RETURNING id"
        ),
        {"n": "CLI-AGENT", "w": ws_id, "c": '{"runtime": "cli", "provider": "claude", "model": "sonnet"}'},
    ).fetchone()[0]
    s.commit()
    s.close()
    yield ws_id, api_agent, cli_agent
    s = new_session.sweep()
    s.execute(text("DELETE FROM board_tasks WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws_id})
    s.execute(text("DELETE FROM cli_hosts WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws_id})
    s.execute(text("DELETE FROM agents WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws_id})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws_id})
    s.commit()
    s.close()


def _seed_task(session, ws_id, agent_id, title, status="assigned"):
    t = BoardTask(
        workspace_id=ws_id, title=title, status=status, priority="medium",
        assigned_agent_id=agent_id, source_type="user", attempts=0,
    )
    session.add(t)
    session.commit()
    return t.id


@pytest.fixture(autouse=True)
def _quiet_side_effects(monkeypatch):
    """Keep the contract test on the contract: no approval-policy lookups, no
    notification/report fan-out (each has its own suite)."""
    import api.board_tasks as bt

    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(bt, "_board_task_blocked_pending_approval", lambda *a, **k: False, raising=True)
    monkeypatch.setattr(bt, "_dispatch_task_complete", _noop, raising=True)
    monkeypatch.setattr(bt, "_dispatch_task_failed", _noop, raising=True)
    monkeypatch.setattr(bt, "_auto_create_task_report", _noop, raising=True)


def test_dispatcher_claim_skips_cli_tickets_and_host_claim_takes_only_them(seeded, new_session):
    ws_id, api_agent, cli_agent = seeded
    s = new_session()
    api_task = _seed_task(s, ws_id, api_agent, "api-work")
    cli_task = _seed_task(s, ws_id, cli_agent, "cli-work")

    dispatcher_got = bd.claim_tasks(s, worker_id="loop", limit=10, lease_seconds=60, workspace_id=ws_id)
    dispatcher_ids = {t.id for t in dispatcher_got if t.workspace_id == uuid.UUID(ws_id)}
    assert api_task in dispatcher_ids and cli_task not in dispatcher_ids

    host_got = bd.claim_tasks(
        s, worker_id="host", limit=10, lease_seconds=60,
        runtime=bd.RUNTIME_CLI, workspace_id=ws_id,
    )
    assert {t.id for t in host_got} == {cli_task}
    s.close()


def test_pairing_round_trip_and_token_resolution(seeded, new_session):
    ws_id, _, _ = seeded
    s = new_session()
    host, code, expires = svc.create_pairing_code(s, uuid.UUID(ws_id), "laptop")
    assert host.status == "pending" and "-" in code and expires is not None

    assert svc.pair_host(s, "WRONG-CODE") is None
    paired, token = svc.pair_host(s, code.lower(), "laptop", {"claude": "2.1.236"})
    assert paired.id == host.id and paired.status == "paired" and paired.pairing_code_hash is None
    assert svc.pair_host(s, code) is None, "a pairing code is single-use"

    assert svc.resolve_host_by_token(s, token).id == host.id
    assert svc.resolve_host_by_token(s, token + "x") is None
    assert svc.resolve_host_by_token(s, "") is None
    s.close()


def test_a_session_file_under_the_workspace_volume_becomes_a_task_deliverable(seeded, new_session, tmp_path, monkeypatch):
    """PRD-234 S2: files_touched under <volume>/<workspace_id>/… are registered as
    deliverables of the ticket (source_type=task, source_id=<task id>); files
    elsewhere stay references only."""
    from config import config as cfg
    ws_id, _api_agent, cli_agent = seeded
    (tmp_path / ws_id / "sessions" / "1").mkdir(parents=True)
    (tmp_path / ws_id / "sessions" / "1" / "hello.py").write_text("print('hi')\n")
    monkeypatch.setattr(cfg, "WORKSPACE_VOLUME_PATH", str(tmp_path), raising=False)
    s = new_session()
    task_id = s.execute(
        text("INSERT INTO board_tasks (workspace_id, title, status, assigned_agent_id, priority) "
             "VALUES (CAST(:w AS uuid), 'deliverable', 'assigned', :a, 'medium') RETURNING id"),
        {"w": ws_id, "a": cli_agent},
    ).fetchone()[0]
    s.commit()
    host, token = svc.pair_host(s, svc.create_pairing_code(s, ws_id, "h-deliv")[1], "h-deliv", {})
    claimed = svc.claim_for_host(s, host, 1)["tasks"]
    assert [t["task_id"] for t in claimed] == [task_id]
    host_path = f"/Users/me/Development/automatos-ai/workspaces/{ws_id}/sessions/1/hello.py"
    out = asyncio.run(svc.apply_result(s, host, task_id, {
        "attempt": claimed[0]["attempt"], "status": "success", "result_text": "wrote hello.py",
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "files_touched": [host_path, "/Users/me/elsewhere/notes.md"],
    }))
    assert out["applied"] is True
    row = s.query(BoardTask).get(task_id)
    assert [d["file_path"] for d in row.runtime_ref["deliverables"]] == ["sessions/1/hello.py"]
    found = s.execute(
        text("SELECT source_type, source_id, artifact_type, file_size_bytes FROM deliverables "
             "WHERE workspace_id = CAST(:w AS uuid) AND file_path = 'sessions/1/hello.py' AND deleted_at IS NULL"),
        {"w": ws_id},
    ).fetchone()
    assert found is not None and tuple(found) == ("task", str(task_id), "code", 12)
    s.execute(text("DELETE FROM deliverables WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws_id})
    s.commit()
    s.close()


def test_host_claim_preassigns_a_session_and_result_applies_once(seeded, new_session):
    ws_id, _, cli_agent = seeded
    s = new_session()
    host, code, _ = svc.create_pairing_code(s, uuid.UUID(ws_id), "laptop")
    host, _token = svc.pair_host(s, code)
    task_id = _seed_task(s, ws_id, cli_agent, "cli-work")

    claimed = svc.claim_for_host(s, host, limit=5)["tasks"]
    assert [c["task_id"] for c in claimed] == [task_id]
    ticket = claimed[0]
    assert ticket["provider"] == "claude" and ticket["model"] == "sonnet"
    assert uuid.UUID(ticket["session_id"])  # a real pre-assigned session id
    row = s.query(BoardTask).get(task_id)
    assert row.status == "in_progress" and row.runtime_ref["host_id"] == str(host.id)
    assert row.runtime_ref["session_id"] == ticket["session_id"]

    # events renew the lease and surface the live tool
    before = row.lease_until
    out = svc.record_events(s, host, task_id, [{"event": "PreToolUse", "tool_name": "Edit", "transcript_path": "/tmp/t.jsonl"}])
    assert out["lease_renewed"] is True and out["control"] == []
    s.refresh(row)
    assert row.runtime_ref["live_tool"] == "Edit" and row.runtime_ref["transcript_path"] == "/tmp/t.jsonl"
    assert row.lease_until is not None and (before is None or row.lease_until >= before)

    # a denial forces review; a duplicate result is a no-op
    first = asyncio.run(svc.apply_result(s, host, task_id, {
        "attempt": ticket["attempt"], "status": "success", "result_text": "changed 2 files",
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "permission_denials": [{"tool": "Bash", "stage": "PreToolUse",
                                "reason": "'python3 hello.py' is outside this ticket's Bash allowlist",
                                "input": {"command": "python3 hello.py", "description": "never kept"}}],
    }))
    assert first == {"applied": True, "status": "review"}
    dup = asyncio.run(svc.apply_result(s, host, task_id, {"attempt": ticket["attempt"], "status": "success"}))
    assert dup["applied"] is False
    s.refresh(row)
    assert row.status == "review" and row.result == "changed 2 files"
    assert row.runtime_ref["denials"] == 1 and row.runtime_ref["exit_reason"] == "success"
    # the ticket can say WHY it is in review — reason + the command, never the whole tool input
    assert row.runtime_ref["permission_denials"] == [{
        "tool": "Bash", "stage": "PreToolUse",
        "reason": "'python3 hello.py' is outside this ticket's Bash allowlist",
        "subject": "python3 hello.py",
    }]
    s.close()


def test_a_host_cannot_touch_a_ticket_it_did_not_claim(seeded, new_session):
    ws_id, _, cli_agent = seeded
    s = new_session()
    host_a, code_a, _ = svc.create_pairing_code(s, uuid.UUID(ws_id), "a")
    host_a, _ = svc.pair_host(s, code_a)
    host_b, code_b, _ = svc.create_pairing_code(s, uuid.UUID(ws_id), "b")
    host_b, _ = svc.pair_host(s, code_b)
    task_id = _seed_task(s, ws_id, cli_agent, "cli-work")
    assert svc.claim_for_host(s, host_a, limit=1)["tasks"][0]["task_id"] == task_id
    with pytest.raises(PermissionError):
        svc.record_events(s, host_b, task_id, [])
    with pytest.raises(LookupError):
        svc.record_events(s, host_a, 999999999, [])
    s.close()


def test_heartbeat_reattaches_a_requeued_ticket_and_flags_stale_ones(seeded, new_session):
    ws_id, _, cli_agent = seeded
    s = new_session()
    host, code, _ = svc.create_pairing_code(s, uuid.UUID(ws_id), "laptop")
    host, _ = svc.pair_host(s, code)
    task_id = _seed_task(s, ws_id, cli_agent, "cli-work")
    ticket = svc.claim_for_host(s, host, limit=1)["tasks"][0]
    # the sweeper requeued it while the host was away
    s.execute(text("UPDATE board_tasks SET status='assigned', lease_until=NULL WHERE id=:id"), {"id": task_id})
    s.commit()
    out = svc.record_heartbeat(s, host, {"claude": "2.1.236"}, [
        {"task_id": task_id, "session_id": ticket["session_id"]},
        {"task_id": 999999999, "session_id": "ghost"},
    ])
    assert out["reattached"] == [task_id] and out["stale"] == [999999999]
    row = s.query(BoardTask).get(task_id)
    assert row.status == "in_progress" and row.lease_until is not None
    assert row.runtime_ref.get("reattached_at")
    s.close()

"""PRD-253 Wave P — the Plan card end to end, against real Postgres (``@integration``).

What the pure suite cannot show: a Plan agent's claim, the turn's final flush
filing ONE real question row (a retried batch files no second), the result
parking the ticket on it with plan.md registered as a Deliverable, the
operator's answer travelling the answer route's own dispatch, and the claim
after Approve working as Edit automatically — the same session, the plan in its
prompt. Reject sends the ticket to review instead.

Skips cleanly when no Postgres is reachable (CI runs it). PRD-158 lesson: seed
``workspaces`` FIRST for every FK'd table.
"""
from __future__ import annotations

import asyncio
import json
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
from core.models.approval_grants import ApprovalGrant  # noqa: E402
from core.models.core import BoardTask  # noqa: E402
from services import cli_host_service as svc  # noqa: E402
from services import session_plans as plans  # noqa: E402

pytestmark = pytest.mark.integration

PLAN = "1. Add hello.txt\n2. Verify it with cat"
# Swept before the workspace, child-first; any table this database lacks is skipped.
_SWEPT_TABLES = (
    "llm_usage", "deliverables", "agent_reports", "notifications",
    "approval_grants", "board_tasks", "cli_hosts", "agents",
)


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            for tbl in ("agents", "board_tasks", "cli_hosts", "workspaces", "approval_grants"):
                c.execute(text(f"SELECT 1 FROM {tbl} LIMIT 1"))
            c.execute(text("SELECT runtime_ref FROM board_tasks LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the Plan card suite needs a reachable Postgres with the S1a schema: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture(autouse=True)
def _quiet_side_effects(monkeypatch):
    """On the Plan card's contract only: no approval-policy lookup at claim, no
    completion fan-out, and the card's bell and Telegram message stay unsent
    (PRD-225's own suites cover them)."""
    import api.board_tasks as bt
    import modules.tools.discovery.handlers_asks as asks

    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(bt, "_board_task_blocked_pending_approval", lambda *a, **k: False, raising=True)
    monkeypatch.setattr(bt, "_dispatch_task_complete", _noop, raising=True)
    monkeypatch.setattr(asks, "_dispatch_question_pending", _noop, raising=True)
    monkeypatch.setattr(asks, "_capture_question_telegram", _noop, raising=True)


@pytest.fixture
def ticket(engine, new_session):
    """A workspace, a Codex session agent in Plan mode and one assigned ticket."""
    ws_id = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"),
              {"id": ws_id, "n": "prd253-plan"})
    s.commit()
    agent_id = s.execute(
        text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
             "VALUES (:n, 'custom', CAST(:w AS uuid), 'active', CAST(:c AS json)) RETURNING id"),
        {"n": f"CODER-{ws_id[:8]}", "w": ws_id,
         "c": json.dumps({"runtime": "cli", "provider": "codex", "model": "gpt-5.5", "permission_mode": "plan"})},
    ).fetchone()[0]
    task = BoardTask(workspace_id=ws_id, title="Say hi", status="assigned", priority="medium",
                     assigned_agent_id=agent_id, source_type="user", attempts=0)
    s.add(task)
    s.commit()
    yield ws_id, task.id
    sweep = new_session.sweep()
    present = {
        row[0] for row in sweep.execute(text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = ANY(:names)"
        ), {"names": list(_SWEPT_TABLES)}).fetchall()
    }
    for table in _SWEPT_TABLES:
        if table in present:
            sweep.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws_id})
    sweep.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws_id})
    sweep.commit()


def _host(session, ws_id):
    """A paired host that announced no CLI list — the claim reads it as "serves every CLI"."""
    _row, code, _ = svc.create_pairing_code(session, uuid.UUID(ws_id), "laptop")
    host, _token = svc.pair_host(session, code)
    return host


def _plan_turn(s, host, ws_id, task_id, plan_file):
    """One Plan turn: claimed in Plan, ends with its plan in the final flush, then its result."""
    first = svc.claim_for_host(s, host, limit=1)["tasks"][0]
    assert first["task_id"] == task_id
    assert first["permission_mode"] == "plan" and first["plan_approved"] is False
    final = [{"event": "Stop", "session_id": "codex-1"}, {"event": plans.PLAN_EVENT, "text": PLAN}]
    asyncio.run(svc.record_events(s, host, task_id, final))
    asyncio.run(svc.record_events(s, host, task_id, final))          # the host retried the batch
    return asyncio.run(svc.apply_result(s, host, task_id, {
        "attempt": first["attempt"], "status": "success", "result_text": PLAN,
        "usage": {"input_tokens": 10, "output_tokens": 5}, "files_touched": [plan_file],
    }))


def _plan_cards(s, ws_id, task_id):
    return s.execute(
        text("SELECT id, status FROM approval_grants WHERE workspace_id = CAST(:w AS uuid) "
             "AND subject_type = 'board_task' AND subject_id = :t ORDER BY id"),
        {"w": ws_id, "t": str(task_id)},
    ).fetchall()


def _answer(s, grant_id, words):
    """The answer route's two steps: the compare-and-swap, then its dispatch."""
    from api.approval_grants import _requeue_subject
    from core.services.approval_grants import answer_pending_grant

    assert answer_pending_grant(s, grant_id, answer_text=words, answered_by="user:1")
    s.commit()
    grant = s.get(ApprovalGrant, grant_id)
    s.refresh(grant)                      # the CAS ran as a bulk UPDATE; read what it wrote
    return asyncio.run(_requeue_subject(s, grant))


def _volume(tmp_path, monkeypatch, ws_id, task_id):
    """plan.md under the workspace volume, as the host saved it; its host-side path."""
    from config import config as cfg

    folder = tmp_path / ws_id / "sessions" / str(task_id)
    folder.mkdir(parents=True)
    (folder / "plan.md").write_text(PLAN + "\n")
    monkeypatch.setattr(cfg, "WORKSPACE_VOLUME_PATH", str(tmp_path), raising=False)
    return f"/Users/me/automatos-ai/workspaces/{ws_id}/sessions/{task_id}/plan.md"


def test_a_plan_parks_on_one_card_and_approve_resumes_the_session_as_edits(ticket, new_session, tmp_path, monkeypatch):
    ws_id, task_id = ticket
    s = new_session()
    host = _host(s, ws_id)
    out = _plan_turn(s, host, ws_id, task_id, _volume(tmp_path, monkeypatch, ws_id, task_id))
    assert out == {"applied": True, "status": "blocked"}

    cards = _plan_cards(s, ws_id, task_id)
    assert [c.status for c in cards] == ["pending"]                   # ONE card, retry or not
    row = s.get(BoardTask, task_id)
    s.refresh(row)
    assert row.status == "blocked"
    assert row.blocked_reason == plans.PARKED_FOR_PLAN_REASON.format(grant_id=cards[0].id)
    assert [d["file_path"] for d in row.runtime_ref["deliverables"]] == [f"sessions/{task_id}/plan.md"]

    assert _answer(s, cards[0].id, "Approve") is True
    s.refresh(row)
    assert row.status == "assigned"

    second = svc.claim_for_host(s, host, limit=1)["tasks"][0]
    assert second["permission_mode"] == "edits" and second["plan_approved"] is True
    assert second["resume_session_id"] == "codex-1"                   # the same session carries it out
    assert "## Your plan was approved — implement it now" in second["prompt"] and PLAN in second["prompt"]
    s.close()


def test_reject_sends_the_ticket_to_review_with_the_reason(ticket, new_session, tmp_path, monkeypatch):
    ws_id, task_id = ticket
    s = new_session()
    host = _host(s, ws_id)
    _plan_turn(s, host, ws_id, task_id, _volume(tmp_path, monkeypatch, ws_id, task_id))
    card = _plan_cards(s, ws_id, task_id)[0]

    assert _answer(s, card.id, "Reject — not this quarter") is False
    row = s.get(BoardTask, task_id)
    s.refresh(row)
    assert row.status == "review"
    assert row.review_feedback == "Plan rejected: Reject — not this quarter"
    assert svc.claim_for_host(s, host, limit=1)["tasks"] == []        # nothing resumes on it
    s.close()

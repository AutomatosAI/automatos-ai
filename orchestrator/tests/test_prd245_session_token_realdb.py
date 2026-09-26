"""PRD-245 S1.1 — the per-ticket session token, against real Postgres (``@integration``).

The token is the whole identity of a call to ``/api/v1/session-tools/mcp``, and
what makes it safe is a property the pure suites cannot show: the LOOKUP is by
hash over ``board_tasks.runtime_ref`` and it only ever resolves a ticket that is
still ``in_progress``. So a token left in a session transcript — the one real
exposure this design has — stops working the moment the ticket ends, whether or
not anything remembered to clear it.

Skips cleanly when no Postgres is reachable (CI runs it). PRD-158 lesson: seed
``workspaces`` FIRST for every FK'd table.
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from core.database.database import get_database_url  # noqa: E402
from api.board_tasks import end_session_claim  # noqa: E402
from core.models.core import BoardTask  # noqa: E402
from services import cli_host_service as svc  # noqa: E402
from services import session_tools as st  # noqa: E402

pytestmark = pytest.mark.integration

# Swept before the workspace, child-first. Every one carries ``workspace_id``;
# any this database does not have is skipped (see the fixture teardown).
_SWEPT_TABLES = (
    "llm_usage", "deliverables", "agent_reports", "notifications",
    "approval_grants", "board_tasks", "cli_hosts", "agents",
)


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            for tbl in ("agents", "board_tasks", "cli_hosts", "workspaces"):
                c.execute(text(f"SELECT 1 FROM {tbl} LIMIT 1"))
            c.execute(text("SELECT runtime_ref FROM board_tasks LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PRD-245 token suite needs a reachable Postgres with the S1a schema: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture(autouse=True)
def _quiet_side_effects(monkeypatch):
    """Keep this on the token contract: no approval-policy lookup (it parks a
    ticket at claim and the claim then returns nothing), no completion fan-out.
    Same shape as the PRD-234 host suite next door."""
    import api.board_tasks as bt

    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(bt, "_board_task_blocked_pending_approval", lambda *a, **k: False, raising=True)
    monkeypatch.setattr(bt, "_dispatch_task_complete", _noop, raising=True)


def _seed_task(session, ws_id, agent_id, title, status="assigned"):
    task = BoardTask(
        workspace_id=ws_id, title=title, status=status, priority="medium",
        assigned_agent_id=agent_id, source_type="user", attempts=0,
    )
    session.add(task)
    session.commit()
    return task.id


@pytest.fixture
def ticket(engine, new_session):
    """A workspace, a ``runtime: cli`` agent and one assigned ticket."""
    ws_id = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"),
              {"id": ws_id, "n": "prd245-token"})
    s.commit()
    agent_id = s.execute(
        text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
             "VALUES (:n, 'custom', CAST(:w AS uuid), 'active', CAST(:c AS json)) RETURNING id"),
        {"n": f"TRACKER-{ws_id[:8]}", "w": ws_id,
         "c": json.dumps({"runtime": "cli", "provider": "claude", "model": "opus"})},
    ).fetchone()[0]
    task_id = _seed_task(s, ws_id, agent_id, "TRACKER — snapshot")
    yield ws_id, agent_id, task_id
    sweep = new_session.sweep()
    # Every table a RESULT writes has to go before the workspace does, or the
    # teardown dies on a foreign key and takes the suite's exit code with it:
    # ``apply_result`` books usage (llm_usage), registers deliverables and writes
    # the task's report, and an ask leaves a grant + a notification behind.
    #
    # Only the tables this database actually HAS: the test schema is not the
    # local one (``agent_reports`` exists in a full local install and not here),
    # and one missing table would abort the whole sweep transaction.
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
    """A paired host. It announces no CLI list, which the claim reads as "serves
    every CLI" — a host that announced an empty list would claim nothing."""
    _host_row, code, _ = svc.create_pairing_code(session, uuid.UUID(ws_id), "laptop")
    host, _token = svc.pair_host(session, code)
    return host


def test_the_claim_hands_the_session_its_own_credential_once(ticket, new_session):
    ws_id, agent_id, task_id = ticket
    s = new_session()
    host = _host(s, ws_id)

    claimed = svc.claim_for_host(s, host, limit=1)["tasks"]
    assert [c["task_id"] for c in claimed] == [task_id]
    payload = claimed[0]

    # the host is told what it may call, where, and with what
    assert payload["session_tools"] == list(st.tool_names())
    assert payload["session_tools_path"] == svc.SESSION_TOOLS_PATH
    token = payload["session_token"]
    assert token and len(token) > 32

    # only the HASH is on the ticket — never the token itself
    row = s.query(BoardTask).get(task_id)
    stored = row.runtime_ref[svc.SESSION_TOKEN_HASH_KEY]
    assert stored == svc.hash_secret(token)
    assert token not in json.dumps(row.runtime_ref)
    # F131: the ticket records that its session was offered the Automatos tools,
    # so a session that never reaches them fails its step instead of "completing"
    assert row.runtime_ref[svc.SESSION_TOOLS_OFFERED_KEY] is True


def test_the_token_resolves_to_its_own_running_ticket_and_nothing_else(ticket, new_session):
    ws_id, agent_id, task_id = ticket
    s = new_session()
    host = _host(s, ws_id)
    token = svc.claim_for_host(s, host, limit=1)["tasks"][0]["session_token"]

    resolved = svc.resolve_session_token(s, token)
    assert resolved is not None
    task, agent = resolved
    assert task.id == task_id and agent is not None and agent.id == agent_id
    assert str(task.workspace_id) == ws_id

    for nonsense in ("", None, "not-a-token", token[:-1], token + "x"):
        assert svc.resolve_session_token(s, nonsense) is None


def test_a_token_stops_working_when_its_ticket_stops_running(ticket, new_session):
    """The property the design leans on: a token in a transcript is dead as soon
    as the ticket stops running, and STAYS dead.

    Status alone was not enough. A ticket leaves ``in_progress`` and comes back
    all the time without a new claim — the operator drags a blocked card back to
    In Progress, a status PATCH moves it, a session sets its own ticket
    ``blocked``. None of those mints a new token, so the old plaintext — still in
    the session transcript, still in that ticket's ``mcp.json`` on disk — started
    authenticating again, from anywhere that can reach the backend, with no
    session running. The rule is a LIVE LEASE: only a claim sets one, the host
    renews it on every event flush, and every path that ends a run nulls it.
    """
    ws_id, _agent_id, task_id = ticket
    s = new_session()
    host = _host(s, ws_id)
    claimed = svc.claim_for_host(s, host, limit=1)["tasks"][0]
    token = claimed["session_token"]
    assert svc.resolve_session_token(s, token) is not None

    row = s.query(BoardTask).get(task_id)
    assert row.lease_until is not None                     # the claim set one

    # an expired lease is not a running session, whatever the status says
    kept = row.lease_until
    row.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
    s.commit()
    assert svc.resolve_session_token(s, token) is None, "an expired claim must not authenticate"
    row.lease_until = kept
    s.commit()
    assert svc.resolve_session_token(s, token) is not None

    # the board gesture: the operator drags the card out of In Progress. The
    # route ends the claim — lease and credential together.
    end_session_claim(row, "in_progress", "review")
    row.status = "review"
    s.commit()
    assert row.lease_until is None
    assert svc.SESSION_TOKEN_HASH_KEY not in (row.runtime_ref or {})
    assert svc.resolve_session_token(s, token) is None, "a token must not outlive its ticket's run"

    # …and dragging it back does not revive anything: no hash, no lease, no claim
    row.status = "in_progress"
    s.commit()
    assert svc.resolve_session_token(s, token) is None, (
        "a ticket moved back to in_progress by hand must NOT revive an old session token"
    )

    # the run still reports its outcome — ending a claim takes the credential,
    # not the result (the ticket is back at in_progress for the same attempt)
    out = asyncio.run(svc.apply_result(s, host, task_id, {
        "attempt": claimed["attempt"], "status": "success", "result_text": "done",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }))
    assert out["applied"] is True
    s.refresh(row)
    assert svc.SESSION_TOKEN_HASH_KEY not in (row.runtime_ref or {})
    assert svc.resolve_session_token(s, token) is None


def test_two_tickets_never_share_a_token(ticket, new_session):
    ws_id, agent_id, first_task = ticket
    s = new_session()
    second = _seed_task(s, ws_id, agent_id, "TRACKER — second")
    host = _host(s, ws_id)

    claimed = svc.claim_for_host(s, host, limit=2)["tasks"]
    assert len(claimed) == 2
    tokens = {c["task_id"]: c["session_token"] for c in claimed}
    assert len(set(tokens.values())) == 2
    for task_id, token in tokens.items():
        task, _agent = svc.resolve_session_token(s, token)
        assert task.id == task_id, "a token must resolve ONLY its own ticket"
    assert second in tokens

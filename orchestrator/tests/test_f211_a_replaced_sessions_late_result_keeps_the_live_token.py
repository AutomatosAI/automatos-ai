"""F211 — a replaced session's late result never kills the live session's credential.

Every claim builds a fresh runtime_ref and mints its own session token, so the
hash on the row is always the newest claim's. apply_result's stale-attempt
branch revoked it: a replaced session's late result (after Run Now, a lease
sweep's requeue, a usage-limit pause) cut the live session off its platform
tools, since PRD-245 fails closed. The stale result is refused and the live
token stays. A result for a ticket that left in_progress still revokes.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

from services import cli_host_service as svc


class _Db:
    def __init__(self):
        self.commits = 0

    def commit(self):
        self.commits += 1


def _ticket(status, attempt):
    return NS(id=5, status=status, attempts=attempt, workspace_id="ws-1", assigned_agent_id=3,
              runtime_ref={svc.SESSION_TOKEN_HASH_KEY: "the-live-claims-hash", "session_id": "s-2",
                           "attempt": attempt, "provider": "claude"})


def _late_result(monkeypatch, task, attempt):
    monkeypatch.setattr(svc, "_owned_task", lambda db, host, task_id: task)
    db = _Db()
    out = asyncio.run(svc.apply_result(db, NS(id="h1"), 5,
                                       {"attempt": attempt, "status": "success", "result": "late words"}))
    return out, db


def test_a_replaced_sessions_late_result_leaves_the_live_token(monkeypatch):
    task = _ticket("in_progress", attempt=3)                  # re-claimed: attempt 3 holds the ticket
    out, db = _late_result(monkeypatch, task, attempt=2)      # attempt 2's session answers late
    assert out == {"applied": False, "reason": "stale attempt", "status": "in_progress"}
    assert task.runtime_ref[svc.SESSION_TOKEN_HASH_KEY] == "the-live-claims-hash"
    assert db.commits == 0


def test_a_result_for_a_ticket_that_left_in_progress_still_revokes(monkeypatch):
    task = _ticket("assigned", attempt=3)                     # moved back while its session ran
    out, db = _late_result(monkeypatch, task, attempt=3)
    assert out == {"applied": False, "reason": "task is assigned", "status": "assigned"}
    assert svc.SESSION_TOKEN_HASH_KEY not in task.runtime_ref
    assert db.commits == 1

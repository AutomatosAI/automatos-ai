"""F083 — a session that hit its CLI's usage limit puts the ticket back.

The claim counted an attempt (``attempts + 1``) and a ticket gets two; a plan
window closing is not the ticket's fault, so the release refunds it. The host
stops claiming for that CLI until the window reopens, so the ticket is not
handed straight back to a closed window.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pytest
from pydantic import ValidationError

from api.cli_hosts import ResultRequest
from services import cli_host_service as svc

REASON = "paused: claude usage limit, resumes ~15:00"
RESETS = "2026-09-23T15:00:00+01:00"


class _Db:
    def __init__(self):
        self.commits = 0

    def commit(self):
        self.commits += 1


def _task(attempts=1, attempt=1):
    return NS(id=5, status="in_progress", attempts=attempts, workspace_id="ws-1", assigned_agent_id=3,
              lease_until=datetime(2026, 9, 23, 13, 0, tzinfo=timezone.utc),
              runtime_ref={svc.SESSION_TOKEN_HASH_KEY: "hash", "session_id": "s-1", "attempt": attempt,
                           "provider": "claude"})


@pytest.fixture
def booked(monkeypatch):
    calls = []
    monkeypatch.setattr(svc, "book_session_usage", lambda task, ref, usage, **kw: calls.append((usage, kw)) or 0)
    return calls


def test_the_ticket_goes_back_to_the_queue_with_its_attempt_refunded(booked):
    task, db = _task(attempts=1), _Db()
    out = svc._release_for_usage_limit(db, task, dict(task.runtime_ref),
                                       {"error": REASON, "resets_at": RESETS, "usage": {"total_tokens": 900}})
    assert out == {"applied": True, "status": "assigned", "released": True, "reason": REASON}
    assert (task.status, task.attempts, task.lease_until) == ("assigned", 0, None)
    assert task.runtime_ref["paused"]["reason"] == REASON and task.runtime_ref["paused"]["resets_at"] == RESETS
    assert task.runtime_ref["exit_reason"] == "usage_limit"
    assert svc.SESSION_TOKEN_HASH_KEY not in task.runtime_ref          # the session's credential dies
    assert db.commits == 1
    assert booked == [({"total_tokens": 900}, {"status": "error", "request_type": svc.LANE_BOARD_TASK,
                                               "execution_id": "board_task:5", "error": REASON})]


def test_attempts_never_go_below_zero(booked):
    task = _task(attempts=0)
    svc._release_for_usage_limit(_Db(), task, dict(task.runtime_ref), {"error": REASON})
    assert task.attempts == 0


def test_apply_result_releases_instead_of_finishing_the_run(monkeypatch, booked):
    import api.board_tasks as board_tasks

    task = _task(attempts=2, attempt=2)
    monkeypatch.setattr(svc, "_owned_task", lambda db, host, task_id: task)

    async def must_not_finish(*_a, **_k):
        raise AssertionError("a usage limit must not finish the ticket's run")

    monkeypatch.setattr(board_tasks, "finalize_board_task_run", must_not_finish)
    out = asyncio.run(svc.apply_result(_Db(), NS(id="h1"), 5,
                                       {"attempt": 2, "status": "usage_limit", "error": REASON, "resets_at": RESETS}))
    assert out["released"] is True and task.status == "assigned" and task.attempts == 1


def test_a_stale_attempt_is_still_refused(monkeypatch, booked):
    task = _task(attempts=2, attempt=2)
    monkeypatch.setattr(svc, "_owned_task", lambda db, host, task_id: task)
    monkeypatch.setattr(svc, "revoke_session_token", lambda db, t: False)
    out = asyncio.run(svc.apply_result(_Db(), NS(id="h1"), 5, {"attempt": 1, "status": "usage_limit", "error": REASON}))
    assert out == {"applied": False, "reason": "stale attempt", "status": "in_progress"}
    assert task.attempts == 2


def test_the_result_schema_accepts_the_new_status_and_nothing_else():
    ok = ResultRequest(status="usage_limit", error=REASON, resets_at=RESETS)
    assert ok.status == "usage_limit" and ok.resets_at == RESETS
    with pytest.raises(ValidationError):
        ResultRequest(status="paused")

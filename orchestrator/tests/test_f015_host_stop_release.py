"""F015 (night 1) — a host that stops mid-session hands the ticket back, saying why.

Night 1 recorded tickets ended by the CLI host stopping (a reinstall, a restart,
SIGTERM) as ``cancelled`` — by no one, for no reason, and final. The owner had
not stopped them. The host now reports ``host_stopped``; the ticket goes back to
the queue with the claim's attempt refunded and the host and reason on it.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pytest
from pydantic import ValidationError

from api.cli_hosts import ResultRequest
from services import cli_host_service as svc

REASON = "the CLI host on gerards-mac stopped (SIGTERM)"


class _Db:
    def __init__(self):
        self.commits = 0

    def commit(self):
        self.commits += 1


def _task(attempts=1, attempt=1):
    return NS(id=9, status="in_progress", attempts=attempts, workspace_id="ws-1", assigned_agent_id=3,
              lease_until=datetime(2026, 9, 23, 1, 0, tzinfo=timezone.utc),
              runtime_ref={svc.SESSION_TOKEN_HASH_KEY: "hash", "session_id": "s-9", "attempt": attempt,
                           "provider": "claude", "host_id": "h1"})


@pytest.fixture
def booked(monkeypatch):
    calls = []
    monkeypatch.setattr(svc, "book_session_usage", lambda task, ref, usage, **kw: calls.append((usage, kw)) or 0)
    return calls


def test_the_ticket_goes_back_to_the_queue_with_the_host_and_reason_on_it(monkeypatch, booked):
    import api.board_tasks as board_tasks

    task = _task(attempts=1)
    monkeypatch.setattr(svc, "_owned_task", lambda db, host, task_id: task)

    async def must_not_finish(*_a, **_k):
        raise AssertionError("a host stopping must not finish the ticket's run")

    monkeypatch.setattr(board_tasks, "finalize_board_task_run", must_not_finish)
    db = _Db()
    out = asyncio.run(svc.apply_result(db, NS(id="h1"), 9, {
        "attempt": 1, "status": "host_stopped", "error": REASON, "usage": {"total_tokens": 1200}}))
    assert out == {"applied": True, "status": "assigned", "released": True, "reason": REASON}
    assert (task.status, task.attempts, task.lease_until) == ("assigned", 0, None)
    released = task.runtime_ref["released"]
    assert (released["reason"], released["by"]) == (REASON, "cli-host:h1") and released["at"]
    assert task.runtime_ref["exit_reason"] == "host_stopped"
    assert svc.SESSION_TOKEN_HASH_KEY not in task.runtime_ref          # the session's credential dies
    assert db.commits == 1
    assert booked[0][0] == {"total_tokens": 1200} and booked[0][1]["error"] == REASON


def test_a_real_cancel_is_still_a_cancel():
    """The operator's own cancel keeps its meaning — only the host's stop is released."""
    assert ResultRequest(status="cancelled").status == "cancelled"
    assert ResultRequest(status="host_stopped", error=REASON).status == "host_stopped"
    with pytest.raises(ValidationError):
        ResultRequest(status="stopped")

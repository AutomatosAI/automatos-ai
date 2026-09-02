"""PRD-234 S1a — the board lane: cli tickets are parked for the host; one completion writer.

Drives ``_launch_task_execution``'s inner ``_run`` the way
``test_prd171_execution_spine.py`` does (stubbed session, neutralised side
effects, the guarded coroutine run inline) and proves:

* a ticket assigned to a ``runtime: cli`` agent is never executed here — it goes
  back to ``assigned`` for the CLI host, claimants are woken, and the factory is
  never even constructed (zero LLM-client invocations);
* an API agent's ticket still runs exactly as before (regression);
* ``finalize_board_task_run`` — the writer extracted from the inline block —
  closes error → ``failed``, success → ``done``/``review``, denials → ``review``,
  cancelled → ``cancelled``, and never writes a row that already left
  ``in_progress`` (idempotency for late results).
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import api.board_tasks as bt  # noqa: E402


class _FakeTask:
    def __init__(self, task_id: int, status: str = "in_progress"):
        self.id = task_id
        self.status = status
        self.result = None
        self.error_message = None
        self.completed_at = None
        self.lease_until = "lease"
        self.blocked_at = None
        self.blocked_reason = None
        self.runtime_ref = None
        self.review_mode = "auto"


class _FakeQuery:
    def __init__(self, task):
        self._task = task

    def get(self, _id):
        return self._task

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._task


class _FakeSession:
    def __init__(self, task):
        self._task = task
        self.committed = 0
        self.closed = False

    def query(self, *a, **k):
        return _FakeQuery(self._task)

    def commit(self):
        self.committed += 1

    def rollback(self):  # pragma: no cover
        pass

    def close(self):
        self.closed = True


class _FactoryMustNotRun:
    def __init__(self, *a, **k):
        raise AssertionError("AgentFactory must never be constructed for a cli agent")


def _wire(monkeypatch, task, *, runtime_kind: str, factory_cls=None, exec_result=None):
    session = _FakeSession(task)
    import core.database.database as dbmod
    monkeypatch.setattr(dbmod, "SessionLocal", lambda: session, raising=False)

    import modules.agents.factory.agent_factory as af
    if factory_cls is None:
        class _StubFactory:
            def __init__(self, db_session=None):
                pass

            async def execute_with_prompt(self, **kwargs):
                return exec_result
        factory_cls = _StubFactory
    monkeypatch.setattr(af, "AgentFactory", factory_cls, raising=True)

    async def _noop(*a, **k):
        return None

    calls: Dict[str, list] = {"notify_available": [], "board_event": []}
    monkeypatch.setattr(bt, "_agent_runtime_kind", lambda db, agent_id: runtime_kind, raising=True)
    monkeypatch.setattr(bt, "_board_task_blocked_pending_approval", lambda *a, **k: False, raising=True)
    monkeypatch.setattr(bt, "_dispatch_task_complete", _noop, raising=True)
    monkeypatch.setattr(bt, "_dispatch_task_failed", _noop, raising=True)
    monkeypatch.setattr(bt, "_auto_create_task_report", _noop, raising=True)
    monkeypatch.setattr(bt, "_lease_heartbeat", _noop, raising=True)
    monkeypatch.setattr(bt, "record_error", lambda **k: None, raising=True)
    monkeypatch.setattr(
        bt, "notify_task_available",
        lambda db, **k: calls["notify_available"].append(k), raising=True,
    )
    monkeypatch.setattr(
        bt, "notify_board_event",
        lambda db, **k: calls["board_event"].append(k), raising=True,
    )
    captured: Dict[str, Any] = {}
    monkeypatch.setattr(bt, "launch_guarded", lambda coro, **kw: captured.setdefault("coro", coro), raising=True)
    bt._launch_task_execution(
        task_id=task.id, agent_id=7, workspace_id="ws-1", prompt="do it", review_mode="auto",
    )
    asyncio.run(captured["coro"])
    return session, calls


# ── the cli branch ───────────────────────────────────────────────────────────

def test_cli_agent_ticket_is_parked_for_the_host_and_never_executed(monkeypatch):
    task = _FakeTask(101, status="in_progress")  # a direct launch flipped it
    session, calls = _wire(monkeypatch, task, runtime_kind="cli", factory_cls=_FactoryMustNotRun)
    assert task.status == "assigned", "the ticket must go back to the queue for the CLI host"
    assert task.lease_until is None
    assert calls["notify_available"] == [{"workspace_id": "ws-1", "task_id": 101}]
    assert calls["board_event"] and calls["board_event"][0]["status"] == "assigned"
    assert session.closed


def test_cli_agent_ticket_already_assigned_is_left_alone_but_claimants_wake(monkeypatch):
    task = _FakeTask(102, status="assigned")
    _, calls = _wire(monkeypatch, task, runtime_kind="cli", factory_cls=_FactoryMustNotRun)
    assert task.status == "assigned"
    assert calls["notify_available"] == [{"workspace_id": "ws-1", "task_id": 102}]
    assert calls["board_event"] == []  # nothing changed → no spurious UI ping


def test_api_agent_ticket_still_runs_through_the_factory(monkeypatch):
    task = _FakeTask(103)
    _wire(monkeypatch, task, runtime_kind="api", exec_result={"status": "success", "result": "done!"})
    assert task.status == "done" and task.result == "done!"


def test_api_agent_error_result_still_fails_honestly(monkeypatch):
    task = _FakeTask(104)
    _wire(monkeypatch, task, runtime_kind="api", exec_result={"status": "error", "error": "boom"})
    assert task.status == "failed" and "boom" in task.error_message


# ── the extracted completion writer ─────────────────────────────────────────

def _finalize(monkeypatch, task, exec_result, **kw) -> Optional[str]:
    async def _noop(*a, **k):
        return None

    events = []
    monkeypatch.setattr(bt, "_dispatch_task_complete", _noop, raising=True)
    monkeypatch.setattr(bt, "_dispatch_task_failed", _noop, raising=True)
    monkeypatch.setattr(bt, "_auto_create_task_report", _noop, raising=True)
    monkeypatch.setattr(bt, "notify_board_event", lambda db, **k: events.append(k), raising=True)
    session = _FakeSession(task)
    terminal = asyncio.run(
        bt.finalize_board_task_run(
            session, task_id=task.id, workspace_id="ws-1", agent_id=7,
            exec_result=exec_result, **kw,
        )
    )
    return terminal


def test_writer_review_mode_and_denials_land_in_review(monkeypatch):
    task = _FakeTask(201)
    assert _finalize(monkeypatch, task, {"status": "success", "result": "x"}, review_mode="human") == "review"
    task = _FakeTask(202)
    assert _finalize(
        monkeypatch, task, {"status": "success", "result": "x", "permission_denials": [{"tool": "Bash"}]},
        review_mode="auto", force_review=True,
    ) == "review"
    assert task.result == "x"


def test_writer_cancelled_result_closes_cancelled(monkeypatch):
    task = _FakeTask(203)
    assert _finalize(monkeypatch, task, {"status": "cancelled"}) == "cancelled"
    assert task.completed_at is not None and task.lease_until is None


@pytest.mark.parametrize("status", ["cancelled", "done", "assigned", "failed"])
def test_writer_never_touches_a_task_that_left_in_progress(monkeypatch, status):
    task = _FakeTask(204, status=status)
    assert _finalize(monkeypatch, task, {"status": "success", "result": "late"}) is None
    assert task.status == status and task.result is None


def test_status_vocabulary_includes_cancelled():
    assert "cancelled" in bt.VALID_STATUSES

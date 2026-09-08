"""PRD-239 S3 — playbooks and missions run a session agent through the ticket
lane and wait for the session to end; the lane returns the exec-result shape
both already read, and a timeout leaves the ticket running and says so.

Pure units: fake session, no sleeping (asyncio.sleep is stubbed)."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from services import cli_ticket_lane as lane  # noqa: E402

WS = uuid4()


class _Query:
    def __init__(self, task):
        self._task = task

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._task


class _DB:
    def __init__(self, task):
        self.task = task
        self.expired = 0

    def expire_all(self):
        self.expired += 1

    def query(self, model):
        return _Query(self.task)


def _ticket(status="in_progress", **over):
    base = dict(id=7, status=status, result="Bob's answer", error_message=None,
                runtime_ref={"usage": {"total_tokens": 42}})
    base.update(over)
    return SimpleNamespace(**base)


def _run(coro):
    return asyncio.run(coro)


# ── the wait ─────────────────────────────────────────────────────────────────

def test_wait_returns_the_ended_tickets_result(monkeypatch):
    task = _ticket()
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            task.status = "done"

    monkeypatch.setattr(lane, "file_cli_ticket", lambda db, **kw: task)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    db = _DB(task)
    res = _run(lane.run_cli_ticket_and_wait(
        db, workspace_id=WS, agent_id=15, title="t", prompt="p", source_type="recipe", source_id="recipe:e:1", poll_s=0.5,
    ))
    assert res["status"] == "success" and res["result"] == "Bob's answer"
    assert res["task_id"] == 7 and res["tokens_used"] == 42 and res["runtime"] == "cli"
    assert sleeps == [0.5, 0.5] and db.expired == 3  # re-read before every look


def test_wait_times_out_and_names_the_still_running_ticket(monkeypatch):
    task = _ticket()
    monkeypatch.setattr(lane, "file_cli_ticket", lambda db, **kw: task)

    async def never(seconds):
        raise AssertionError("must not sleep past the timeout")

    monkeypatch.setattr(asyncio, "sleep", never)
    res = _run(lane.run_cli_ticket_and_wait(
        _DB(task), workspace_id=WS, agent_id=15, title="t", prompt="p", source_type="mission", source_id="m:1", timeout_s=0,
    ))
    assert res["status"] == "error" and res["timed_out"] is True
    assert "ticket #7 is still running" in res["error"] and "carries on" in res["error"]


def test_wait_reads_failed_and_cancelled_endings(monkeypatch):
    for status, expected in (("failed", "error"), ("cancelled", "cancelled")):
        task = _ticket(status=status, error_message="bad")
        monkeypatch.setattr(lane, "file_cli_ticket", lambda db, **kw: task)
        res = _run(lane.run_cli_ticket_and_wait(
            _DB(task), workspace_id=WS, agent_id=15, title="t", prompt="p", source_type="recipe", source_id="r:1",
        ))
        assert res["status"] == expected and res["task_id"] == 7


def test_wait_passes_the_lane_kwargs_through_to_the_filing(monkeypatch):
    seen = {}

    def fake_file(db, **kw):
        seen.update(kw)
        return _ticket(status="done")

    monkeypatch.setattr(lane, "file_cli_ticket", fake_file)
    _run(lane.run_cli_ticket_and_wait(
        _DB(_ticket(status="done")), workspace_id=WS, agent_id=15, title="t", prompt="p", source_type="mission",
        source_id="mission:r:t", tags=["mission"], orchestration_run_id="r", orchestration_task_id="t",
    ))
    assert seen["tags"] == ["mission"] and seen["orchestration_run_id"] == "r" and seen["orchestration_task_id"] == "t"


# ── the playbook step ────────────────────────────────────────────────────────

def test_a_playbook_step_for_a_session_agent_files_a_ticket_and_waits(monkeypatch):
    import api.recipe_executor as rx

    seen = {}

    async def fake_wait(db, **kw):
        seen.update(kw)
        return {"status": "success", "result": "done by Bob", "execution": {"tokens_used": 0, "tool_calls": [], "messages": []}}

    monkeypatch.setattr(lane, "is_cli_agent", lambda db, aid: True)
    monkeypatch.setattr(lane, "run_cli_ticket_and_wait", fake_wait)
    res = _run(rx._execute_step(
        db=SimpleNamespace(), agent=SimpleNamespace(id=15), clean_prompt="Write the weekly digest\nwith the numbers",
        workspace_id=WS, step_order=2, recipe_name="Weekly", recipe_execution_id="exec-1", max_iterations=3,
    ))
    assert res["status"] == "success" and res["result"] == "done by Bob"
    assert seen["source_type"] == "recipe" and seen["source_id"] == "recipe:exec-1:2"
    assert seen["title"] == "Weekly · step 2: Write the weekly digest" and seen["tags"] == ["playbook"]
    assert seen["agent_id"] == 15 and seen["prompt"].startswith("Write the weekly digest")


def test_cli_step_title_shape():
    import api.recipe_executor as rx

    assert rx._cli_step_title("", 1, "") == "Playbook · step 1"
    long = rx._cli_step_title("Weekly", 3, "x" * 80)
    assert long.startswith("Weekly · step 3: " + "x" * 60) and long.endswith("…")


# ── the mission task ─────────────────────────────────────────────────────────

def test_a_mission_task_for_a_session_agent_runs_the_lane_on_its_own_session(monkeypatch):
    from services import coordinator_service as cs
    import core.database.database as dbmod

    owner = next(v for v in vars(cs).values() if isinstance(v, type) and hasattr(v, "_run_cli_ticket"))
    closed = []

    class _Own:
        def close(self):
            closed.append(True)

    monkeypatch.setattr(dbmod, "SessionLocal", lambda: _Own())
    seen = {}

    async def fake_wait(db, **kw):
        seen["db"] = db
        seen.update(kw)
        return {"status": "success", "result": "ok"}

    monkeypatch.setattr(lane, "run_cli_ticket_and_wait", fake_wait)
    task = SimpleNamespace(id="t1", title="Research the market")
    res = _run(owner._run_cli_ticket(task, "the prompt", 15, WS, "run-1", 600))
    assert res == {"status": "success", "result": "ok"}
    assert isinstance(seen["db"], _Own) and closed == [True]
    assert seen["source_type"] == "mission" and seen["source_id"] == "mission:run-1:t1"
    assert seen["orchestration_run_id"] == "run-1" and seen["orchestration_task_id"] == "t1"
    assert seen["timeout_s"] == 600.0 and seen["title"] == "Research the market" and seen["tags"] == ["mission"]


def test_a_mission_lane_failure_is_an_error_result_and_the_session_still_closes(monkeypatch):
    from services import coordinator_service as cs
    import core.database.database as dbmod

    owner = next(v for v in vars(cs).values() if isinstance(v, type) and hasattr(v, "_run_cli_ticket"))
    closed = []

    class _Own:
        def close(self):
            closed.append(True)

    monkeypatch.setattr(dbmod, "SessionLocal", lambda: _Own())

    async def boom(db, **kw):
        raise RuntimeError("board unavailable")

    monkeypatch.setattr(lane, "run_cli_ticket_and_wait", boom)
    res = _run(owner._run_cli_ticket(SimpleNamespace(id="t1", title=None), "p", 15, WS, "run-1", None))
    assert res == {"status": "error", "error": "board unavailable"} and closed == [True]

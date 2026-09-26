"""F135 (night 4) — a switched-off agent never runs a step, a refused run says
why in words, and deleting an agent clears what still names it.

B67, B87: agent #303 was switched off at 14:08:11Z and ran a playbook step at
14:08:16Z; nothing warned at save or at Run. The save checks and the run's agent
check asked only whether the agent existed. B59: a fourth run came back as a
bare 429 carrying "Running executions (3) >= limit (3)". The cap stays: the run
queue is per process, so queueing past it would make the limit per worker.
Gerard holds the global queue. B66: once the playbook that used an agent was
deleted, deleting the agent answered 500, because tool_routing_edges and
tool_routing_affinities name it with no ON DELETE.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from tests.helpers_playbook_run import run_playbook

WS = UUID("00000000-0000-0000-0000-0000000000c1")
MONITOR = NS(id=303, name="Green Coffee Stock Monitor", status="inactive", workspace_id=WS)
SECRETARY = NS(id=7, name="CLUB SECRETARY", status="active", workspace_id=WS)
REFUSAL = "3 of 3 runs are going in this workspace, so this run was not started. Start it again when one finishes."


class _Rows:
    def __init__(self, first=None, all_rows=()):
        self._first, self._all = first, list(all_rows)

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._first

    def all(self):
        return list(self._all)


# ── a switched-off agent: refused at save, and a run using it never starts ──

def test_auto_cannot_put_a_switched_off_agent_on_a_step():
    from modules.tools.discovery import handlers_playbooks

    db = NS(query=lambda *entities: _Rows(first=MONITOR, all_rows=[SECRETARY]))
    reply = asyncio.run(handlers_playbooks.add_playbook_step(
        db, WS, {"playbook_id": 82, "prompt_template": "Check the green stock.", "agent_id": 303}))
    assert reply["success"] is False
    assert reply["error"] == ("agent_id 303 (Green Coffee Stock Monitor) is switched off (inactive). "
                              "Switch it on, or pick an active agent: [7=CLUB SECRETARY]")


def test_a_run_with_a_switched_off_agent_fails_before_any_step(monkeypatch):
    calls = []
    execution, _card = run_playbook(monkeypatch, outcomes=[], step_seconds=5, exec_config={}, calls=calls,
                                    agent_status="inactive")
    assert calls == []
    assert execution.status == "failed"
    assert execution.error_message.startswith("Switched off: #7 CLUB SECRETARY (inactive).")


def test_the_route_refuses_a_switched_off_agent_on_save():
    from fastapi import HTTPException

    from api.workflow_recipes import _check_step_agents

    db = NS(query=lambda *entities: _Rows(all_rows=[MONITOR]))
    with pytest.raises(HTTPException) as refused:
        _check_step_agents(db, WS, [{"order": 1, "agent_id": 303}])
    assert refused.value.status_code == 400
    assert refused.value.detail.startswith("Switched off: #303 Green Coffee Stock Monitor (inactive).")


# ── a refused run says why in words ─────────────────────────────────────────

class _GuardDb:
    def query(self, *entities):
        return _Rows(first=NS(plan_limits={}))

    def execute(self, statement, params=None):
        return NS(fetchall=lambda: [("running", 3)])


def test_the_run_limit_says_in_words_what_happened_and_what_to_do():
    from services.concurrency_guard import check_concurrency

    result = asyncio.run(check_concurrency(WS, _GuardDb()))
    assert result.allowed is False
    assert result.reason == "3 of 3 runs are going in this workspace"
    assert result.refusal == REFUSAL


def test_autos_run_tool_passes_the_words_on(monkeypatch):
    import services.concurrency_guard as guard
    from modules.tools.discovery import handlers_playbooks

    monkeypatch.setattr(guard, "check_concurrency", lambda ws, db: _refused())
    db = NS(query=lambda *entities: _Rows(first=NS(id=83, name="Friday café payment chase", steps=[{"order": 1}])))
    reply = asyncio.run(handlers_playbooks.execute_playbook(db, WS, {"playbook_id": 83}))
    assert reply["error"] == REFUSAL


async def _refused():
    from services.concurrency_guard import ConcurrencyResult

    return ConcurrencyResult(allowed=False, reason="3 of 3 runs are going in this workspace", current_running=3)


def test_every_429_the_routes_raise_carries_the_words():
    source = (Path(__file__).resolve().parents[1] / "api" / "workflow_recipes.py").read_text()
    assert source.count('"detail": concurrency.refusal') == 3
    assert '"detail": concurrency.reason' not in source


# ── deleting an agent clears the routing rows that name it ──────────────────

class _Savepoint:
    def commit(self):
        pass

    def rollback(self):
        pass


class _DeleteDb:
    def __init__(self, agent):
        self.agent, self.sql = agent, []

    def query(self, *entities):
        return _Rows(first=self.agent)

    def execute(self, statement, params=None):
        self.sql.append(" ".join(str(statement).split()))
        return NS(fetchall=lambda: [])

    def begin_nested(self):
        return _Savepoint()

    def delete(self, obj):
        self.sql.append(f"<delete agent {obj.id}>")

    def commit(self):
        pass

    def rollback(self):
        pass


def test_deleting_an_agent_clears_its_routing_rows_before_the_agent():
    from api.agents import delete_agent

    db = _DeleteDb(MONITOR)
    asyncio.run(delete_agent(303, NS(workspace_id=WS), db))
    gone = db.sql.index("<delete agent 303>")
    for table in ("tool_routing_edges", "tool_routing_affinities"):
        cleared = db.sql.index(f"DELETE FROM {table} WHERE agent_id = :agent_id")
        assert cleared < gone

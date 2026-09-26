"""Run Now leaves a mission's steps to the mission.

The mission engine runs its steps (PRD-171 F025). Run Now refused a step only
while it said in progress (F176), so an assigned or blocked step went back
through the board: reset to 'assigned', dispatched outside its mission. Run Now,
and a watch's re-run, now refuse every mission ticket, naming the mission and
where to retry it.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException
from sqlalchemy import text

GOAL = "Launch the spring menu"


@pytest.fixture
def mission(db_session, seed_workspace, monkeypatch):
    import api.board_tasks as bt
    from core.models.orchestration import OrchestrationRun, OrchestrationTask
    from services.orchestration_board_bridge import create_mission_board_task, create_task_board_task

    db = db_session
    ws = UUID(seed_workspace())
    agent = db.execute(
        text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
             "VALUES ('Content Creator', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"),
        {"w": str(ws)}).scalar()
    run = OrchestrationRun(workspace_id=ws, goal=GOAL, state="running", state_type="active", created_by="user_test")
    db.add(run)
    db.flush()
    step = OrchestrationTask(run_id=run.id, title="Write the menu copy", description="Four dishes",
                             sequence_number=1, agent_role="writer", state="assigned", state_type="active",
                             assigned_agent_id=agent)
    db.add(step)
    db.flush()
    parent = create_mission_board_task(db, run)
    child = create_task_board_task(db, run, step)
    woken, launched = [], []
    monkeypatch.setattr(bt, "notify_task_available", lambda db, **kw: woken.append(kw))
    monkeypatch.setattr(bt, "_launch_task_execution", lambda **kw: launched.append(kw["task_id"]))
    return NS(db=db, run=run, parent=parent, child=child, agent=agent, woken=woken, launched=launched,
              ctx=NS(workspace_id=ws, user=NS(id="owner@cafe.test", email="owner@cafe.test")))


def _run_now(mission, ticket):
    import api.board_tasks as bt

    with pytest.raises(HTTPException) as refused:
        asyncio.run(bt.run_task_now(ticket.id, ctx=mission.ctx, db=mission.db))
    return refused.value


@pytest.mark.parametrize("status", ["assigned", "blocked", "failed"])
def test_run_now_never_sends_a_missions_step_through_the_board(mission, status):
    mission.child.status = status
    mission.db.flush()

    refused = _run_now(mission, mission.child)

    mission.db.refresh(mission.child)
    assert mission.child.status == status and mission.woken == []
    assert refused.status_code == 409
    assert f"Ticket #{mission.child.id} is a step of the mission “{GOAL}”" in refused.detail
    assert f"(/missions/{mission.run.id})" in refused.detail


def test_run_now_points_the_missions_own_ticket_to_the_mission(mission):
    refused = _run_now(mission, mission.parent)
    assert refused.status_code == 409
    assert f"Ticket #{mission.parent.id} is the mission “{GOAL}”" in refused.detail


def test_a_watch_never_re_runs_a_missions_step(mission, monkeypatch):
    import services.watch_actions as wa
    import services.watch_service as ws_mod

    mission.child.status = "blocked"
    mission.db.flush()
    escalated = []

    async def _escalate(db, watch, *, reason):
        escalated.append(reason)

    monkeypatch.setattr(wa, "escalate_watch_now", _escalate)
    monkeypatch.setattr(ws_mod.WatchService, "record_action",
                        staticmethod(lambda *a, **k: pytest.fail("a refused re-run spends no budget")))
    watch = NS(id="w-1", workspace_id=mission.ctx.workspace_id, target_type="board_task",
               target_id=str(mission.child.id), actions_taken=0, action_budget=2)

    outcome = asyncio.run(wa.run_board_task_action(mission.db, watch, "rerun", diagnosis="below bar"))

    assert outcome.escalated is True and mission.woken == []
    assert escalated and f"(/missions/{mission.run.id})" in escalated[0]


# ── review HIGH: every other door ───────────────────────────────────────────

class _Request:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


def test_the_dispatcher_never_claims_a_missions_mirror(mission):
    """Whoever set it 'assigned' (a PATCH, a tool, a grant's re-queue), the board does not run it."""
    from core.models.core import BoardTask
    from services.board_dispatcher import claim_tasks

    mission.child.status = "assigned"
    ticket = BoardTask(workspace_id=mission.ctx.workspace_id, title="Reorder oat milk", priority="medium",
                       source_type="user", status="assigned", assigned_agent_id=mission.agent, attempts=0)
    mission.db.add(ticket)
    mission.db.flush()

    claimed = claim_tasks(mission.db, worker_id="w-1", limit=10, lease_seconds=60,
                          workspace_id=mission.ctx.workspace_id)

    assert [task.id for task in claimed] == [ticket.id]


@pytest.mark.parametrize("status", ["assigned", "in_progress"])
@pytest.mark.parametrize("door", ["tool", "status PATCH", "PATCH"])
def test_no_one_starts_a_missions_step_by_hand(mission, door, status):
    import api.board_tasks as bt
    from modules.tools.discovery.handlers_board_tasks import update_board_task_status

    mission.child.status = "blocked"
    mission.db.flush()
    if door == "tool":
        reply = asyncio.run(update_board_task_status(
            mission.db, mission.ctx.workspace_id, {"task_id": mission.child.id, "status": status}))
        assert reply["success"] is False and "the mission runs its steps" in reply["error"]
    else:
        route = bt.update_task_status if door == "status PATCH" else bt.update_task
        with pytest.raises(HTTPException) as refused:
            asyncio.run(route(mission.child.id, _Request({"status": status}), ctx=mission.ctx, db=mission.db))
        assert refused.value.status_code == 409 and "the mission runs its steps" in refused.value.detail

    mission.db.refresh(mission.child)
    assert mission.child.status == "blocked" and mission.launched == []


def test_run_now_leaves_a_cli_agents_mission_step_to_its_mission(mission):
    """A Claude Code agent's mission step is filed as a 'mission' ticket its host runs."""
    from core.models.core import BoardTask

    ticket = BoardTask(workspace_id=mission.ctx.workspace_id, title="Write the menu copy", priority="medium",
                       source_type="mission", source_id=f"mission:{mission.run.id}:1", status="blocked",
                       assigned_agent_id=mission.agent, attempts=0, orchestration_run_id=mission.run.id)
    mission.db.add(ticket)
    mission.db.flush()

    refused = _run_now(mission, ticket)

    assert refused.status_code == 409 and f"(/missions/{mission.run.id})" in refused.detail


def test_only_the_dispatch_loop_is_barred_from_a_mission_mirror():
    """A CLI agent's step runs on its host for the mission; its card may be the mirror (F094)."""
    from core.cli_runtime import RUNTIME_API, RUNTIME_CLI
    from services.board_dispatcher import mission_mirror_exclusion_sql

    assert mission_mirror_exclusion_sql(RUNTIME_API, "t") == "AND t.source_type NOT IN ('orchestration', 'orchestration_task')"
    # review LOW: a host may claim a step's mirror, never the mission's own ticket
    assert mission_mirror_exclusion_sql(RUNTIME_CLI, "t") == "AND t.source_type NOT IN ('orchestration')"

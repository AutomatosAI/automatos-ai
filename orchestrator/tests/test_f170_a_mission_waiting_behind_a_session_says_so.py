"""F170 (night 5, B52/B62) — a mission waiting behind another's Claude Code step says so.

Mission 81e8f37a was approved at 17:34:19 and shown running, its first steps
queued, and nothing was said until 18:08:02, seven seconds after mission
fb8e0ef4's Claude Code step #1046 ended. The coordinator tick awaits a Claude
Code step inline (F160, held), so no other mission's step is dispatched while
one runs. The wait is now named on the mission (a RUN_WAITING event the mission
page shows), in the chat it came from, and in Auto's reply to the approval. On
the real schema.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID

import pytest

from core.models import Agent
from core.models.orchestration import OrchestrationEvent, OrchestrationRun, OrchestrationTask
from core.models.orchestration_enums import RunState, TaskState
from services import cli_ticket_lane as lane
from services import coordinator_service as cs
from services.orchestration_board_bridge import create_mission_board_task, create_task_board_task

BLOCKING_STEP = "Synthesize all pending items for Monday morning"
BLOCKING_GOAL = "three loose ends from tonight, then one page for me to pick up on Monday"


@pytest.fixture
def narrated(monkeypatch):
    lines = []
    monkeypatch.setattr(cs, "_narrate_mission", lambda db, run, text, **kw: lines.append((str(run.id), text)))
    monkeypatch.setattr(lane, "host_online", lambda db, ws: True)
    monkeypatch.setattr(lane, "no_cli_host_reason_for", lambda db, ws, cli: None)
    monkeypatch.setattr(lane, "_notify", lambda *args, **kwargs: None)
    return lines


def _agent(db, ws, name):
    agent = Agent(name=name, agent_type="chatbot", description="", status="active", configuration={"runtime": "cli"},
                  model_config=None, workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws))
    db.add(agent)
    db.flush()
    return agent


def _session_running(db, ws):
    """Mission 5's shape: running, its step on a Claude Code session (#1046's card)."""
    agent = _agent(db, ws, "WRITER")
    run = OrchestrationRun(workspace_id=ws, goal=BLOCKING_GOAL, state=RunState.RUNNING.value, created_by="user_test",
                           config={})
    db.add(run)
    db.flush()
    task = OrchestrationTask(run_id=run.id, title=BLOCKING_STEP, description="Pull it together.", sequence_number=1,
                             state=TaskState.RUNNING.value, state_type="active", assigned_agent_id=agent.id)
    db.add(task)
    db.flush()
    create_mission_board_task(db, run)
    create_task_board_task(db, run, task)
    card = lane.file_cli_ticket(db, workspace_id=ws, agent_id=agent.id, title=task.title, prompt="Do it.",
                                source_type="mission", source_id=f"mission:{run.id}:{task.id}", tags=["mission"],
                                orchestration_run_id=run.id, orchestration_task_id=task.id)
    card.status, card.started_at = "in_progress", datetime(2026, 9, 25, 17, 39, 32, tzinfo=timezone.utc)
    db.flush()
    return run, task, card


def _awaiting_approval(db, ws):
    """81e8f37a's shape: a plan waiting for the owner, first step ready to queue."""
    run = OrchestrationRun(workspace_id=ws, goal="the big one: the Christmas box launch",
                           state=RunState.AWAITING_APPROVAL.value, created_by="user_test", config={})
    db.add(run)
    db.flush()
    db.add(OrchestrationTask(run_id=run.id, title="Split the box for the club", description="Split it.",
                             sequence_number=1, state=TaskState.PENDING.value, state_type="initial"))
    db.flush()
    return run


def _waiting_events(db, run):
    return [e.payload for e in db.query(OrchestrationEvent).filter(
        OrchestrationEvent.run_id == run.id, OrchestrationEvent.event_type == "run_waiting")]


def test_an_approved_mission_behind_a_claude_code_step_says_what_it_waits_for(db_session, seed_workspace, narrated):
    ws = UUID(seed_workspace())
    blocking, _task, card = _session_running(db_session, ws)
    waiting = _awaiting_approval(db_session, ws)

    cs.CoordinatorService().approve_plan(db_session, waiting.id, "user_test")

    (event,) = _waiting_events(db_session, waiting)                     # old: nothing until the step ran
    said = event["stop_detail"]
    assert said.startswith("Approved. It probably starts when")         # inferred: no record in this process
    assert f"'{BLOCKING_STEP}' of mission '{BLOCKING_GOAL}'" in said and f"ticket #{card.id}" in said
    assert "since 17:39 UTC" in said
    assert (str(waiting.id), said) in narrated                          # the chat it came from hears it too


def test_the_tick_record_names_the_exact_step_and_the_reply_says_it(db_session, seed_workspace, narrated):
    from modules.tools.discovery.handlers_missions import approve_mission
    from services.mission_wait import awaiting_session_step

    ws = UUID(seed_workspace())
    blocking, task, card = _session_running(db_session, ws)
    waiting = _awaiting_approval(db_session, ws)

    with awaiting_session_step(run_id=blocking.id, task_id=task.id, workspace_id=ws, title=task.title) as saw:
        saw(card)
        reply = asyncio.run(approve_mission(db_session, ws, {"mission_id": str(waiting.id), "_created_by": "user_test"}))

    (event,) = _waiting_events(db_session, waiting)
    assert event["stop_detail"].startswith(f"Approved. It starts when '{BLOCKING_STEP}'")   # certain: no "probably"
    assert f"ticket #{card.id}" in event["stop_detail"]
    assert reply["success"] and event["stop_detail"] in reply["message"] and reply["waiting"] == event["stop_detail"]


def test_another_workspaces_step_is_only_another_mission(db_session, seed_workspace, narrated):
    from services.mission_wait import awaiting_session_step

    elsewhere = UUID(seed_workspace())
    blocking, task, card = _session_running(db_session, elsewhere)
    ws = UUID(seed_workspace())
    waiting = _awaiting_approval(db_session, ws)

    with awaiting_session_step(run_id=blocking.id, task_id=task.id, workspace_id=elsewhere, title=task.title) as saw:
        saw(card)
        cs.CoordinatorService().approve_plan(db_session, waiting.id, "user_test")

    (event,) = _waiting_events(db_session, waiting)
    assert event["stop_detail"] == ("Approved. It starts when another mission's step finishes: while a Claude Code "
                                    "step runs, other missions' steps wait.")
    assert event["ticket_id"] is None


def test_nothing_is_said_when_nothing_holds_the_tick(db_session, seed_workspace, narrated):
    ws = UUID(seed_workspace())
    waiting = _awaiting_approval(db_session, ws)
    cs.CoordinatorService().approve_plan(db_session, waiting.id, "user_test")
    assert _waiting_events(db_session, waiting) == [] and not any("starts when" in text for _, text in narrated)


def test_a_wait_that_begins_after_approval_is_said_once(db_session, seed_workspace, narrated):
    from services.task_reconciler import TaskReconciler

    ws = UUID(seed_workspace())
    waiting = _awaiting_approval(db_session, ws)
    cs.CoordinatorService().approve_plan(db_session, waiting.id, "user_test")   # nothing held the tick yet
    _session_running(db_session, ws)                                            # then a session step starts
    later = datetime.now(timezone.utc) + timedelta(minutes=10)

    TaskReconciler._note_waiting_missions(db_session, later, 300)               # the reconciler's own job
    TaskReconciler._note_waiting_missions(db_session, later + timedelta(minutes=1), 300)

    (event,) = _waiting_events(db_session, waiting)                            # one event per wait
    assert event["stop_detail"].startswith("Waiting: this mission's next steps probably start when")
    assert [text for run_id, text in narrated if run_id == str(waiting.id) and "Waiting:" in text] == [
        event["stop_detail"]]


def test_the_tick_awaits_the_session_exactly_as_before_and_the_record_is_only_a_record(monkeypatch):
    """(3): the lane gets the same call (plus on_poll) and its answer comes back
    unchanged; the record exists for the wait's length and is gone after, also
    when the lane raises."""
    from services.mission_wait import awaited_steps
    import core.database.database as database

    class _Own:
        closed = False

        def close(self):
            self.closed = True

    own = _Own()
    monkeypatch.setattr(database, "SessionLocal", lambda: own)
    seen = {}

    async def _lane(db, **kwargs):
        seen.update(kwargs, during=awaited_steps())
        kwargs["on_poll"](SimpleNamespace(id=1046))
        seen["after_poll"] = awaited_steps()
        return {"status": "success", "result": "the page"}

    monkeypatch.setattr(lane, "run_cli_ticket_and_wait", _lane)
    task = SimpleNamespace(id="t-1046", title=BLOCKING_STEP)
    result = asyncio.run(cs.CoordinatorService._run_cli_ticket(task, "Do it.", 272, "ws-1", "run-5", 600, 3600))

    assert result == {"status": "success", "result": "the page"} and own.closed and awaited_steps() == []
    assert [(s.run_id, s.title, s.ticket_id) for s in seen["during"]] == [("run-5", BLOCKING_STEP, None)]
    assert [s.ticket_id for s in seen["after_poll"]] == [1046]
    assert (seen["source_id"], seen["timeout_s"], seen["hard_timeout_s"]) == ("mission:run-5:t-1046", 600.0, 3600.0)

    async def _boom(db, **kwargs):
        raise RuntimeError("board unavailable")

    monkeypatch.setattr(lane, "run_cli_ticket_and_wait", _boom)
    failed = asyncio.run(cs.CoordinatorService._run_cli_ticket(task, "Do it.", 272, "ws-1", "run-5", 600, 3600))
    assert failed == {"status": "error", "error": "board unavailable"} and awaited_steps() == []

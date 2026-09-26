"""F094 follow-ups (TESTER, 09-26) — a step's card ends on its session's outcome.

Once the session lane claims a mission step's card, the lane alone writes its
status. The code review worried that a card could then stay claimed after its
session ended: the retry and the stall recovery clear the step's agent while the
session may still be working. Whatever the mission does meanwhile — retries the
step, recovers it from a stall, stops waiting for it, or is cancelled — the
session's result lands on the card and the card ends on that outcome. When the
mission stopped waiting or was cancelled, the card also says so, in a note, and
its status stays the session's. On the real schema, through the host's own
claim and result.
"""
from __future__ import annotations

import asyncio
from uuid import UUID

import pytest

from core.models import Agent
from core.models.orchestration import OrchestrationRun, OrchestrationTask
from core.models.orchestration_enums import RunState, TaskState
from modules.coordination.dispatcher import MissionDispatcher
from modules.coordination.reconciler import MissionReconciler
from services import cli_host_service as svc
from services import cli_ticket_lane as lane
from services import coordinator_service as cs
from services.orchestration_board_bridge import create_mission_board_task, create_task_board_task

STOPPED_WAITING = ("The mission stopped waiting for this step after 60 minutes. The session is still working, "
                   "and its result will land here.")
CANCELLED = ("The mission was cancelled while this step ran. The session is still working, and its result "
             "will land here.")


@pytest.fixture
def quiet(monkeypatch):
    """The board's fan-out (approvals, notices, reports, the Canvas) is its own suites' business."""
    import api.board_tasks as bt

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(bt, "_board_task_blocked_pending_approval", lambda *a, **k: False)
    monkeypatch.setattr(bt, "_dispatch_task_complete", _noop)
    monkeypatch.setattr(bt, "_dispatch_task_failed", _noop)
    monkeypatch.setattr(bt, "_auto_create_task_report", _noop)
    published = []
    monkeypatch.setattr(svc, "publish_canvas_events", lambda ws, events: published.extend(events))
    monkeypatch.setattr(lane, "_notify", lambda *args, **kwargs: None)
    monkeypatch.setattr(cs, "_narrate_mission", lambda *args, **kwargs: None)
    return published


def _session_working(db, ws):
    """A mission step on a Claude Code agent, its card claimed by the lane and by a host."""
    agent = Agent(name="NEWSROOM", agent_type="chatbot", description="", status="active",
                  configuration={"runtime": "cli", "provider": "claude", "model": "sonnet"}, model_config=None,
                  workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws))
    db.add(agent)
    db.flush()
    run = OrchestrationRun(workspace_id=ws, goal="A Christmas box offer for the cafés", state=RunState.RUNNING.value,
                           created_by="user_test", config={})
    db.add(run)
    db.flush()
    task = OrchestrationTask(run_id=run.id, title="Synthesize Christmas Box Offer Details", description="Do it.",
                             sequence_number=2, state=TaskState.RUNNING.value, state_type="active",
                             assigned_agent_id=agent.id, max_retries=3)
    db.add(task)
    db.flush()
    create_mission_board_task(db, run)
    create_task_board_task(db, run, task)
    card = lane.file_cli_ticket(db, workspace_id=ws, agent_id=agent.id, title=task.title, prompt="Work on it.",
                                source_type="mission", source_id=f"mission:{run.id}:{task.id}", tags=["mission"],
                                orchestration_run_id=run.id, orchestration_task_id=task.id)
    host, code, _ = svc.create_pairing_code(db, ws, "laptop")
    host, _token = svc.pair_host(db, code)
    (ticket,) = svc.claim_for_host(db, host, 1)["tasks"]
    assert ticket["task_id"] == card.id and card.status == "in_progress"
    return run, task, card, host, ticket


def _retry(db, run, task):
    MissionDispatcher.record_task_completion(db, task, {"status": "error", "error": "the lane lost the database"})


def _stall_recovery(db, run, task):
    task.state = TaskState.STALLED.value
    db.flush()
    asyncio.run(MissionReconciler._recover_stalled_task(db, task))


def _stopped_waiting(db, run, task):
    MissionDispatcher.record_task_completion(db, task, {
        "status": "error", "timed_out": True, "still_running": True, "waited_s": 3600,
        "error": "ticket is still running after 3600 s — the Claude Code session carries on"})


def _cancelled(db, run, task):
    cs.CoordinatorService().cancel_mission(db, run.id, "user_test")


@pytest.mark.parametrize("what_the_mission_did, note", [
    (_retry, None), (_stall_recovery, None), (_stopped_waiting, STOPPED_WAITING), (_cancelled, CANCELLED),
], ids=["retry", "stall-recovery", "stopped-waiting", "cancelled"])
def test_no_card_the_lane_runs_stays_claimed_after_its_session_ends(db_session, seed_workspace, quiet,
                                                                     what_the_mission_did, note):
    ws = UUID(seed_workspace())
    run, task, card, host, ticket = _session_working(db_session, ws)

    what_the_mission_did(db_session, run, task)            # the mission moves on while the session works
    db_session.refresh(card)
    assert card.status == "in_progress"                    # the card is still the session's

    out = asyncio.run(svc.apply_result(db_session, host, card.id, {
        "attempt": ticket["attempt"], "status": "success", "result_text": "The offer, drafted.",
        "usage": {"input_tokens": 10, "output_tokens": 5}}))
    db_session.refresh(card)

    assert out["applied"] is True
    assert (card.status, card.result) == ("done", "The offer, drafted.")   # it ends on the session's outcome
    notes = [(n["by"], n["note"]) for n in card.runtime_ref.get("session_notes") or []]
    assert notes == ([("the mission", note)] if note else [])
    assert [e["data"]["note"] for e in quiet if e["data"].get("note")] == ([note] if note else [])   # live, too


def test_the_notes_on_a_card_survive_the_claim_that_resumes_its_run(db_session, seed_workspace, quiet):
    """A session that parked on a question is claimed again to resume the same
    run; its card's notes, the mission's verdict among them, are kept."""
    ws = UUID(seed_workspace())
    run, task, card, host, ticket = _session_working(db_session, ws)
    _stopped_waiting(db_session, run, task)
    db_session.refresh(card)
    card.status = "assigned"                               # the answer re-queued it
    db_session.flush()

    (again,) = svc.claim_for_host(db_session, host, 1)["tasks"]
    db_session.refresh(card)

    assert again["task_id"] == card.id
    assert [n["note"] for n in card.runtime_ref["session_notes"]] == [STOPPED_WAITING]


def test_a_note_the_database_refuses_never_costs_the_mission_its_cancel(db_session, seed_workspace, quiet,
                                                                       monkeypatch):
    """The note is written inside the transaction that holds the cancel. A note
    statement that fails there must not abort that transaction (code review of
    75e084cbf): each note has its own savepoint."""
    from sqlalchemy import text

    ws = UUID(seed_workspace())
    run, task, card, host, ticket = _session_working(db_session, ws)

    def _refused(db, **kwargs):
        db.execute(text("SELECT 1 / 0"))                   # the note's statement fails in Postgres

    monkeypatch.setattr(svc, "append_session_note", _refused)
    cs.CoordinatorService().cancel_mission(db_session, run.id, "user_test")
    db_session.flush()
    db_session.refresh(run)                                # old: the transaction was aborted here

    assert run.state == RunState.CANCELLED.value

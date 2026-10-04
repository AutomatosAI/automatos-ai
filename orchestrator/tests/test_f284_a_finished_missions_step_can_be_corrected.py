"""F284 (night 8) — a step of a finished mission can be corrected; its mission opens again.

#0305.2 and #0408.1 (wrong margins: 53.5% for 31.6%, 82.35% for 62.35%) were sent back
after their missions completed and refused: "…which has finished: a mission only runs
its steps while it runs. Re-run the mission from its page". The wrong numbers stayed
on the mission cards. #0250.1, sent back 2 s after its mission completed, got through
and stuck "in progress" for ever.
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from core.models.orchestration_enums import RunState, TaskState
from tests import test_f116_a_cancelled_run_stops_its_sessions as f116
from tests.test_f245_cancel_stops_a_mission import _mission

# F116's Postgres and workspace, as fixtures of this module too (the race needs two sessions).
engine = f116.engine
workspace = f116.workspace

WRONG = "Margin: £16.47 on £20.00, 82.35%."
RIGHT = "Margin: £12.47 on £20.00, 62.35%."
NOTE = "The profit has to come off the price without VAT: 62.35%, not 82.35%."


def _owner(ws):
    return NS(workspace_id=uuid.UUID(str(ws)), user_id="2", auth_type="anonymous", user=NS(id="2"))


def _body(payload):
    async def _json():
        return payload
    return NS(json=_json)


def _finished(db, ws, state=RunState.COMPLETED):
    """A mission that ended, its card done with the wrong margin, and its verified step."""
    from datetime import datetime, timezone

    run, card, steps = _mission(db, ws, state=state)
    run.started_at = run.completed_at = datetime.now(timezone.utc)
    run.output_summary = {"tasks_completed": 1}
    run.config = {"progress_ledger": {"rounds": 3}, "output_document_id": 41}
    card.status, card.result = ("done" if state == RunState.COMPLETED else "failed"), WRONG
    task, step_card = steps[TaskState.VERIFIED]
    task.output = step_card.result = WRONG
    db.flush()
    return run, card, task, step_card


def test_a_step_sent_back_after_its_mission_completed_opens_the_mission_again(db_session, seed_workspace):
    from api.board_tasks import reject_task
    from core.models.orchestration import OrchestrationEvent

    ws = UUID(seed_workspace())
    run, card, task, step_card = _finished(db_session, ws)

    asyncio.run(reject_task(step_card.id, _body({"feedback": NOTE}), ctx=_owner(ws), db=db_session))

    for row in (run, card, task, step_card):
        db_session.refresh(row)
    assert run.state == RunState.RUNNING.value                               # night: 409 "…has finished"
    assert (run.completed_at, run.output_summary) == (None, None)
    assert "progress_ledger" not in run.config and "output_document_id" not in run.config
    assert task.state == TaskState.RETRYING.value and step_card.status == "in_progress"
    assert NOTE in task.input_context["verification_feedback"]["reasoning"]
    assert task.input_context["previous_output"] == WRONG
    assert (card.status, card.result) == ("in_progress", None)               # the wrong margin is off the card
    assert card.planning_data["previous_runs"][-1]["result"] == WRONG        # and kept in its history
    resumed = db_session.query(OrchestrationEvent).filter(OrchestrationEvent.run_id == run.id,
                                                          OrchestrationEvent.event_type == "run_resumed").count()
    assert resumed == 1


def test_the_mission_completes_again_with_the_corrected_result_on_its_card(db_session, seed_workspace):
    from api.board_tasks import reject_task
    from core.models.orchestration_enums import ActorType
    from services.orchestration_state import transition_run

    ws = UUID(seed_workspace())
    run, card, task, step_card = _finished(db_session, ws)
    asyncio.run(reject_task(step_card.id, _body({"feedback": NOTE}), ctx=_owner(ws), db=db_session))
    db_session.refresh(task)

    task.output, task.state = RIGHT, TaskState.VERIFIED.value                # the redo, verified
    db_session.flush()
    for state in (RunState.VERIFYING, RunState.COMPLETED):
        transition_run(db=db_session, run=run, new_state=state, actor_type=ActorType.COORDINATOR,
                       actor_id="coordinator", reason="All tasks verified")

    db_session.refresh(card)
    assert (card.status, card.result) == ("done", RIGHT)                     # night: 82.35% stayed for good


def test_a_step_of_a_failed_mission_reopens_it_as_a_retry(db_session, seed_workspace):
    from api.board_tasks import reject_task

    ws = UUID(seed_workspace())
    run, card, task, step_card = _finished(db_session, ws, state=RunState.FAILED)
    card.error_message = "Tasks failed: Step 3"
    db_session.flush()

    asyncio.run(reject_task(step_card.id, _body({"feedback": NOTE}), ctx=_owner(ws), db=db_session))

    for row in (run, card, task):
        db_session.refresh(row)
    assert run.state == RunState.RUNNING.value and task.state == TaskState.RETRYING.value
    assert (card.status, card.error_message) == ("in_progress", None)


def test_a_step_of_a_cancelled_mission_is_refused_naming_a_button_that_exists(db_session, seed_workspace):
    from api.board_tasks import reject_task

    ws = UUID(seed_workspace())
    run, _card, steps = _mission(db_session, ws, state=RunState.CANCELLED)
    task, step_card = steps[TaskState.VERIFIED]
    db_session.flush()

    with pytest.raises(HTTPException) as refused:
        asyncio.run(reject_task(step_card.id, _body({"feedback": NOTE}), ctx=_owner(ws), db=db_session))

    assert refused.value.status_code == 409 and f"/missions/{run.id}" in refused.value.detail
    assert "Re-run on the mission's page" in refused.value.detail            # the page has Re-run when it ended
    assert "Re-run the mission from its page" not in refused.value.detail
    db_session.refresh(task)
    assert task.state == TaskState.VERIFIED.value


# --- #0250.1: the send-back and the mission's end, two transactions ---------------------


def _committed_mission(new_session, ws, state=RunState.RUNNING):
    """A mission whose two steps passed, committed, with its cards."""
    from datetime import datetime, timezone

    from core.models.orchestration import OrchestrationRun, OrchestrationTask
    from services.orchestration_board_bridge import create_mission_board_task, create_task_board_task

    s = new_session()
    run = OrchestrationRun(workspace_id=uuid.UUID(ws), goal="Price the Christmas box", state=state.value,
                           created_by="user_test", config={}, started_at=datetime.now(timezone.utc))
    s.add(run)
    s.flush()
    create_mission_board_task(s, run)
    cards = []
    for n in (1, 2):
        step = OrchestrationTask(run_id=run.id, title=f"Step {n}", description="Do it.", sequence_number=n,
                                 state=TaskState.VERIFIED.value, state_type="blocked", max_retries=3, output=WRONG)
        s.add(step)
        s.flush()
        cards.append(create_task_board_task(s, run, step))
    for card in cards:
        card.status, card.review_feedback = "in_progress", NOTE           # as send_back leaves it for the redo
    s.commit()
    return NS(run=run.id, card=cards[0].id, step=cards[0].orchestration_task_id)


def _state(new_session, table, row_id):
    return new_session().execute(text(f"SELECT state FROM {table} WHERE id = :i"), {"i": row_id}).scalar()


def _hold_the_mission(new_session, run_id):
    """The coordinator's tick, ending the mission: it holds the mission's row."""
    tick = new_session()
    tick.execute(text("SELECT id FROM orchestration_runs WHERE id = :i FOR UPDATE"), {"i": run_id})
    return tick


def _complete(tick, run_id):
    tick.execute(text("UPDATE orchestration_runs SET state = 'completed', state_type = 'terminal', "
                      "completed_at = now(), version_id = version_id + 1 WHERE id = :i"), {"i": run_id})
    tick.commit()


def _send_back(new_session, mission):
    from core.models.core import BoardTask
    from services.run_redo import start_redo

    mine = new_session()
    return start_redo(mine, mine.get(BoardTask, mission.card), by="user:2")


def test_a_send_back_as_the_mission_completes_waits_and_opens_it_again(workspace, new_session):
    mission = _committed_mission(new_session, workspace)
    tick = _hold_the_mission(new_session, mission.run)
    outcome = []
    redo = threading.Thread(target=lambda: outcome.append(_send_back(new_session, mission)))
    redo.start()
    time.sleep(0.3)
    _complete(tick, mission.run)                                             # the mission ends under the send-back
    redo.join(timeout=10)

    assert outcome and "went back to its mission" in outcome[0]
    assert _state(new_session, "orchestration_runs", mission.run) == RunState.RUNNING.value
    assert _state(new_session, "orchestration_tasks", mission.step) == TaskState.RETRYING.value  # night: stuck


def test_a_send_back_while_the_mission_is_held_too_long_changes_nothing(workspace, new_session, monkeypatch):
    from services import mission_reopen
    from services.run_redo import RedoRefused

    monkeypatch.setattr(mission_reopen, "LOCK_WAIT", "200ms")
    mission = _committed_mission(new_session, workspace)
    tick = _hold_the_mission(new_session, mission.run)

    with pytest.raises(RedoRefused) as refused:
        _send_back(new_session, mission)

    assert "nothing was changed" in str(refused.value) and "again in a moment" in str(refused.value)
    tick.rollback()
    assert _state(new_session, "orchestration_tasks", mission.step) == TaskState.VERIFIED.value


def test_a_tick_that_read_the_mission_before_the_send_back_does_not_end_it(workspace, new_session):
    from core.models.orchestration import OrchestrationRun, OrchestrationTask
    from modules.coordination.reconciler import MissionReconciler

    mission = _committed_mission(new_session, workspace)
    tick = new_session()
    run = tick.get(OrchestrationRun, mission.run)
    steps = tick.query(OrchestrationTask).filter(OrchestrationTask.run_id == mission.run).all()  # all passed
    _send_back(new_session, mission)                                         # lands before the tick decides

    out = MissionReconciler._advance_run_on_completion(
        db=tick, run=run, all_tasks=steps, failed_tasks=[], stalls_detected=0, stalls_recovered=0, tasks_failed=0)
    tick.rollback()

    assert not out.run_advanced
    assert _state(new_session, "orchestration_runs", mission.run) == RunState.RUNNING.value

"""F245 (night 7) — cancelling a mission stops it, and its cards end Cancelled.

#0119 was cancelled from its page while #0119.2 and #0119.3 ran: they kept
going (14 model calls after the cancel), finished their drafts and stayed In
progress, while the mission's card and its five unstarted steps went to Done.
#0098 was cancelled on the board while its step ran: the step finished, the
mission said "All tasks verified successfully" and the card went from
Cancelled to Done five seconds later.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException

from core.models.core import BoardTask
from core.models.orchestration import OrchestrationRun, OrchestrationTask
from core.models.orchestration_enums import RunState, TaskState
from services import coordinator_service as cs
from services.orchestration_board_bridge import create_mission_board_task, create_task_board_task
from tests.helpers_mission_lane import quiet_board, session_working as _session_working

STATES = (TaskState.PENDING, TaskState.QUEUED, TaskState.RUNNING, TaskState.COMPLETED, TaskState.VERIFIED)


@pytest.fixture
def quiet(monkeypatch):
    """The board's fan-out is its own suites' business (tests/helpers_mission_lane.py)."""
    return quiet_board(monkeypatch)


@pytest.fixture(autouse=True)
def no_narration(monkeypatch):
    """The launching thread's lines are their own suite's business."""
    monkeypatch.setattr(cs, "_narrate_run_terminal", lambda *a, **k: None)
    monkeypatch.setattr(cs, "_narrate_mission", lambda *a, **k: None)


def _owner(ws):
    return NS(workspace_id=ws, user_id="2", auth_type="anonymous", user=NS(id="2"))


def _mission(db, ws, state=RunState.RUNNING):
    run = OrchestrationRun(workspace_id=ws, goal="Get the Christmas gift box ready for pre-orders",
                           state=state.value, created_by="user_test", config={})
    db.add(run)
    db.flush()
    card = create_mission_board_task(db, run)
    steps = {}
    for n, step_state in enumerate(STATES, start=1):
        task = OrchestrationTask(run_id=run.id, title=f"Step {n}: {step_state.value}", description="Do it.",
                                 sequence_number=n, state=step_state.value, max_retries=3)
        db.add(task)
        db.flush()
        steps[step_state] = (task, create_task_board_task(db, run, task))
    db.flush()
    return run, card, steps


def _card(db, card):
    db.expire(card)
    return db.get(BoardTask, card.id)


def test_a_cancelled_mission_cancels_every_unfinished_step(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    run, card, steps = _mission(db_session, ws)

    cs.CoordinatorService().cancel_mission(db_session, run.id, "user_test")

    assert _card(db_session, card).status == "cancelled"                # night: Done
    for state in (TaskState.PENDING, TaskState.QUEUED, TaskState.RUNNING, TaskState.COMPLETED):
        task, step_card = steps[state]
        db_session.refresh(task)
        step_card = _card(db_session, step_card)
        assert (task.state, step_card.status) == (TaskState.SKIPPED.value, "cancelled"), state
        assert step_card.runtime_ref["cancelled"]["by"] == "user:user_test"
    verified_task, verified_card = steps[TaskState.VERIFIED]
    assert _card(db_session, verified_card).status == "done"            # finished work stays done


def test_a_missions_cancel_is_committed_by_its_caller_as_one(db_session, seed_workspace, monkeypatch):
    """Review of #885: committed card by card, a cancel that failed half way left
    the mission terminal (so it could not be cancelled again) and its other cards
    open. The mission and every card now commit together, or not at all."""
    ws = UUID(seed_workspace())
    run, _card_row, _steps = _mission(db_session, ws)
    commits = []
    monkeypatch.setattr(db_session, "commit", lambda: commits.append(True))

    cs.CoordinatorService().cancel_mission(db_session, run.id, "user_test")

    assert commits == []


def test_a_step_result_that_arrives_after_the_cancel_is_not_recorded(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    run, _card_row, steps = _mission(db_session, ws)
    running, running_card = steps[TaskState.RUNNING]
    cs.CoordinatorService().cancel_mission(db_session, run.id, "user_test")

    asyncio.run(cs.CoordinatorService()._record_task_result(
        db_session, run, running, 7, {"status": "success", "result": "The gift box page, drafted."}))

    db_session.refresh(running)
    assert running.state == TaskState.SKIPPED.value                     # night: In progress, then a draft
    assert (_card(db_session, running_card).status, _card(db_session, running_card).result) == ("cancelled", None)


def test_a_running_step_stops_when_its_mission_is_cancelled(monkeypatch):
    from modules.coordination import mission_cancel

    stopped = []

    async def _drafting():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            stopped.append(True)
            raise

    reads = iter([False, True])
    monkeypatch.setattr(mission_cancel, "mission_cancelled", lambda run_id: next(reads))

    out = asyncio.run(mission_cancel.until_mission_cancelled(_drafting(), uuid.uuid4(), poll_seconds=0.01))

    assert out == mission_cancel.cancelled_step_result() and stopped == [True]   # the model call is dropped


def test_a_step_that_finishes_first_keeps_its_result(monkeypatch):
    from modules.coordination import mission_cancel

    async def _quick():
        return {"status": "success", "result": "Done."}

    monkeypatch.setattr(mission_cancel, "mission_cancelled", lambda run_id: False)
    out = asyncio.run(mission_cancel.until_mission_cancelled(_quick(), uuid.uuid4(), poll_seconds=0.01))

    assert out == {"status": "success", "result": "Done."}


def test_cancel_on_a_missions_card_cancels_the_mission(db_session, seed_workspace):
    from api.board_tasks import cancel_task

    ws = UUID(seed_workspace())
    run, card, _steps = _mission(db_session, ws)

    out = asyncio.run(cancel_task(card.id, ctx=_owner(ws), db=db_session))

    db_session.refresh(run)
    assert run.state == RunState.CANCELLED.value                        # night (#0098): the mission ran on
    assert (out["status"], _card(db_session, card).status) == ("cancelled", "cancelled")


def test_a_step_of_a_live_mission_is_the_missions_to_cancel(db_session, seed_workspace):
    from api.board_tasks import cancel_task

    ws = UUID(seed_workspace())
    run, _card_row, steps = _mission(db_session, ws)
    _task, queued_card = steps[TaskState.QUEUED]

    with pytest.raises(HTTPException) as refused:
        asyncio.run(cancel_task(queued_card.id, ctx=_owner(ws), db=db_session))

    assert refused.value.status_code == 409 and f"/missions/{run.id}" in refused.value.detail
    assert _card(db_session, queued_card).status == "inbox"


def test_a_step_left_open_by_an_ended_mission_can_be_cancelled(db_session, seed_workspace):
    """#0119.3 and #0176.9-.12: the board is where they are tidied away."""
    from api.board_tasks import cancel_task

    ws = UUID(seed_workspace())
    _run_row, _card_row, steps = _mission(db_session, ws, state=RunState.FAILED)
    _task, queued_card = steps[TaskState.QUEUED]

    out = asyncio.run(cancel_task(queued_card.id, ctx=_owner(ws), db=db_session))

    assert (out["applied"], out["status"]) == (True, "cancelled")


def test_a_claude_code_step_stops_with_its_mission(db_session, seed_workspace, quiet):
    """Its card is cancelled, so the host's next event batch tells the session to
    stop, and the session's late result never reopens the card."""
    from services import cli_host_service as svc

    ws = UUID(seed_workspace())
    run, _task, card, host, ticket = _session_working(db_session, ws)

    cs.CoordinatorService().cancel_mission(db_session, run.id, "user_test")
    db_session.refresh(card)
    assert card.status == "cancelled" and card.runtime_ref.get("cancel_requested_at")

    out = asyncio.run(svc.apply_result(db_session, host, card.id, {
        "attempt": ticket["attempt"], "status": "success", "result_text": "The offer, drafted.",
        "usage": {"input_tokens": 10, "output_tokens": 5}}))
    db_session.refresh(card)
    assert out.get("applied") is not True and card.status == "cancelled"


def test_the_board_names_a_cancelled_mission_and_a_skipped_step_cancelled():
    from services.orchestration_board_bridge import _RUN_STATE_TO_BOARD_STATUS, _resolve_board_status

    assert _RUN_STATE_TO_BOARD_STATUS[RunState.CANCELLED.value] == "cancelled"
    assert _resolve_board_status(TaskState.SKIPPED) == "cancelled"      # a step that never ran

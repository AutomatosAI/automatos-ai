"""F308 (night 9): approving or sending back a mission's step through Auto is the board's Approve or Reject.

- "Step 1 of mission 27 (the green check) is fine - approve it with the note: Guji 118 kg
  is plenty, carry on." → platform_approve_mission {mission_id: 27}. The run went from
  paused to running, but card #1875 stayed in Review and steps 2-3 stayed queued until the
  owner pressed Approve on the board.
- "Reject it, then — send it back with: I need the three cafés and their kilos, or a plain
  line saying you can't. Don't guess column names." → platform_reject_mission {mission_id:
  35, reason: "Agent thinking aloud, …"}. #0035 was cancelled, step 2 skipped, #1888 left
  in Review; Auto said a cancelled mission can't be brought back.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from core.models.orchestration_enums import RunState, TaskState
from tests.test_f242_wait_for_me_holds import _held
from tests.test_f242_wait_for_me_holds import quiet as _quiet
from tests.test_f245_cancel_stops_a_mission import _mission

quiet = _quiet  # the board's fan-out on an approval is its own suites' business

GUJI_NOTE = "Guji 118 kg is plenty, carry on. Then let the intro step start."
APPROVE_STEP_1 = f"Step 1 of mission 27 (the green check) is fine - approve it with the note: {GUJI_NOTE}"
CAFES = "I need the three cafés and their kilos, or a plain line saying you can't. Don't guess column names."
SEND_IT_BACK = f"Reject it, then — send it back with: {CAFES}"
AUTOS_REASON = ("Agent thinking aloud, not providing required output. Needs to return the three cafés and their "
                "kilos, or explicitly state it can't, without guessing column names.")
CHAT = uuid4()
OWNER = "owner@local"


@pytest.fixture
def said(monkeypatch):
    """The owner's words in the chat the turn runs in (found through its usage scope)."""
    import modules.tools.discovery.handlers_board_task_review as review

    words = []
    monkeypatch.setattr(review, "owner_words", lambda db, ws, chat_id: list(words) if chat_id == CHAT else [])
    return words


def _in_the_chat(handler, db, ws, **params):
    from core.llm.usage_context import LANE_CHAT, usage_scope

    with usage_scope(request_type=LANE_CHAT, execution_id=f"chat:{CHAT}"):
        return asyncio.run(handler(db, ws, {"_created_by": OWNER, **params}))


def _number(card):
    return int(card.workspace_seq)                                  # night 9: Auto sent 27 and 35


def _reload(db, *rows):
    for row in rows:
        db.refresh(row)


def test_approving_step_1_through_auto_is_the_boards_approve(db_session, seed_workspace, quiet, said):
    from modules.tools.discovery.handlers_missions import approve_mission

    ws = UUID(seed_workspace())
    run, card, task, step_card = _held(db_session, ws, config={"check_each_step": True})
    said.append(APPROVE_STEP_1)

    out = _in_the_chat(approve_mission, db_session, ws, mission_id=_number(card))

    _reload(db_session, run, task, step_card)
    assert out["success"] is True and "approved" in out["message"]
    assert (task.state, step_card.status) == (TaskState.VERIFIED.value, "done")   # night 9: #1875 stayed in Review
    assert run.state == RunState.RUNNING.value                                     # the mission carries on
    assert GUJI_NOTE in step_card.runtime_ref["session_notes"][-1]["note"]          # the owner's words, on the card


def test_sending_step_1_back_through_auto_redoes_it_and_never_cancels(db_session, seed_workspace, said):
    from modules.tools.discovery.handlers_missions import reject_mission

    ws = UUID(seed_workspace())
    run, card, task, step_card = _held(db_session, ws, config={"check_each_step": True})
    pending, _pending_card = _steps_of(db_session, run)[0]
    said.append(SEND_IT_BACK)

    out = _in_the_chat(reject_mission, db_session, ws, mission_id=_number(card), reason=AUTOS_REASON)

    _reload(db_session, run, task, step_card, card, pending)
    assert out["success"] is True and "was not cancelled" in out["message"]
    assert run.state == RunState.RUNNING.value                                    # night 9: cancelled
    assert card.status != "cancelled" and pending.state == TaskState.PENDING.value  # night 9: step 2 skipped
    assert task.state == TaskState.RETRYING.value and step_card.status == "in_progress"
    assert CAFES in task.input_context["verification_feedback"]["reasoning"]     # the owner's words, not Auto's


def test_a_step_named_by_its_card_number_is_sent_back_through_its_mission(db_session, seed_workspace, said):
    from modules.tools.discovery.handlers_missions import reject_mission
    from services.ticket_numbers import ticket_number

    ws = UUID(seed_workspace())
    run, _card, task, step_card = _held(db_session, ws, config={"check_each_step": True})
    said.append(SEND_IT_BACK)

    out = _in_the_chat(reject_mission, db_session, ws, mission_id=ticket_number(db_session, step_card),
                       reason=CAFES)

    _reload(db_session, run, task)
    assert out["success"] is True and run.state == RunState.RUNNING.value
    assert task.state == TaskState.RETRYING.value


def test_a_step_of_a_finished_mission_sent_back_opens_the_mission_again(db_session, seed_workspace, said):
    from modules.tools.discovery.handlers_missions import reject_mission

    ws = UUID(seed_workspace())
    run, card, steps = _mission(db_session, ws, state=RunState.COMPLETED)
    run.started_at = run.completed_at = datetime.now(timezone.utc)
    task, step_card = steps[TaskState.VERIFIED]
    task.output = step_card.result = card.result = "Quay 410 kg, Lantern 380 kg, Kiln 300 kg."
    step_card.status = card.status = "done"
    db_session.flush()
    said.append(f"Send step 5 back with: {CAFES}")

    out = _in_the_chat(reject_mission, db_session, ws, mission_id=str(run.id), step="5", reason=CAFES)

    _reload(db_session, run, task)
    assert out["success"] is True and "opened again" in out["message"]
    assert run.state == RunState.RUNNING.value and task.state == TaskState.RETRYING.value


def test_rejecting_a_started_mission_with_no_step_to_redo_cancels_nothing(db_session, seed_workspace, said):
    from modules.tools.discovery.handlers_missions import reject_mission

    ws = UUID(seed_workspace())
    run, card, _steps = _mission(db_session, ws, state=RunState.RUNNING)
    said.append("Reject it.")

    out = _in_the_chat(reject_mission, db_session, ws, mission_id=str(run.id), reason="Reject it.")

    _reload(db_session, run, card)
    assert out["success"] is False and "nothing was done" in out["error"] and "step" in out["error"]
    assert run.state == RunState.RUNNING.value and card.status != "cancelled"


def test_a_plan_awaiting_approval_is_still_rejected_and_cancelled(db_session, seed_workspace, said):
    from modules.tools.discovery.handlers_missions import reject_mission

    ws = UUID(seed_workspace())
    run, _card, _steps = _mission(db_session, ws, state=RunState.AWAITING_APPROVAL)

    out = _in_the_chat(reject_mission, db_session, ws, mission_id=str(run.id), reason="Not this one.")

    db_session.refresh(run)
    assert out["success"] is True and run.state == RunState.CANCELLED.value


def test_moving_a_held_steps_card_to_done_lets_the_step_through(db_session, seed_workspace, monkeypatch):
    """What mission_refs tells Auto to do with a step's card: platform_update_task_status done."""
    from contextlib import contextmanager

    from core.database import database
    from modules.tools.discovery.handlers_board_task_done import update_board_task_status

    @contextmanager
    def this_session():                    # the ticket's change notes, in the test's own transaction
        yield db_session

    async def _not_filed(db, workspace_id, task):
        return None

    monkeypatch.setattr(database, "get_db_session", this_session)
    monkeypatch.setattr("services.report_knowledge.file_done_ticket", _not_filed)
    ws = UUID(seed_workspace())
    run, _card, task, step_card = _held(db_session, ws, config={"check_each_step": True})

    out = asyncio.run(update_board_task_status(db_session, ws, {"task_id": step_card.id, "status": "done",
                                                                "_user_id": "2"}))

    _reload(db_session, run, task)
    assert out["success"] is True
    assert (task.state, run.state) == (TaskState.VERIFIED.value, RunState.RUNNING.value)   # was: held, paused


def _steps_of(db, run):
    """The mission's pending step and its card."""
    from core.models.core import BoardTask
    from core.models.orchestration import OrchestrationTask

    task = db.query(OrchestrationTask).filter(OrchestrationTask.run_id == run.id,
                                              OrchestrationTask.state == TaskState.PENDING.value).first()
    card = db.query(BoardTask).filter(BoardTask.orchestration_task_id == task.id).first()
    return [(task, card)]

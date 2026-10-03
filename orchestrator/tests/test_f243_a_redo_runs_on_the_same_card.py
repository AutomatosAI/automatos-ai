"""F243 (night 7) — a redo on a playbook's or a mission's card runs on that card, or is refused up front.

Six of six rejected playbook cards never ran again (#0106, #0112, #0149, #0070);
Run now and a re-brief said "started" when nothing started (#0112, #0123,
#0095); rejected mission steps never ran again (#0083.1, #0105.3); and a redo
that failed for credit took the draft the owner was correcting off the card's
face (#0171, #0174).
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from core.models.orchestration_enums import RunState, TaskState
from tests import test_f116_a_cancelled_run_stops_its_sessions as f116
from tests.test_f116_a_cancelled_run_stops_its_sessions import _recipe, _run
from tests.test_f245_cancel_stops_a_mission import _mission

# F116's Postgres and workspace, as fixtures of this module too.
engine = f116.engine
workspace = f116.workspace

DRAFT = "Monday stock: 14 lines, oat milk low."
NOTE = "Add the milk order for Harbour Street."


def _owner(ws):
    return NS(workspace_id=uuid.UUID(str(ws)), user_id="2", auth_type="anonymous", user=NS(id="2"))


def _body(payload):
    async def _json():
        return payload
    return NS(json=_json)


@pytest.fixture
def launched(monkeypatch):
    """The playbook runs the engine was asked to start (services/watch_rerun)."""
    import services.watch_rerun as watch_rerun

    runs = []
    monkeypatch.setattr(watch_rerun, "launch_execution", lambda execution: runs.append(execution.execution_id))
    return runs


def _finished_playbook(new_session, ws, *, status="review", run_status="completed"):
    from core.models.core import BoardTask

    s = new_session()
    recipe = _recipe(s, ws)
    run = _run(s, ws, recipe.id)
    s.flush()
    s.execute(text("UPDATE recipe_executions SET status = :st WHERE execution_id = :e"), {"st": run_status, "e": run})
    card = BoardTask(workspace_id=ws, title="Recipe: Monday Stock Report", status=status, priority="medium",
                     source_type="recipe", source_id=run, review_mode="auto", result=DRAFT)
    s.add(card)
    s.commit()
    return NS(run=run, card=card.id)


def _execution(new_session, execution_id):
    return new_session().execute(text(
        "SELECT execution_id, status, retry_of, execution_metadata FROM recipe_executions WHERE execution_id = :e"),
        {"e": execution_id}).first()


def _card(new_session, card_id):
    return new_session().execute(text("SELECT status, source_id, result FROM board_tasks WHERE id = :i"),
                                 {"i": card_id}).first()


def test_a_rejected_playbook_card_runs_its_playbook_again_on_that_card(workspace, new_session, launched):
    from api.board_tasks import reject_task
    from services.playbook_owner_ask import REDO_KEY

    pb = _finished_playbook(new_session, workspace)

    asyncio.run(reject_task(pb.card, _body({"feedback": NOTE}), ctx=_owner(workspace), db=new_session()))

    card = _card(new_session, pb.card)
    assert card.status == "in_progress" and launched == [card.source_id]     # night: Assigned, never ran
    rerun = _execution(new_session, card.source_id)
    assert rerun.retry_of == pb.run                                          # the same job, run again
    words = rerun.execution_metadata[REDO_KEY]
    assert NOTE in words and DRAFT in words                                  # the owner's words and the draft


def test_run_now_on_a_failed_playbook_card_runs_it_again(workspace, new_session, launched):
    from api.board_tasks import run_task_now

    pb = _finished_playbook(new_session, workspace, status="failed", run_status="failed")

    out = asyncio.run(run_task_now(pb.card, ctx=_owner(workspace), db=new_session()))

    card = _card(new_session, pb.card)
    assert out["started"] is True and launched == [card.source_id]          # night: "started", nothing ran
    assert "running again" in out["message"] and card.status == "in_progress"


def test_a_rebrief_on_a_playbook_card_runs_it_with_the_agreed_brief(workspace, new_session, launched):
    from api.board_task_rebrief import RebriefBody, rebrief_task
    from services.playbook_owner_ask import REDO_KEY

    pb = _finished_playbook(new_session, workspace, status="done")
    brief = "Tom's Monday dispatch checklist, for Monday 5 October."

    rebrief_task(pb.card, RebriefBody(brief=brief), ctx=_owner(workspace), db=new_session())

    card = _card(new_session, pb.card)
    assert launched == [card.source_id]                                      # night (#0095): nothing ran
    assert brief in _execution(new_session, card.source_id).execution_metadata[REDO_KEY]


def test_a_playbook_still_running_is_refused_before_anything_changes(workspace, new_session, launched):
    from api.board_tasks import reject_task

    pb = _finished_playbook(new_session, workspace, run_status="running")

    with pytest.raises(HTTPException) as refused:
        asyncio.run(reject_task(pb.card, _body({"feedback": NOTE}), ctx=_owner(workspace), db=new_session()))

    assert refused.value.status_code == 409 and "still running" in refused.value.detail
    assert (_card(new_session, pb.card).status, launched) == ("review", [])


def test_every_step_of_the_redo_is_told_the_owners_words():
    from services.playbook_owner_ask import ANSWERS_KEY, REDO_KEY, owner_answers_block

    block = owner_answers_block({REDO_KEY: f"## Redo\n{NOTE}",
                                 ANSWERS_KEY: [{"step": 1, "question": "Which café?", "answer": "Harbour Street"}]})

    assert block.startswith("## Redo") and NOTE in block and "Harbour Street" in block


def _step(steps, state, output="The gift box page, drafted."):
    task, card = steps[state]
    task.output = output
    return task, card


def test_a_rejected_step_of_a_running_mission_goes_back_to_its_mission(db_session, seed_workspace):
    from api.board_tasks import reject_task

    ws = UUID(seed_workspace())
    _run_row, _mission_card, steps = _mission(db_session, ws)
    task, card = _step(steps, TaskState.VERIFIED)
    db_session.flush()

    asyncio.run(reject_task(card.id, _body({"feedback": NOTE}), ctx=_owner(ws), db=db_session))

    db_session.refresh(task)
    db_session.refresh(card)
    assert (task.state, card.status) == (TaskState.RETRYING.value, "in_progress")  # night: never ran again
    assert task.input_context["previous_output"] == "The gift box page, drafted."
    assert NOTE in task.input_context["verification_feedback"]["reasoning"]


def test_a_rejected_step_of_an_ended_mission_is_refused_before_anything_changes(db_session, seed_workspace):
    from api.board_tasks import reject_task

    ws = UUID(seed_workspace())
    run, _mission_card, steps = _mission(db_session, ws, state=RunState.COMPLETED)
    task, card = _step(steps, TaskState.VERIFIED)
    db_session.flush()

    with pytest.raises(HTTPException) as refused:
        asyncio.run(reject_task(card.id, _body({"feedback": NOTE}), ctx=_owner(ws), db=db_session))

    assert refused.value.status_code == 409 and f"/missions/{run.id}" in refused.value.detail
    assert "finished" in refused.value.detail and "Re-run the mission" in refused.value.detail
    db_session.refresh(card)
    assert (card.status, task.state) == ("done", TaskState.VERIFIED.value)


def test_a_rebrief_on_a_step_of_a_running_mission_goes_back_to_its_mission(db_session, seed_workspace):
    from api.board_task_rebrief import RebriefBody, rebrief_task

    ws = UUID(seed_workspace())
    _run_row, _mission_card, steps = _mission(db_session, ws)
    task, card = _step(steps, TaskState.VERIFIED)
    db_session.flush()

    rebrief_task(card.id, RebriefBody(brief="A shorter gift box page."), ctx=_owner(ws), db=db_session)

    db_session.refresh(task)
    assert task.state == TaskState.RETRYING.value                           # night (#0126.2): refused
    assert "A shorter gift box page." in task.input_context["verification_feedback"]["reasoning"]


def test_a_failed_redo_keeps_the_last_draft_on_the_card():
    from core.models.core import BoardTask
    from services.board_task_view import board_dict

    previous = [{"why": "sent back", "result": "Hi Rosa, ... Gerard", "at": "2026-10-03T07:01:00+00:00"}]
    failed = BoardTask(id=171, title="Rosa's apology", status="failed", result=None,
                       error_message="The AI provider's account ran out of credit",
                       planning_data={"previous_runs": previous})

    assert board_dict(failed)["kept_draft"] == "Hi Rosa, ... Gerard"         # night: only in its history
    redone = BoardTask(id=172, title="Rosa's apology", status="failed", result="A new draft",
                       planning_data={"previous_runs": previous})
    assert board_dict(redone)["kept_draft"] is None                          # its own result shows instead

"""F245 (night 7) — Cancel on a playbook's card stops its run, and the card stays Cancelled.

#0096 was cancelled 9 s after it started and #0150 a second after: both runs
kept going (seven model calls after the cancel), and each card went from
Cancelled to Done with the run's result. The board's Cancel only flipped the
card; the playbooks page's own cancel stopped the run but filed the card as
failed. A failed ticket could not be cancelled at all ("without a word").
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from tests import test_f116_a_cancelled_run_stops_its_sessions as f116
from tests.test_f116_a_cancelled_run_stops_its_sessions import _recipe, _run, _step, _tickets

# F116's Postgres and workspace, as fixtures of this module too.
engine = f116.engine
workspace = f116.workspace

OWNER = "user:2"


def _owner(ws):
    """The local edition's one person, who may stop any run (core/auth/workspace_permission)."""
    return NS(workspace_id=uuid.UUID(ws), user_id="2", auth_type="anonymous", user=NS(id="2"))


def _sdk_key(ws):
    """A caller whose role grants no run's cancel (an SDK key never holds a workspace role)."""
    return NS(workspace_id=uuid.UUID(ws), user_id="9", auth_type="sdk_key", user=NS(id="9"))


@pytest.fixture
def stopped_here(monkeypatch):
    """The runs this worker was told to stop at once (api/recipe_executor)."""
    import api.recipe_executor as executor

    told = []
    monkeypatch.setattr(executor, "request_execution_cancel", lambda execution_id: told.append(execution_id) or True)
    return told


def _running_playbook(new_session, ws):
    from core.models.core import BoardTask

    s = new_session()
    recipe = _recipe(s, ws)
    run = _run(s, ws, recipe.id)
    card = BoardTask(workspace_id=ws, title="Recipe: Monday Stock Report", status="in_progress", priority="medium",
                     source_type="recipe", source_id=run, review_mode="auto")
    s.add(card)
    s.flush()
    step = _step(s, ws, run, 1, "in_progress")
    s.commit()
    return NS(recipe=recipe, run=run, card=card.id, step=step)


def _execution_status(new_session, run):
    return new_session().execute(
        text("SELECT status FROM recipe_executions WHERE execution_id = :e"), {"e": run}).scalar()


def test_cancel_on_a_playbooks_card_stops_its_run(workspace, new_session, stopped_here):
    from api.board_tasks import cancel_task

    pb = _running_playbook(new_session, workspace)

    out = asyncio.run(cancel_task(pb.card, ctx=_owner(workspace), db=new_session()))

    assert out == {"id": pb.card, "status": "cancelled", "applied": True, "previous_status": "in_progress"}
    assert _execution_status(new_session, pb.run) == "cancelled"        # night: the run kept going
    assert stopped_here == [pb.run]                                     # stopped at once on this worker
    rows = _tickets(new_session, [pb.card, pb.step])
    assert rows[pb.card].runtime_ref["cancelled"]["by"] == OWNER
    assert rows[pb.step].status == "cancelled"                          # its session step stops with it


def test_a_run_that_ends_after_the_cancel_leaves_the_card_cancelled(workspace, new_session, stopped_here):
    from api.board_tasks import cancel_task
    from services.board_task_bridge import complete_recipe_board_task

    pb = _running_playbook(new_session, workspace)
    asyncio.run(cancel_task(pb.card, ctx=_owner(workspace), db=new_session()))

    complete_recipe_board_task(new_session(), pb.run, success=True, result="Stock report: 14 lines")

    row = _tickets(new_session, [pb.card])[pb.card]
    assert row.status == "cancelled"                                    # night: Cancelled → Done
    result = new_session().execute(text("SELECT result FROM board_tasks WHERE id = :i"), {"i": pb.card}).scalar()
    assert result is None


def test_the_playbooks_pages_cancel_files_the_card_as_cancelled(workspace, new_session, stopped_here):
    from api.workflow_recipes import cancel_execution

    pb = _running_playbook(new_session, workspace)

    asyncio.run(cancel_execution(str(pb.recipe.id), pb.run, ctx=_owner(workspace), db=new_session()))

    assert _tickets(new_session, [pb.card])[pb.card].status == "cancelled"   # was filed as failed
    assert _execution_status(new_session, pb.run) == "cancelled" and stopped_here == [pb.run]


def test_a_drag_to_cancelled_does_what_cancel_does(workspace, new_session, stopped_here):
    from api.board_tasks import update_task_status

    pb = _running_playbook(new_session, workspace)

    async def _body():
        return {"status": "cancelled"}

    out = asyncio.run(update_task_status(pb.card, NS(json=_body), ctx=_owner(workspace), db=new_session()))

    assert out["status"] == "cancelled" and _execution_status(new_session, pb.run) == "cancelled"


def test_whoever_may_not_stop_a_playbook_cannot_stop_it_from_the_board(workspace, new_session, stopped_here):
    from api.board_tasks import cancel_task

    pb = _running_playbook(new_session, workspace)

    with pytest.raises(HTTPException) as refused:
        asyncio.run(cancel_task(pb.card, ctx=_sdk_key(workspace), db=new_session()))

    assert refused.value.status_code == 403 and "playbooks:execute" in refused.value.detail
    assert _execution_status(new_session, pb.run) == "running" and stopped_here == []
    assert _tickets(new_session, [pb.card])[pb.card].status == "in_progress"


def test_a_cancel_that_fails_part_way_changes_nothing(workspace, new_session, stopped_here, monkeypatch):
    """Review of #885: the run, its card and its step tickets are cancelled in one
    commit, so a failure on a step ticket leaves the run running, to be cancelled
    again, rather than marked cancelled with its card still open."""
    import services.board_cancel as board_cancel
    from api.board_tasks import cancel_task

    pb = _running_playbook(new_session, workspace)
    staged = board_cancel.stage_ticket_cancel

    def _step_fails(db, task, **kwargs):
        if task.id == pb.step:
            raise RuntimeError("the step ticket could not be written")
        return staged(db, task, **kwargs)

    monkeypatch.setattr(board_cancel, "stage_ticket_cancel", _step_fails)
    with pytest.raises(RuntimeError):
        asyncio.run(cancel_task(pb.card, ctx=_owner(workspace), db=new_session()))

    assert _execution_status(new_session, pb.run) == "running" and stopped_here == []
    assert _tickets(new_session, [pb.card])[pb.card].status == "in_progress"


def test_a_failed_ticket_can_be_cancelled(workspace, new_session):
    """Night 7: the Cancel button refused a failed card without a word; F246 keeps
    a failure in Needs you until the owner deals with it."""
    from api.board_tasks import cancel_task
    from core.models.core import BoardTask

    s = new_session()
    failed = BoardTask(workspace_id=workspace, title="Weekly numbers", status="failed", priority="medium",
                       source_type="user")
    s.add(failed)
    s.commit()

    out = asyncio.run(cancel_task(failed.id, ctx=_owner(workspace), db=new_session()))

    assert (out["applied"], out["status"]) == (True, "cancelled")

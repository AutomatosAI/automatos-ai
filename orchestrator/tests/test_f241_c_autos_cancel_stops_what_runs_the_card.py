"""F241 with F245 (night 7): Auto's cancel stops what runs the card, as the board's does.

Auto's platform_update_task_status('cancelled') only wrote the status. A playbook's
run went on under its cancelled card, and so did a mission; F245 fixed this for the
board's Cancel (#0096, #0098, #0150). A cancel through Auto now takes the same path.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest


@pytest.fixture
def board(db_session, seed_workspace, monkeypatch):
    import modules.tools.discovery.handlers_board_tasks as handlers

    for quiet in ("_notify_board_safe", "_notify_dispatch_safe", "_consent_for_chat_filed"):
        monkeypatch.setattr(handlers, quiet, lambda *a, **k: None)
    return NS(db=db_session, ws=UUID(seed_workspace()), handlers=handlers)


def _playbook_card(board):
    """A playbook run that is still going, and its card."""
    from core.models.core import BoardTask, RecipeExecution, WorkflowTemplate

    playbook = WorkflowTemplate(template_id=f"f241c-{uuid.uuid4().hex[:8]}", name="Weekly social posts",
                                description="Posts.", workspace_id=board.ws, template_definition={"steps": []},
                                steps=[], created_by="f241")
    board.db.add(playbook)
    board.db.flush()
    run = RecipeExecution(execution_id=f"exec-{uuid.uuid4().hex[:12]}", recipe_id=playbook.id,
                          workspace_id=board.ws, status="running", input_data={}, triggered_by="manual")
    board.db.add(run)
    board.db.flush()
    card = BoardTask(workspace_id=board.ws, title="Recipe: Weekly social posts", status="in_progress",
                     priority="medium", source_type="recipe", source_id=run.execution_id)
    board.db.add(card)
    board.db.flush()
    return run, card


def _cancel(board, owner="1", **params):
    """platform_update_task_status('cancelled'), with the chat's person behind it unless ``owner`` is None."""
    driver = {"_user_id": owner} if owner else {}
    return asyncio.run(board.handlers.update_board_task_status(
        board.db, board.ws, {"status": "cancelled", **driver, **params}))


def test_cancelling_a_playbooks_card_stops_its_run(board):
    run, card = _playbook_card(board)

    reply = _cancel(board, task_id=card.id)

    assert reply["success"] is True and reply["message"] == "Cancelled, and its playbook run stopped."
    board.db.refresh(run)
    board.db.refresh(card)
    assert (run.status, card.status) == ("cancelled", "cancelled")


def test_a_call_no_person_drives_cannot_stop_a_run(board):
    run, card = _playbook_card(board)

    reply = _cancel(board, owner=None, task_id=card.id)

    assert reply["success"] is False and "playbooks:execute" in reply["error"]
    board.db.refresh(run)
    assert run.status == "running"


def test_a_live_missions_step_is_the_missions_to_stop(board):
    from core.models.orchestration import OrchestrationRun, OrchestrationTask
    from services.orchestration_board_bridge import create_mission_board_task, create_task_board_task

    run = OrchestrationRun(workspace_id=board.ws, goal="Prepare the cafes for the price change", state="running",
                           created_by="user_test", config={})
    board.db.add(run)
    board.db.flush()
    step = OrchestrationTask(run_id=run.id, title="List every cafe", description="Do it.", sequence_number=1,
                             state="running", state_type="active")
    board.db.add(step)
    board.db.flush()
    create_mission_board_task(board.db, run)
    step_card = create_task_board_task(board.db, run, step)
    board.db.flush()

    reply = _cancel(board, task_id=step_card.id)

    assert reply["success"] is False and "the mission runs its steps" in reply["error"]
    board.db.refresh(run)
    assert run.state == "running"


def test_a_bulk_cancel_stops_the_run_and_cancels_the_plain_card(board):
    from core.models.core import BoardTask

    run, card = _playbook_card(board)
    plain = BoardTask(workspace_id=board.ws, title="Reply to The Salt House", status="inbox", priority="medium",
                      source_type="user")
    board.db.add(plain)
    board.db.flush()

    reply = _cancel(board, task_ids=[plain.id, card.id])

    assert reply["success"] is True and sorted(reply["updated"]) == sorted([plain.id, card.id])
    board.db.refresh(run)
    board.db.refresh(plain)
    assert (run.status, plain.status) == ("cancelled", "cancelled")

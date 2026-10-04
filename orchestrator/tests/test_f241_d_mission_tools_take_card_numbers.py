"""F241 (night 7b): a mission tool given a card's number acts on that card's mission.

Night 7b, Auto sent the owner's card numbers as mission ids, and every call failed:
platform_approve_mission("0177") and platform_cancel_mission("0193") on task cards,
platform_get_mission(188) for mission #0188's own card, platform_get_mission("0188.3")
for its third step, and platform_update_task(task_id=188.3) with the step's number sent
as a JSON number.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest


@pytest.fixture
def board(db_session, seed_workspace):
    """A mission awaiting approval with its card and two steps' cards, and a task card in Review."""
    from core.models.core import BoardTask
    from core.models.orchestration import OrchestrationRun, OrchestrationTask
    from services.orchestration_board_bridge import create_mission_board_task, create_task_board_task

    ws = UUID(seed_workspace())
    run = OrchestrationRun(workspace_id=ws, goal="Get the Christmas gift subscription ready",
                           state="awaiting_approval", created_by="user_test", config={})
    db_session.add(run)
    db_session.flush()
    card = create_mission_board_task(db_session, run)
    steps = []
    for n, title in enumerate(("Work out the coffee", "Draft the shop words"), start=1):
        step = OrchestrationTask(run_id=run.id, title=title, description="Do it.", sequence_number=n,
                                 state="pending", state_type="initial")
        db_session.add(step)
        db_session.flush()
        steps.append(create_task_board_task(db_session, run, step))
    task = BoardTask(workspace_id=ws, title="Break-even on the gift box print run", status="review",
                     description="Work it out.")
    db_session.add(task)
    db_session.flush()
    return NS(db=db_session, ws=ws, run=run, card=card, steps=steps, task=task,
              number=f"#{card.workspace_seq:04d}", task_number=f"#{task.workspace_seq:04d}")


def _call(handler, board, said, **params):
    return asyncio.run(handler(board.db, board.ws, {"mission_id": said, **params}))


def test_a_missions_card_number_is_its_mission(board):
    from modules.tools.discovery.handlers_missions import get_mission

    seq = board.card.workspace_seq
    for said in (board.number, f"{seq:04d}", str(seq), seq):       # night 7b: get_mission(188)
        out = _call(get_mission, board, said)
        assert out["success"] is True and out["mission"]["id"] == board.run.id, said


def test_a_steps_number_reads_its_mission_and_names_the_step(board):
    from modules.tools.discovery.handlers_missions import get_mission

    for said in (f"{board.number}.2", f"{board.card.workspace_seq:04d}.2", float(f"{board.card.workspace_seq}.2")):
        out = _call(get_mission, board, said)                    # night 7b: get_mission("0188.3")
        assert out["mission"]["id"] == board.run.id, said
        assert f"{board.number}.2" in out["asked_about"] and "Draft the shop words" in out["asked_about"]


def test_reading_a_task_cards_number_as_a_mission_gives_the_card(board):
    from modules.tools.discovery.handlers_missions import get_mission

    out = _call(get_mission, board, board.task_number)
    assert out["success"] is True and out["task"]["id"] == board.task.id
    assert "not a mission" in out["note"]


def test_approving_a_task_card_as_a_mission_names_the_call_that_approves_it(board):
    from modules.tools.discovery.handlers_missions import approve_mission

    out = _call(approve_mission, board, board.task_number.lstrip("#"))  # night 7b: approve_mission("0177")
    assert out["success"] is False
    assert "is a task card, not a mission" in out["error"] and "nothing was done" in out["error"]
    assert f'platform_update_task_status with task_id "{board.task_number}", status "done"' in out["error"]
    board.db.refresh(board.task)
    assert board.task.status == "review"                                 # nothing moved


def test_cancelling_a_task_card_as_a_mission_names_the_cancel_that_does_it(board):
    from modules.tools.discovery.handlers_missions import cancel_mission

    out = _call(cancel_mission, board, board.task_number)                # night 7b: cancel_mission("0193")
    assert out["success"] is False
    assert f'task_id "{board.task_number}" and status "cancelled"' in out["error"]


def test_a_step_is_not_cancelled_or_approved_alone(board):
    from modules.tools.discovery.handlers_missions import approve_mission, cancel_mission

    step = f"{board.number}.1"
    cancelled = _call(cancel_mission, board, step)
    approved = _call(approve_mission, board, step)

    assert cancelled["success"] is False and "A step stops with its mission" in cancelled["error"]
    assert f'mission_id "{board.number}"' in cancelled["error"]
    assert approved["success"] is False and f'task_id "{step}", status "done"' in approved["error"]


def test_approving_by_the_missions_card_number_approves_the_mission(board, monkeypatch):
    from modules.tools.discovery import handlers_missions
    from services import coordinator_service

    approved = []

    def approve_plan(self, db, run_id, actor_id):
        approved.append(run_id)
        return NS(id=run_id, state="running")

    monkeypatch.setattr(coordinator_service.CoordinatorService, "approve_plan", approve_plan)
    monkeypatch.setattr("services.mission_wait.wait_note_of", lambda db, run_id: "")
    out = _call(handlers_missions.approve_mission, board, board.number, _created_by="owner@local")

    assert out["success"] is True and approved == [board.run.id]


def test_a_number_that_names_no_card_says_so(board):
    from modules.tools.discovery.handlers_missions import get_mission

    out = _call(get_mission, board, "#9999")
    assert out["success"] is False and "No ticket #9999" in out["error"]


def test_a_missions_uuid_still_works(board):
    from modules.tools.discovery.handlers_missions import get_mission

    assert _call(get_mission, board, str(board.run.id))["mission"]["id"] == board.run.id


def test_a_widget_turn_reads_no_number(board):
    from core.security.surface import WIDGET, turn_surface
    from modules.tools.discovery.handlers_missions import get_mission

    with turn_surface(WIDGET):
        out = _call(get_mission, board, board.number)
    assert out["success"] is False and "Invalid mission_id" in out["error"]


def test_a_steps_number_sent_as_a_json_number_is_the_step(board):
    """Night 7b: platform_update_task(task_id=188.3) was read as ticket 188."""
    from services.ticket_refs import ticket_id_named

    found, error = ticket_id_named(board.db, board.ws, float(f"{board.card.workspace_seq}.2"))
    assert error is None and found == board.steps[1].id


def test_a_json_number_that_lost_its_zero_is_refused_when_it_could_be_two_steps(board):
    from core.models.orchestration import OrchestrationTask
    from services.orchestration_board_bridge import create_task_board_task
    from services.ticket_refs import ticket_id_named

    for n in range(3, 11):                                    # ten steps: .1 could be .10
        step = OrchestrationTask(run_id=board.run.id, title=f"Step {n}", description="Do it.",
                                 sequence_number=n, state="pending", state_type="initial")
        board.db.add(step)
        board.db.flush()
        create_task_board_task(board.db, board.run, step)
    board.db.flush()

    found, error = ticket_id_named(board.db, board.ws, float(f"{board.card.workspace_seq}.1"))
    assert found is None and f"{board.number}.1 or {board.number}.10" in error


def test_the_mission_tools_say_they_take_a_card_number():
    from modules.tools.discovery.action_registry import ActionRegistry
    from modules.tools.discovery.actions_mission_resume import register_mission_resume_action
    from modules.tools.discovery.actions_missions import register_mission_actions

    registry = ActionRegistry()
    register_mission_actions(registry)
    register_mission_resume_action(registry)
    for name in ("platform_get_mission", "platform_approve_mission", "platform_cancel_mission",
                 "platform_resume_mission"):
        described = registry.get(name).parameters["properties"]["mission_id"]["description"]
        assert "card's number" in described, name
    assert "platform_update_task_status" in registry.get("platform_approve_mission").description

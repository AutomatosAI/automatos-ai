"""F249 (night 8): a card sent back runs its redo with the owner's correction, whoever starts it.

Auto's move to In progress launched a sent-back card from its bare brief, so the redo
lost the owner's words. The board's dispatcher runs it with them; Auto's move now
leaves it to the dispatcher.
"""
from __future__ import annotations

import asyncio
from uuid import UUID

from core.models import Agent
from core.models.core import BoardTask
from modules.tools.discovery.ticket_run_now import redo_keeps_the_correction


def _card(db_session, ws, **fields):
    agent = Agent(name="Support Agent", agent_type="chatbot", description="Answers customers.", status="active",
                  configuration={}, model_config=None, workspace_id=ws, created_by="test", owner_type="workspace",
                  owner_id=str(ws))
    db_session.add(agent)
    db_session.flush()
    card = BoardTask(workspace_id=ws, title="Reply to Hannah", source_type="user", assigned_agent_id=agent.id,
                     **fields)
    db_session.add(card)
    db_session.flush()
    return card


def _move(db_session, ws, card):
    launched = []

    async def handler(db, workspace_id, params):
        launched.append(params)
        return {"success": True, "task_id": card.id, "status": "in_progress", "triggered_execution": True}

    out = asyncio.run(redo_keeps_the_correction(handler)(db_session, ws, {"task_id": card.id, "status": "in_progress"}))
    return out, launched


def test_a_card_waiting_for_its_redo_is_left_to_the_board(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    card = _card(db_session, ws, status="assigned", review_feedback="Just the email: nothing before To:.")

    out, launched = _move(db_session, ws, card)

    assert launched == [] and out["success"] and out["triggered_execution"] is False
    assert "correction waiting" in out["message"]
    db_session.refresh(card)
    assert (card.status, card.review_feedback) == ("assigned", "Just the email: nothing before To:.")


def test_a_card_with_no_correction_waiting_runs_as_before(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    card = _card(db_session, ws, status="assigned", review_feedback=None)

    out, launched = _move(db_session, ws, card)

    assert len(launched) == 1 and out["triggered_execution"] is True


def test_autos_status_moves_go_through_it():
    from modules.tools.discovery.handlers_board_tasks import update_board_task_status

    wrapped, seen = update_board_task_status, []
    while wrapped is not None:
        seen.append(getattr(wrapped, "__code__", None))
        wrapped = getattr(wrapped, "__wrapped__", None)
    assert redo_keeps_the_correction(lambda: None).__code__ in seen

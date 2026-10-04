"""F309 (night 9): giving an answered card to another agent runs it again, with that agent.

"Please give card 1859 to the Shopify Business Analyst instead - the Analyst couldn't
find the data." Auto's platform_assign_task changed #1859's agent and nothing else: the
card sat in Review with the Analyst's answer ("I don't have access to your specific
data") until the owner sent it back by hand. It now goes back to Assigned for its new
agent, the way the board's Reject sends a card back: the old answer on record, the
owner's corrections kept, the run told the card is its own now.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from tests import test_1094_a_ticket_with_no_agent_is_never_in_progress as f1094

board = f1094.board        # a workspace, its Content Creator, Auto's real tools
OWNER = "owner@cafe.test"
BRIEF = "How many club boxes went out late in September? Say where the figure came from."
OLD = "I don't have access to your specific data. I will explain the steps I would take."
FROM_THE_SHOP = "Look in the shop system: subscription_orders has shipped_late."


@pytest.fixture
def shop(board, monkeypatch):
    """A second agent; ticket notes written in the test's session; no dispatcher to wake."""
    import api.board_tasks as bt
    import core.database.database as database

    @contextmanager
    def this_session():
        yield board.db

    monkeypatch.setattr(database, "get_db_session", this_session)
    monkeypatch.setattr(bt, "notify_task_available", lambda *a, **k: board.woken.append("redo"))
    board.analyst = board.db.execute(
        text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
             "VALUES ('Shopify Business Analyst', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) "
             "RETURNING id"), {"w": str(board.ws)}).scalar()
    return board


def _answered(shop, status="review", **fields):
    return f1094._ticket(shop, status=status, assigned_agent_id=shop.agent, description=BRIEF, raw_prompt=BRIEF,
                         result=OLD, completed_at=datetime.now(timezone.utc), **fields)


def _give(shop, task, agent_name):
    """Auto's platform_assign_task, as the platform executor runs it."""
    from modules.tools.discovery.handlers_board_task_assign import assign_board_task

    return asyncio.run(assign_board_task(shop.db, shop.ws, {"task_id": task.id, "agent_name": agent_name,
                                                             "_user_id": OWNER}))


def test_a_card_in_review_given_to_another_agent_runs_again_with_it(shop):
    from services.ticket_redo import GIVEN_TO_YOU, GIVEN_WHY, redo_block, times_sent_back

    task = _answered(shop, planning_data={"owner_corrections": [
        {"note": FROM_THE_SHOP, "by": f"user:{OWNER}", "at": "2026-10-04T12:39:00+00:00"}]})

    out = _give(shop, task, "Shopify Business Analyst")

    shop.db.refresh(task)
    assert out["success"] is True and out["runs_again"] is True and "runs again" in out["message"]
    assert (task.status, task.assigned_agent_id, task.result) == ("assigned", shop.analyst, None)   # night 9: review
    kept = task.planning_data["previous_runs"][-1]
    assert (kept["result"], kept["why"]) == (OLD, GIVEN_WHY)          # the old answer is on record
    assert task.review_feedback == GIVEN_TO_YOU and times_sent_back(task) == 0
    assert "redo" in shop.woken                                         # the dispatcher is woken for it
    told = redo_block(task)
    assert "## This card was given to you" in told and f"1. {FROM_THE_SHOP}" in told
    assert "was sent back" not in told and OLD not in told


def test_a_done_card_given_to_another_agent_runs_again_too(shop):
    task = _answered(shop, status="done")

    _give(shop, task, "Shopify Business Analyst")

    shop.db.refresh(task)
    assert (task.status, task.assigned_agent_id) == ("assigned", shop.analyst)


def test_the_same_agent_again_changes_nothing(shop):
    task = _answered(shop)

    out = _give(shop, task, "Content Creator")

    shop.db.refresh(task)
    assert out["success"] is True and "runs_again" not in out
    assert (task.status, task.result) == ("review", OLD)


def test_an_inbox_card_is_assigned_as_before(shop):
    task = f1094._ticket(shop)

    out = _give(shop, task, "Shopify Business Analyst")

    shop.db.refresh(task)
    assert out["success"] is True and "runs_again" not in out
    assert (task.status, task.assigned_agent_id) == ("assigned", shop.analyst)


def test_autos_assign_tool_is_the_one_that_runs_it_again():
    from modules.tools.discovery import handlers_board_task_assign, platform_executor

    assert platform_executor.assign_board_task is handlers_board_task_assign.assign_board_task

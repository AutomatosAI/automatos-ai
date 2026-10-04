"""F279 (night 8): Auto's send-back replaced the card's brief with the owner's correction.

Auto called platform_update_task {description: <the owner's correction>, send_back: true}
7 times of 7, twice after "keep its brief as it is". #0347's correction "take out the
line It starts with To:" became the whole brief, and the agent deleted the To: address
line; #0402's redo invented a shop notice; #0376, #0380 and #0449 asked for their own
draft. The board's Reject keeps the brief and adds the owner's words; so does Auto's.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

from tests import test_1094_a_ticket_with_no_agent_is_never_in_progress as f1094

board = f1094.board        # a workspace, its Content Creator, Auto's real tools
OWNER = "owner@cafe.test"
BRIEF = ("Invoice reminder to Fernhill Bakery for invoice 1187, £84.20, due 25 September. Start at To: "
         "accounts@fernhill.example and sign it Gerard.")
DRAFT = "It starts with To:.\nTo: accounts@fernhill.example\nHi Ellie, a reminder about invoice 1187…"
WORDS = "Take out the line It starts with To:. Nothing at all before To:. The rest of the email is right, keep it."
NEW_BRIEF = "Write the intro for the 7 January club newsletter, 85 to 95 words. Start with Hello from Gerard."


@pytest.fixture
def shop(board, monkeypatch):
    """Ticket notes are written in a session of their own; here, in the test's."""
    import api.board_tasks as bt
    import core.database.database as database

    @contextmanager
    def this_session():
        yield board.db

    monkeypatch.setattr(database, "get_db_session", this_session)
    monkeypatch.setattr(bt, "notify_task_available", lambda *a, **k: None)
    return board


def _in_review(shop):
    return f1094._ticket(shop, status="review", assigned_agent_id=shop.agent, description=BRIEF, raw_prompt=BRIEF,
                         result=DRAFT, completed_at=datetime.now(timezone.utc))


def _update(shop, **params):
    from modules.tools.discovery.handlers_board_tasks import update_board_task

    return asyncio.run(update_board_task(shop.db, shop.ws, params))


def _row(shop, task):
    shop.db.refresh(task)
    return task


@pytest.mark.parametrize("words_in", ["description", "note"])
def test_send_back_keeps_the_brief_and_the_owners_words_are_what_the_redo_fixes(shop, words_in):
    """#0347: the call night 8 saw (the words in description), and the one the tool asks for (in note)."""
    task = _in_review(shop)

    out = _update(shop, task_id=task.id, send_back=True, _user_id=OWNER, **{words_in: WORDS})

    row = _row(shop, task)
    assert out["success"] is True
    assert (row.status, row.description) == ("assigned", BRIEF)                         # night 8: WORDS
    assert row.planning_data["owner_corrections"][-1]["note"] == WORDS
    assert row.planning_data["previous_runs"][-1]["result"] == DRAFT                    # the draft is kept


def test_send_back_with_a_brief_and_words_does_nothing_and_says_why(shop):
    task = _in_review(shop)

    out = _update(shop, task_id=task.id, send_back=True, description=NEW_BRIEF, note=WORDS, _user_id=OWNER)

    assert out["success"] is False and "takes no description" in out["error"]
    assert (_row(shop, task).status, task.description) == ("review", BRIEF)


def test_a_new_brief_on_a_card_its_agent_worked_on_is_the_boards_rebrief(shop):
    """#0377: Auto's "update" changed the description only, kept no old brief and re-ran nothing."""
    task = _in_review(shop)

    out = _update(shop, task_id=task.id, description=NEW_BRIEF, _user_id=OWNER)

    row = _row(shop, task)
    assert out["success"] is True and out["updated"]["back_to_its_agent"] is True
    assert (row.status, row.description) == ("assigned", NEW_BRIEF)
    assert row.planning_data["previous_briefs"][-1]["description"] == BRIEF
    assert row.planning_data["previous_runs"][-1]["result"] == DRAFT


def test_a_new_brief_on_a_card_nobody_worked_on_is_an_edit(shop):
    task = f1094._ticket(shop, description=BRIEF)

    out = _update(shop, task_id=task.id, description=NEW_BRIEF)

    assert out["success"] is True
    assert (_row(shop, task).status, task.description) == ("inbox", NEW_BRIEF)


def test_the_tool_says_send_back_keeps_the_brief():
    from modules.tools.discovery import get_action_registry

    params = get_action_registry().get("platform_update_task").parameters["properties"]
    assert "its brief stays" in params["send_back"]["description"]
    assert "Re-brief" in params["description"]["description"]

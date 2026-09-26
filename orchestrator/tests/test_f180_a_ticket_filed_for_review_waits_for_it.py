"""F180 — a ticket Auto files for a person's review waits for it.

Night 6: the owner asked four times for review ("set to 'human' so it waits for
me", #1110 at 03:03:44Z). Auto said it was set each time, and every ticket ran
straight to Done: platform_create_task took only 'auto' and 'manual' and made
anything else 'auto' without a word, while the board, the REST API and
platform_update_task speak 'human', 'llm' and 'auto'. Both tools now take the
board's words ('manual' is 'human'), refuse a value that is none of them, and
the create result says what was kept.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text


@pytest.fixture
def board(db_session, seed_workspace, monkeypatch):
    from modules.tools.discovery import handlers_board_tasks as handlers

    ws = UUID(seed_workspace())
    db_session.execute(
        text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
             "VALUES ('Shopify Support Agent', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json))"),
        {"w": str(ws)})
    monkeypatch.setattr(handlers, "_notify_dispatch_safe", lambda *a, **k: None)
    return NS(db=db_session, ws=ws)


def _file(board, **params):
    from modules.tools.discovery.handlers_board_tasks import create_board_task

    return asyncio.run(create_board_task(board.db, board.ws, {
        "title": "Draft friendly reminder for invoice HL-2291 to The Ropewalk",
        "description": "Draft only, don't send.", "assigned_agent_name": "Shopify Support Agent", **params}))


def _review_mode(board, task_id):
    return board.db.execute(text("SELECT review_mode FROM board_tasks WHERE id = :i"), {"i": task_id}).scalar()


def test_1110_a_ticket_filed_for_a_persons_review_keeps_it(board):
    reply = _file(board, review_mode="human")
    assert reply["success"] is True
    assert _review_mode(board, reply["task_id"]) == "human"          # night 6: 'auto', so it never waited
    assert reply["review_mode"] == "human"                           # and the reply says what was kept


def test_the_old_word_manual_is_a_persons_review(board):
    reply = _file(board, review_mode="manual")
    assert _review_mode(board, reply["task_id"]) == "human"


def test_a_review_mode_that_is_none_of_the_boards_is_refused_not_dropped(board):
    reply = _file(board, review_mode="strict")
    assert reply["success"] is False and "'human' waits in Review for a person" in reply["error"]
    assert board.db.execute(text("SELECT count(*) FROM board_tasks WHERE workspace_id = :w"),
                            {"w": str(board.ws)}).scalar() == 0


def test_no_review_mode_still_closes_it_done(board):
    reply = _file(board)
    assert _review_mode(board, reply["task_id"]) == "auto"
    assert reply["review_mode"] == "auto"


def test_updating_takes_the_same_words(board):
    from modules.tools.discovery.handlers_board_tasks import update_board_task

    task_id = _file(board)["task_id"]
    reply = asyncio.run(update_board_task(board.db, board.ws, {"task_id": task_id, "review_mode": "manual"}))
    assert reply["success"] is True and _review_mode(board, task_id) == "human"


def test_both_tools_offer_the_boards_words():
    from api.board_tasks import VALID_REVIEW_MODES
    from modules.tools.discovery import get_action_registry

    registry = get_action_registry()
    for name in ("platform_create_task", "platform_update_task"):
        offered = registry.get(name).parameters["properties"]["review_mode"]["enum"]
        assert sorted(offered) == sorted(VALID_REVIEW_MODES), name


def test_a_session_ticket_speaks_the_boards_word():
    """Review LOW: the session lane wrote 'manual' straight to the row."""
    import inspect

    from services import cli_ticket_lane

    assert 'review_mode="manual"' not in inspect.getsource(cli_ticket_lane)

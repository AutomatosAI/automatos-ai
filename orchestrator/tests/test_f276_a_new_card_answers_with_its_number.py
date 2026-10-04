"""F276 (night 7b, report friction 3) — creating a card answers with its number.

The owner filed #0178 to #0181, and later #0197, on the board (POST /api/v1/tasks),
and every answer said ``number: null``: "Creating a card doesn't tell me its number.
I had to list the board." The create answered the new row's own columns, while the
list and the ticket view give every ticket its number (#0042). It now answers as the
list does. Auto's platform_create_task already gave the number (#0182: "number
right, on the board"), and still does.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException

TITLE = "Reply to Priya Shah - pause December, decaf in January (wait for me)"


class _Request:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


@pytest.fixture
def board(db_session, seed_workspace, monkeypatch):
    from modules.tools.discovery import handlers_board_tasks as handlers

    for quiet in ("_notify_board_safe", "_notify_dispatch_safe", "_consent_for_chat_filed"):
        monkeypatch.setattr(handlers, quiet, lambda *a, **k: None)
    ws = UUID(seed_workspace())
    return NS(db=db_session, ws=ws,
              ctx=NS(workspace_id=ws, user=NS(id="owner@cafe.test", clerk_user_id=None, email="owner@cafe.test")))


def _create(board, title=TITLE, **body):
    from api.board_tasks import create_task

    return asyncio.run(create_task(_Request({"title": title, **body}), ctx=board.ctx, db=board.db))


def _listed(board):
    """Each ticket's number, as the board's list serves it."""
    from api.board_tasks import list_tasks

    served = list_tasks(ctx=board.ctx, db=board.db, status=None, agent_id=None, priority=None, search=None,
                        parent_task_id=None, limit=100, offset=0, finished_limit=None)
    return {t["id"]: t["number"] for t in served["tasks"]}


def test_a_card_made_on_the_board_answers_with_its_number(board):
    first = _create(board)
    second = _create(board)                         # the same title: told apart by their numbers

    assert (first["number"], second["number"]) == ("#0001", "#0002")      # night 7b: null
    assert _listed(board) == {first["id"]: "#0001", second["id"]: "#0002"}
    assert (first["title"], first["status"]) == (TITLE, "inbox")          # the rest of the answer as before


def test_autos_new_card_answers_with_its_number_too(board):
    """#0182: platform_create_task gave the number, and it is the board's numbering."""
    from modules.tools.discovery.handlers_board_task_review import create_board_task

    _create(board, title="Price list for The Lantern Room")
    reply = asyncio.run(create_board_task(board.db, board.ws, {
        "title": "Calculate margin per 250g bag for October coffees", "description": "Every October coffee."}))

    assert reply["success"] is True and reply["number"] == "#0002"
    assert _listed(board)[reply["task_id"]] == "#0002"


def test_planning_data_that_is_not_an_object_is_refused_before_anything_is_filed(board):
    """The board reads a card's planning data as an object wherever it shows the
    card, the create's answer now included: a list is a 422, and nothing is filed."""
    with pytest.raises(HTTPException) as refused:
        _create(board, planning_data=["approve it"])

    assert refused.value.status_code == 422 and "planning_data" in refused.value.detail
    assert _listed(board) == {}

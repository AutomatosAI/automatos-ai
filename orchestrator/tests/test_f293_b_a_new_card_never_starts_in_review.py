"""F293 (night 8) — a card nobody has worked on never starts in Review or Done.

Auto made #0251 (02:21) and #0386 (05:54) straight into Review: no run, no answer,
and Needs you counted each as the owner's to judge. COPILOT's F289 stops Auto
asking; the board's own create (POST /api/v1/tasks) took no status at all, so a
request that asked for Review was quietly filed in the Inbox. It is now refused and
says where a new card goes, before anything is filed. A card that carries an action
for the owner to approve still starts in Review: that approval is what it asks for.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException

TITLE = "Reply to Hana about the March box"


class _Request:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


@pytest.fixture
def board(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    return NS(db=db_session, ws=ws,
              ctx=NS(workspace_id=ws, user=NS(id="owner@cafe.test", clerk_user_id=None, email="owner@cafe.test")))


def _create(board, **body):
    from api.board_tasks import create_task

    return asyncio.run(create_task(_Request({"title": TITLE, **body}), ctx=board.ctx, db=board.db))


def _cards(board):
    from core.models.core import BoardTask

    return board.db.query(BoardTask).filter(BoardTask.workspace_id == board.ws).count()


@pytest.mark.parametrize("column,shown", [("review", "Review"), ("done", "Done")])
def test_a_new_card_asked_into_a_finished_column_is_refused_and_nothing_is_filed(board, column, shown):
    with pytest.raises(HTTPException) as refused:
        _create(board, status=column)

    assert refused.value.status_code == 422
    assert f"can't start in {shown}" in refused.value.detail
    assert "Inbox" in refused.value.detail and "Nothing was filed" in refused.value.detail
    assert _cards(board) == 0


def test_a_new_card_with_an_action_to_approve_still_starts_in_review(board):
    created = _create(board, status="review",
                      planning_data={"approval_action": {"type": "publish_blog", "post_id": "b6f1"}})

    assert created["status"] == "review"


def test_a_new_card_that_asks_for_no_finished_column_is_filed_as_before(board):
    assert _create(board)["status"] == "inbox"
    assert _create(board, status="inbox")["status"] == "inbox"

"""F194 — a request files only a person's ticket.

POST /api/v1/tasks took any source_type, so a workspace member could file a
ticket that claimed to be a mission's step ('orchestration_task'), a Claude Code
agent's mission step or session ('mission', 'chat') or a playbook's step
('recipe'): kinds the dispatcher and a CLI host treat as the platform's own. The
clients send 'user' (the board's create, the sim harness) or 'activity' (a
Command Centre follow-up); anything else is a 422 and nothing is filed.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException
from sqlalchemy import text


class _Request:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


@pytest.fixture
def board(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    return NS(db=db_session, ws=ws,
              ctx=NS(workspace_id=ws, user=NS(id=7, clerk_user_id=None, email="owner@cafe.test")))


def _create(board, **body):
    import api.board_tasks as bt

    return asyncio.run(bt.create_task(_Request({"title": "Chase invoice HL-2291", **body}), ctx=board.ctx, db=board.db))


def _kinds(board):
    return board.db.execute(text("SELECT source_type FROM board_tasks WHERE workspace_id = CAST(:w AS uuid)"),
                            {"w": str(board.ws)}).scalars().all()


@pytest.mark.parametrize("kind", ["orchestration_task", "orchestration", "mission", "chat", "recipe", "lane"])
def test_a_platform_kind_is_refused_and_nothing_is_filed(board, kind):
    with pytest.raises(HTTPException) as refused:
        _create(board, source_type=kind, source_id="run-7:task-3")
    assert refused.value.status_code == 422 and "filed by the platform" in refused.value.detail
    assert _kinds(board) == []


@pytest.mark.parametrize("body,kind", [({}, "user"), ({"source_type": "user"}, "user"),
                                       ({"source_type": "activity", "source_id": "mission:42"}, "activity")],
                         ids=["the board's create", "the sim harness", "a Command Centre follow-up"])
def test_what_the_clients_send_still_files(board, body, kind):
    _create(board, **body)
    assert _kinds(board) == [kind]

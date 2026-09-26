"""F195's candidates — a malformed board request is a 4xx, never a 500.

After F195 (b): POST /api/v1/tasks with a title that is not text (its .strip())
or an assigned_agent_id that is not an id (its int()), the general PATCH with a
status or title that is not text, PATCH /status with a status that is not text,
and approve/reject's re-read of a ticket deleted as it was decided were all
500s. Each is now refused as a bad request (422), or not found (404).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import InvalidRequestError

WS = UUID("00000000-0000-0000-0000-0000000000c1")
CTX = NS(workspace_id=WS, user=NS(id=1, clerk_user_id="u1", email="owner@cafe.test"))


class _Req:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


class _Rows:
    def __init__(self, row):
        self._row = row

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._row

    def update(self, *a, **k):
        return 1


class _Session:
    """One ticket; a refresh after the commit finds it gone when ``deleted``."""

    def __init__(self, row=None, deleted=False):
        self._row, self._deleted = row, deleted

    def query(self, _model):
        return _Rows(self._row)

    def commit(self):
        pass

    def rollback(self):
        pass

    def refresh(self, _row):
        if self._deleted:
            raise InvalidRequestError("Could not refresh instance")


def _ticket(status="inbox", **kw):
    from core.models.core import BoardTask

    return BoardTask(id=21, workspace_id=WS, title="Order more oat milk", status=status, priority="medium",
                     review_mode="auto", source_type="user", **kw)


def _refused(call):
    with pytest.raises(HTTPException) as refused:                   # night: TypeError/ValueError/AttributeError
        asyncio.run(call)
    return refused.value


@pytest.mark.parametrize("body, says", [
    ({"title": 5}, "title must be text"),
    ({"title": ["Order oat milk"]}, "title must be text"),
    ({"title": "Order oat milk", "assigned_agent_id": "Numbers"}, "assigned_agent_id must be an agent's id"),
    ({"title": "Order oat milk", "assigned_agent_id": [7]}, "assigned_agent_id must be an agent's id"),
    ({"title": "Order oat milk", "assigned_agent_id": True}, "assigned_agent_id must be an agent's id"),
])
def test_creating_a_ticket_with_a_malformed_field_is_a_422(body, says):
    from api import board_tasks as bt

    refused = _refused(bt.create_task(_Req(body), ctx=CTX, db=_Session()))
    assert refused.status_code == 422 and says in refused.detail


@pytest.mark.parametrize("body, says", [
    ({"status": ["done"]}, "Invalid status"),
    ({"status": 5}, "Invalid status"),
    ({"title": 5}, "title must be text"),
])
def test_patching_a_ticket_with_a_malformed_field_is_a_422(body, says):
    from api import board_tasks as bt

    task = _ticket()
    refused = _refused(bt.update_task(21, _Req(body), ctx=CTX, db=_Session(task)))
    assert refused.status_code == 422 and says in refused.detail
    assert (task.status, task.title) == ("inbox", "Order more oat milk")


def test_a_status_patch_that_is_not_text_is_a_422():
    from api import board_tasks as bt

    refused = _refused(bt.update_task_status(21, _Req({"status": 5}), ctx=CTX, db=_Session(_ticket())))
    assert refused.status_code == 422


def test_a_ticket_deleted_as_it_was_approved_or_sent_back_is_not_found(monkeypatch):
    from api import board_tasks as bt

    async def _no_notice(*a, **k):
        return None

    monkeypatch.setattr(bt, "_dispatch_task_complete", _no_notice)
    monkeypatch.setattr(bt, "notify_task_available", lambda db, **kw: None)

    approved = _refused(bt.approve_task(21, _Req({}), ctx=CTX, db=_Session(_ticket("review"), deleted=True)))
    sent_back = _refused(bt.reject_task(21, _Req({"feedback": "Shorter."}), ctx=CTX,
                                        db=_Session(_ticket("review", assigned_agent_id=5), deleted=True)))

    assert (approved.status_code, sent_back.status_code) == (404, 404)
    assert "deleted as it was decided" in approved.detail

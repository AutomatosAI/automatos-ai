"""F195 (LOW) — a board field sent as a list or an object is a 422, not a 500.

POST /api/v1/tasks and its PATCH looked a sent ``priority``, ``review_mode`` or
``source_type`` up in a set; a JSON list or object is unhashable, so the lookup
raised TypeError and the request came back a 500. Each is now type-checked and
refused as a bad request.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException

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


class _Session:
    def __init__(self, row=None):
        self._row = row

    def query(self, _model):
        return _Rows(self._row)


def _refusal(call):
    with pytest.raises(HTTPException) as refused:                 # night: TypeError, a 500
        asyncio.run(call)
    return refused.value


@pytest.mark.parametrize("field, value", [
    ("priority", ["high"]),
    ("review_mode", {"mode": "human"}),
    ("source_type", ["user"]),
])
def test_creating_a_ticket_with_a_field_that_is_not_a_name_is_a_422(field, value):
    from api import board_tasks as bt

    refused = _refusal(bt.create_task(_Req({"title": "Order more oat milk", field: value}), ctx=CTX, db=_Session()))

    assert refused.status_code == 422 and field in refused.detail


@pytest.mark.parametrize("field, value", [("priority", ["high"]), ("review_mode", {"mode": "auto"})])
def test_patching_a_ticket_with_a_field_that_is_not_a_name_is_a_422(field, value):
    from api import board_tasks as bt
    from core.models.core import BoardTask

    task = BoardTask(id=5, workspace_id=WS, title="Order more oat milk", status="inbox", priority="medium",
                     review_mode="auto", source_type="user")
    refused = _refusal(bt.update_task(5, _Req({field: value}), ctx=CTX, db=_Session(task)))

    assert refused.status_code == 422 and field in refused.detail
    assert (task.priority, task.review_mode) == ("medium", "auto")

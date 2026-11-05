"""F105 (4 Nov): the workflow streams hand their request's connection back before they stream.

``GET /api/workflows/executions/{id}/stream``, ``GET …/stream/aisdk`` and
``POST /api/workflows/stream`` check the execution on the request's ``get_db`` session,
then stream from the in-memory stream manager, which reads no database. FastAPI closes
that session only when the response ends, so its pooled connection sat idle in
transaction for as long as the stream stayed open. Each route now ends the transaction
before it returns the stream, and is a plain ``def`` (FastAPI's threadpool), so neither
the check nor the hand-back runs on the event loop. The frames are the stream's, unchanged.

DB-free: no database, model or socket.
"""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace as NS

import pytest

from api import workflows
from consumers.workflows import streaming

FRAMES = ('0:"Stage 1"\n', 'd:{"stage": 1}\n')


class _RequestSession:
    """The request's ``get_db`` session: one running execution; records when its transaction ends."""

    def __init__(self, events):
        self._events = events
        self.execution = NS(id=7, status="running", completed_at=None)

    def query(self, *_):
        return self

    def filter(self, *_):
        return self

    def first(self):
        return self.execution

    def rollback(self):
        self._events.append("request transaction ended")


def _streams_recording(events, monkeypatch):
    async def _stream(execution_id):
        events.append(f"stream of {execution_id} started")
        for frame in FRAMES:
            yield frame

    monkeypatch.setattr(streaming, "stream_workflow_execution", _stream)
    monkeypatch.setattr(streaming, "stream_workflow_as_aisdk", _stream)


ROUTES = (
    ("stream_execution_updates", lambda db: workflows.stream_execution_updates(execution_id=7, db=db)),
    ("stream_execution_aisdk", lambda db: workflows.stream_execution_aisdk(execution_id=7, db=db)),
    ("stream_workflow_chat", lambda db: workflows.stream_workflow_chat(request={"executionId": 7}, db=db)),
)


@pytest.mark.parametrize("name,call", ROUTES, ids=[name for name, _ in ROUTES])
def test_the_request_transaction_ends_before_the_stream_starts(name, call, monkeypatch):
    events = []
    _streams_recording(events, monkeypatch)

    response = call(_RequestSession(events))
    events.append("route returned")

    async def drain():
        return [chunk async for chunk in response.body_iterator]

    frames = asyncio.run(drain())

    assert events == ["request transaction ended", "route returned", "stream of 7 started"]
    assert tuple(frames) == FRAMES


@pytest.mark.parametrize("name", [name for name, _ in ROUTES])
def test_the_stream_routes_run_in_the_threadpool(name):
    route = getattr(workflows, name)
    assert not inspect.iscoroutinefunction(route), f"{name} must be a plain def (F105)"


def test_a_finished_execution_is_still_refused_before_anything_streams(monkeypatch):
    events = []
    _streams_recording(events, monkeypatch)
    db = _RequestSession(events)
    db.execution = NS(id=7, status="completed", completed_at=None)

    with pytest.raises(workflows.HTTPException) as refused:
        workflows.stream_execution_aisdk(execution_id=7, db=db)

    assert refused.value.status_code == 400
    assert events == []

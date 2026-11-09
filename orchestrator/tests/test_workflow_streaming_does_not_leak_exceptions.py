"""The SSE and AI-SDK workflow streams never yield an exception's own text to
the client (CodeQL py/stack-trace-exposure #52 #53 #54).

Both generators already log the exception (exc_info=True) before yielding an
error event/frame — that event used to carry str(e) itself. A frontend
listener (or a curious caller) sees the stream's raw bytes directly, so this
is a real exposure, not just a log line.
"""
from __future__ import annotations

import asyncio
import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import consumers.workflows.streaming as streaming  # noqa: E402

SECRET = "db password is correct-horse-battery-staple at 10.0.0.5:5432"


class _BoomQueue:
    async def get(self):
        raise RuntimeError(SECRET)


class _FakeManager:
    def __init__(self):
        self.removed = []

    async def create_stream(self, execution_id):
        return _BoomQueue()

    async def remove_stream(self, execution_id, queue):
        self.removed.append(execution_id)


async def _collect(agen):
    out = []
    async for chunk in agen:
        out.append(chunk)
    return out


def test_sse_stream_error_event_does_not_leak_the_exception(monkeypatch):
    monkeypatch.setattr(streaming, "get_stream_manager", lambda: _FakeManager())

    chunks = asyncio.run(_collect(streaming.stream_workflow_execution(7)))

    joined = "".join(chunks)
    assert SECRET not in joined
    assert "connected" in joined  # the initial event still went out
    assert '"message": "A streaming error occurred"' in joined


def test_aisdk_stream_error_frame_does_not_leak_the_exception(monkeypatch):
    monkeypatch.setattr(streaming, "get_stream_manager", lambda: _FakeManager())

    chunks = asyncio.run(_collect(streaming.stream_workflow_as_aisdk(7)))

    joined = "".join(chunks)
    assert SECRET not in joined
    assert "A streaming error occurred" in joined

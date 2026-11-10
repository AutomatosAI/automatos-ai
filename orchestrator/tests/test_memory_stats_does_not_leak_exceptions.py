"""api/memory_stats.py never echoes an exception's own text (CodeQL
py/stack-trace-exposure #200 #36 #281 #282).

browse_memories, delete_memory and consolidate_memories (both of its
try/except blocks) each already logged with exc_info=True, then ALSO put
str(e) into the "error" field returned to the caller. Each now returns a
fixed message.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from api import memory_stats  # noqa: E402

WS = uuid.uuid4()
SECRET = "db password is correct-horse-battery-staple at 10.0.0.5:5432"


def _ctx():
    return SimpleNamespace(workspace_id=WS)


def test_browse_failure_does_not_leak_the_exception():
    service = MagicMock()
    with patch.object(memory_stats, "_get_memory_service", return_value=service), \
         patch.object(memory_stats, "_get_agent_ids", side_effect=RuntimeError(SECRET)):
        out = asyncio.run(memory_stats.browse_memories(ctx=_ctx(), db=MagicMock()))

    assert out["success"] is False
    assert SECRET not in out["error"]
    assert out["error"] == "Memory browse failed"


def test_delete_failure_does_not_leak_the_exception():
    import modules.memory.forget as forget_mod

    async def _boom(*args, **kwargs):
        raise RuntimeError(SECRET)

    with patch.object(memory_stats, "_get_memory_service", return_value=MagicMock()), \
         patch.object(forget_mod, "forget", _boom), \
         patch("core.auth.principal.resolve_user_pk", return_value=1):
        out = asyncio.run(memory_stats.delete_memory("mem-1", ctx=_ctx(), db=MagicMock()))

    assert out["success"] is False
    assert SECRET not in out["error"]
    assert out["error"] == "Memory delete failed"


def _consolidate(service, fetch_side_effect=None, store_side_effect=None):
    body = memory_stats.ConsolidateRequest(memory_ids=["a", "b"], strategy="merge")
    with patch.object(memory_stats, "_get_memory_service", return_value=service), \
         patch.object(memory_stats, "_get_agent_ids", return_value=[7]), \
         patch.object(memory_stats, "_fetch_all_scoped_memories",
                       AsyncMock(side_effect=fetch_side_effect) if fetch_side_effect else
                       AsyncMock(return_value=[
                           ("global", {"id": "a", "memory": "likes tea"}),
                           ("global", {"id": "b", "memory": "likes coffee"}),
                       ])):
        return asyncio.run(memory_stats.consolidate_memories(body, ctx=_ctx(), db=MagicMock()))


def test_consolidate_fetch_failure_does_not_leak_the_exception():
    service = MagicMock()
    out = _consolidate(service, fetch_side_effect=RuntimeError(SECRET))

    assert out["success"] is False
    assert SECRET not in out["error"]
    assert out["error"] == "Failed to fetch memories for consolidation"


def test_consolidate_store_failure_does_not_leak_the_exception():
    service = MagicMock()
    service.store_long_term = AsyncMock(side_effect=RuntimeError(SECRET))

    out = _consolidate(service)

    assert out["success"] is False
    assert SECRET not in out["error"]
    assert out["error"] == "Failed to store consolidated memory"

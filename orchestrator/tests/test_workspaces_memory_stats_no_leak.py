"""GET /api/workspaces/current/memory-stats never echoes its own exception
text (CodeQL py/stack-trace-exposure #55).
"""
from __future__ import annotations

import os
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import api.workspaces as workspaces  # noqa: E402

SECRET = "db password is correct-horse-battery-staple at 10.0.0.5:5432"


def test_a_query_failure_does_not_leak_the_exception():
    db = MagicMock()
    db.query.side_effect = RuntimeError(SECRET)
    ctx = SimpleNamespace(workspace_id=uuid.uuid4())

    out = workspaces.get_memory_stats(ctx=ctx, db=db)

    assert SECRET not in out["error"]
    assert out["error"] == "Could not read memory stats"

"""GET /api/workflows/{workflow_id}/live-progress never echoes its own
exception text (CodeQL py/stack-trace-exposure #45). Polled continuously
during a run, so a failure's text (a DB error, an internals detail) must
stay server-side.
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

import api.workflow_history as workflow_history  # noqa: E402

SECRET = "db password is correct-horse-battery-staple at 10.0.0.5:5432"


def test_a_query_failure_does_not_leak_the_exception():
    db = MagicMock()
    db.query.side_effect = RuntimeError(SECRET)
    ctx = SimpleNamespace(workspace_id=uuid.uuid4())

    out = workflow_history.get_live_progress(workflow_id=1, ctx=ctx, db=db)

    assert out["status"] == "error"
    assert SECRET not in out["error"]
    assert out["error"] == "Could not fetch live progress"

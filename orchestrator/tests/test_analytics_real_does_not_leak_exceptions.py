"""analytics_real.py's dashboard tiles never echo an exception's own text
(CodeQL py/stack-trace-exposure #165 #17 #18 #19).

Each of these endpoints already catches, but used to put str(e) straight
into the "error" field of the dict it returns — reachable by any
workspace member hitting a dashboard tile. Each now logs the trace
(logger.exception) and returns a fixed message.
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

import api.analytics_real as analytics  # noqa: E402

SECRET = "db password is correct-horse-battery-staple at 10.0.0.5:5432"


def _ctx():
    return SimpleNamespace(workspace_id=uuid.uuid4())


def test_success_rate_failure_does_not_leak_the_exception():
    db = MagicMock()
    db.query.side_effect = RuntimeError(SECRET)

    out = analytics.get_agent_success_rate(ctx=_ctx(), db=db)

    assert SECRET not in out["error"]
    assert out["error"] == "Could not calculate success rate"


def test_system_load_failure_does_not_leak_the_exception(monkeypatch):
    def _boom(interval=None):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(analytics.psutil, "cpu_percent", _boom)

    out = analytics.get_system_load_trend(ctx=_ctx(), db=MagicMock())

    assert SECRET not in out["error"]
    assert out["error"] == "Could not read system load"


def test_queue_depth_failure_does_not_leak_the_exception():
    db = MagicMock()
    db.query.side_effect = RuntimeError(SECRET)

    out = analytics.get_queue_depth(ctx=_ctx(), db=db)

    assert SECRET not in out["error"]
    assert out["error"] == "Could not read queue depth"


def test_efficiency_score_failure_does_not_leak_the_exception(monkeypatch):
    def _boom(interval=None):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(analytics.psutil, "cpu_percent", _boom)

    out = analytics.get_efficiency_score(ctx=_ctx(), db=MagicMock())

    assert SECRET not in out["error"]
    assert out["error"] == "Could not calculate efficiency score"

"""PRD-185 S1: telemetry user_id type-poison repair.

``ToolExecutionLog.user_id`` is ``Column(Integer)`` (no FK), but the chat lane
threads a Clerk *string* principal (``driving_clerk``, service.py:132) into
``caller_context["user_id"]``. Binding the raw string made every logged-in tool
call's INSERT fail — swallowed at DEBUG — so the table carried 0 organic rows
across 21 workspaces and the whole learning plane (operating graph, affinities,
SLOs, uplift eval) starved.

Fix: coerce the principal to ``users.id`` at the write boundary (ints pass
through, a Clerk id resolves via a cached ``users`` lookup, unresolvable -> None
so the row STILL lands), and make the write failure loud (WARNING, not DEBUG).

Pure unit test — telemetry.py is loaded via importlib with a fake
ToolExecutionLog and a stubbed ``core.models.core.User``, exactly like
tests/test_prd177_composio_telemetry.py. No DB / network.
"""
import importlib.util
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

_telemetry_path = Path(_orchestrator_root) / "modules" / "tools" / "execution" / "telemetry.py"
_spec = importlib.util.spec_from_file_location("telemetry_mod_p2w0", _telemetry_path)
telemetry_mod = importlib.util.module_from_spec(_spec)


class MockToolExecutionLog:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeUser:
    id = "User.id"
    clerk_user_id = "User.clerk_user_id"


_mock_composio_cache = MagicMock()
_mock_composio_cache.ToolExecutionLog = MockToolExecutionLog
_mock_core = MagicMock()
_mock_core.User = _FakeUser

_spec.loader.exec_module(telemetry_mod)

write_telemetry = telemetry_mod.write_telemetry
_coerce_user_id = telemetry_mod._coerce_user_id

_saved = {}


def setup_module(module):
    for name, stub in (
        ("core.models.composio_cache", _mock_composio_cache),
        ("core.models.core", _mock_core),
    ):
        _saved[name] = sys.modules.get(name)
        sys.modules[name] = stub
    telemetry_mod._CLERK_ID_CACHE.clear()


def teardown_module(module):
    for name, prev in _saved.items():
        if prev is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.add = MagicMock()
    db.commit = MagicMock()
    db.rollback = MagicMock()
    return db


# ---------------------------------------------------------------------------
# _coerce_user_id — the pure boundary
# ---------------------------------------------------------------------------

def test_coerce_int_passthrough(mock_db):
    assert _coerce_user_id(mock_db, 5) == 5


def test_coerce_none(mock_db):
    assert _coerce_user_id(mock_db, None) is None


def test_coerce_empty_string(mock_db):
    assert _coerce_user_id(mock_db, "   ") is None


def test_coerce_digit_string(mock_db):
    assert _coerce_user_id(mock_db, "42") == 42


def test_coerce_clerk_string_resolves(mock_db):
    telemetry_mod._CLERK_ID_CACHE.clear()
    mock_db.query.return_value.filter.return_value.first.return_value = (99,)
    assert _coerce_user_id(mock_db, "user_2abcDEF") == 99


def test_coerce_clerk_string_unresolved_is_none(mock_db):
    telemetry_mod._CLERK_ID_CACHE.clear()
    mock_db.query.return_value.filter.return_value.first.return_value = None
    assert _coerce_user_id(mock_db, "user_missing") is None


def test_coerce_caches_resolution(mock_db):
    telemetry_mod._CLERK_ID_CACHE.clear()
    mock_db.query.return_value.filter.return_value.first.return_value = (7,)
    assert _coerce_user_id(mock_db, "user_cache") == 7
    mock_db.query.reset_mock()
    # Second call for the same principal must hit the cache, not re-query.
    assert _coerce_user_id(mock_db, "user_cache") == 7
    mock_db.query.assert_not_called()


# ---------------------------------------------------------------------------
# write_telemetry end-to-end — the headline: the row now lands
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_logged_in_clerk_user_row_is_written(mock_db):
    """A Clerk-string principal no longer breaks the INSERT — a row lands with
    user_id resolved to the integer users.id."""
    telemetry_mod._CLERK_ID_CACHE.clear()
    mock_db.query.return_value.filter.return_value.first.return_value = (123,)
    await write_telemetry(
        session_factory=lambda: mock_db,
        tool_name="platform_list_agents",
        parameters={"workspace_id": "x"},
        agent_id=None,
        workspace_id=uuid4(),
        result={"success": True},
        execution_time_ms=10,
        caller_context={"user_id": "user_2abcDEF", "user_query": "list agents"},
    )
    assert mock_db.add.call_count == 1
    assert mock_db.commit.call_count == 1
    log = mock_db.add.call_args[0][0]
    assert log.user_id == 123  # resolved int, not the raw Clerk string


@pytest.mark.asyncio
async def test_unresolved_user_still_writes_row(mock_db):
    """Even when the principal can't be resolved, the row STILL lands (a null-user
    row beats no row) — the learning plane must not starve on attribution."""
    telemetry_mod._CLERK_ID_CACHE.clear()
    mock_db.query.return_value.filter.return_value.first.return_value = None
    await write_telemetry(
        session_factory=lambda: mock_db,
        tool_name="platform_list_agents",
        parameters={},
        agent_id=None,
        workspace_id=uuid4(),
        result={"success": True},
        execution_time_ms=10,
        caller_context={"user_id": "user_unknown"},
    )
    assert mock_db.add.call_count == 1
    log = mock_db.add.call_args[0][0]
    assert log.user_id is None


@pytest.mark.asyncio
async def test_write_failure_is_loud(mock_db, caplog):
    """A write failure logs at WARNING (not the DEBUG that hid the 2-month outage)."""
    mock_db.commit.side_effect = RuntimeError("boom")
    with caplog.at_level(logging.WARNING):
        await write_telemetry(
            session_factory=lambda: mock_db,
            tool_name="platform_list_agents",
            parameters={},
            agent_id=None,
            workspace_id=uuid4(),
            result={"success": True},
            execution_time_ms=10,
            caller_context={"user_id": 1},
        )
    assert any(
        r.levelno == logging.WARNING
        and "Failed to write tool execution log" in r.getMessage()
        for r in caplog.records
    ), "telemetry write failure must be logged at WARNING"
    assert mock_db.rollback.call_count == 1

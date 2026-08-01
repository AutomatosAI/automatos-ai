"""The telemetry writer owns its session — a background flush can never touch foreground work.

2026-08-01 Inbuild incident: ``fire_telemetry(self.db, ...)`` scheduled a
detached task holding the CALLER's session. When the telemetry INSERT failed
(NotNullViolation — ``tool_execution_logs.user_id`` was NOT NULL in prod for
user-less heartbeat calls), the task's rollback poisoned the caller's live
transaction: board-task creation died with "This Session's transaction has
been rolled back due to a previous exception during flush".

These tests pin the repaired contract:

1. write_telemetry opens its session from a factory and closes it — commit,
   rollback, and close all happen on the session it owns.
2. A write failure rolls back and closes the OWNED session and never raises.
3. Neither write_telemetry nor fire_telemetry accepts a session — only a
   factory. The signature IS the contract; this fails loudly if someone
   reintroduces a ``db`` parameter.
4. The user-less row (user_id=None) builds and commits — the PRD-185 S1
   intent ("the row STILL lands") that the prod NOT NULL drift was breaking.

Same loading pattern as tests/test_prd139_telemetry.py: telemetry.py is
loaded directly with a stubbed ToolExecutionLog, no app boot, no DB.
"""
import asyncio
import importlib.util
import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

_telemetry_path = Path(_orchestrator_root) / "modules" / "tools" / "execution" / "telemetry.py"
_spec = importlib.util.spec_from_file_location("telemetry_ownership_mod", _telemetry_path)
telemetry_mod = importlib.util.module_from_spec(_spec)


class _CapturedLog:
    """Stands in for ToolExecutionLog; captures constructor kwargs."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture(autouse=True)
def _stub_models(monkeypatch):
    fake_models = MagicMock()
    fake_models.ToolExecutionLog = _CapturedLog
    monkeypatch.setitem(sys.modules, "core.models.composio_cache", fake_models)
    fake_user_mod = MagicMock()
    monkeypatch.setitem(sys.modules, "core.models.core", fake_user_mod)
    _spec.loader.exec_module(telemetry_mod)
    yield


def _kwargs(**over):
    base = dict(
        tool_name="platform_update_agent",
        parameters={"agent_id": 7},
        agent_id=540,
        workspace_id=uuid4(),
        result={"success": True},
        execution_time_ms=42,
    )
    base.update(over)
    return base


@pytest.mark.asyncio
async def test_write_uses_factory_session_and_closes_it():
    own = MagicMock()
    factory = MagicMock(return_value=own)

    await telemetry_mod.write_telemetry(session_factory=factory, **_kwargs())

    factory.assert_called_once()
    assert own.add.call_count == 1
    own.commit.assert_called_once()
    own.close.assert_called_once()
    own.rollback.assert_not_called()


@pytest.mark.asyncio
async def test_write_failure_rolls_back_own_session_never_raises():
    own = MagicMock()
    own.commit.side_effect = RuntimeError("NotNullViolation: user_id")

    # Must not raise — fire-and-forget contract.
    await telemetry_mod.write_telemetry(
        session_factory=lambda: own, **_kwargs()
    )

    own.rollback.assert_called_once()
    own.close.assert_called_once()


@pytest.mark.asyncio
async def test_userless_row_still_lands():
    """Heartbeat/cadence calls have no user. The row must build with
    user_id=None and commit — the intent the prod NOT NULL drift broke."""
    own = MagicMock()

    await telemetry_mod.write_telemetry(
        session_factory=lambda: own,
        **_kwargs(caller_context={}),  # no user_id at all
    )

    own.commit.assert_called_once()
    row = own.add.call_args[0][0]
    assert row.user_id is None


def test_signatures_accept_no_session():
    """The signature is the contract: a session cannot be handed in, only a
    factory. Reintroducing a ``db`` parameter is the incident vector."""
    for fn in (telemetry_mod.write_telemetry, telemetry_mod.fire_telemetry):
        params = inspect.signature(fn).parameters
        assert "db" not in params, f"{fn.__name__} must not accept a session"
        assert "session_factory" in params
        # Everything is keyword-only — no positional slot a session could
        # silently occupy.
        positional = [
            p for p in params.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        assert positional == [], f"{fn.__name__} must be keyword-only"


@pytest.mark.asyncio
async def test_fire_telemetry_background_write_lands_on_factory_session():
    own = MagicMock()
    telemetry_mod.fire_telemetry(session_factory=lambda: own, **_kwargs())
    await asyncio.sleep(0.05)
    assert own.add.call_count == 1
    own.close.assert_called_once()

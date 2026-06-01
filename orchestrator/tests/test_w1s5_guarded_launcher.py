"""PRD-142 Wave 1 · WS-C · W1-S5 — guarded fire-and-forget launcher.

``asyncio.create_task`` is a footgun for fire-and-forget work: the loop keeps
only a *weak* reference, so an un-referenced task can be GC-cancelled mid-flight,
and an uncaught exception surfaces only as a GC-time "never retrieved" warning —
invisible to telemetry.

``launch_guarded`` replaces every bare ``create_task`` site (board task run,
wizard scrape pipeline, workflow local fallback). These tests prove the
contract:

  - the task is strongly referenced while in flight (no GC-cancellation);
  - an *uncaught* exception fires ``record_error`` with the caller's
    subsystem/operation and is then cleaned up;
  - a clean completion records nothing;
  - cancellation is NOT treated as a recordable error.
"""
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

# Importing the launcher pulls in core.utils.exception_telemetry → SessionLocal,
# which builds the SQLAlchemy engine and refuses to without POSTGRES_* creds.
# These tests never touch a real DB; setdefault means a real .env still wins.
for _k, _v in {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)

import core.utils.background_tasks as bg  # noqa: E402


async def _settle():
    """Let scheduled done-callbacks (loop.call_soon) run."""
    for _ in range(3):
        await asyncio.sleep(0)


def test_holds_strong_ref_while_in_flight(monkeypatch):
    monkeypatch.setattr(bg, "record_error", MagicMock())

    async def _drive():
        started = asyncio.Event()

        async def _work():
            started.set()
            await asyncio.sleep(0.05)

        task = bg.launch_guarded(_work(), subsystem="x", operation="y")
        await started.wait()
        # Strongly referenced while pending — this is the GC-cancellation guard.
        assert task in bg._BACKGROUND_TASKS
        await asyncio.gather(task, return_exceptions=True)
        await _settle()
        return task

    task = asyncio.run(_drive())
    assert task not in bg._BACKGROUND_TASKS  # cleaned up on completion


def test_records_uncaught_exception(monkeypatch):
    rec = MagicMock()
    monkeypatch.setattr(bg, "record_error", rec)

    async def _boom():
        raise RuntimeError("kaboom")

    async def _drive():
        task = bg.launch_guarded(_boom(), subsystem="board", operation="execute_task")
        await asyncio.gather(task, return_exceptions=True)
        await _settle()
        return task

    task = asyncio.run(_drive())
    assert task not in bg._BACKGROUND_TASKS
    rec.assert_called_once()
    kw = rec.call_args.kwargs
    assert kw["subsystem"] == "board"
    assert kw["operation"] == "execute_task"
    assert "kaboom" in str(kw["error"])


def test_clean_completion_records_nothing(monkeypatch):
    rec = MagicMock()
    monkeypatch.setattr(bg, "record_error", rec)

    async def _ok():
        return "fine"

    async def _drive():
        task = bg.launch_guarded(_ok(), subsystem="wizard", operation="scrape_pipeline")
        await asyncio.gather(task, return_exceptions=True)
        await _settle()
        return task

    task = asyncio.run(_drive())
    rec.assert_not_called()
    assert task not in bg._BACKGROUND_TASKS


def test_cancellation_is_not_recorded(monkeypatch):
    rec = MagicMock()
    monkeypatch.setattr(bg, "record_error", rec)

    async def _drive():
        async def _hang():
            await asyncio.sleep(10)

        task = bg.launch_guarded(_hang(), subsystem="x", operation="y")
        await asyncio.sleep(0)      # let it start
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await _settle()
        return task

    task = asyncio.run(_drive())
    rec.assert_not_called()         # cancellation != failure
    assert task not in bg._BACKGROUND_TASKS


def test_workspace_and_extra_are_plumbed(monkeypatch):
    rec = MagicMock()
    monkeypatch.setattr(bg, "record_error", rec)

    async def _boom():
        raise ValueError("nope")

    async def _drive():
        task = bg.launch_guarded(
            _boom(),
            subsystem="workflow",
            operation="execute",
            workspace_id="ws-123",
            agent_id=7,
            extra={"execution_id": 42},
        )
        await asyncio.gather(task, return_exceptions=True)
        await _settle()

    asyncio.run(_drive())
    kw = rec.call_args.kwargs
    assert kw["workspace_id"] == "ws-123"
    assert kw["agent_id"] == 7
    assert kw["extra"] == {"execution_id": 42}

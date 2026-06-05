"""PRD-142 Wave 1 · WS-A · W1-S1 — record_error adoption (Tier 2: board + wizard).

Companion to ``test_w1s1_hotpath_telemetry.py`` (Tier 1: planner/verification/
widget, clean seams). These two hot-paths need heavier mocking because both are
fire-and-forget background coroutines that own their own DB session and reach
their failure handler only after a real dependency blows up:

- **board** (``api.board_tasks._launch_task_execution``): schedules an inner
  ``_run()`` via ``asyncio.create_task``. We intercept ``create_task`` to capture
  the coroutine, stub ``SessionLocal``, and make ``AgentFactory`` raise so ``_run``
  falls into its terminal ``except``.
- **wizard** (``api.wizard._run_scrape_pipeline``): an async pipeline whose first
  step is ``_firecrawl_client()``. We make that raise to drive straight to the
  top-level ``except`` and stub the DB/progress side-effects in that handler.

Both subsystems fail open (the background task swallows the exception so the
request that spawned it already returned). The proof here is purely that
``record_error`` fires with the right ``subsystem``/``operation`` on the terminal
failure — that is what lights up the ERRORS-by-subsystem dashboard tile.
"""
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

# Importing api.board_tasks / api.wizard eagerly constructs the SQLAlchemy
# engine, which refuses to build without POSTGRES_* creds. These tests mock the
# DB entirely (no query ever runs), so provide inert placeholders; setdefault
# means a real .env (when present) still wins.
for _k, _v in {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)


# ---------------------------------------------------------------------------
# board (subsystem="board") — fire-and-forget background coroutine
# ---------------------------------------------------------------------------

def test_board_task_failure_emits_board_error(monkeypatch):
    from api import board_tasks as bt_mod
    import core.database.database as db_mod
    import core.utils.background_tasks as bg_mod
    import modules.agents.factory.agent_factory as factory_mod

    rec = MagicMock()
    monkeypatch.setattr(bt_mod, "record_error", rec, raising=False)

    # _launch_task_execution schedules its work via launch_guarded (W1-S5), which
    # calls asyncio.create_task in the background_tasks module. Capture the
    # coroutine there instead of letting it loose on a loop, then drive it
    # ourselves. (launch_guarded also calls .add_done_callback on the result, so
    # a MagicMock stands in for the Task fine.)
    captured: dict = {}
    monkeypatch.setattr(
        bg_mod.asyncio,
        "create_task",
        lambda coro: captured.__setitem__("coro", coro) or MagicMock(),
    )

    # SessionLocal() is called OUTSIDE the try → must succeed (return a mock).
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: MagicMock())

    # AgentFactory is constructed as the first statement inside the try; make it
    # raise so _run() falls into its terminal except hot-path. (Local import in
    # _run resolves the name off the already-loaded module, so this patch wins.)
    monkeypatch.setattr(
        factory_mod,
        "AgentFactory",
        MagicMock(side_effect=RuntimeError("agent unavailable")),
    )

    ws = str(uuid4())
    bt_mod._launch_task_execution(
        task_id=123,
        agent_id=7,
        workspace_id=ws,
        prompt="do the thing",
    )

    assert "coro" in captured, "create_task was never invoked"
    asyncio.run(captured["coro"])

    assert rec.call_count >= 1, "record_error never called on terminal board failure"
    kw = rec.call_args.kwargs
    assert kw["subsystem"] == "board"
    assert kw["operation"] == "execute_task"
    assert kw["workspace_id"] == ws


def test_board_task_success_does_not_emit(monkeypatch):
    """Guardrail: a clean agent execution must NOT record an error."""
    from api import board_tasks as bt_mod
    import core.database.database as db_mod
    import core.utils.background_tasks as bg_mod
    import modules.agents.factory.agent_factory as factory_mod

    rec = MagicMock()
    monkeypatch.setattr(bt_mod, "record_error", rec, raising=False)

    # See failure test: launch_guarded (W1-S5) owns the create_task call now.
    captured: dict = {}
    monkeypatch.setattr(
        bg_mod.asyncio,
        "create_task",
        lambda coro: captured.__setitem__("coro", coro) or MagicMock(),
    )
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: MagicMock())

    # Factory returns a normal result → no exception, no record_error. The
    # completion block is guarded by task.status == "in_progress"; the mocked
    # task's status won't match, so we never touch report/dispatch helpers.
    factory = MagicMock()
    factory.execute_with_prompt = AsyncMock(return_value={"result": "all good"})
    monkeypatch.setattr(factory_mod, "AgentFactory", lambda **kw: factory)

    bt_mod._launch_task_execution(
        task_id=123,
        agent_id=7,
        workspace_id=str(uuid4()),
        prompt="do the thing",
    )
    asyncio.run(captured["coro"])

    rec.assert_not_called()


# ---------------------------------------------------------------------------
# wizard (subsystem="wizard") — fire-and-forget intake pipeline
# ---------------------------------------------------------------------------

def test_wizard_pipeline_failure_emits_wizard_error(monkeypatch):
    from api import wizard as wiz_mod

    rec = MagicMock()
    monkeypatch.setattr(wiz_mod, "record_error", rec, raising=False)

    # _firecrawl_client() is the first call inside the pipeline's try; make it
    # raise to drive straight to the top-level except.
    monkeypatch.setattr(
        wiz_mod,
        "_firecrawl_client",
        MagicMock(side_effect=RuntimeError("firecrawl unavailable")),
    )

    # The except marks the profile failed (DB) and emits a terminal progress
    # event — stub both so no real session/SSE is touched.
    monkeypatch.setattr(wiz_mod, "get_db_session", MagicMock())
    monkeypatch.setattr(wiz_mod, "progress_emit", AsyncMock())

    ws = str(uuid4())
    pid = str(uuid4())
    asyncio.run(
        wiz_mod._run_scrape_pipeline(
            profile_id=pid,
            workspace_id=ws,
            domain="example.com",
            archetype_slug=None,
            selected_urls=["https://example.com"],
            user_goals=[],
        )
    )

    assert rec.call_count >= 1, "record_error never called on wizard pipeline failure"
    kw = rec.call_args.kwargs
    assert kw["subsystem"] == "wizard"
    assert kw["operation"] == "scrape_pipeline"
    assert kw["workspace_id"] == ws


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

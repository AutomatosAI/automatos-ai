"""PRD-239 S3b — a playbook step for a session agent, after the first live run
failed as "Stalled": the host may claim recipe tickets (the exclusion is for the
API runtime only), the step gets a session-sized deadline, and the run stamps
progress while it waits so the stall watchdog leaves it alone."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")
_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.cli_runtime import RUNTIME_API, RUNTIME_CLI  # noqa: E402
from services import cli_ticket_lane as lane  # noqa: E402
from services.board_dispatcher import recipe_exclusion_sql  # noqa: E402
from api.recipe_executor import _stamp_progress, _step_deadline  # noqa: E402


def test_recipe_tickets_are_excluded_for_the_api_runtime_only():
    assert recipe_exclusion_sql(RUNTIME_API) == "AND board_tasks.source_type <> 'recipe'"
    assert recipe_exclusion_sql(RUNTIME_API, alias="t") == "AND t.source_type <> 'recipe'"
    assert recipe_exclusion_sql(RUNTIME_CLI) == ""
    assert recipe_exclusion_sql(RUNTIME_CLI, alias="t") == ""


def test_a_session_step_gets_a_session_sized_deadline_and_api_steps_keep_theirs():
    cfg = SimpleNamespace(CLI_LANE_STEP_TIMEOUT_SECONDS=1800)
    assert _step_deadline(300, False, cfg) == 300.0
    assert _step_deadline(300, True, cfg) == 1800.0
    assert _step_deadline(3600, True, cfg) == 3600.0          # a longer recipe timeout stands
    assert _step_deadline(300, True, SimpleNamespace()) == 1800.0  # default when the knob is missing


def test_the_progress_stamp_rebuilds_the_metadata_and_never_raises():
    class _DB:
        commits = 0

        def commit(self):
            self.commits += 1

    execution = SimpleNamespace(execution_metadata={"kept": 1})
    before = execution.execution_metadata
    db = _DB()
    _stamp_progress(db, execution)
    assert execution.execution_metadata["kept"] == 1 and "last_progress_at" in execution.execution_metadata
    assert execution.execution_metadata is not before and db.commits == 1

    class _Broken:
        def commit(self):
            raise RuntimeError("db down")

    _stamp_progress(_Broken(), SimpleNamespace(execution_metadata=None))  # logged, not raised


def test_the_wait_loop_reports_progress_on_every_poll_and_tolerates_a_failing_hook(monkeypatch):
    states = iter(["assigned", "in_progress", "in_progress", "done"])
    ticket = SimpleNamespace(id=101, status="assigned", runtime_ref={}, result_summary=None, error_message=None,
                             description="", title="t", completed_at=None)

    class _Q:
        def filter(self, *a, **k):
            return self

        def first(self):
            ticket.status = next(states)
            return ticket

    class _DB:
        def expire_all(self):
            pass

        def query(self, model):
            return _Q()

    monkeypatch.setattr(lane, "file_cli_ticket", lambda db, **kw: ticket)
    monkeypatch.setattr(lane, "exec_result_for", lambda t: {"status": "success", "task_id": t.id})
    seen = []

    def hook(t):
        seen.append(t.status)
        if len(seen) == 2:
            raise RuntimeError("stamp failed once")

    out = asyncio.run(lane.run_cli_ticket_and_wait(
        _DB(), workspace_id="ws", agent_id=15, title="t", prompt="p", source_type="recipe", source_id="recipe:x:1",
        poll_s=0.01, on_poll=hook,
    ))
    assert out == {"status": "success", "task_id": 101}
    assert seen == ["assigned", "in_progress", "in_progress"]  # every non-terminal poll, the failure swallowed

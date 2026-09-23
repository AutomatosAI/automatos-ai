"""F114 (run 4) — a playbook run waiting on a session survives a database blip,
is not stalled while its step ticket is worked, and when a run is stalled its
session steps are named, not repeated.

06:43:56Z the local Postgres crash-restarted (~10 s in recovery). Every run
waiting on a Claude Code step died in the same second: the wait's poll raised,
the step retry failed at once, the run's failure could not be written, and the
executor exited with the rows still 'running'. Five minutes later the task
reconciler failed them (right — nothing drove them) and retried each from step
1, filing a second session for a step whose first session was still working.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError, ProgrammingError

from services import cli_ticket_lane as lane

# ── the wait ────────────────────────────────────────────────────────────────

CLOSED = OperationalError("SELECT board_tasks", {}, Exception("server closed the connection unexpectedly"))
RECOVERING = OperationalError("SELECT board_tasks", {}, Exception("FATAL: the database system is in recovery mode"))


class _FlakyDB:
    """What the wait sees poll by poll: an exception raises, a ticket returns."""

    def __init__(self, *outcomes):
        self.outcomes, self.rollbacks = list(outcomes), 0

    def expire_all(self):
        pass

    def query(self, model):
        return self

    def filter(self, *args):
        return self

    def first(self):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def rollback(self):
        self.rollbacks += 1


def _ticket(status):
    return SimpleNamespace(id=688, status=status, runtime_ref={}, result="Newsletter drafted.",
                           error_message=None, lease_until=None)


def _wait(monkeypatch, db, polls):
    async def instant(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", instant)
    monkeypatch.setattr(lane, "file_cli_ticket", lambda db, **kwargs: SimpleNamespace(id=688))
    return asyncio.run(lane.run_cli_ticket_and_wait(
        db, workspace_id="ws", agent_id=274, title="newsletter · step 1", prompt="draft it",
        source_type="recipe", source_id="recipe:exec-ed423d878813:1", poll_s=5, on_poll=polls.append))


def test_the_wait_rides_out_a_database_restart(monkeypatch):
    db = _FlakyDB(CLOSED, RECOVERING, _ticket("in_progress"), _ticket("done"))
    polls = []
    result = _wait(monkeypatch, db, polls)
    assert result["board_status"] == "done" and result["task_id"] == 688
    assert db.rollbacks == 2
    assert len(polls) == 1                      # progress was marked again once the database was back


def test_any_other_error_still_ends_the_wait(monkeypatch):
    with pytest.raises(RuntimeError):
        _wait(monkeypatch, _FlakyDB(RuntimeError("bug")), [])


def test_an_outage_longer_than_the_grace_ends_it(monkeypatch):
    monkeypatch.setattr(lane, "_db_outage_grace_s", lambda: -1.0)
    with pytest.raises(OperationalError):
        _wait(monkeypatch, _FlakyDB(CLOSED, CLOSED), [])


def test_what_counts_as_the_database_being_unreachable():
    assert lane.is_database_unreachable(CLOSED) and lane.is_database_unreachable(RECOVERING)
    assert lane.is_database_unreachable(InterfaceError("SELECT 1", {}, Exception("connection already closed")))
    invalidated = DBAPIError("SELECT 1", {}, Exception("gone"), connection_invalidated=True)
    assert lane.is_database_unreachable(invalidated)
    assert not lane.is_database_unreachable(ProgrammingError("SELECT nope", {}, Exception("syntax error")))
    assert not lane.is_database_unreachable(ValueError("not a database error"))


# ── the reconciler, against Postgres ────────────────────────────────────────

@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the reconciler tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def workspace(engine, new_session):
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'f114')"), {"id": ws})
    s.commit()
    yield ws
    s = new_session.sweep()
    s.execute(text("DELETE FROM board_tasks WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM recipe_executions WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM workflow_recipes WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


def _run(s, ws, recipe_id, *, stamped_minutes_ago=10, max_retries=None):
    from core.models.core import RecipeExecution

    execution_id = f"exec-{uuid.uuid4().hex[:12]}"
    long_ago = datetime.utcnow() - timedelta(minutes=stamped_minutes_ago)
    meta = {"last_progress_at": long_ago.replace(microsecond=0).isoformat()}
    if max_retries is not None:
        meta["execution_config"] = {"max_retries": max_retries}
    s.add(RecipeExecution(execution_id=execution_id, recipe_id=recipe_id, workspace_id=ws, status="running",
                          input_data={}, attempt_count=1, triggered_by="f114", started_at=long_ago - timedelta(minutes=20),
                          execution_metadata=meta))
    return execution_id


def _step_ticket(s, ws, execution_id, step, status):
    from core.models.core import BoardTask

    task = BoardTask(workspace_id=ws, title=f"f114 · step {step}", status=status, priority="medium",
                     source_type="recipe", source_id=f"recipe:{execution_id}:{step}")
    s.add(task)
    s.flush()
    return task.id


def _state(new_session, execution_id):
    return new_session().execute(
        text("SELECT status, error_message, execution_metadata FROM recipe_executions WHERE execution_id = :e"),
        {"e": execution_id},
    ).first()


def test_the_reconciler_leaves_a_live_session_step_and_stalls_a_dead_run(workspace, new_session):
    from core.models.core import WorkflowTemplate
    from services.task_reconciler import TaskReconciler

    s = new_session()
    recipe = WorkflowTemplate(template_id=f"f114-{uuid.uuid4().hex[:8]}", name="Fortnightly club newsletter",
                              description="f114", workspace_id=workspace, template_definition={"steps": []},
                              steps=[], created_by="f114")
    s.add(recipe)
    s.flush()
    waiting = _run(s, workspace, recipe.id)                            # step 1's session still working
    _step_ticket(s, workspace, waiting, 1, "in_progress")
    dead = _run(s, workspace, recipe.id, max_retries=1)                # nobody driving, nothing filed
    stranded = _run(s, workspace, recipe.id)                           # nobody driving, steps already ran
    done_id = _step_ticket(s, workspace, stranded, 1, "done")
    review_id = _step_ticket(s, workspace, stranded, 2, "review")
    s.commit()

    asyncio.run(TaskReconciler()._tick())

    assert _state(new_session, waiting).status == "running"           # survives 300 s on a live ticket
    status, error, _meta = _state(new_session, dead)
    assert status == "failed" and error == "Stalled: no progress for 300s (status was 'running')"
    status, error, meta = _state(new_session, stranded)
    assert status == "failed" and f"#{done_id} step 1 done" in error and f"#{review_id} step 2 review" in error
    assert [t["id"] for t in meta["step_tickets"]] == [done_id, review_id]
    retries = new_session().execute(
        text("SELECT count(*) FROM recipe_executions WHERE retry_of = :e"), {"e": stranded}).scalar()
    assert retries == 0                                                # its sessions are not run again

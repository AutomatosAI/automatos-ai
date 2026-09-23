"""F116 (run 4) — cancelling a playbook run stops the sessions working its steps.

The owner cancelled two runs at 06:54; their session step tickets carried on —
#725 was still in progress on a playbook nobody was running. Now the run's
cancel closes each live ``recipe:<run>:<step>`` ticket through the board's own
cancel (the host's next event batch gets ``control: cancel``), saying who
cancelled and that it went with the run. A finished step keeps its result; a
backend restart still does not kill the sessions.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

from services.cli_host_service import SESSION_TOKEN_HASH_KEY as TOKEN_KEY  # noqa: E402


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the cancel tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def workspace(engine, new_session):
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'f116')"), {"id": ws})
    s.commit()
    yield ws
    s = new_session.sweep()
    s.execute(text("DELETE FROM board_tasks WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM recipe_executions WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM workflow_recipes WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


def _recipe(s, ws):
    from core.models.core import WorkflowTemplate

    recipe = WorkflowTemplate(template_id=f"f116-{uuid.uuid4().hex[:8]}", name="Roast day write-up",
                              description="f116", workspace_id=ws, template_definition={"steps": []},
                              steps=[], created_by="f116")
    s.add(recipe)
    s.flush()
    return recipe


def _run(s, ws, recipe_id):
    from core.models.core import RecipeExecution

    execution_id = f"exec-{uuid.uuid4().hex[:12]}"
    s.add(RecipeExecution(execution_id=execution_id, recipe_id=recipe_id, workspace_id=ws, status="running",
                          input_data={}, attempt_count=1, triggered_by="f116"))
    return execution_id


def _step(s, ws, execution_id, step, status):
    from core.models.core import BoardTask

    task = BoardTask(workspace_id=ws, title=f"Roast day write-up · step {step}", status=status, priority="medium",
                     source_type="recipe", source_id=f"recipe:{execution_id}:{step}",
                     runtime_ref={TOKEN_KEY: "hash-of-a-live-session"} if status == "in_progress" else {})
    s.add(task)
    s.flush()
    return task.id


def _tickets(new_session, ids):
    rows = new_session().execute(
        text("SELECT id, status, runtime_ref FROM board_tasks WHERE id = ANY(:ids)"), {"ids": list(ids)}).fetchall()
    return {row.id: row for row in rows}


def test_cancelling_a_run_leaves_no_session_step_running(workspace, new_session):
    from api.workflow_recipes import cancel_execution

    s = new_session()
    recipe = _recipe(s, workspace)
    run = _run(s, workspace, recipe.id)
    reviewed = _step(s, workspace, run, 1, "review")
    finished = _step(s, workspace, run, 2, "done")
    working = _step(s, workspace, run, 3, "in_progress")
    queued = _step(s, workspace, run, 4, "assigned")
    other_run = _run(s, workspace, recipe.id)
    elsewhere = _step(s, workspace, other_run, 1, "in_progress")
    s.commit()

    ctx = SimpleNamespace(workspace_id=uuid.UUID(workspace), user_id="2")
    result = asyncio.run(cancel_execution(str(recipe.id), run, ctx=ctx, db=new_session()))
    assert result["status"] == "cancelled"

    rows = _tickets(new_session, [reviewed, finished, working, queued, elsewhere])
    live = [t for t in (working, queued) if rows[t].status in ("inbox", "assigned", "in_progress", "blocked")]
    assert live == []                                                    # nothing of this run still runs
    for ticket in (working, queued):
        ref = rows[ticket].runtime_ref
        assert rows[ticket].status == "cancelled" and ref.get("cancel_requested_at")   # the host stops the session
        assert ref["cancelled"]["by"] == "user:2" and ref["cancelled"]["reason"] == f"cancelled with run {run}"
    assert TOKEN_KEY not in rows[working].runtime_ref                   # its session credential is gone
    assert (rows[reviewed].status, rows[finished].status) == ("review", "done")   # finished steps keep results
    assert rows[elsewhere].status == "in_progress"                       # another run's session is untouched


def test_the_boards_own_cancel_now_says_who_and_why(workspace, new_session):
    from api.board_tasks import cancel_task

    s = new_session()
    recipe = _recipe(s, workspace)
    run = _run(s, workspace, recipe.id)
    working = _step(s, workspace, run, 1, "in_progress")
    finished = _step(s, workspace, run, 2, "done")
    s.commit()

    ctx = SimpleNamespace(workspace_id=uuid.UUID(workspace), user_id="2")
    assert asyncio.run(cancel_task(working, ctx=ctx, db=new_session()))["applied"] is True
    assert asyncio.run(cancel_task(finished, ctx=ctx, db=new_session()))["applied"] is False
    ref = _tickets(new_session, [working])[working].runtime_ref
    assert ref["cancelled"]["by"] == "user:2" and ref["cancelled"]["reason"] == "cancelled on the board"


def test_a_backend_restart_does_not_kill_the_sessions(workspace, new_session):
    """The executor marks its run cancelled on any CancelledError — a shutdown
    included. It cannot tell that from the owner, so it leaves the sessions."""
    from api.recipe_executor import _mark_execution_cancelled

    s = new_session()
    recipe = _recipe(s, workspace)
    run = _run(s, workspace, recipe.id)
    working = _step(s, workspace, run, 1, "in_progress")
    s.commit()

    asyncio.run(_mark_execution_cancelled(run, None))
    status = new_session().execute(
        text("SELECT status FROM recipe_executions WHERE execution_id = :e"), {"e": run}).scalar()
    assert status == "cancelled" and _tickets(new_session, [working])[working].status == "in_progress"

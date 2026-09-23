"""A playbook run whose task is killed without the cancel endpoint is marked stopped.

``_mark_execution_cancelled`` (c6ba1dbe0) set ``completed_at = sa_func.now()``
with ``sa_func`` never imported in recipe_executor: every call NameError'd into
its "best-effort" warning, and a run stopped by a shutdown stayed 'running'
until the task reconciler stalled it (found writing F116's tests, 23 Sep).
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import create_engine, text


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"needs a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def runs(engine, new_session):
    from core.models.core import RecipeExecution, WorkflowTemplate

    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'stopped')"), {"id": ws})
    recipe = WorkflowTemplate(template_id=f"stop-{uuid.uuid4().hex[:8]}", name="stopped", description="stopped",
                              workspace_id=ws, template_definition={"steps": []}, steps=[], created_by="test")
    s.add(recipe)
    s.flush()
    ids = {}
    for key, status, error in (("open", "running", None), ("owner", "cancelled", "Cancelled by user")):
        ids[key] = f"exec-{uuid.uuid4().hex[:12]}"
        s.add(RecipeExecution(execution_id=ids[key], recipe_id=recipe.id, workspace_id=ws, status=status,
                              error_message=error, input_data={}, attempt_count=1, triggered_by="test"))
    s.commit()
    yield ids
    s = new_session.sweep()
    s.execute(text("DELETE FROM recipe_executions WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM workflow_recipes WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


def _row(new_session, execution_id):
    return new_session().execute(
        text("SELECT status, error_message, completed_at FROM recipe_executions WHERE execution_id = :e"),
        {"e": execution_id},
    ).first()


def test_a_run_stopped_without_the_endpoint_is_marked_stopped(runs, new_session):
    from api.recipe_executor import RUN_STOPPED_TEXT, _mark_execution_cancelled

    asyncio.run(_mark_execution_cancelled(runs["open"], None))
    status, error, completed_at = _row(new_session, runs["open"])
    assert (status, error) == ("cancelled", RUN_STOPPED_TEXT) and completed_at is not None


def test_a_run_the_owner_cancelled_keeps_what_the_endpoint_wrote(runs, new_session):
    from api.recipe_executor import _mark_execution_cancelled

    asyncio.run(_mark_execution_cancelled(runs["owner"], None))
    assert _row(new_session, runs["owner"])[:2] == ("cancelled", "Cancelled by user")

"""PRD-163 S2 — plan mode + plan import.

Pure: the MIN_TASKS=3 floor is gone (a 1-task plan is legal).
Integration: import_plan persists the EXACT given DAG (no re-decomposition).
"""

from __future__ import annotations

import os
import sys
import types
import uuid

for _k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
    os.environ.setdefault(_k, "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
sys.modules.setdefault("camelot", types.ModuleType("camelot"))

import pytest  # noqa: E402


def test_min_tasks_floor_removed():
    """Q55: a single-task plan no longer trips the minimum-tasks validator."""
    from modules.coordination.planner import _validate_plan, PlannedTask, MIN_TASKS

    assert MIN_TASKS == 1
    task = PlannedTask(
        temp_id="t1", title="Do the thing", description="", agent_role="generalist",
        sequence_number=1, task_type="execution", verification_criteria=[],
        required_tools=[], dependencies=[],
    )
    errors = _validate_plan([task], [], [])
    # Other structural checks may complain (e.g. unknown agent role), but the
    # min-tasks floor must NOT be one of them.
    assert not any("minimum" in e.lower() for e in errors)


@pytest.mark.integration
def test_import_plan_persists_exact_dag(db_session, seed_workspace):
    """An imported plan creates exactly the given tasks + dependency edges,
    without re-running the planner."""
    from sqlalchemy import text
    from services.coordinator_service import CoordinatorService
    from core.models.orchestration import OrchestrationTask, OrchestrationTaskDependency
    from core.models.orchestration_enums import RunState

    ws = uuid.UUID(seed_workspace())
    plan = {
        "tasks": [
            {"temp_id": "a", "title": "Research", "agent_role": "researcher", "sequence_number": 1},
            {"temp_id": "b", "title": "Write", "agent_role": "writer", "sequence_number": 2, "dependencies": ["a"]},
        ],
        "dependencies": [{"from": "a", "to": "b"}],
    }

    run = CoordinatorService().import_plan(
        db=db_session, workspace_id=ws, goal="research then write",
        plan=plan, created_by="user_test",
    )
    db_session.flush()

    assert run.state == RunState.AWAITING_APPROVAL.value
    assert run.config.get("imported_plan") is True

    tasks = (
        db_session.query(OrchestrationTask)
        .filter(OrchestrationTask.run_id == run.id)
        .order_by(OrchestrationTask.sequence_number)
        .all()
    )
    assert [t.title for t in tasks] == ["Research", "Write"]   # exact, no re-decomposition

    # the single dependency edge (Write depends on Research) exists
    write = next(t for t in tasks if t.title == "Write")
    research = next(t for t in tasks if t.title == "Research")
    edge = (
        db_session.query(OrchestrationTaskDependency)
        .filter(
            OrchestrationTaskDependency.task_id == write.id,
            OrchestrationTaskDependency.depends_on_task_id == research.id,
        )
        .first()
    )
    assert edge is not None


@pytest.mark.integration
def test_import_plan_rejects_empty(db_session, seed_workspace):
    from services.coordinator_service import CoordinatorService

    ws = uuid.UUID(seed_workspace())
    with pytest.raises(ValueError):
        CoordinatorService().import_plan(
            db=db_session, workspace_id=ws, goal="x", plan={"tasks": []}, created_by="user_test",
        )

"""PRD-163 S4/Q57 — approval-time plan editing (the apply-path).

`apply_plan_task_edits` is the pure core of `CoordinatorService.update_mission_plan`:
it mutates the OrchestrationTask rows the dispatcher will execute and mirrors the
change into the run.plan snapshot. These tests prove an edited agent_role persists
onto the task row (S4 acceptance: "edited agent override persists into execution")
without needing a live DB.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

for _k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
    os.environ.setdefault(_k, "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")

from services.coordinator_service import apply_plan_task_edits  # noqa: E402


def _task(id_, seq, agent_role="generalist", title="t", description="d"):
    return SimpleNamespace(
        id=id_, sequence_number=seq, agent_role=agent_role,
        title=title, description=description,
    )


def _plan(tasks):
    return {
        "tasks": [
            {"temp_id": f"t{t.sequence_number}", "sequence_number": t.sequence_number,
             "agent_role": t.agent_role, "title": t.title, "description": t.description}
            for t in tasks
        ],
        "dependencies": [],
    }


def test_agent_override_persists_to_task_row():
    tasks = [_task("id-1", 1, agent_role="generalist"), _task("id-2", 2, agent_role="generalist")]
    plan = _plan(tasks)

    new_plan, changed = apply_plan_task_edits(
        tasks, plan, [{"sequence_number": 1, "agent_role": "researcher"}]
    )

    assert changed == 1
    # The executable row is mutated — this is what the dispatcher reads.
    assert tasks[0].agent_role == "researcher"
    assert tasks[1].agent_role == "generalist"
    # Snapshot mirrors the row.
    assert new_plan["tasks"][0]["agent_role"] == "researcher"
    assert new_plan["tasks"][1]["agent_role"] == "generalist"


def test_match_by_task_id_and_temp_id():
    tasks = [_task("id-1", 1), _task("id-2", 2)]
    plan = _plan(tasks)

    _, by_id = apply_plan_task_edits(tasks, plan, [{"task_id": "id-2", "title": "edited"}])
    assert by_id == 1 and tasks[1].title == "edited"

    _, by_temp = apply_plan_task_edits(tasks, plan, [{"temp_id": "t1", "agent_role": "writer"}])
    assert by_temp == 1 and tasks[0].agent_role == "writer"


def test_multi_field_and_multi_task_edits():
    tasks = [_task("id-1", 1), _task("id-2", 2), _task("id-3", 3)]
    plan = _plan(tasks)

    new_plan, changed = apply_plan_task_edits(tasks, plan, [
        {"sequence_number": 1, "agent_role": "researcher", "title": "research"},
        {"sequence_number": 3, "description": "final write-up"},
    ])

    assert changed == 3  # 2 fields on task1 + 1 on task3
    assert tasks[0].agent_role == "researcher" and tasks[0].title == "research"
    assert tasks[2].description == "final write-up"
    assert new_plan["tasks"][2]["description"] == "final write-up"


def test_noop_when_value_unchanged_or_unmatched():
    tasks = [_task("id-1", 1, agent_role="researcher")]
    plan = _plan(tasks)

    # same value -> no change counted
    _, same = apply_plan_task_edits(tasks, plan, [{"sequence_number": 1, "agent_role": "researcher"}])
    assert same == 0

    # unmatched task -> ignored, no crash
    _, miss = apply_plan_task_edits(tasks, plan, [{"sequence_number": 99, "agent_role": "x"}])
    assert miss == 0

    # non-editable field is ignored (only agent_role/title/description honoured)
    _, ignored = apply_plan_task_edits(tasks, plan, [{"sequence_number": 1, "task_type": "hacked"}])
    assert ignored == 0


def test_empty_edits_safe():
    tasks = [_task("id-1", 1)]
    plan = _plan(tasks)
    new_plan, changed = apply_plan_task_edits(tasks, plan, [])
    assert changed == 0 and new_plan == plan

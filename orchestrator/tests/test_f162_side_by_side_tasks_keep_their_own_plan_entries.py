"""F162 (night 5, persona B2) — tasks that run side by side keep their own plan entries.

After the owner fixed a plan's staffing, the plan the approval card shows repeated
one step's title for several steps: mission b2dac26b's three step-2 tasks
("Draft personalized tasting visit notes…", "List free November slots…", "Verify
numbers…") all read "List free November slots for visits" and all named agent
274, while the task rows kept their own titles. The plan snapshot paired plan
tasks with rows by sequence_number, which tasks side by side share, so an edit
to one copied its title, role and description onto every sibling (and the agent
match previews did the same). A row now carries its plan task's temp_id.
"""
from __future__ import annotations

from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from services import coordinator_service as cs
from services.coordinator_service import apply_plan_task_edits

STEP_2 = [("task_2", "Draft personalized tasting visit notes for each café", "WRITER", "drafting"),
          ("task_3", "List free November slots for visits", "OPS", "scheduling"),
          ("task_4", "Verify numbers in draft notes and Christmas letter", "COUNTINGHOUSE", "verification")]


def _plan():
    return {"tasks": [{"temp_id": temp, "title": title, "description": f"{title}.", "agent_role": "analyst",
                       "sequence_number": 2, "parallel_group": group} for temp, title, _role, group in STEP_2]}


def _rows(*, stamped=True):
    return [NS(id=f"row-{temp}", title=title, description=f"{title}.", agent_role="analyst", sequence_number=2,
               input_context={"plan_temp_id": temp} if stamped else {})
            for temp, title, _role, _group in STEP_2]


@pytest.mark.parametrize("stamped", [True, False], ids=["rows-made-since-f162", "older-rows"])
def test_restaffing_one_side_by_side_task_changes_only_its_plan_entry(stamped):
    rows = _rows(stamped=stamped)
    plan, changed = apply_plan_task_edits(rows, _plan(), [{"temp_id": "task_3", "agent_role": "OPS"}])

    assert changed == 1
    assert [t.agent_role for t in rows] == ["analyst", "OPS", "analyst"]
    assert [pt["title"] for pt in plan["tasks"]] == [title for _t, title, _r, _g in STEP_2]
    assert [pt["agent_role"] for pt in plan["tasks"]] == ["analyst", "OPS", "analyst"]


def test_a_step_number_names_no_one_task_when_several_run_side_by_side():
    with pytest.raises(ValueError, match="Step 2 has 3 tasks that run side by side; name the one to change"):
        apply_plan_task_edits(_rows(), _plan(), [{"sequence_number": 2, "agent_role": "OPS"}])


def test_each_side_by_side_task_shows_its_own_agent_match():
    rows = _rows()
    run = NS(id="b2dac26b", plan=_plan())
    agents = {temp: NS(agent_id=270 + i, agent_name=role, total_score=0.9, reason=f"{role} fits",
                       is_override=False) for i, (temp, _title, role, _group) in enumerate(STEP_2)}

    def _rank(db, task, agents, task_spec, semantic=None):
        return [by_temp[task.input_context["plan_temp_id"]]]

    by_temp = agents
    with patch.object(cs.AgentMatcher, "rank", side_effect=_rank):
        cs.CoordinatorService._annotate_match_previews(cs.CoordinatorService.__new__(cs.CoordinatorService),
                                                       None, run, [], rows)
    assert [pt["match_agent"] for pt in run.plan["tasks"]] == ["WRITER", "OPS", "COUNTINGHOUSE"]

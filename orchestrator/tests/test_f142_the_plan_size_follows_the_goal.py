"""F142: a plan is as long as its goal needs.

The validator has accepted a one-task plan since PRD-163 (MIN_TASKS = 1). Both
planner prompts still ordered "between 3 and 20 tasks", so the 25 Sep goal
(WRITER drafts, COUNTINGHOUSE checks, OPS lists slots) was padded with extra
agents and a synthesis step. Both prompts now state the validator's range,
built from MIN_TASKS and MAX_TASKS, and say that no task is added only to
reach a count.
"""
from __future__ import annotations

import re


def test_the_planner_prompts_state_the_range_the_validator_enforces():
    from modules.coordination import planner

    for prompt in (planner._SYSTEM_PROMPT, planner._REPLAN_SYSTEM_PROMPT):
        stated = tuple(int(n) for n in re.search(r"between (\d+) and (\d+) tasks", prompt).groups())
        assert stated == (planner.MIN_TASKS, planner.MAX_TASKS)
        assert "never added only to reach a count" in prompt

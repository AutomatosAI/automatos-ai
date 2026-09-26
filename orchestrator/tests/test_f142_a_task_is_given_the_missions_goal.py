"""F142 (d1): a mission's task is given the mission's goal.

The owner asked for 20-minute visits; the plan made them two hours. The
executing agents never saw the goal, only the planner's task text. Each
task's prompt now carries the goal, in the owner's words, capped at
MISSION_GOAL_PROMPT_CHARS. That covers a first attempt and a revision.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

GOAL = ("WRITER drafts the club newsletter, COUNTINGHOUSE checks every number, OPS lists the free "
        "November slots and books nothing. Visits are 20 minutes.")


def _task(input_context=None):
    return NS(id="t-1", title="List the free November slots", description="From the club calendar.",
              input_context=input_context or {}, verification_criteria=None)


def test_a_task_prompt_carries_the_owners_goal():
    from modules.coordination.dispatcher import MissionDispatcher

    prompt = MissionDispatcher.build_task_prompt(_task(), goal=GOAL)
    assert "## The mission's goal" in prompt and GOAL in prompt
    assert "## The mission's goal" not in MissionDispatcher.build_task_prompt(_task())


def test_a_revision_keeps_the_goal():
    from modules.coordination.dispatcher import MissionDispatcher

    revision = _task({"previous_output": "Two-hour visits on 3, 10 and 17 November.",
                      "verification_feedback": {"reasoning": "The visits are too long.", "attempt": 1}})
    prompt = MissionDispatcher.build_task_prompt(revision, goal=GOAL)
    assert prompt.startswith("# Revision Request") and GOAL in prompt


def test_a_long_goal_is_capped():
    from modules.coordination.dispatcher import MISSION_GOAL_PROMPT_CHARS, MissionDispatcher

    prompt = MissionDispatcher.build_task_prompt(_task(), goal="x" * (MISSION_GOAL_PROMPT_CHARS + 500))
    assert "x" * MISSION_GOAL_PROMPT_CHARS + " …" in prompt
    assert "x" * (MISSION_GOAL_PROMPT_CHARS + 1) not in prompt

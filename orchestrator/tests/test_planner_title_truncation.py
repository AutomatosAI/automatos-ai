"""Regression net: task titles must fit orchestration_tasks.title (VARCHAR 500).

The decomposition templates build a task title by embedding the mission goal,
e.g. ``"Research scope and criteria for: {goal}"``. With a long goal (a
detailed research-led brief runs to thousands of characters) the rendered
title blew past the ``VARCHAR(500)`` column and aborted mission creation with
``StringDataRightTruncation`` — so a mission could not be (re-)created.

``_parse_plan`` is the single choke point both the template path and the LLM
path flow through, so it caps every title to ``MAX_TASK_TITLE_LEN``. The full
goal is never lost: it lives in the task ``description`` (a TEXT column).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.models.orchestration import OrchestrationTask  # noqa: E402
from modules.coordination.planner import (  # noqa: E402
    MAX_TASK_TITLE_LEN,
    _parse_plan,
)
from modules.coordination.templates import (  # noqa: E402
    TEMPLATE_REGISTRY,
    render_template,
)

# The real constraint we must never overflow: the DB column width.
_TITLE_COLUMN_LEN = OrchestrationTask.__table__.c.title.type.length

# A goal long enough to overflow VARCHAR(500) once embedded in a title.
_LONG_GOAL = (
    "Build a detailed research-led blog post comparing agentic AI "
    "orchestration frameworks across reliability, cost, and developer "
    "experience. " * 80
)


def test_long_title_is_capped_with_ellipsis():
    """A title past the cap is truncated and marked with an ellipsis."""
    long_title = "x" * (MAX_TASK_TITLE_LEN + 500)
    raw = {"tasks": [{"temp_id": "task_1", "title": long_title, "agent_role": "writer"}]}
    errors: list = []

    tasks, _ = _parse_plan(raw, errors)

    assert errors == []
    assert len(tasks) == 1
    assert len(tasks[0].title) <= MAX_TASK_TITLE_LEN
    assert tasks[0].title.endswith("…")


def test_short_title_passes_through_unchanged():
    """A normal-length title is left exactly as-is."""
    raw = {"tasks": [{"temp_id": "task_1", "title": "Draft the report", "agent_role": "writer"}]}
    errors: list = []

    tasks, _ = _parse_plan(raw, errors)

    assert tasks[0].title == "Draft the report"


def test_cap_is_within_db_column_width():
    """The cap must fit the orchestration_tasks.title column."""
    assert MAX_TASK_TITLE_LEN <= _TITLE_COLUMN_LEN


def test_research_template_long_goal_does_not_overflow_column():
    """Reproduce the incident: research_and_report + a long goal stays in-bounds."""
    template = next(t for t in TEMPLATE_REGISTRY if t.id == "research_and_report")
    raw_tasks = render_template(template, _LONG_GOAL)
    errors: list = []

    tasks, _ = _parse_plan({"tasks": raw_tasks}, errors)

    assert errors == []
    assert tasks, "template should yield tasks"
    for task in tasks:
        assert len(task.title) <= MAX_TASK_TITLE_LEN
        assert len(task.title) <= _TITLE_COLUMN_LEN


def test_full_goal_preserved_in_description():
    """Truncating the title must not lose the goal — it stays in the description."""
    template = next(t for t in TEMPLATE_REGISTRY if t.id == "research_and_report")
    raw_tasks = render_template(template, _LONG_GOAL)
    errors: list = []

    tasks, _ = _parse_plan({"tasks": raw_tasks}, errors)

    # At least one task embeds the goal in its description; the full goal text
    # survives there even though the title was capped.
    assert any(_LONG_GOAL in task.description for task in tasks)


def test_all_templates_long_goal_fit_column():
    """Every template that embeds {goal} must stay within the title column."""
    errors_total: list = []
    for template in TEMPLATE_REGISTRY:
        raw_tasks = render_template(template, _LONG_GOAL)
        errors: list = []
        tasks, _ = _parse_plan({"tasks": raw_tasks}, errors)
        assert errors == [], f"{template.id}: {errors}"
        for task in tasks:
            assert len(task.title) <= _TITLE_COLUMN_LEN, (
                f"{template.id}/{task.temp_id} title overflows column"
            )

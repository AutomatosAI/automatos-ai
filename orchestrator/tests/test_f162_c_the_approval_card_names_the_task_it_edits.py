"""F162 (c) (night 5, JEV) — the in-chat approval card names the task it edits.

MissionApprovalWidget sent ``{sequence_number: seq, agent_role}`` for the task a
person re-staffed. Since F162 an edit by step number is refused when several
tasks run side by side at that step (81e8f37a had seven at step 1), so the card
could not re-staff any of them. The card's tasks come from the create-mission
tool result (``_plan_task_summary``); each now carries its plan ``temp_id``,
which the card sends back (the widget half is its own frontend commit).
"""
from __future__ import annotations

from types import SimpleNamespace as NS

from modules.tools.discovery.handlers_missions import _plan_task_summary
from services.coordinator_service import apply_plan_task_edits

STEP_1 = [("task_1", "Split the box for the club"), ("task_2", "Draft Club Member Email for Christmas Box"),
          ("task_3", "Draft the web shop page")]


def _plan():
    return {"tasks": [{"temp_id": temp, "title": title, "description": title, "agent_role": "writer",
                       "sequence_number": 1} for temp, title in STEP_1]}


def test_each_task_on_the_card_carries_its_temp_id():
    summary = _plan_task_summary(_plan()["tasks"])
    assert [entry.get("temp_id") for entry in summary] == ["task_1", "task_2", "task_3"]


def test_the_cards_edit_reaches_its_own_side_by_side_task():
    """What the card now sends: the temp_id it was given, and the role typed."""
    (_first, second, _third) = _plan_task_summary(_plan()["tasks"])
    rows = [NS(id=f"row-{temp}", title=title, description=title, agent_role="writer", sequence_number=1,
               input_context={"plan_temp_id": temp}) for temp, title in STEP_1]

    plan, changed = apply_plan_task_edits(rows, _plan(), [{"temp_id": second["temp_id"],
                                                           "agent_role": "CLUB SECRETARY"}])

    assert changed == 1
    assert [pt["agent_role"] for pt in plan["tasks"]] == ["writer", "CLUB SECRETARY", "writer"]

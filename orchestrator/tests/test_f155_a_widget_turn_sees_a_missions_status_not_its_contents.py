"""F155: a widget turn with missions:read sees a mission's status, not its contents.

missions:read is offered for public keys, and grants platform_list_missions
and platform_get_mission. platform_get_mission returned the mission as the
owner sees it:
- its config (now the owner's staffing, in their words, plus budget and origin);
- its plan;
- each task's output (up to 500 characters), its errors and its match reason;
- who started it.
platform_list_missions named who started each mission. On a widget turn both
now return a visitor's view: what the mission is for and how far it has got.
"""
from __future__ import annotations

import asyncio
from uuid import UUID

PRIVATE = "£42k"


def _widget_turn():
    from core.security.surface import WIDGET, turn_surface

    return turn_surface(WIDGET, ("chat", "missions:read"), None)


def test_a_widget_turn_sees_a_missions_status_not_its_contents(db_session, seed_workspace):
    from core.models.orchestration import OrchestrationRun, OrchestrationTask
    from modules.tools.discovery.handlers_missions import get_mission, list_missions

    ws = UUID(seed_workspace())
    run = OrchestrationRun(
        workspace_id=ws, goal="Hire the new head chef", state="running", created_by="user_owner",
        config={"staffing": [{"agent_id": 7, "agent_name": "WRITER", "does": f"drafts the offer at {PRIVATE}"}]},
        plan={"tasks": [{"title": "Draft the offer", "description": f'The owner\'s words: "offer {PRIVATE}"'}]})
    db_session.add(run)
    db_session.flush()
    db_session.add_all([
        OrchestrationTask(run_id=run.id, sequence_number=1, title=f"Draft the offer at {PRIVATE}", state="verified",
                          output=f"Dear chef, we offer {PRIVATE}."),
        OrchestrationTask(run_id=run.id, sequence_number=2, title="Check the numbers", state="running"),
    ])
    db_session.flush()

    with _widget_turn():
        mission = asyncio.run(get_mission(db_session, ws, {"mission_id": str(run.id)}))["mission"]
        listed = asyncio.run(list_missions(db_session, ws, {}))["missions"]
    assert set(mission) == {"id", "goal", "state", "task_count", "tasks_done", "created_at", "completed_at"}
    assert (mission["goal"], mission["task_count"], mission["tasks_done"]) == ("Hire the new head chef", 2, 1)
    assert PRIVATE not in str(mission)
    assert [m["id"] for m in listed] == [run.id] and "user_owner" not in str(listed)

    owner = asyncio.run(get_mission(db_session, ws, {"mission_id": str(run.id)}))["mission"]
    assert owner["config"]["staffing"] and owner["tasks"][0]["result_summary"].startswith("Dear chef")

"""F241 (night 7): Auto names a mission's card and a playbook run's card by number.

It never once did on night 7. A mission came back as its run's id ("The mission ID
is c0a86f4e-…"), or as a number Auto made up ("142"). A playbook run came back as "I
don't have a way to directly retrieve the card number of a playbook execution",
though its card was #0145.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest


@pytest.fixture
def mission(db_session, seed_workspace):
    """A mission awaiting approval, with its card and one step's card, as planning files them."""
    from core.models.orchestration import OrchestrationRun, OrchestrationTask
    from services.orchestration_board_bridge import create_mission_board_task, create_task_board_task

    ws = UUID(seed_workspace())
    run = OrchestrationRun(workspace_id=ws, goal="Get the wholesale cafes ready for the price change",
                           state="awaiting_approval", created_by="user_test", config={})
    db_session.add(run)
    db_session.flush()
    step = OrchestrationTask(run_id=run.id, title="List every cafe and what it owes", description="Do it.",
                             sequence_number=1, state="pending", state_type="initial")
    db_session.add(step)
    db_session.flush()
    card = create_mission_board_task(db_session, run)
    create_task_board_task(db_session, run, step)
    db_session.flush()
    return NS(db=db_session, ws=ws, run=run, card=card)


def _call(handler, mission, **params):
    return asyncio.run(handler(mission.db, mission.ws, {"mission_id": str(mission.run.id), **params}))


def test_a_missions_card_and_its_steps_are_named_by_number(mission):
    from modules.tools.discovery.handlers_missions import get_mission, list_missions

    number = f"#{mission.card.workspace_seq:04d}"
    got = _call(get_mission, mission)["mission"]
    listed = asyncio.run(list_missions(mission.db, mission.ws, {}))["missions"]

    assert got["number"] == number and got["tasks"][0]["number"] == f"{number}.1"
    assert [m["number"] for m in listed] == [number]


def test_a_mission_tools_answer_names_its_card(mission):
    from modules.tools.discovery.handlers_missions import reject_mission

    reply = _call(reject_mission, mission, reason="Not this week")
    assert reply["success"] is True and reply["number"] == f"#{mission.card.workspace_seq:04d}"


def test_a_widget_turn_is_given_no_number(mission):
    from core.security.surface import WIDGET, turn_surface
    from modules.tools.discovery.handlers_missions import list_missions

    with turn_surface(WIDGET):
        listed = asyncio.run(list_missions(mission.db, mission.ws, {}))["missions"]
    assert "number" not in listed[0]


def test_a_playbook_runs_card_is_made_when_it_starts_and_named_by_number(db_session, seed_workspace, monkeypatch):
    import modules.tools.discovery.handlers_watches as watches
    import services.concurrency_guard as guard
    import services.playbook_engine as engine
    from core.models.core import WorkflowTemplate
    from modules.tools.discovery.handlers_playbooks import execute_playbook, get_playbook_execution

    async def allowed(workspace_id, db):
        return NS(allowed=True, reason="")

    monkeypatch.setattr(guard, "check_concurrency", allowed)
    monkeypatch.setattr(engine, "get_playbook_engine", lambda: NS(launch=lambda **kw: None))
    monkeypatch.setattr(watches, "auto_create_watch", lambda *a, **k: None)
    ws = UUID(seed_workspace())
    playbook = WorkflowTemplate(template_id=f"f241-{uuid.uuid4().hex[:8]}", name="Monday Stock Report",
                                description="What to reorder.", workspace_id=ws, template_definition={"steps": []},
                                steps=[], created_by="f241")
    db_session.add(playbook)
    db_session.flush()

    started = asyncio.run(execute_playbook(db_session, ws, {"playbook_id": playbook.id}))
    looked_up = asyncio.run(get_playbook_execution(db_session, ws, {"execution_id": started["execution_id"]}))

    assert started["success"] is True and started["number"].startswith("#")
    assert looked_up["execution"]["number"] == started["number"]

"""F261 (night 7b): Auto ran the agent-less one of two namesake playbooks, twice.

Two playbooks are called "New Cafe Onboarding": 102 has the Analyst on both steps, 103
has no agent on its steps. Asked to run "the one that actually has an agent on its
steps", Auto started 103 (#0183, #0184, both failing in 0.3 s) while saying it had
picked the one with the Analyst, and it read one of them and said neither had an agent.
Each tool looked the name up with ILIKE and took whichever row came first.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest

NAME = "New Cafe Onboarding"
LANTERN = {"cafe_name": "The Lantern Room", "contact_name": "Ade Bello"}


@pytest.fixture
def roastery(db_session, seed_workspace, monkeypatch):
    import modules.tools.discovery.handlers_watches as watches
    import services.concurrency_guard as guard
    import services.playbook_engine as engine
    from core.models.core import WorkflowTemplate
    from sqlalchemy import text

    async def allowed(workspace_id, db):
        return NS(allowed=True, reason="")

    launched = []
    monkeypatch.setattr(guard, "check_concurrency", allowed)
    monkeypatch.setattr(engine, "get_playbook_engine", lambda: NS(launch=lambda **kw: launched.append(kw)))
    monkeypatch.setattr(watches, "auto_create_watch", lambda *a, **k: None)
    ws = UUID(seed_workspace())
    analyst = db_session.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, configuration, owner_type) "
        "VALUES ('Analyst', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json), 'workspace') RETURNING id"),
        {"w": str(ws)}).scalar()

    def playbook(agent_id):
        steps = [{"prompt_template": "Welcome {cafe_name}.", "agent_id": agent_id},
                 {"prompt_template": "Draft the email to {contact_name}.", "agent_id": agent_id}]
        made = WorkflowTemplate(template_id=f"f261-{uuid4().hex[:8]}", name=NAME, workspace_id=ws,
                                description="Onboard a cafe.", template_definition={"steps": steps}, steps=steps,
                                created_by="f261")
        db_session.add(made)
        db_session.flush()
        return made

    staffed, unstaffed = playbook(analyst), playbook(None)
    return NS(db=db_session, ws=ws, staffed=staffed, unstaffed=unstaffed, launched=launched, analyst=analyst)


def test_a_name_two_playbooks_share_runs_the_one_with_an_agent_on_every_step(roastery):
    from modules.tools.discovery.handlers_playbooks import execute_playbook

    out = asyncio.run(execute_playbook(roastery.db, roastery.ws, {"playbook_name": NAME, "input_data": LANTERN}))

    assert out["success"] is True and out["playbook_id"] == roastery.staffed.id
    assert [kw["recipe_id"] for kw in roastery.launched] == [roastery.staffed.id]
    assert f"Playbook {roastery.staffed.id} ran" in out["chosen_because"]
    assert f"{roastery.unstaffed.id} (step 1 has no agent, step 2 has no agent)" in out["chosen_because"]


def test_two_namesakes_that_can_both_run_are_refused_naming_both(roastery):
    from modules.tools.discovery.handlers_playbooks import execute_playbook

    roastery.unstaffed.steps = [dict(s, agent_id=roastery.analyst) for s in roastery.unstaffed.steps]
    roastery.db.flush()
    out = asyncio.run(execute_playbook(roastery.db, roastery.ws, {"playbook_name": NAME, "input_data": LANTERN}))

    assert out["success"] is False and "Nothing was started" in out["error"]
    assert str(roastery.staffed.id) in out["error"] and str(roastery.unstaffed.id) in out["error"]
    assert roastery.launched == []


def test_reading_the_name_gives_both_with_their_steps_agents(roastery):
    from modules.tools.discovery.handlers_playbooks import get_playbook

    out = asyncio.run(get_playbook(roastery.db, roastery.ws, {"playbook_name": NAME}))

    assert out["success"] is True and "2 playbooks are called" in out["namesakes"]
    agents = {p["id"]: [s["agent"] for s in p["steps"]] for p in out["playbooks"]}
    assert agents == {roastery.staffed.id: ["Analyst", "Analyst"], roastery.unstaffed.id: [None, None]}


def test_a_playbook_id_that_is_a_name_is_read_as_the_name(roastery):
    from modules.tools.discovery.playbook_lookup import _name_in_id

    assert _name_in_id({"playbook_id": "Weekly Instagram posts"}) == {"playbook_name": "Weekly Instagram posts"}
    assert _name_in_id({"playbook_id": "113"}) == {"playbook_id": "113"}
    assert _name_in_id({"playbook_id": 113}) == {"playbook_id": 113}

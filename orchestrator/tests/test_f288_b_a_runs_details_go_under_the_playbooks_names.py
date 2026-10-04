"""F288 (night 8): the details Auto passes reach the run under the playbook's own names.

Auto ran New Cafe Onboarding with owner_name, owner_email, first_order and
delivery_day where the playbook named contact_person and its other inputs. The run
asked the owner again (#0366), or went out with the template's own words (Thursday,
"the crew"; #0440). A call like that is now sent back before anything starts, naming
the playbook's inputs; resent under them, it runs. A playbook that declares nothing,
or whose steps read the whole input, is never held up by it.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest

CAFE = {"cafe_name": {"required": True, "description": "The café's name"},
        "contact_person": {"required": True},
        "delivery_day": {"required": False, "default": "Thursday"}}
EMAIL = [{"prompt_template": "Write to {{contact_person}} at {{cafe_name}}: deliveries on {{delivery_day}}, "
                             "first to {{location}}.", "agent_id": None}]


@pytest.fixture
def roastery(db_session, seed_workspace, monkeypatch):
    import modules.tools.discovery.handlers_watches as watches
    import services.concurrency_guard as guard
    import services.playbook_engine as engine
    from core.models.core import WorkflowTemplate

    async def allowed(workspace_id, db):
        return NS(allowed=True, reason="")

    launched = []
    monkeypatch.setattr(guard, "check_concurrency", allowed)
    monkeypatch.setattr(engine, "get_playbook_engine", lambda: NS(launch=lambda **kw: launched.append(kw)))
    monkeypatch.setattr(watches, "auto_create_watch", lambda *a, **k: None)
    ws = UUID(seed_workspace())

    def playbook(name, *, inputs=None, steps=EMAIL):
        made = WorkflowTemplate(template_id=f"f288b-{uuid4().hex[:8]}", name=name, workspace_id=ws,
                                description=name, template_definition={"steps": steps}, steps=steps,
                                created_by="f288", inputs=inputs)
        db_session.add(made)
        db_session.flush()
        return made

    return NS(db=db_session, ws=ws, launched=launched, playbook=playbook)


def _run(roastery, playbook, details):
    from modules.tools.discovery.handlers_playbooks import execute_playbook

    return asyncio.run(execute_playbook(roastery.db, roastery.ws, {"playbook_id": playbook.id, "input_data": details}))


def _runs(roastery, playbook):
    from core.models.core import RecipeExecution

    return roastery.db.query(RecipeExecution).filter(RecipeExecution.recipe_id == playbook.id).all()


@pytest.mark.parametrize("details, left", [
    ({"cafe_name": "Larder & Loaf", "owner_name": "Ffion Hughes", "deliveries": "Tuesdays"},
     "contact_person, delivery_day"),
    ({"cafe_name": "Larder & Loaf", "contact_person": "Ffion Hughes", "deliveries": "Tuesdays"}, "delivery_day"),
], ids=["a-required-one-under-another-name", "only-the-default-left"])
def test_details_under_other_names_are_sent_back_before_the_run_starts(roastery, details, left):
    onboarding = roastery.playbook("New Cafe Onboarding", inputs=CAFE)

    out = _run(roastery, onboarding, details)

    assert out["success"] is False and _runs(roastery, onboarding) == [] and roastery.launched == []
    assert f"leaving {left} to" in out["error"] and "deliveries" in out["error"]
    assert "- delivery_day (default 'Thursday')" in out["error"]
    assert "- cafe_name (required): The café's name" in out["error"]
    assert '"cafe_name": "Larder & Loaf"' in out["error"] and "Ask the owner only for" in out["error"]


def test_resent_under_the_playbooks_names_the_run_gets_them(roastery):
    onboarding = roastery.playbook("New Cafe Onboarding", inputs=CAFE)
    details = {"cafe_name": "Larder & Loaf", "contact_person": "Ffion Hughes", "delivery_day": "Tuesday"}

    out = _run(roastery, onboarding, details)

    assert out["success"] is True
    (run,) = _runs(roastery, onboarding)
    assert run.input_data == details and roastery.launched[0]["input_data"] == details


@pytest.mark.parametrize("case", ["no-contract", "whole-input", "default-left-alone", "extra-alongside-all",
                                  "a-blank-reads-it"])
def test_a_call_the_playbook_can_read_is_never_held_up(roastery, case):
    playbooks = {
        "no-contract": (None, EMAIL, {"owner_name": "Ffion Hughes", "first_order": "4 kg"}),
        "whole-input": (None, [{"prompt_template": "Welcome them: {input}", "agent_id": None}],
                        {"owner_name": "Ffion Hughes", "first_order": "4 kg"}),
        "default-left-alone": (CAFE, EMAIL, {"cafe_name": "Larder & Loaf", "contact_person": "Ffion Hughes"}),
        "extra-alongside-all": (CAFE, EMAIL, {"cafe_name": "Larder & Loaf", "contact_person": "Ffion Hughes",
                                              "delivery_day": "Tuesday", "start_date": "8 October"}),
        "a-blank-reads-it": (CAFE, EMAIL, {"cafe_name": "Larder & Loaf", "contact_person": "Ffion Hughes",
                                           "location": "Portishead"}),
    }
    inputs, steps, details = playbooks[case]
    onboarding = roastery.playbook(f"Onboarding {case}", inputs=inputs, steps=steps)

    out = _run(roastery, onboarding, details)

    assert out["success"] is True and len(_runs(roastery, onboarding)) == 1

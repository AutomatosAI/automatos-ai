"""F203 — a step that is not there names the playbook's namesakes.

Night 6 (03:54Z): the owner's "New Cafe Onboarding" was two playbooks, #102
(ask the café's details, draft the welcome email) and #103 (draft the welcome
email, create the record card), made 48 s apart. Asked to fix the steps, Auto
edited 102's two, was refused "step_index 2 out of range (0-1)" for a third,
and told the owner it had put the record card back. Nothing was deleted, but
nothing said the step was 103's. A read of a playbook, and a step refused as out
of range, now name the namesakes.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

NAME = "New Cafe Onboarding"


@pytest.fixture
def cafe(db_session, seed_workspace):
    return NS(db=db_session, ws=UUID(seed_workspace()))


def _playbook(cafe, *step_prompts, name=NAME):
    from core.models.core import WorkflowTemplate

    steps = [{"step_number": i + 1, "prompt_template": p, "error_handling": "stop"} for i, p in enumerate(step_prompts)]
    playbook = WorkflowTemplate(name=name, template_id=f"custom-{uuid.uuid4().hex[:8]}", description="d",
                                workspace_id=cafe.ws, owner_type="workspace", owner_id=str(cafe.ws),
                                created_by="platform", tags=[], template_definition={"steps": steps}, steps=steps)
    cafe.db.add(playbook)
    cafe.db.flush()
    return playbook.id


def _night(cafe):
    first = _playbook(cafe, "Ask for the new café's details", "Draft the welcome email")
    second = _playbook(cafe, "Draft the welcome email", "Create the one-page record card")
    return first, second


def test_a_step_out_of_range_names_the_namesake_it_may_be_on(cafe):
    from modules.tools.discovery.handlers_playbooks import delete_playbook_step, update_playbook_step

    first, second = _night(cafe)

    for handler, params in ((update_playbook_step, {"prompt_template": "Create the record card"}), (delete_playbook_step, {})):
        refused = asyncio.run(handler(cafe.db, cafe.ws, {"playbook_id": first, "step_index": 2, **params}))
        assert refused["success"] is False and refused["error"].startswith("step_index 2 out of range (0-1)")
        assert (f"2 playbooks are named '{NAME}' (ids {first}, {second}); the step you want may be on {second}"
                in refused["error"])                                           # night: the bare range


def test_reading_a_playbook_names_its_namesakes(cafe):
    from modules.tools.discovery.handlers_playbooks import get_playbook

    first, second = _night(cafe)

    read = asyncio.run(get_playbook(cafe.db, cafe.ws, {"playbook_id": first}))

    assert read["success"] is True and read["playbook"]["id"] == first
    assert read["namesakes"] == (f"2 playbooks are named '{NAME}' (ids {first}, {second}); "
                                 f"the one you want may be on {second}.")    # night: no word of 103


def test_a_playbook_with_no_namesake_reads_and_refuses_as_before(cafe):
    from modules.tools.discovery.handlers_playbooks import get_playbook, update_playbook_step

    only = _playbook(cafe, "Draft the welcome email", name="Weekly Numbers")

    assert "namesakes" not in asyncio.run(get_playbook(cafe.db, cafe.ws, {"playbook_id": only}))
    refused = asyncio.run(update_playbook_step(cafe.db, cafe.ws, {"playbook_id": only, "step_index": 1,
                                                                   "prompt_template": "x"}))
    assert refused["error"] == "step_index 1 out of range (0-0)"

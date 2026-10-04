"""F292 (night 8) — a playbook is edited at the number the list shows, and its steps say their order.

Ten of ten ``PUT /api/workflow-recipes/113`` (switching a timer off, editing
playbook 101's steps) answered 404 "Recipe '113' not found" while reading 113
worked; F277 had covered the read, the run and the use, not the edit. By its
template id the edit then answered 400 "Step 0 missing required field: order":
the steps Auto adds carry step_number and no order, so the playbook's own read
gave steps the edit refused.

F294 (night 8): the Run button answered "Recipe execution started (direct mode)"
and an exec code; the owner works by card number. Its card was titled "Recipe: …".
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from tests import test_f277_a_playbook_is_found_by_the_number_the_list_shows as f277

cafe = f277.cafe             # F277's workspace with its playbooks, a fixture
_playbook = f277._playbook

DISPATCH = "Generate the content for Tom's Monday Dispatch Checklist."


def _edit(cafe, address, data):
    from api.workflow_recipes import update_workflow_recipe

    return asyncio.run(update_workflow_recipe(address, ctx=cafe.ctx, recipe_data=data, db=cafe.db))


def _as_auto_adds_them(cafe, playbook):
    """Steps the way platform_add_playbook_step writes them: step_number, no order."""
    playbook.steps = [{"step_id": f"s{n}", "step_number": n, "agent_id": cafe.analyst, "prompt_template": text,
                       "error_handling": "stop"} for n, text in ((1, DISPATCH), (2, "Copy step 1 exactly."))]
    cafe.db.flush()


def test_an_edit_at_the_number_the_list_shows_lands_on_that_playbook(cafe):
    playbook = _playbook(cafe, name="Tom's Monday Dispatch Checklist")

    answer = _edit(cafe, str(playbook.id), {"name": "Monday dispatch checklist"})   # night: 404

    cafe.db.refresh(playbook)
    assert playbook.name == "Monday dispatch checklist"
    assert answer["recipe"]["id"] == playbook.id


def test_an_edit_at_its_template_id_still_lands(cafe):
    playbook = _playbook(cafe)

    _edit(cafe, playbook.template_id, {"description": "Welcome a new wholesale cafe."})

    cafe.db.refresh(playbook)
    assert playbook.description == "Welcome a new wholesale cafe."


@pytest.mark.parametrize("address", ["neighbours", "99999999"])
def test_an_edit_of_another_workspaces_playbook_or_none_is_not_found(cafe, address):
    neighbours = _playbook(cafe, ws=cafe.neighbour)
    asked = str(neighbours.id) if address == "neighbours" else address

    with pytest.raises(HTTPException) as missing:
        _edit(cafe, asked, {"name": "Mine now"})

    assert (missing.value.status_code, missing.value.detail) == (404, f"Playbook '{asked}' not found in this workspace")
    cafe.db.refresh(neighbours)
    assert neighbours.name == "New Cafe Onboarding"


def test_the_read_gives_every_step_its_order(cafe):
    from api.workflow_recipes import get_workflow_recipe

    playbook = _playbook(cafe)
    _as_auto_adds_them(cafe, playbook)

    read = get_workflow_recipe(str(playbook.id), ctx=cafe.ctx, db=cafe.db)

    assert [step["order"] for step in read["steps"]] == [1, 2]
    assert [step["order"] for step in playbook.to_dict()["steps"]] == [1, 2]   # the list reads to_dict too


def test_steps_sent_back_as_the_read_gave_them_are_taken(cafe):
    """The night's 02:26 edit: read the playbook, change a step, send the steps back."""
    from api.workflow_recipes import get_workflow_recipe

    playbook = _playbook(cafe)
    _as_auto_adds_them(cafe, playbook)
    steps = get_workflow_recipe(str(playbook.id), ctx=cafe.ctx, db=cafe.db)["steps"]
    steps[0] = {**steps[0], "prompt_template": DISPATCH + " The first job is checking Thursday's roast."}

    _edit(cafe, str(playbook.id), {"steps": steps})                                # night: 400, order

    cafe.db.refresh(playbook)
    assert [step["order"] for step in playbook.steps] == [1, 2]
    assert playbook.steps[0]["prompt_template"].endswith("checking Thursday's roast.")
    assert all("agent" not in step for step in playbook.steps), "the read's agent summary is never stored"


def test_steps_with_no_order_at_all_are_ordered_by_their_place(cafe):
    playbook = _playbook(cafe)
    steps = [{"step_id": "a", "agent_id": cafe.analyst, "prompt_template": "Count the bags."},
             {"step_id": "b", "agent_id": cafe.analyst, "prompt_template": "Write the order."}]

    _edit(cafe, playbook.template_id, {"steps": steps})

    cafe.db.refresh(playbook)
    assert [(step["step_id"], step["order"]) for step in playbook.steps] == [("a", 1), ("b", 2)]


def test_a_step_that_is_not_a_step_is_still_refused(cafe):
    playbook = _playbook(cafe)

    with pytest.raises(HTTPException) as refused:
        _edit(cafe, str(playbook.id), {"steps": ["Count the bags."]})

    assert refused.value.status_code == 400
    assert "Step 0 must be an object" in refused.value.detail


def test_the_run_button_answers_with_the_card_it_runs_on(cafe):
    from core.models.core import BoardTask
    from services.ticket_numbers import ticket_number

    from api.workflow_recipes import execute_recipe

    playbook = _playbook(cafe)

    started = asyncio.run(execute_recipe(str(playbook.id), ctx=cafe.ctx, db=cafe.db, body={}))

    card = cafe.db.query(BoardTask).filter(BoardTask.source_id == started["recipe_execution_id"]).one()
    assert started["number"] == ticket_number(cafe.db, card) and started["task_id"] == card.id
    assert started["message"] == f'Playbook {playbook.id} "New Cafe Onboarding" is running on card {started["number"]}.'
    assert card.title == "Playbook: New Cafe Onboarding"
    assert [kw["recipe_execution_id"] for kw in cafe.launched] == [started["recipe_execution_id"]]


class _Session:
    """What start_run_row asks of a session; the run's row is committed before its card."""

    def __init__(self):
        self.calls = []

    def add(self, row):
        self.calls.append(("add", type(row).__name__))

    def commit(self):
        self.calls.append(("commit",))

    def rollback(self):
        self.calls.append(("rollback",))


def test_a_run_starts_when_its_card_cannot_be_made_at_its_start(cafe, monkeypatch):
    import services.board_task_bridge as bridge
    from api.playbook_run_start import start_run_row

    def _refused(db, recipe, execution):
        raise RuntimeError("the board is busy")

    monkeypatch.setattr(bridge, "create_recipe_board_task", _refused)
    session = _Session()
    playbook = _playbook(cafe)

    execution_id = start_run_row(session, cafe.ctx, playbook, {}, None)

    assert execution_id.startswith("exec-")
    assert session.calls == [("add", "RecipeExecution"), ("commit",), ("rollback",)]


def test_a_run_whose_card_is_not_made_yet_is_answered_without_a_number(cafe):
    from api.playbook_run_start import run_started

    playbook = _playbook(cafe)

    answer = run_started(cafe.db, playbook, "exec-not-made-yet")

    assert answer == {"message": f'Playbook {playbook.id} "New Cafe Onboarding" is running.',
                      "task_id": None, "number": None}

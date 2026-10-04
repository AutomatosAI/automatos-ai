"""F277 (night 7b) — a playbook's address takes the number the list shows, as well as its template id.

To run playbook 102 ("New Cafe Onboarding") herself, the owner had to read the
code: the run address took only its template id (custom-aa07d786), and
``GET /api/workflow-recipes/102`` answered "Recipe '102' not found". The routes
under /api/workflow-recipes/{recipe_id} now find the caller's workspace
playbook by either form, the template id first, and never another workspace's.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException
from sqlalchemy import text


@pytest.fixture
def cafe(db_session, seed_workspace, monkeypatch):
    """Harbourline's workspace with the Analyst, and a neighbour's workspace."""
    import services.playbook_engine as playbook_engine

    ws, neighbour = UUID(seed_workspace()), UUID(seed_workspace())
    analyst = db_session.execute(
        text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
             "VALUES ('Analyst', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"),
        {"w": str(ws)}).scalar()
    launched = []
    monkeypatch.setattr(playbook_engine, "get_playbook_engine", lambda: NS(launch=lambda **kw: launched.append(kw)))
    ctx = NS(workspace_id=ws, user_id="2", user=NS(id="2", email="gerard@harbourline.example"))
    return NS(db=db_session, ws=ws, neighbour=neighbour, analyst=analyst, launched=launched, ctx=ctx)


def _playbook(cafe, *, ws=None, template_id=None, name="New Cafe Onboarding"):
    from core.models.core import WorkflowTemplate

    steps = [{"step_id": "s1", "order": 1, "agent_id": cafe.analyst, "prompt_template": "Draft the welcome email."}]
    row = WorkflowTemplate(template_id=template_id or f"custom-{uuid.uuid4().hex[:8]}", name=name,
                           description="F277", workspace_id=ws or cafe.ws, template_definition={"steps": []},
                           steps=steps, created_by="f277")
    cafe.db.add(row)
    cafe.db.flush()
    return row


def test_get_finds_the_playbook_by_its_number_and_still_by_its_template_id(cafe):
    from api.workflow_recipes import get_workflow_recipe

    playbook = _playbook(cafe)

    by_number = get_workflow_recipe(str(playbook.id), ctx=cafe.ctx, db=cafe.db)          # night: not found
    by_template = get_workflow_recipe(playbook.template_id, ctx=cafe.ctx, db=cafe.db)

    assert (by_number["id"], by_number["template_id"]) == (playbook.id, playbook.template_id)
    assert by_template["id"] == playbook.id
    assert by_number["steps"][0]["agent"]["name"] == "Analyst"


def test_the_run_address_takes_the_number(cafe):
    from api.workflow_recipes import execute_recipe, get_recipe_execution_detail, list_recipe_executions

    playbook = _playbook(cafe)

    started = asyncio.run(execute_recipe(str(playbook.id), ctx=cafe.ctx, db=cafe.db, body={}))

    assert [kw["recipe_id"] for kw in cafe.launched] == [playbook.id]
    runs = list_recipe_executions(str(playbook.id), ctx=cafe.ctx, status=None, skip=0, limit=20, db=cafe.db)
    assert [run["execution_id"] for run in runs["items"]] == [started["recipe_execution_id"]]
    number = runs["items"][0]["id"]                     # a run is found by its number too
    detail = get_recipe_execution_detail(str(playbook.id), str(number), ctx=cafe.ctx, db=cafe.db)
    assert detail["execution_id"] == started["recipe_execution_id"]


def test_a_use_is_counted_on_the_playbook_at_the_number(cafe):
    from api.workflow_recipes import record_recipe_usage

    playbook = _playbook(cafe)

    record_recipe_usage(str(playbook.id), ctx=cafe.ctx, db=cafe.db)

    cafe.db.refresh(playbook)
    assert playbook.use_count == 1


def test_a_template_id_that_is_all_digits_still_means_its_own_playbook(cafe):
    from api.workflow_recipes import get_workflow_recipe

    first = _playbook(cafe, name="Weekly Instagram posts")
    named_by_digits = _playbook(cafe, template_id=str(first.id), name="Price list")

    found = get_workflow_recipe(str(first.id), ctx=cafe.ctx, db=cafe.db)

    assert found["id"] == named_by_digits.id


@pytest.mark.parametrize("address", ["neighbours", "99999999999", "0102x"])
def test_another_workspaces_playbook_or_no_playbook_is_not_found(cafe, address):
    from api.workflow_recipes import get_workflow_recipe

    neighbours = _playbook(cafe, ws=cafe.neighbour)
    asked = str(neighbours.id) if address == "neighbours" else address

    with pytest.raises(HTTPException) as missing:
        get_workflow_recipe(asked, ctx=cafe.ctx, db=cafe.db)

    assert (missing.value.status_code, missing.value.detail) == (404, f"Playbook '{asked}' not found in this workspace")

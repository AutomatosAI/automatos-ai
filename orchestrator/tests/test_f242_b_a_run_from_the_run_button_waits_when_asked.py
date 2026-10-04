"""F242 (night 7b) — a playbook run started from the Run button waits for the owner when asked.

Night 7's fix (#890) made mission steps wait (#0188 step by step) and a timer's
card wait (#0195 on playbook 113, set to wait, stopped in Review at 20:25:00).
The Run button was left out: the owner ran playbook 102 "New Cafe Onboarding"
for The Lantern Room from it (POST /api/workflow-recipes/{id}/execute), and
#0185 went straight to Done in 5.6 s with review_mode auto. The route's body
had no way to ask, and the Run button had no switch.

The body's ``wait_for_me`` is now the run's own choice, kept on the run where
services/playbook_wait.py reads it when the run makes its card. Left out, the
playbook's own setting decides, as it does for a timer's run and for Auto's.
These tests start the run through the real route, then make and finish its card
through the real board bridge, as the run does.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException

WELCOME = "Hi Ade,\n\nWelcome to Harbourline: your first 4 kg of Harbour Blend ships Monday.\n\nGerard"


@pytest.fixture
def cafe(db_session, seed_workspace, monkeypatch):
    """A workspace with the Analyst on both steps of New Cafe Onboarding (102 on the
    night). The engine's background launch is captured, not run."""
    from sqlalchemy import text

    import services.playbook_engine as engine

    ws = UUID(seed_workspace())
    analyst = db_session.execute(
        text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
             "VALUES ('Analyst', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"),
        {"w": str(ws)}).scalar()
    launched = []
    monkeypatch.setattr(engine, "get_playbook_engine", lambda: NS(launch=lambda **kw: launched.append(kw)))
    ctx = NS(workspace_id=ws, user_id="2", user=NS(id="2", email="gerard@harbourline.example"))
    return NS(db=db_session, ws=ws, analyst=analyst, launched=launched, ctx=ctx)


def _onboarding(cafe, *, waits=None):
    from core.models.core import WorkflowTemplate

    steps = [{"step_id": "s1", "order": 1, "agent_id": cafe.analyst, "prompt_template": "Set up {cafe_name}."},
             {"step_id": "s2", "order": 2, "agent_id": cafe.analyst, "prompt_template": "Draft the welcome email."}]
    playbook = WorkflowTemplate(template_id=f"custom-{uuid.uuid4().hex[:8]}", name="New Cafe Onboarding",
                                description="F242", workspace_id=cafe.ws, template_definition={"steps": []},
                                steps=steps, created_by="f242",
                                execution_config={"wait_for_me": waits} if waits is not None else {})
    cafe.db.add(playbook)
    cafe.db.flush()
    return playbook


def _run_button(cafe, playbook, **body):
    from api.workflow_recipes import execute_recipe

    return asyncio.run(execute_recipe(playbook.template_id, ctx=cafe.ctx, db=cafe.db,
                                      body={"input_data": {"cafe_name": "The Lantern Room"}, **body}))


def _the_run_ends(cafe, playbook, started):
    """The run makes its card and finishes well, through the bridge the executor calls."""
    from core.models.core import BoardTask, RecipeExecution
    from services.board_task_bridge import complete_recipe_board_task, create_recipe_board_task

    run = cafe.db.query(RecipeExecution).filter(
        RecipeExecution.execution_id == started["recipe_execution_id"]).one()
    create_recipe_board_task(cafe.db, playbook, run)
    complete_recipe_board_task(cafe.db, run.execution_id, success=True, result=WELCOME)
    card = cafe.db.query(BoardTask).filter(BoardTask.source_type == "recipe",
                                           BoardTask.source_id == run.execution_id).one()
    cafe.db.refresh(card)
    return run, card


def test_a_run_button_run_asked_to_wait_stops_in_review(cafe):
    from core.services.ticket_reasons import ASKED, review_reason

    playbook = _onboarding(cafe)                                  # 102: no setting of its own

    started = _run_button(cafe, playbook, wait_for_me=True)
    run, card = _the_run_ends(cafe, playbook, started)

    assert run.execution_metadata["wait_for_me"] is True
    assert (card.status, card.review_mode, review_reason(card)) == ("review", "human", ASKED)   # #0185: Done
    assert [kw["recipe_execution_id"] for kw in cafe.launched] == [run.execution_id]


def test_left_out_the_playbooks_own_setting_decides(cafe):
    """The default the timer's run and Auto's run follow (#0195 waited on 113's setting)."""
    waits, closes = _onboarding(cafe, waits=True), _onboarding(cafe)

    _, waited = _the_run_ends(cafe, waits, _run_button(cafe, waits))
    _, closed = _the_run_ends(cafe, closes, _run_button(cafe, closes))

    assert (waited.status, closed.status) == ("review", "done")


def test_switched_off_a_run_closes_itself_though_its_playbook_waits(cafe):
    playbook = _onboarding(cafe, waits=True)

    run, card = _the_run_ends(cafe, playbook, _run_button(cafe, playbook, wait_for_me=False))

    assert (run.execution_metadata["wait_for_me"], card.review_mode, card.status) == (False, "auto", "done")


def test_wait_for_me_must_be_true_or_false_and_nothing_starts_otherwise(cafe):
    from core.models.core import RecipeExecution

    playbook = _onboarding(cafe)

    with pytest.raises(HTTPException) as refused:
        _run_button(cafe, playbook, wait_for_me="yes please")

    assert (refused.value.status_code, refused.value.detail) == (400, "wait_for_me must be true or false")
    assert cafe.db.query(RecipeExecution).filter(RecipeExecution.recipe_id == playbook.id).count() == 0
    assert cafe.launched == []

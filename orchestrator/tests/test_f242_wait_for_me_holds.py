"""F242 (night 7) — "wait for me" holds on every kind of card.

Playbook runs were always auto: #0112 was set to wait and went Done in 8 s; a
timer's runs went straight to Done (#0057's £3,500-4,000 "order today", #0162,
#0170); a run that ended on a question to the owner went Done (#0145); a step
that asked for what it needed without a question mark ran on (#0123). Mission
steps ignored step_by_step / each_task (#0139, #0176) and a step card set to
wait (#0113.2).
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from core.models.orchestration_enums import RunState, TaskState
from tests.test_f094_a_step_card_ends_on_its_sessions_outcome import quiet as _quiet
from tests.test_f245_cancel_stops_a_mission import _mission

quiet = _quiet  # the board's fan-out on an approval (notices, reports) is its own suites' business

ORDER_TODAY = "Kiambu AA is short: order 100 kg today from Saltmarsh, about £3,500-4,000."
OFFER = "Would you like me to help draft the actual purchase orders to send to Saltmarsh Green Coffee Co.?"


def _owner(ws):
    return NS(workspace_id=uuid.UUID(str(ws)), user_id="2", auth_type="anonymous", user=NS(id="2"))


def _body(payload):
    async def _json():
        return payload
    return NS(json=_json)


def _playbook_run(db, ws, *, waits=None, run_asks=None, triggered_by="cron_scheduler"):
    from core.models.core import RecipeExecution, WorkflowTemplate

    recipe = WorkflowTemplate(template_id=f"f242-{uuid.uuid4().hex[:8]}", name="Monday Stock Report",
                              description="f242", workspace_id=ws, template_definition={"steps": []},
                              steps=[{"order": 1, "agent_id": None, "prompt_template": "The stock report."}],
                              created_by="f242", execution_config={"wait_for_me": waits} if waits is not None else {})
    db.add(recipe)
    db.flush()
    run = RecipeExecution(execution_id=f"cron-{uuid.uuid4().hex[:12]}", recipe_id=recipe.id, workspace_id=ws,
                          status="running", input_data={}, attempt_count=1, triggered_by=triggered_by,
                          execution_metadata={"wait_for_me": run_asks} if run_asks is not None else {})
    db.add(run)
    db.flush()
    return recipe, run


def _card_of(db, run):
    from core.models.core import BoardTask

    return db.query(BoardTask).filter(BoardTask.source_type == "recipe", BoardTask.source_id == run.execution_id).first()


def _ended(db, ws, *, result=ORDER_TODAY, **how):
    from services.board_task_bridge import complete_recipe_board_task, create_recipe_board_task

    recipe, run = _playbook_run(db, ws, **how)
    create_recipe_board_task(db, recipe, run)
    complete_recipe_board_task(db, run.execution_id, success=True, result=result)
    card = _card_of(db, run)
    db.refresh(card)
    return card


def test_a_timer_run_of_a_playbook_set_to_wait_ends_in_review(db_session, seed_workspace):
    from core.services.ticket_reasons import ASKED, review_reason

    card = _ended(db_session, UUID(seed_workspace()), waits=True)

    assert (card.status, card.review_mode, review_reason(card)) == ("review", "human", ASKED)   # night: Done


def test_a_run_asked_to_wait_waits_though_its_playbook_does_not(db_session, seed_workspace):
    card = _ended(db_session, UUID(seed_workspace()), waits=False, run_asks=True, triggered_by="platform_action")

    assert card.status == "review"


def test_a_card_set_to_wait_while_its_run_works_waits(db_session, seed_workspace):
    """#0112: set to wait, Done in 8 s anyway."""
    from services.board_task_bridge import complete_recipe_board_task, create_recipe_board_task

    recipe, run = _playbook_run(db_session, UUID(seed_workspace()))
    create_recipe_board_task(db_session, recipe, run)
    _card_of(db_session, run).review_mode = "human"
    db_session.flush()

    complete_recipe_board_task(db_session, run.execution_id, success=True, result=ORDER_TODAY)

    assert _card_of(db_session, run).status == "review"


def test_a_run_whose_answer_ends_on_a_question_waits_for_the_owner(db_session, seed_workspace):
    from core.services.ticket_reasons import ENDS_ON_A_QUESTION, review_reason

    card = _ended(db_session, UUID(seed_workspace()), result=f"{ORDER_TODAY}\n\n{OFFER}")

    assert (card.status, review_reason(card)) == ("review", ENDS_ON_A_QUESTION)                 # night (#0145): Done


def test_a_run_that_closes_itself_still_does(db_session, seed_workspace):
    ws = UUID(seed_workspace())

    assert _ended(db_session, ws).status == "done"
    drafted = "Hi Rosa,\n\nYour order ships Thursday. Could you confirm the delivery address?\n\nGerard"
    assert _ended(db_session, ws, result=drafted).status == "done"                             # a draft's question


def test_a_step_that_only_mentions_emails_still_asks_the_owner():
    """Review of #887: only a step whose job is writing to someone keeps its question."""
    from services.playbook_owner_ask import writes_to_someone

    assert writes_to_someone("Draft a one-line text message to the café.")
    assert writes_to_someone("You are the support agent.\nWrite the reply to Rosa's email.")
    assert not writes_to_someone("Summarize this week's customer emails; is there anything urgent?")


def test_a_step_that_asks_for_what_it_needs_without_a_question_mark_stops():
    """#0123 ended Done with "Please provide this information" in it."""
    from services.playbook_owner_ask import owner_question

    asked = ("To set up the café I need a few details first:\n1. The café's legal name\n"
             "2. Its delivery address\n3. Who receives the invoices\nPlease provide this information.")

    assert owner_question(asked, {}, "Onboard the new café: set up its account and first order.") is not None


def test_a_playbooks_setting_must_be_true_or_false():
    from core.models.core import WorkflowTemplate

    assert WorkflowTemplate(execution_config={"wait_for_me": "yes"}).validate_execution_config()[0] is False
    assert WorkflowTemplate(execution_config={"wait_for_me": True}).validate_execution_config() == (True, None)


def test_the_playbook_tools_keep_the_owners_wait_for_me(db_session, seed_workspace):
    from modules.tools.discovery.action_registry import nothing_changed
    from modules.tools.discovery.wait_for_me import keeps_wait_for_me, updates_wait_for_me

    ws = UUID(seed_workspace())
    recipe, run = _playbook_run(db_session, ws)

    async def _made(db, workspace_id, params):
        assert "wait_for_me" not in params
        return {"success": True, "playbook": {"id": recipe.id}}

    async def _nothing_else(db, workspace_id, params):
        return {"success": False, "error": nothing_changed("platform_update_playbook", "playbook_id"),
                "playbook_id": recipe.id}

    async def _started(db, workspace_id, params):
        return {"success": True, "execution_id": run.execution_id}

    asyncio.run(keeps_wait_for_me(_made)(db_session, ws, {"name": "Weekly posts", "wait_for_me": True}))
    assert recipe.execution_config["wait_for_me"] is True
    out = asyncio.run(updates_wait_for_me(_nothing_else)(db_session, ws, {"playbook_id": recipe.id,
                                                                        "wait_for_me": False}))
    assert out["success"] is True and recipe.execution_config["wait_for_me"] is False
    asyncio.run(keeps_wait_for_me(_started)(db_session, ws, {"playbook_id": recipe.id, "wait_for_me": True}))
    db_session.refresh(run)
    assert run.execution_metadata["wait_for_me"] is True


def test_auto_is_told_how_the_owner_says_wait_for_me():
    from modules.tools.discovery.action_registry import ActionRegistry
    from modules.tools.discovery.actions_mission_create import register_mission_create_action
    from modules.tools.discovery.actions_playbook_runs import register_playbook_run_actions

    registry = ActionRegistry()
    register_playbook_run_actions(registry)
    register_mission_create_action(registry)

    for tool in ("platform_create_playbook", "platform_update_playbook", "platform_execute_playbook"):
        assert registry.get(tool).parameters["properties"]["wait_for_me"]["type"] == "boolean"
    config = registry.get("platform_create_mission").parameters["properties"]["config"]["description"]
    assert "check_each_step" in config


def _held(db, ws, *, config=None, card_waits=False):
    from modules.coordination.reconciler import MissionReconciler
    from services.orchestration_state import transition_task
    from core.models.orchestration_enums import ActorType

    run, card, steps = _mission(db, ws)
    run.config = config or {}
    task, step_card = steps[TaskState.COMPLETED]
    task.output = "The price-change letter, drafted."
    if card_waits:
        step_card.review_mode = "human"
    transition_task(db=db, task=task, new_state=TaskState.VERIFYING, actor_type=ActorType.COORDINATOR,
                    actor_id="reconciler", reason="test")
    db.flush()
    asyncio.run(MissionReconciler._apply_verdict_pass(db, task))
    db.flush()
    return run, card, task, step_card


@pytest.mark.parametrize("config", [{"check_each_step": True}, {"approval_mode": "step_by_step"},
                                    {"review_mode": "each_task"}])
def test_a_mission_checking_each_step_holds_a_step_for_the_owner(db_session, seed_workspace, config):
    from core.services.ticket_reasons import ASKED, OWNER_CHECK, blocked_code, review_reason

    run, card, task, step_card = _held(db_session, UUID(seed_workspace()), config=config)

    assert task.state == TaskState.VERIFYING.value                                             # night: verified
    assert (step_card.status, review_reason(step_card)) == ("review", ASKED)
    db_session.refresh(card)
    assert run.state == RunState.PAUSED.value and blocked_code(card) == OWNER_CHECK


def test_a_step_card_set_to_wait_holds_only_that_step(db_session, seed_workspace):
    """#0113.2 was set to wait before it ran, and went straight to Done."""
    run, _card, task, step_card = _held(db_session, UUID(seed_workspace()), card_waits=True)

    assert (task.state, step_card.status, run.state) == (TaskState.VERIFYING.value, "review", RunState.PAUSED.value)


def test_a_mission_that_does_not_ask_verifies_as_before(db_session, seed_workspace):
    run, _card, task, _step_card = _held(db_session, UUID(seed_workspace()))

    assert (task.state, run.state) == (TaskState.VERIFIED.value, RunState.RUNNING.value)


def test_approve_lets_the_step_through_and_the_mission_carries_on(db_session, seed_workspace, quiet):
    from api.board_tasks import approve_task

    ws = UUID(seed_workspace())
    run, _card, task, step_card = _held(db_session, ws, config={"check_each_step": True})

    asyncio.run(approve_task(step_card.id, _body({}), ctx=_owner(ws), db=db_session))

    db_session.refresh(task)
    db_session.refresh(run)
    assert (task.state, run.state) == (TaskState.VERIFIED.value, RunState.RUNNING.value)


def test_reject_sends_the_held_step_back_and_the_mission_carries_on(db_session, seed_workspace):
    from api.board_tasks import reject_task

    ws = UUID(seed_workspace())
    run, _card, task, step_card = _held(db_session, ws, config={"check_each_step": True})

    asyncio.run(reject_task(step_card.id, _body({"feedback": "Sign it Gerard & the Harbourline crew."}),
                            ctx=_owner(ws), db=db_session))

    db_session.refresh(task)
    db_session.refresh(run)
    assert (task.state, run.state) == (TaskState.RETRYING.value, RunState.RUNNING.value)
    assert "waiting_for_owner" not in task.input_context


def test_resume_cannot_skip_a_step_waiting_for_the_owners_check(db_session, seed_workspace):
    """Review of #887: Resume (the mission page, Auto's tool) put a held mission back
    to running with its step still unchecked."""
    from modules.coordination.owner_checks import WaitsForTheOwnersCheck
    from services import coordinator_service as cs

    run, _card, task, _step_card = _held(db_session, UUID(seed_workspace()), config={"check_each_step": True})

    with pytest.raises(WaitsForTheOwnersCheck) as refused:
        cs.CoordinatorService().resume_mission(db_session, run.id, "user_test")

    assert isinstance(refused.value, ValueError) and "approve that step" in str(refused.value)
    db_session.refresh(run)
    assert run.state == RunState.PAUSED.value

"""F182 (night 6) — a playbook run that is missing an input it needs asks the
owner for it, before step 1.

Run 207 of "New Cafe Onboarding" started with no inputs, and its first step
wrote to another café's contact from memory. A playbook's ``inputs`` column
already declared what each run needs, and nothing read it: the Run Now route
filled every missing one with ''. Now the run checks it once before step 1. When
the playbook declares nothing, the steps' {input.<name>} and {input} are the
contract. A declared default fills in, and '' never does. A required input
still missing stops the run through F140's ask, naming each one and showing the
answer's format. The answer's rerun is given the values, and anything still
missing is asked for again. Auto's tool refuses such a run up front, and
create, update and get playbook carry the inputs.
"""
from __future__ import annotations

import asyncio
import inspect
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID

import pytest

from api import recipe_executor as rex
from core.services import playbook_inputs as pi
from modules.tools.discovery import handlers_asks
from tests.helpers_playbook_run import run_playbook

CAFE = {"cafe_name": {"required": True, "description": "The café's name"},
        "contact_person": {"required": True},
        "delivery_day": {"required": False, "default": "Thursday"}}
STEP = {"step_id": "s1", "order": 1, "agent_id": 7, "error_handling": "stop", "max_retries": 0,
        "prompt_template": "Draft a welcome email for {{cafe_name}}, to {{contact_person}}."}
QUESTION = ("'Monday dispatch' needs these before it can run:\n- cafe_name: The café's name\n- contact_person\n\n"
            "Answer with lines like:\ncafe_name: …\ncontact_person: …")


# ── the contract ────────────────────────────────────────────────────────────

def test_the_declared_inputs_are_read_as_the_editor_saves_them():
    assert pi.declared_inputs('{"cafe_name": {"required": true}}') == {"cafe_name": {"required": True}}
    assert pi.declared_inputs({"cafe_name": "The café's name", "draft_only": True}) == {
        "cafe_name": {"description": "The café's name"}, "draft_only": {"required": True}}
    assert pi.declared_inputs("not json") == {} and pi.declared_inputs(None) == {}


def test_without_a_declaration_the_steps_input_placeholders_are_the_contract():
    steps = [{"prompt_template": "Chase {input.cafe_name} about {{invoice}} by {current_date}."},
             {"prompt_template": "Summarise {input}."}]
    assert pi.derived_inputs(steps) == {
        "cafe_name": {"required": True, "derived": True},
        "input": {"required": True, "derived": True, "description": "the text this playbook works on"}}
    # a declaration wins
    assert pi.input_contract({"week": {"required": True}}, steps) == {"week": {"required": True}}


def test_a_default_fills_in_and_an_empty_string_never_does():
    assert pi.with_defaults(CAFE, {}) == {"delivery_day": "Thursday"}
    assert pi.with_defaults({"cafe_name": {"required": True, "default": ""}}, {}) == {}


def test_what_is_missing():
    assert pi.missing_inputs(CAFE, {"cafe_name": "Gull & Anchor", "contact_person": "  "}) == ["contact_person"]
    derived = pi.derived_inputs([{"prompt_template": "Summarise {input}."}])
    assert pi.missing_inputs(derived, {}) == ["input"]
    assert pi.missing_inputs(derived, {"cafe_name": "Gull & Anchor"}) == []   # {input} reads the whole input


def test_the_question_shows_the_answers_format():
    assert pi.inputs_question("Monday dispatch", ["cafe_name", "contact_person"], CAFE) == QUESTION
    assert pi.inputs_question("Monday dispatch", ["contact_person"], CAFE).endswith(
        "Answer with lines like:\ncontact_person: …\n(or just the value)")


@pytest.mark.parametrize("answer, missing, given", [
    ("cafe_name: Gull & Anchor\ncontact_person: Priya Shah", ["cafe_name", "contact_person"],
     {"cafe_name": "Gull & Anchor", "contact_person": "Priya Shah"}),
    ("- Cafe name = Gull & Anchor\nnotes: be quick\nhttps://gullandanchor.example", ["cafe_name", "contact_person"],
     {"cafe_name": "Gull & Anchor"}),
    ("  Priya Shah  ", ["contact_person"], {"contact_person": "Priya Shah"}),
    ("It's Priya and the café is Gull & Anchor", ["cafe_name", "contact_person"], {}),
    ("delivery_day: Friday 10:30", ["contact_person"], {"delivery_day": "Friday 10:30"}),
], ids=["lines", "loose-lines", "lone-value", "prose", "colon-in-value"])
def test_the_answer_as_input_values(answer, missing, given):
    assert pi.inputs_from_answer(answer, missing, CAFE) == given


# ── the run ─────────────────────────────────────────────────────────────────

@pytest.fixture
def staged(monkeypatch):
    questions = []

    async def _stage(db, workspace_id, **kwargs):
        questions.append(kwargs)
        kwargs["park"].status = "blocked"
        return {"success": True, "ask_id": 41, "parked": True}

    monkeypatch.setattr(handlers_asks, "stage_question", _stage)
    return questions


@pytest.fixture
def failures(monkeypatch):
    calls = []

    async def _fail(db, execution_id, error_message, step_results=None, review_card=None):
        calls.append(error_message)

    monkeypatch.setattr(rex, "_fail_execution", _fail)
    return calls


def _said(output):
    return {"status": "success", "result": output, "execution": {"tokens_used": 900, "tool_calls": []}}


def test_a_run_without_its_inputs_asks_for_them_before_step_1(monkeypatch, staged, failures):
    calls = []
    execution, card = run_playbook(monkeypatch, outcomes=[_said("Drafted.")], step_seconds=5, exec_config={},
                                   steps=[STEP], calls=calls, inputs=CAFE)

    assert calls == []                                   # no step ran
    assert (execution.status, execution.error_message) == ("failed", f"Needs you: {QUESTION}")
    (question,) = staged
    assert question["question"] == f"{QUESTION}\n\nAnswering runs the whole playbook again from step 1."
    assert question["details"]["playbook_ask"]["inputs"] == ["cafe_name", "contact_person"]
    assert card.status == "blocked" and failures == []


def test_a_run_given_its_inputs_runs_with_the_defaults_filled_in(monkeypatch, staged, failures):
    calls = []
    execution, _card = run_playbook(monkeypatch, outcomes=[_said("Drafted.")], step_seconds=5, exec_config={},
                                    steps=[STEP], calls=calls, inputs=CAFE,
                                    input_data={"cafe_name": "Gull & Anchor", "contact_person": "Priya Shah"})

    assert execution.status == "completed" and staged == []
    assert calls[0]["clean_prompt"].startswith("Draft a welcome email for Gull & Anchor, to Priya Shah.")
    assert calls[0]["input_data"]["delivery_day"] == "Thursday"


def test_a_run_that_may_not_ask_fails_naming_what_it_needs(monkeypatch, staged, failures):
    import core.security.surface as surface

    monkeypatch.setattr(surface, "widget_turn", lambda: True)
    calls = []
    run_playbook(monkeypatch, outcomes=[_said("Drafted.")], step_seconds=5, exec_config={}, steps=[STEP],
                 calls=calls, inputs=CAFE)
    assert calls == [] and staged == [] and failures == [f"Needs you: {QUESTION}"]


def test_run_now_never_fills_an_input_with_an_empty_string():
    from api import workflow_recipes

    source = inspect.getsource(workflow_recipes)
    assert "param_def.get('default', '')" not in source
    assert "input_data = with_defaults(contract_of(recipe), body.get('input_data') or {})" in source


# ── the answer ──────────────────────────────────────────────────────────────

@pytest.fixture
def launched(monkeypatch):
    """No bells, chat lines, Telegram or local runs: the rerun's launch is captured."""
    import services.chat_messenger as messenger
    import services.watch_rerun as wr
    from core.services.notification_dispatcher import NotificationDispatcher

    async def _no_telegram(*args, **kwargs):
        return None

    async def _no_bell(self, event_type, title, message=None, **kwargs):
        return {"dispatched_to": []}

    monkeypatch.setattr(NotificationDispatcher, "dispatch", _no_bell)
    monkeypatch.setattr(messenger, "deliver_background_message", lambda db, **kw: None)
    monkeypatch.setattr(handlers_asks, "_capture_question_telegram", _no_telegram)
    runs = []
    monkeypatch.setattr(wr, "launch_execution", lambda execution: runs.append(execution.execution_id))
    return runs


def _stopped_for_inputs(db, ws, needed):
    from core.models import WorkflowTemplate
    from core.models.approval_grants import ApprovalGrant
    from core.models.core import BoardTask, RecipeExecution
    from services import playbook_owner_ask as ask
    from services.board_task_bridge import create_recipe_board_task

    recipe = WorkflowTemplate(
        template_id=f"f182-{uuid.uuid4().hex[:10]}", name="New Cafe Onboarding", description="F182",
        workspace_id=ws, template_definition={"steps": []}, created_by="user_test", inputs=CAFE,
        steps=[{k: v for k, v in STEP.items() if k != "agent_id"}])  # no agent row on this schema
    db.add(recipe)
    db.flush()
    execution = RecipeExecution(execution_id=f"exec-{uuid.uuid4().hex[:12]}", recipe_id=recipe.id,
                                workspace_id=ws, status="running", input_data={"week": "40"}, attempt_count=1,
                                triggered_by="platform_action", started_at=datetime.now(timezone.utc))
    db.add(execution)
    db.flush()
    create_recipe_board_task(db, recipe, execution)
    card = db.query(BoardTask).filter(BoardTask.source_type == "recipe",
                                      BoardTask.source_id == execution.execution_id).one()
    asked = asyncio.run(ask.stop_for_owner(
        db, execution=execution, recipe=recipe, step_order=1, agent_id=None, agent_name=None,
        ask={"question": pi.inputs_question(recipe.name, needed, CAFE), "options": None},
        step_results=[], step_calls=[], inputs=needed))
    assert asked is True
    return db.query(ApprovalGrant).filter(ApprovalGrant.subject_type == "board_task",
                                          ApprovalGrant.subject_id == str(card.id)).one()


@pytest.mark.parametrize("needed, answer, given", [
    (["cafe_name", "contact_person"], "cafe_name: Gull & Anchor\ncontact_person: Priya Shah",
     {"cafe_name": "Gull & Anchor", "contact_person": "Priya Shah"}),
    (["contact_person"], "Priya Shah", {"contact_person": "Priya Shah"}),
], ids=["lines", "lone-value"])
def test_the_answers_rerun_is_given_the_inputs(db_session, seed_workspace, launched, needed, answer, given):
    from api.approval_grants import apply_question_answer
    from core.models.core import RecipeExecution

    grant = _stopped_for_inputs(db_session, UUID(seed_workspace()), needed)
    outcome = asyncio.run(apply_question_answer(db_session, grant, answer_text=answer, answered_by="user:owner"))

    assert outcome.applied and outcome.resumed
    (rerun_id,) = launched
    rerun = db_session.query(RecipeExecution).filter(RecipeExecution.execution_id == rerun_id).one()
    assert rerun.input_data == {"week": "40", **given}


# ── Auto's tools ────────────────────────────────────────────────────────────

class _Query:
    def __init__(self, found):
        self.found = found

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self.found

    def order_by(self, *args, **kwargs):
        return self

    def all(self):
        return []

    def count(self):
        return 0


class _DB:
    def __init__(self, playbook):
        self.playbook, self.added = playbook, []

    def query(self, *entities):
        return _Query(self.playbook)

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        pass

    def commit(self):
        pass

    def rollback(self):
        pass


def _playbook(**overrides):
    return SimpleNamespace(**{"id": 102, "name": "New Cafe Onboarding", "inputs": CAFE, "steps": [STEP],
                              "template_id": "custom-102", "description": "Onboard a café", "tags": [],
                              **overrides})


def test_the_run_tool_refuses_a_run_without_its_inputs(monkeypatch):
    import services.playbook_engine as engine
    from modules.tools.discovery.handlers_playbooks import execute_playbook

    launched = []
    monkeypatch.setattr(engine, "get_playbook_engine", lambda: SimpleNamespace(launch=lambda **kw: launched.append(kw)))
    db = _DB(_playbook())
    result = asyncio.run(execute_playbook(db, "ws-c1", {"playbook_id": 102, "input_data": {"cafe_name": "Gull"}}))

    assert result["success"] is False and db.added == [] and launched == []
    assert result["error"].startswith("'New Cafe Onboarding' needs contact_person before it can run, and this "
                                      "call did not give it. Nothing was started.")
    assert '{"input_data": {"contact_person": "…"}}' in result["error"]


def test_create_and_update_carry_the_inputs_and_get_shows_them():
    from modules.tools.discovery.handlers_playbooks import create_playbook, get_playbook, update_playbook

    db = _DB(None)
    created = asyncio.run(create_playbook(db, "ws-c1", {"name": "New Cafe Onboarding", "description": "Onboard",
                                                        "inputs": CAFE}))
    assert created["success"] is True and db.added[0].inputs == CAFE

    playbook = _playbook(inputs=None, steps=[{"prompt_template": "Chase {input.cafe_name}."}])
    shown = asyncio.run(get_playbook(_DB(playbook), "ws-c1", {"playbook_id": 102}))
    assert shown["playbook"]["inputs"] == {"cafe_name": {"required": True, "derived": True}}

    updated = asyncio.run(update_playbook(_DB(playbook), "ws-c1", {"playbook_id": 102, "inputs": CAFE}))
    assert updated["success"] is True and playbook.inputs == CAFE and pi.contract_of(playbook) == CAFE


def test_bad_inputs_change_nothing():
    from modules.tools.discovery.handlers_playbooks import update_playbook

    playbook = _playbook()
    result = asyncio.run(update_playbook(_DB(playbook), "ws-c1", {"playbook_id": 102, "name": "Renamed",
                                                                  "inputs": {"cafe name": {"required": True}}}))
    assert result["success"] is False and "cafe name" in result["error"]
    assert (playbook.name, playbook.inputs) == ("New Cafe Onboarding", CAFE)

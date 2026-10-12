"""P256-FIX-T4 (Gerard, 11 Oct, benchmark F321-a): running a playbook from chat waits for the owner's click.

Benchmark F321-a on Sonnet 5.5: "Set the Weekly Green Coffee Stock Check to run every Monday, and run
it once now" → the timer waited on a card, but platform_execute_playbook ran at once (card #1280,
GREEN BUYER started): ``owner_only.OWNER_ONLY_ACTIONS`` listed a playbook made, timed or deleted, not
run. A run from the owner's chat now raises the grant card first; the card names the playbook (its
number and name) and the run's inputs, and the click starts exactly one run, whose execution id and
card number reach the next turn (FX-009). The owner's Run button, a timer's fire and a watch's rerun
of a run a person started do not ask.

Each chat call runs through the executor's whole stack (the guards, the role gates, the rate limit,
the ask) onto the real run tool; the engine's background launch is captured, not run.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

RUN, SCHEDULE = "platform_execute_playbook", "platform_schedule_playbook"
NAME = "Weekly Green Coffee Stock Check"
INPUTS = {"origin": "Guji"}
TIMER = {"cron_expression": "0 8 * * 1", "timezone": "Europe/London"}
OWNER_EMAIL = "gerard@harbourline.example"


class _Kept:
    """The test's session, handed to the scheduler as its own; its close leaves the test's open."""

    def __init__(self, db):
        self._db = db

    def __getattr__(self, name):
        return getattr(self._db, name)

    def close(self):
        return None


@pytest.fixture
def roastery(db_session, seed_workspace, monkeypatch):
    """A workspace whose owner drives the chat, GREEN BUYER, and the stock check playbook on one
    step that reads {input.origin}. The dial is on (an owner-only call still asks)."""
    from sqlalchemy import text

    import modules.tools.discovery.follows_the_owner as guard
    import modules.tools.discovery.handlers_watches as watches
    import services.concurrency_guard as concurrency
    import services.playbook_engine as engine
    from core.models.core import WorkflowTemplate
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    ws = UUID(seed_workspace())
    tag = uuid4().hex[:8]
    owner = db_session.execute(text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
                               {"e": f"t4-{tag}@harbourline.test", "u": f"t4-{tag}"}).scalar()
    db_session.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                            "VALUES (CAST(:ws AS uuid), :user, 'owner', TRUE)"), {"ws": str(ws), "user": owner})
    buyer = db_session.execute(
        text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
             "VALUES ('GREEN BUYER', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"),
        {"w": str(ws)}).scalar()
    steps = [{"step_id": "s1", "order": 1, "agent_id": buyer, "prompt_template": "Count the {input.origin} stock."}]
    playbook = WorkflowTemplate(template_id=f"t4-{tag}", name=NAME, description="Count the green coffee.",
                                workspace_id=ws, owner_type="workspace", owner_id=str(ws), created_by="t4",
                                steps=steps, schedule_config={"type": "manual"},
                                template_definition={"steps": steps, "agents": [], "config": {}, "variables": []})
    db_session.add(playbook)
    db_session.flush()

    async def allowed(workspace_id, db):
        return NS(allowed=True, reason="", limits={})

    launched = []
    monkeypatch.setattr(concurrency, "check_concurrency", allowed)
    monkeypatch.setattr(engine, "get_playbook_engine", lambda: NS(launch=lambda **kw: launched.append(kw)))
    monkeypatch.setattr(watches, "auto_create_watch", lambda *a, **k: None)
    monkeypatch.setattr(guard, "owner_turn", lambda db, ws, ctx: None)
    executor = PlatformActionExecutor(db_session, ws)
    executor._full_autonomy = lambda: True
    chat = {"driving_user_id": str(owner), "user_id": "user_owner", "conversation_id": str(uuid4()), "turn_id": "t-1"}
    return NS(db=db_session, ws=ws, owner=owner, buyer=buyer, playbook=playbook, launched=launched, executor=executor,
              chat=chat)


def _call(roastery, action, params, caller_context=None):
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(roastery.executor.execute(action, dict(params), caller_context or roastery.chat))


def _grants(roastery):
    from core.models.approval_grants import ApprovalGrant

    return roastery.db.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == roastery.ws).all()


def _runs(roastery):
    from core.models.core import RecipeExecution

    return (roastery.db.query(RecipeExecution)
            .filter(RecipeExecution.recipe_id == roastery.playbook.id, RecipeExecution.workspace_id == roastery.ws)
            .all())


def _asked(reply):
    return (reply["success"] is False and reply.get("requires_confirmation") is True
            and reply.get("owner_only") is True and isinstance(reply.get("grant_id"), int))


def _click(roastery, ask):
    from core.models.approval_grants import ApprovalGrant
    from core.services.approval_grants import grant_grant

    grant_grant(roastery.db.get(ApprovalGrant, ask["grant_id"]), granted_by=f"user:{roastery.owner}")
    roastery.db.flush()


# ── F321-a: Auto's run from the owner's chat ────────────────────────────────────────

def test_the_list_holds_a_run_with_its_verb_and_its_card_lines():
    from modules.tools.discovery import owner_only
    from modules.tools.discovery.card_question import READERS
    from modules.tools.discovery.playbook_binding import BINDS

    assert RUN in owner_only.OWNER_ONLY_ACTIONS and owner_only.is_owner_only(RUN, {"playbook_id": 3})
    assert owner_only.VERBS[RUN] == "run a playbook" and RUN in READERS and RUN in BINDS


def test_autos_run_raises_one_card_naming_the_playbook_and_its_inputs_and_starts_nothing(roastery):
    from consumers.chatbot.receipts import WAITING, receipt

    params = {"playbook_name": NAME, "input_data": INPUTS}
    reply = _call(roastery, RUN, params)

    assert _asked(reply), reply
    assert len(_grants(roastery)) == 1
    assert _runs(roastery) == [] and roastery.launched == []                  # F321-a: #1280 started at once
    head, *lines = reply["question_md"].splitlines()
    assert head.startswith(f"Run a playbook '{NAME}' (playbook #{roastery.playbook.id})"), head
    assert f"- playbook: '{NAME}' (playbook #{roastery.playbook.id})" in lines
    assert "- input origin: Guji" in lines
    assert reply["params"]["playbook_id"] == roastery.playbook.id and "playbook_name" not in reply["params"]
    assert receipt(RUN, params, reply)["status"] == WAITING                  # FX-004: waiting, never a failure


def test_the_click_starts_exactly_one_run_and_its_ids_reach_the_next_turn(roastery):
    from services.click_results import click_line, executed_summary

    ask = _call(roastery, RUN, {"playbook_name": NAME, "input_data": INPUTS})
    _click(roastery, ask)

    ran = _call(roastery, RUN, ask["params"])

    assert ran["success"] is True and ran["approved_via_grant_id"] == ask["grant_id"], ran
    (run,) = _runs(roastery)
    assert run.execution_id == ran["execution_id"] and run.input_data == INPUTS
    assert len(roastery.launched) == 1
    summary = executed_summary(ran)                                           # FX-009: what the click ran
    assert summary["ids"]["execution_id"] == ran["execution_id"] and summary["ids"]["number"] == ran["number"]
    said = click_line(NS(question_md=ask["question_md"], tool_name=RUN), summary)
    assert said.startswith(f"Your click ran: run a playbook '{NAME}'") and ran["number"] in said
    again = _call(roastery, RUN, ask["params"])                               # one click, one run
    assert _asked(again) and again["grant_id"] != ask["grant_id"], again
    assert len(_runs(roastery)) == 1 and len(roastery.launched) == 1


def test_set_a_timer_and_run_it_now_raises_two_cards_and_nothing_runs_before_the_clicks(roastery):
    timer = _call(roastery, SCHEDULE, {"playbook_name": NAME, **TIMER})
    run = _call(roastery, RUN, {"playbook_name": NAME, "input_data": INPUTS})

    assert _asked(timer) and _asked(run), (timer, run)
    assert sorted(grant.tool_name for grant in _grants(roastery)) == [RUN, SCHEDULE]
    roastery.db.refresh(roastery.playbook)
    assert roastery.playbook.schedule_config == {"type": "manual"}
    assert _runs(roastery) == [] and roastery.launched == []


def test_a_run_the_click_could_not_start_is_refused_before_any_card(roastery):
    missing = _call(roastery, RUN, {"playbook_id": roastery.playbook.id})
    elsewhere = _call(roastery, RUN, {"playbook_name": "Bravo Stock Take", "input_data": INPUTS})

    assert missing["success"] is False and "requires_confirmation" not in missing
    assert "needs origin before it can run" in missing["error"]               # F182: Auto asks in the chat
    assert elsewhere["success"] is False and "requires_confirmation" not in elsewhere
    assert _grants(roastery) == [] and _runs(roastery) == []


def _namesake(roastery, steps):
    from core.models.core import WorkflowTemplate

    playbook = WorkflowTemplate(template_id=f"t4-{uuid4().hex[:8]}", name=NAME, description="A copy.",
                                workspace_id=roastery.ws, owner_type="workspace", owner_id=str(roastery.ws),
                                created_by="t4", steps=steps, template_definition={"steps": []})
    roastery.db.add(playbook)
    roastery.db.flush()
    return playbook


def test_namesakes_are_bound_to_the_one_with_an_agent_on_every_step(roastery):
    _namesake(roastery, [{"step_id": "s1", "prompt_template": "Count."}])

    reply = _call(roastery, RUN, {"playbook_name": NAME, "input_data": INPUTS})

    assert _asked(reply) and reply["params"]["playbook_id"] == roastery.playbook.id, reply


def test_namesakes_none_can_be_picked_from_are_refused_with_their_agents_and_no_card(roastery):
    """F261: the refusal lists each playbook with its steps' agents, as the handler's does."""
    copy = _namesake(roastery, list(roastery.playbook.steps))

    reply = _call(roastery, RUN, {"playbook_name": NAME, "input_data": INPUTS})

    assert reply["success"] is False and "requires_confirmation" not in reply, reply
    assert f"{roastery.playbook.id} (1 steps, agents: GREEN BUYER)" in reply["error"]
    assert f"{copy.id} (1 steps, agents: GREEN BUYER)" in reply["error"] and "Nothing was started" in reply["error"]
    assert _grants(roastery) == []


def test_a_playbook_id_of_another_workspace_is_never_bound_or_run(roastery, seed_workspace):
    from core.models.core import WorkflowTemplate

    theirs = WorkflowTemplate(template_id=f"t4-{uuid4().hex[:8]}", name="Bravo Stock Take", description="Bravo's.",
                              workspace_id=UUID(seed_workspace()), created_by="t4", steps=list(roastery.playbook.steps),
                              template_definition={"steps": []})
    roastery.db.add(theirs)
    roastery.db.flush()

    reply = _call(roastery, RUN, {"playbook_id": theirs.id, "input_data": INPUTS})

    assert reply["success"] is False and f"No playbook #{theirs.id} in this workspace" in reply["error"]
    assert "Bravo" not in str(reply) and _grants(roastery) == []


def test_details_under_names_the_playbook_never_reads_are_refused_before_any_card(roastery):
    """F288: the run would not start on them, so the click is never asked for one that fails."""
    reply = _call(roastery, RUN, {"playbook_id": roastery.playbook.id, "input_data": {"bean": "Guji"}})

    assert reply["success"] is False and "it reads its details under its own names" in reply["error"], reply
    assert _grants(roastery) == [] and _runs(roastery) == []


def test_the_inputs_alias_and_this_runs_wait_for_me_are_on_the_card(roastery):
    reply = _call(roastery, RUN, {"playbook_id": roastery.playbook.id, "inputs": INPUTS, "wait_for_me": True})

    assert _asked(reply), reply
    lines = reply["question_md"].splitlines()
    assert "- input origin: Guji" in lines and "- this run waits for your check: True" in lines


# ── The owner's own Run button, a timer's fire and a watch's rerun do not ask ──────────

def test_the_run_button_route_runs_at_once(roastery):
    from api.workflow_recipes import execute_recipe

    ctx = NS(workspace_id=roastery.ws, user_id=str(roastery.owner), user=NS(id=str(roastery.owner), email=OWNER_EMAIL))
    started = asyncio.run(execute_recipe(roastery.playbook.template_id, ctx=ctx, db=roastery.db,
                                         body={"input_data": INPUTS}))

    assert started["status"] == "started"
    assert [run.execution_id for run in _runs(roastery)] == [started["recipe_execution_id"]]
    assert len(roastery.launched) == 1 and _grants(roastery) == []


def test_a_timers_fire_runs_at_once(roastery, monkeypatch):
    import core.database.database as database
    from services.playbook_scheduler import PlaybookSchedulerService

    roastery.playbook.schedule_config = {"type": "cron", **TIMER}
    roastery.db.flush()
    monkeypatch.setattr(database, "SessionLocal", lambda: _Kept(roastery.db))

    asyncio.run(PlaybookSchedulerService()._fire_playbook(roastery.playbook.id, str(roastery.ws)))

    (run,) = _runs(roastery)
    assert run.triggered_by == "cron_scheduler"
    assert len(roastery.launched) == 1 and _grants(roastery) == []


def test_a_watchs_rerun_of_a_run_a_person_started_runs_at_once(roastery):
    from core.models.core import RecipeExecution
    from services.watch_rerun import TRIGGERED_BY_WATCH, request_rerun

    original = RecipeExecution(execution_id=f"exec-{uuid4().hex[:12]}", recipe_id=roastery.playbook.id,
                               workspace_id=roastery.ws, status="failed", input_data=INPUTS,
                               triggered_by=OWNER_EMAIL)
    roastery.db.add(original)
    roastery.db.flush()

    outcome = asyncio.run(request_rerun(roastery.db, workspace_id=roastery.ws, recipe=roastery.playbook,
                                        original=original, triggered_by=TRIGGERED_BY_WATCH,
                                        override_auto_approve=True))

    assert outcome.launched is True and outcome.grant_id is None
    rerun = next(run for run in _runs(roastery) if run.execution_id == outcome.execution_id)
    assert rerun.triggered_by == TRIGGERED_BY_WATCH and rerun.retry_of == original.execution_id
    assert len(roastery.launched) == 1 and _grants(roastery) == []

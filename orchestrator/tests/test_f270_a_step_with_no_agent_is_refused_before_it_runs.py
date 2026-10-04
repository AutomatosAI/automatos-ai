"""F270 (night 7b) — a playbook with a step that has no agent is refused before it runs, saying so.

The owner asked Auto to run "New Cafe Onboarding" for The Lantern Room. Auto ran
playbook 103, whose two steps have no agent (102, also "New Cafe Onboarding",
has the Analyst on both). #0183 and #0184 each failed in 0.2-0.3 s with "Step 1
failed: The run stopped on an internal error, so nothing after it ran. The
details are in the server log." Run now on #0183 ran it again, and it failed
the same way 7 s later. Nothing said which playbook was broken, or what to do.

Now the Run button's route, the rerun route and the board's Run now refuse
before anything changes. Every other start (Auto's tool, the timer, a trigger)
fails before its first step. All of them say the same words: the playbook's
number and name, the steps with no agent, and what to do.
"""
from __future__ import annotations

import asyncio
import sys
import types
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from tests import test_f116_a_cancelled_run_stops_its_sessions as f116

# F116's Postgres and committed workspace, for the board's Run now (it commits).
engine = f116.engine
workspace = f116.workspace

NAME = "New Cafe Onboarding"
NO_AGENTS = [{"step_id": "s1", "order": 1, "agent_id": None, "prompt_template": "Set up {cafe_name}."},
             {"step_id": "s2", "order": 2, "agent_id": None, "prompt_template": "Draft the welcome email."}]


def _words(number):
    return (f'Playbook {number} "{NAME}" can\'t run: steps 1 and 2 have no agent. '
            "Give each of them an agent, then run it again.")


# ── the words ───────────────────────────────────────────────────────────────

def test_the_words_name_the_playbook_by_number_and_each_step_with_no_agent():
    from core.models.core import PLAYBOOK_DOCUMENT_STEP
    from services.playbook_run_refusal import run_refusal

    analyst = {"order": 1, "agent_id": 325, "prompt_template": "Set up the account."}
    assert run_refusal(NS(id=103, name=NAME, steps=NO_AGENTS)) == _words(103)
    assert run_refusal(NS(id=103, name=NAME, steps=[analyst, {"order": 2, "agent_id": None}])) == (
        f'Playbook 103 "{NAME}" can\'t run: step 2 has no agent. Give step 2 an agent, then run it again.')
    three = [{"order": 3}, {"order": 1}, {"order": 2, "agent_id": 0}]
    assert "steps 1, 2 and 3 have no agent" in run_refusal(NS(id=7, name="Roast day", steps=three))
    # 102 runs; so does a fixed generate_document step, which never has an agent (PRD-251 US-117)
    assert run_refusal(NS(id=102, name=NAME, steps=[analyst, {**analyst, "order": 2}])) is None
    assert run_refusal(NS(id=9, name="Price list", steps=[analyst, {"order": 2, "type": PLAYBOOK_DOCUMENT_STEP}])) is None


# ── the Run button and the rerun route: refused before anything changes ─────

@pytest.fixture
def cafe(db_session, seed_workspace, monkeypatch):
    """Two playbooks called New Cafe Onboarding: 102's steps have the Analyst, 103's have none."""
    import services.playbook_engine as playbook_engine
    from core.models.core import WorkflowTemplate

    ws = UUID(seed_workspace())
    analyst = db_session.execute(
        text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
             "VALUES ('Analyst', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"),
        {"w": str(ws)}).scalar()

    def _playbook(steps):
        row = WorkflowTemplate(template_id=f"custom-{uuid.uuid4().hex[:8]}", name=NAME, description="F270",
                               workspace_id=ws, template_definition={"steps": []}, steps=steps, created_by="f270")
        db_session.add(row)
        db_session.flush()
        return row

    with_analyst = [{**step, "agent_id": analyst} for step in NO_AGENTS]
    launched = []
    monkeypatch.setattr(playbook_engine, "get_playbook_engine", lambda: NS(launch=lambda **kw: launched.append(kw)))
    ctx = NS(workspace_id=ws, user_id="2", user=NS(id="2", email="gerard@harbourline.example"))
    return NS(db=db_session, ws=ws, ctx=ctx, launched=launched,
              good=_playbook(with_analyst), broken=_playbook(NO_AGENTS))


def _runs_of(cafe, playbook):
    from core.models.core import RecipeExecution

    return cafe.db.query(RecipeExecution).filter(RecipeExecution.recipe_id == playbook.id).count()


def test_the_run_button_refuses_103_in_words_and_runs_102(cafe):
    from api.workflow_recipes import execute_recipe

    with pytest.raises(HTTPException) as refused:
        asyncio.run(execute_recipe(cafe.broken.template_id, ctx=cafe.ctx, db=cafe.db, body={}))

    assert (refused.value.status_code, refused.value.detail) == (400, _words(cafe.broken.id))  # night: internal error
    assert (_runs_of(cafe, cafe.broken), cafe.launched) == (0, [])                           # nothing started
    started = asyncio.run(execute_recipe(str(cafe.good.id), ctx=cafe.ctx, db=cafe.db, body={}))
    assert started["status"] == "started" and [kw["recipe_id"] for kw in cafe.launched] == [cafe.good.id]


def test_a_rerun_of_103_is_refused_before_anything_changes(cafe):
    from api.workflow_recipes import rerun_recipe_execution
    from core.models.core import RecipeExecution

    failed = RecipeExecution(execution_id=f"exec-{uuid.uuid4().hex[:12]}", recipe_id=cafe.broken.id,
                             workspace_id=cafe.ws, status="failed", input_data={}, attempt_count=1,
                             triggered_by="platform_action")
    cafe.db.add(failed)
    cafe.db.flush()

    with pytest.raises(HTTPException) as refused:
        asyncio.run(rerun_recipe_execution(str(cafe.broken.id), failed.execution_id, ctx=cafe.ctx, body={},
                                           db=cafe.db))

    assert (refused.value.status_code, refused.value.detail) == (400, _words(cafe.broken.id))
    assert (_runs_of(cafe, cafe.broken), cafe.launched) == (1, [])


# ── the board's Run now on #0183's failed card ──────────────────────────────

@pytest.fixture
def relaunched(monkeypatch):
    """The reruns the board's redo asked the engine for (services/watch_rerun)."""
    import services.watch_rerun as watch_rerun

    runs = []
    monkeypatch.setattr(watch_rerun, "launch_execution", lambda execution: runs.append(execution.execution_id))
    return runs


def _failed_card_of_103(new_session, ws):
    from core.models.core import BoardTask, WorkflowTemplate

    s = new_session()
    playbook = WorkflowTemplate(template_id=f"custom-{uuid.uuid4().hex[:8]}", name=NAME, description="F270",
                                workspace_id=ws, template_definition={"steps": []}, steps=NO_AGENTS,
                                created_by="f270")
    s.add(playbook)
    s.flush()
    run = f116._run(s, ws, playbook.id)
    s.flush()
    s.execute(text("UPDATE recipe_executions SET status = 'failed' WHERE execution_id = :e"), {"e": run})
    card = BoardTask(workspace_id=ws, title=f"Recipe: {NAME}", status="failed", priority="medium",
                     source_type="recipe", source_id=run, review_mode="auto",
                     error_message="Step 1 failed: The run stopped on an internal error, so nothing after it ran.")
    s.add(card)
    s.commit()
    return NS(playbook=playbook.id, run=run, card=card.id)


def test_run_now_on_103s_failed_card_says_why_and_changes_nothing(workspace, new_session, relaunched):
    from api.board_tasks import run_task_now

    ctx = NS(workspace_id=UUID(workspace), user_id="2", auth_type="anonymous", user=NS(id="2"))
    pb = _failed_card_of_103(new_session, workspace)

    with pytest.raises(HTTPException) as refused:
        asyncio.run(run_task_now(pb.card, ctx=ctx, db=new_session()))

    assert (refused.value.status_code, refused.value.detail) == (409, _words(pb.playbook))   # night: ran, failed again
    card = new_session().execute(text("SELECT status, source_id FROM board_tasks WHERE id = :i"),
                                 {"i": pb.card}).first()
    assert (card.status, card.source_id, relaunched) == ("failed", pb.run, [])


# ── every other start (Auto's tool, the timer): failed before its first step ─

@pytest.fixture
def run_edges(monkeypatch, new_session):
    """The engine's entry point on this Postgres. The run's outside edges (the bell,
    the report, the memory, the heartbeat, the watch) are quiet, and its step loop
    is recorded instead of run."""
    from api import recipe_executor as rex
    from services import playbook_engine_heartbeat

    async def _nothing(*args, **kwargs):
        return None

    class _Memory:
        def __init__(self, db=None):
            pass

        async def store_execution_memory(self, *args, **kwargs):
            return None

    memory = types.ModuleType("core.services.playbook_memory_service")
    memory.PlaybookMemoryService = _Memory
    monkeypatch.setitem(sys.modules, memory.__name__, memory)
    monkeypatch.setattr(rex, "SessionLocal", new_session)
    monkeypatch.setattr(rex, "_dispatch_playbook_event", _nothing)
    monkeypatch.setattr(rex, "_auto_create_playbook_report", _nothing)
    monkeypatch.setattr(rex, "_ingest_playbook_terminal_watch", lambda *a, **k: None)
    monkeypatch.setattr(playbook_engine_heartbeat, "_emit_playbooks_primitive", lambda *a, **k: None)
    ran = []

    async def _the_run(execution_id, *args):
        ran.append(execution_id)

    monkeypatch.setattr(rex, "_execute_recipe_inner", _the_run)
    return ran


def _a_timers_run(new_session, ws, steps):
    """A run as the timer starts it: its row, pending, and no card yet."""
    from core.models.core import RecipeExecution, WorkflowTemplate

    s = new_session()
    playbook = WorkflowTemplate(template_id=f"custom-{uuid.uuid4().hex[:8]}", name=NAME, description="F270",
                                workspace_id=ws, template_definition={"steps": []}, steps=steps, created_by="f270")
    s.add(playbook)
    s.flush()
    run = RecipeExecution(execution_id=f"cron-{uuid.uuid4().hex[:12]}", recipe_id=playbook.id, workspace_id=ws,
                          status="pending", input_data={}, triggered_by="cron_scheduler",
                          execution_metadata={"execution_type": "cron_scheduler"})
    s.add(run)
    s.commit()
    return NS(playbook=playbook.id, run=run.execution_id)


def _launched(pb, ws):
    from api import recipe_executor as rex

    asyncio.run(rex.execute_recipe_direct(pb.run, pb.playbook, UUID(ws), {}))


def test_a_run_started_elsewhere_fails_before_its_first_step_with_the_same_words(workspace, new_session, run_edges):
    pb = _a_timers_run(new_session, workspace, NO_AGENTS)

    _launched(pb, workspace)

    run = new_session().execute(text("SELECT status, error_message FROM recipe_executions WHERE execution_id = :e"),
                                {"e": pb.run}).first()
    card = new_session().execute(text("SELECT status, error_message FROM board_tasks "
                                      "WHERE source_type = 'recipe' AND source_id = :e"), {"e": pb.run}).first()
    assert (run.status, run.error_message) == ("failed", _words(pb.playbook))     # night: "internal error"
    assert (card.status, card.error_message) == ("failed", _words(pb.playbook))   # on the board, saying why
    assert run_edges == []                                                         # its steps never began


def test_a_playbook_whose_steps_have_agents_runs_as_before(workspace, new_session, run_edges):
    pb = _a_timers_run(new_session, workspace, [{**step, "agent_id": 325} for step in NO_AGENTS])

    _launched(pb, workspace)

    status = new_session().execute(text("SELECT status FROM recipe_executions WHERE execution_id = :e"),
                                   {"e": pb.run}).scalar()
    assert (run_edges, status) == ([pb.run], "pending")

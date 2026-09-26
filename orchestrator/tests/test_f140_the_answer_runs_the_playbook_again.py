"""F140 (night 4) — the owner's answer runs a stopped playbook again from step 1.

Gerard's call (25 Sep): no parked state. A step that asks ends its run failed
with "Needs you: <question>"; the question goes on the run's card, in the
owner's Questions, and says what answering does; the answer runs the whole
playbook again (PRD-204's rerun, retry_of) on the same card, with the answer
in every step's prompt. The tester's rules: one bell, the question's; the watch
never scores the stop and follows the rerun; a dismissed question closes the
run as the owner's no, as F143 closes a rejected plan; a question left open
does not stop the next scheduled run. On the real schema.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID

import pytest

from core.models import WorkflowTemplate
from core.models.approval_grants import ApprovalGrant
from core.models.core import BoardTask, RecipeExecution
from core.models.watch_enums import WatchStatus
from services import playbook_owner_ask as ask
from services.watch_service import WatchService

QUESTION = ("I am unable to directly read the file from Dropbox without a specified path. Could you please "
            "provide the full path to 'green-coffee-list-autumn-2026.csv' in your Dropbox?")
ANSWER = "Dropbox/Stock/green-coffee-list-autumn-2026.csv"
NOW = datetime.now(timezone.utc)


@pytest.fixture
def bells(monkeypatch):
    """Every notification, at the dispatcher seam (the table is migration-only)."""
    from core.services.notification_dispatcher import NotificationDispatcher

    sent = []

    async def _capture(self, event_type, title, message=None, **kwargs):
        sent.append(event_type)
        return {"dispatched_to": ["in_app"]}

    monkeypatch.setattr(NotificationDispatcher, "dispatch", _capture)
    return sent


@pytest.fixture
def said(monkeypatch):
    """What the owner reads in chat, and no Telegram."""
    import services.chat_messenger as messenger
    from modules.tools.discovery import handlers_asks

    lines = []

    async def _no_telegram(*args, **kwargs):
        return None

    monkeypatch.setattr(messenger, "deliver_background_message", lambda db, **kw: lines.append(kw["text"]))
    monkeypatch.setattr(handlers_asks, "_capture_question_telegram", _no_telegram)
    return lines


@pytest.fixture
def launched(monkeypatch):
    """No local playbook runs: the engine launch is captured."""
    import services.watch_rerun as wr

    runs = []
    monkeypatch.setattr(wr, "launch_execution", lambda execution: runs.append(execution.execution_id))
    return runs


def _playbook(db, ws):
    recipe = WorkflowTemplate(
        template_id=f"f140-{uuid.uuid4().hex[:10]}", name="Monday green-coffee reorder", description="F140",
        workspace_id=ws, template_definition={"steps": []}, created_by="user_test",
        steps=[{"step_id": "s1", "order": 1, "prompt_template": "Read the green coffee list."}],
    )
    db.add(recipe)
    db.flush()
    return recipe


def _run(db, ws, recipe, *, status="running", started=None):
    execution = RecipeExecution(execution_id=f"exec-{uuid.uuid4().hex[:12]}", recipe_id=recipe.id,
                                workspace_id=ws, status=status, input_data={"week": "40"}, attempt_count=1,
                                triggered_by="schedule", started_at=started or NOW)
    db.add(execution)
    db.flush()
    return execution


def _stopped(db, ws, recipe=None, *, deadline=None, started=None):
    """A run of the playbook whose step 1 asked the owner, with its card and live watch."""
    from services.board_task_bridge import create_recipe_board_task

    recipe = recipe or _playbook(db, ws)
    execution = _run(db, ws, recipe, started=started)
    create_recipe_board_task(db, recipe, execution)
    card = db.query(BoardTask).filter(BoardTask.source_type == "recipe",
                                      BoardTask.source_id == execution.execution_id).one()
    watch = WatchService.create_watch(db, workspace_id=ws, watch_type="playbook_execution",
                                      target_type="playbook_execution", target_id=execution.execution_id,
                                      title="Watch: Monday green-coffee reorder", deadline_at=deadline)
    asked = asyncio.run(ask.stop_for_owner(
        db, execution=execution, recipe=recipe, step_order=1, agent_id=None, agent_name="GREEN COFFEE STOCK",
        ask={"question": QUESTION, "options": None},
        step_results=[{"order": 1, "status": "failed", "agent_id": None, "tool_calls_summary": []}],
        step_calls=[]))
    assert asked is True
    grant = db.query(ApprovalGrant).filter(ApprovalGrant.subject_type == "board_task",
                                           ApprovalGrant.subject_id == str(card.id)).one()
    return recipe, execution, card, watch, grant


# ── the stop ────────────────────────────────────────────────────────────────

def test_the_stop_puts_one_question_on_the_runs_card(db_session, seed_workspace, bells, said):
    ws = UUID(seed_workspace())
    _recipe, execution, card, _watch, grant = _stopped(db_session, ws)
    assert (execution.status, execution.error_message) == ("failed", f"Needs you: {QUESTION}")
    assert card.status == "blocked" and card.blocked_reason == f"Needs you: {QUESTION} (ask #{grant.id})"
    assert (grant.kind, grant.status) == ("question", "pending")
    assert grant.question_md == f"{QUESTION}\n\nAnswering runs the whole playbook again from step 1."
    assert execution.execution_metadata["needs_owner"] == {"ask_id": grant.id, "step": 1, "card_id": card.id}
    assert bells == ["question_pending"]                # the one bell


@pytest.fixture
def committed_workspace():
    """A workspace on a committing engine (the fault below is about what one
    commit saved), and every row it gathers deleted after."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    from core.database.database import get_database_url

    engine = create_engine(get_database_url())
    ws = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'f140-part-way')"),
                     {"id": str(ws)})
    yield sessionmaker(bind=engine), ws
    with engine.begin() as conn:
        for table in ("approval_grants", "board_tasks", "recipe_executions", "workflow_recipes"):
            conn.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:id AS uuid)"), {"id": str(ws)})
        conn.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": str(ws)})
    engine.dispose()


def test_a_fault_after_the_question_is_saved_leaves_the_stop_standing(committed_workspace, bells, monkeypatch):
    """stage_question commits twice. A fault after its first commit (here the
    Telegram send) finds the run, its card and the question already saved: the
    stop stands, so the caller never fails the run with the side effects it avoids."""
    from modules.tools.discovery import handlers_asks
    from services.board_task_bridge import create_recipe_board_task

    async def _telegram_down(*args, **kwargs):
        raise RuntimeError("telegram is down")

    monkeypatch.setattr(handlers_asks, "_capture_question_telegram", _telegram_down)
    make_session, ws = committed_workspace
    db = make_session()
    try:
        recipe = _playbook(db, ws)
        execution = _run(db, ws, recipe)
        create_recipe_board_task(db, recipe, execution)
        execution_id = execution.execution_id
        asked = asyncio.run(ask.stop_for_owner(
            db, execution=execution, recipe=recipe, step_order=1, agent_id=None, agent_name="GREEN COFFEE STOCK",
            ask={"question": QUESTION, "options": None}, step_results=[], step_calls=[]))
    finally:
        db.close()

    assert asked is True
    fresh = make_session()
    try:
        run = fresh.query(RecipeExecution).filter(RecipeExecution.execution_id == execution_id).one()
        assert (run.status, run.error_message) == ("failed", f"Needs you: {QUESTION}")
        assert ask.waiting_for_owner(run.status, run.execution_metadata)
        assert fresh.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == ws,
                                                 ApprovalGrant.status == "pending").count() == 1
    finally:
        fresh.close()


# ── the answer ──────────────────────────────────────────────────────────────

def test_the_answer_runs_the_playbook_again_on_the_same_card(db_session, seed_workspace, bells, said, launched):
    from api.approval_grants import apply_question_answer

    ws = UUID(seed_workspace())
    _recipe, original, card, watch, grant = _stopped(db_session, ws)
    outcome = asyncio.run(apply_question_answer(db_session, grant, answer_text=ANSWER, answered_by="user:owner"))

    assert outcome.applied and outcome.resumed
    (rerun_id,) = launched
    rerun = db_session.query(RecipeExecution).filter(RecipeExecution.execution_id == rerun_id).one()
    assert (rerun.retry_of, rerun.triggered_by, rerun.status) == (original.execution_id, "rerun", "pending")
    assert rerun.input_data == {"week": "40"}
    assert rerun.execution_metadata["owner_answers"] == [
        {"step": 1, "question": QUESTION, "answer": ANSWER, "ask_id": grant.id}]
    db_session.refresh(card)
    assert (card.source_id, card.status, card.blocked_reason) == (rerun_id, "in_progress", None)
    assert card.result == f"Re-run after your answer: the playbook runs again from step 1 ({rerun_id})."
    db_session.refresh(original)
    assert original.status == "failed" and original.execution_metadata["needs_owner"]["rerun"] == rerun_id
    db_session.refresh(watch)
    assert (watch.target_id, watch.status, watch.final_score) == (rerun_id, WatchStatus.WATCHING.value, None)
    assert said[-1] == "Answered — the playbook runs again from step 1 with your answer."


def test_a_long_answer_is_kept_up_to_f037s_cap(db_session, seed_workspace, bells, said, launched):
    from api.approval_grants import apply_question_answer

    ws = UUID(seed_workspace())
    *_, grant = _stopped(db_session, ws)
    long_answer = "k" * 16_500
    asyncio.run(apply_question_answer(db_session, grant, answer_text=long_answer, answered_by="user:owner"))
    rerun = db_session.query(RecipeExecution).filter(RecipeExecution.execution_id == launched[0]).one()
    kept = rerun.execution_metadata["owner_answers"][0]["answer"]
    assert kept == ("k" * 16_000 + "\n\n[The owner's answer is cut here at 16,000 characters; "
                    f"500 more are on question #{grant.id}.]")


def test_a_question_on_any_other_card_still_requeues_it(db_session, seed_workspace, bells, said, launched):
    from api.approval_grants import apply_question_answer
    from modules.tools.discovery.handlers_asks import ask_human

    ws = UUID(seed_workspace())
    task = BoardTask(workspace_id=ws, title="Price the Christmas box", status="in_progress")
    db_session.add(task)
    db_session.flush()
    staged = asyncio.run(ask_human(db_session, ws, {"subject_type": "board_task", "subject_id": str(task.id),
                                                     "question": "Which box size?"}))
    grant = db_session.query(ApprovalGrant).filter(ApprovalGrant.id == staged["ask_id"]).one()
    outcome = asyncio.run(apply_question_answer(db_session, grant, answer_text="The 1 kg box",
                                                answered_by="user:owner"))
    db_session.refresh(task)
    assert outcome.resumed and task.status == "assigned" and launched == []


# ── a dismissed question ────────────────────────────────────────────────────

def test_dismissing_the_question_closes_the_run_as_the_owners_no(db_session, seed_workspace, bells, said,
                                                                  launched):
    from api.approval_grants import deny_approval

    ws = UUID(seed_workspace())
    _recipe, execution, card, watch, grant = _stopped(db_session, ws)
    ctx = SimpleNamespace(workspace_id=ws, user=SimpleNamespace(id="user_owner"), user_id=None,
                          internal_user_id=None)
    asyncio.run(deny_approval(grant.id, ctx=ctx, db=db_session))

    reason = f"Dismissed by the owner: {QUESTION}"
    db_session.refresh(execution)
    db_session.refresh(card)
    db_session.refresh(watch)
    assert (execution.status, execution.error_message) == ("cancelled", reason)
    assert (card.status, card.error_message) == ("done", reason)
    assert watch.status == WatchStatus.CANCELLED.value and watch.final_score is None
    assert reason in watch.final_verdict
    assert launched == [] and bells == ["question_pending"]


# ── the watch ───────────────────────────────────────────────────────────────

@pytest.fixture
def no_scoring(monkeypatch):
    import services.watch_decider as decider

    def _never(*args, **kwargs):
        raise AssertionError("a stop to ask the owner was handed to the scorer")

    monkeypatch.setattr(decider, "get_watch_decider", _never)


def test_the_watch_waits_for_the_answer_and_never_scores_the_stop(db_session, seed_workspace, bells, said,
                                                                   no_scoring):
    from services.watch_ticker import WatchTicker

    ws = UUID(seed_workspace())
    *_, watch, _grant = _stopped(db_session, ws, deadline=NOW + timedelta(days=1))
    asyncio.run(WatchTicker()._check_watch(db_session, watch, NOW))
    assert watch.status == WatchStatus.WATCHING.value and watch.final_score is None


def test_an_unanswered_question_closes_the_watch_unscored_at_its_deadline(db_session, seed_workspace, bells,
                                                                           said, no_scoring):
    from services.watch_ticker import WatchTicker

    ws = UUID(seed_workspace())
    *_, watch, _grant = _stopped(db_session, ws, deadline=NOW - timedelta(minutes=1))
    asyncio.run(WatchTicker()._check_watch(db_session, watch, NOW))
    assert watch.status == WatchStatus.EXPIRED.value and watch.final_score is None
    assert watch.final_verdict == ("The run stopped to ask the owner, and the question had no answer "
                                   "by the watch's deadline.")
    assert bells == ["question_pending"]                # no 'watch expired' bell


# ── the next scheduled run ──────────────────────────────────────────────────

def test_stops_to_ask_never_trip_the_repeated_failure_breaker(db_session, seed_workspace, bells, said):
    """Each scheduled fire asks its own question; the breaker would stop the cron."""
    from config import config
    from services.playbook_breaker import breaker_is_open

    ws = UUID(seed_workspace())
    recipe = _playbook(db_session, ws)
    for hours in range(config.PLAYBOOK_BREAKER_THRESHOLD):
        _stopped(db_session, ws, recipe, started=NOW - timedelta(hours=hours + 1))
    assert breaker_is_open(db_session, recipe.id) is False
    for minutes in range(config.PLAYBOOK_BREAKER_THRESHOLD):
        _run(db_session, ws, recipe, status="failed", started=NOW + timedelta(minutes=minutes))
    assert breaker_is_open(db_session, recipe.id) is True


def test_a_stop_is_no_flip_for_a_scheduled_playbooks_watch(db_session, seed_workspace, bells, said):
    from services.watch_decider import DECIDED_NOOP, get_watch_decider

    ws = UUID(seed_workspace())
    recipe = _playbook(db_session, ws)
    _run(db_session, ws, recipe, status="completed", started=NOW - timedelta(days=7))
    _stopped(db_session, ws, recipe, started=NOW)
    watch = WatchService.create_watch(db_session, workspace_id=ws, watch_type="scheduled_playbook",
                                      target_type="scheduled_playbook", target_id=str(recipe.id),
                                      title="Watch: Monday green-coffee reorder", policy="persistent")
    decision = asyncio.run(get_watch_decider().observe_scheduled(db_session, watch, recipe, NOW))
    assert decision == DECIDED_NOOP and bells == ["question_pending"]

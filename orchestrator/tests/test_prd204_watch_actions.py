"""PRD-204 S8 -- direction-change actions: replan / reassign / spawn_agent /
escalate.

Each action under full_auto (auto-executes) and always_ask (grant card);
budget exhaustion -> straight to escalate; reassign excludes the failed
agent and escalates on no_capable_agent; spawn is ALWAYS grant-gated except
full_auto, honours blueprint `rules` defaulting (onboarding-wall regression
guard), and strict-mode blueprint failures roll the spawn back.

The coordinator replan (LLM planner) and AgentMatcher.rank are stubbed;
notifications are captured at the dispatcher seam. DB-backed, skips cleanly
without Postgres.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from core.database.database import get_database_url
from core.models.core import Agent, BoardTask
from core.models.approval_grants import ApprovalGrant, SUBJECT_PLAYBOOK_RUN
from core.models.orchestration import (
    OrchestrationRun,
    OrchestrationTask,
    OrchestrationTaskDependency,
)
from core.models.watch_enums import WatchStatus
from services.watch_service import WatchService

FROZEN_NOW = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1 FROM watches LIMIT 1"))
            c.execute(text("SELECT 1 FROM orchestration_runs LIMIT 1"))
            c.execute(text("SELECT 1 FROM agent_blueprints LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"watch actions suite needs a reachable Postgres with schema: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def new_session(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def workspace(new_session):
    ws_id = str(uuid.uuid4())
    s = new_session()
    s.execute(
        text(
            "INSERT INTO workspaces (id, name) "
            "VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"
        ),
        {"id": ws_id, "n": "prd204-watch-actions"},
    )
    s.commit()
    s.close()

    yield ws_id

    s = new_session()
    for stmt in (
        "DELETE FROM watch_events WHERE watch_id IN "
        "(SELECT id FROM watches WHERE workspace_id = CAST(:w AS uuid))",
        "DELETE FROM watches WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM approval_grants WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM board_tasks WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM orchestration_events WHERE run_id IN "
        "(SELECT id FROM orchestration_runs WHERE workspace_id = CAST(:w AS uuid))",
        "DELETE FROM orchestration_task_dependencies WHERE task_id IN "
        "(SELECT id FROM orchestration_tasks WHERE run_id IN "
        "(SELECT id FROM orchestration_runs WHERE workspace_id = CAST(:w AS uuid)))",
        "DELETE FROM orchestration_tasks WHERE run_id IN "
        "(SELECT id FROM orchestration_runs WHERE workspace_id = CAST(:w AS uuid))",
        "DELETE FROM orchestration_runs WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM agent_blueprints WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM agents WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM workspaces WHERE id = CAST(:w AS uuid)",
    ):
        s.execute(text(stmt), {"w": ws_id})
    s.commit()
    s.close()


@pytest.fixture
def capture_notifications(monkeypatch):
    from core.services.notification_dispatcher import NotificationDispatcher

    sent = []

    async def _capture(self, event_type, title, message=None, **kwargs):
        sent.append({"event_type": event_type, "title": title, "message": message})
        return {"dispatched_to": ["in_app"]}

    monkeypatch.setattr(NotificationDispatcher, "dispatch", _capture)
    return sent


@pytest.fixture
def stub_replan(monkeypatch):
    """The coordinator replan path is LLM-backed -- stub it, capture args."""
    from services.coordinator_service import CoordinatorService

    calls = []

    async def _fake_replan(self, db, run_id, actor_id, notes=None, **kwargs):
        calls.append({"run_id": run_id, "actor_id": actor_id,
                      "notes": notes, **kwargs})
        return db.query(OrchestrationRun).filter(OrchestrationRun.id == run_id).first()

    monkeypatch.setattr(CoordinatorService, "replan_mission", _fake_replan)
    return calls


def _rank_stub(monkeypatch, results):
    from modules.coordination import agent_matcher

    monkeypatch.setattr(
        agent_matcher.AgentMatcher,
        "rank",
        staticmethod(lambda db, task, agents, task_spec=None, semantic=None: list(results)),
    )


def _match(agent_id, agent_name):
    from modules.coordination.agent_matcher import MatchResult

    return MatchResult(
        agent_id=agent_id,
        agent_name=agent_name,
        total_score=0.9,
        tool_coverage=1.0,
        skill_match=0.8,
        model_fit=1.0,
        availability=1.0,
        history=0.5,
        reason="stubbed",
    )


def _set_policy(s, ws_id, settings):
    s.execute(
        text("UPDATE workspaces SET settings = CAST(:cfg AS jsonb) WHERE id = CAST(:w AS uuid)"),
        {"cfg": json.dumps(settings), "w": ws_id},
    )
    s.commit()


FULL_AUTO_SETTINGS = {
    "approval_policy": {"policy": "full_auto"},
    "autonomy": {"level": "full"},
}


def _seed_failed_mission(s, ws_id, *, with_dependent=True):
    run = OrchestrationRun(
        workspace_id=ws_id,
        goal="produce the market analysis",
        state="failed",
        state_type="terminal",
        created_by="user_test",
        budget_spent={"cost": 2.5, "tokens": 5000},
    )
    s.add(run)
    s.commit()

    pred = OrchestrationTask(
        run_id=run.id,
        title="gather data",
        description="collect the inputs",
        sequence_number=1,
        agent_role="researcher",
        state="verified",
        state_type="blocked",
        output="data gathered",
    )
    failed = OrchestrationTask(
        run_id=run.id,
        title="write analysis",
        description="write the market analysis",
        sequence_number=2,
        agent_role="writer",
        state="failed",
        state_type="terminal",
        failure_detail="agent produced empty output",
        input_context={"agent_match": {"agent_id": 111, "agent_name": "alpha"}},
    )
    s.add_all([pred, failed])
    s.commit()

    dependent = None
    s.add(OrchestrationTaskDependency(task_id=failed.id, depends_on_task_id=pred.id))
    if with_dependent:
        dependent = OrchestrationTask(
            run_id=run.id,
            title="publish analysis",
            description="publish it",
            sequence_number=3,
            agent_role="publisher",
            state="pending",
            state_type="initial",
        )
        s.add(dependent)
        s.commit()
        s.add(OrchestrationTaskDependency(task_id=dependent.id, depends_on_task_id=failed.id))
    s.commit()
    return run, pred, failed, dependent


def _make_watch(s, ws_id, run, **overrides):
    params = dict(
        workspace_id=ws_id,
        watch_type="mission",
        target_type="mission",
        target_id=str(run.id),
        title="Watch: market analysis mission",
        success_criteria="a complete market analysis",
        policy="score_and_improve",
        now=FROZEN_NOW,
    )
    params.update(overrides)
    watch = WatchService.create_watch(s, **params)
    s.commit()
    return watch


def _run_action(s, watch, action, **kwargs):
    from services.watch_actions import run_mission_action

    return asyncio.run(run_mission_action(s, watch, action, **kwargs))


def _escalation_cards(s, ws_id, watch):
    return (
        s.query(BoardTask)
        .filter(
            BoardTask.workspace_id == ws_id,
            BoardTask.tags.contains(["escalation", f"watch:{watch.id}"]),
        )
        .all()
    )


# ---------------------------------------------------------------------------
# replan
# ---------------------------------------------------------------------------


def test_replan_full_auto_executes_and_counts_budget(
    workspace, new_session, stub_replan, capture_notifications
):
    s = new_session()
    run, _, _, _ = _seed_failed_mission(s, workspace)
    watch = _make_watch(s, workspace, run)
    _set_policy(s, workspace, FULL_AUTO_SETTINGS)

    outcome = _run_action(s, watch, "replan", diagnosis="planner missed a data step")
    s.commit()

    assert outcome.executed is True
    assert outcome.parked is False
    assert len(stub_replan) == 1
    call = stub_replan[0]
    assert call["run_id"] == run.id
    assert call["actor_id"] == "watcher"
    assert call["trigger"] == "watch"
    assert "planner missed a data step" in call["notes"]

    s.refresh(watch)
    assert watch.actions_taken == 1
    assert watch.status == WatchStatus.ACTING.value  # corrective attempt in flight
    s.close()


def test_replan_always_ask_parks_grant(
    workspace, new_session, stub_replan, capture_notifications
):
    s = new_session()
    run, _, _, _ = _seed_failed_mission(s, workspace)
    watch = _make_watch(s, workspace, run)
    # Default policy: always_ask.

    outcome = _run_action(s, watch, "replan", diagnosis="needs a different plan")
    s.commit()

    assert outcome.parked is True
    assert outcome.executed is False
    assert stub_replan == []  # nothing ran

    grant = s.query(ApprovalGrant).get(outcome.grant_id)
    assert grant.subject_type == SUBJECT_PLAYBOOK_RUN
    assert grant.subject_id == str(run.id)
    assert grant.tool_name == "watch_replan"
    assert grant.details["watch_action"] == "replan"
    assert grant.details["watch_id"] == str(watch.id)
    assert grant.details["spec"]["diagnosis"] == "needs a different plan"

    s.refresh(watch)
    assert watch.status == WatchStatus.AWAITING_APPROVAL.value
    assert watch.actions_taken == 1  # the attempt is counted at initiation

    pending = [n for n in capture_notifications if n["event_type"] == "approval_pending"]
    assert len(pending) == 1
    s.close()


def test_granted_replan_resumes_via_playbook_run_branch(
    workspace, new_session, stub_replan, capture_notifications
):
    from core.services.approval_grants import grant_grant
    from services.watch_rerun import resume_playbook_run_grant

    s = new_session()
    run, _, _, _ = _seed_failed_mission(s, workspace)
    watch = _make_watch(s, workspace, run)
    outcome = _run_action(s, watch, "replan", diagnosis="fix the plan")
    s.commit()
    grant = s.query(ApprovalGrant).get(outcome.grant_id)

    grant_grant(grant, granted_by="user:42")
    asyncio.run(resume_playbook_run_grant(s, grant))
    s.commit()

    assert grant.details["executed_result"]["success"] is True
    assert len(stub_replan) == 1  # the stored spec executed
    s.refresh(watch)
    assert watch.status == WatchStatus.WATCHING.value
    assert watch.actions_taken == 1  # no second budget charge on resume
    s.close()


# ---------------------------------------------------------------------------
# reassign
# ---------------------------------------------------------------------------


def test_reassign_requeues_to_different_capable_agent(
    workspace, new_session, monkeypatch, capture_notifications
):
    s = new_session()
    run, pred, failed, dependent = _seed_failed_mission(s, workspace)
    watch = _make_watch(s, workspace, run)
    _set_policy(s, workspace, FULL_AUTO_SETTINGS)
    # Prior agent 111 ranks first but MUST be excluded; beta (222) is next.
    _rank_stub(monkeypatch, [_match(111, "alpha"), _match(222, "beta")])

    outcome = _run_action(s, watch, "reassign", diagnosis="alpha keeps timing out")
    s.commit()

    assert outcome.executed is True, outcome.error
    s.refresh(run)
    assert run.state == "running"

    s.refresh(failed)
    assert failed.state == "skipped"
    assert failed.failure_reason_code == "reassigned_by_watch"

    clone = (
        s.query(OrchestrationTask)
        .filter(
            OrchestrationTask.run_id == run.id,
            OrchestrationTask.title == failed.title,
            OrchestrationTask.id != failed.id,
        )
        .one()
    )
    assert clone.agent_role == "beta"  # the explicit-override pin
    assert clone.state == "queued"
    assert clone.sequence_number == failed.sequence_number
    assert clone.input_context["watch_reassign"]["excluded_agent_id"] == 111

    # Dependency wiring: clone inherits the prerequisite; the dependent
    # task now depends on the clone (not the skipped original).
    clone_deps = (
        s.query(OrchestrationTaskDependency)
        .filter(OrchestrationTaskDependency.task_id == clone.id)
        .all()
    )
    assert [d.depends_on_task_id for d in clone_deps] == [pred.id]
    dependent_deps = (
        s.query(OrchestrationTaskDependency)
        .filter(OrchestrationTaskDependency.task_id == dependent.id)
        .all()
    )
    assert [d.depends_on_task_id for d in dependent_deps] == [clone.id]
    s.close()


def test_reassign_no_capable_agent_escalates(
    workspace, new_session, monkeypatch, capture_notifications
):
    s = new_session()
    run, _, _, _ = _seed_failed_mission(s, workspace)
    watch = _make_watch(s, workspace, run)
    _set_policy(s, workspace, FULL_AUTO_SETTINGS)
    _rank_stub(monkeypatch, [_match(111, "alpha")])  # only the failed agent

    outcome = _run_action(s, watch, "reassign")
    s.commit()

    assert outcome.escalated is True
    assert "no_capable_agent" in (outcome.error or "")
    s.refresh(watch)
    assert watch.status == WatchStatus.ESCALATED.value
    assert len(_escalation_cards(s, workspace, watch)) == 1
    escalations = [n for n in capture_notifications if n["event_type"] == "watch_escalation"]
    assert len(escalations) == 1
    s.close()


# ---------------------------------------------------------------------------
# spawn_agent
# ---------------------------------------------------------------------------


def _seed_blueprint(s, ws_id, rules=None, is_default=True):
    from core.models.blueprints import AgentBlueprint

    bp = AgentBlueprint(
        workspace_id=ws_id,
        name="workspace standard",
        rules=rules if rules is not None else {},  # handler-default shape
        is_default=is_default,
    )
    s.add(bp)
    s.commit()
    return bp


def test_spawn_agent_always_grant_gated_even_below_ceiling(
    workspace, new_session, capture_notifications
):
    """Section 8 Q5: auto_below_budget's auto lane must NOT auto-approve a
    spawn -- only full_auto does."""
    s = new_session()
    run, _, _, _ = _seed_failed_mission(s, workspace)
    watch = _make_watch(s, workspace, run)
    _seed_blueprint(s, workspace)
    _set_policy(
        s,
        workspace,
        {"approval_policy": {"policy": "auto_below_budget", "approval_dollar_ceiling": 100.0}},
    )

    outcome = _run_action(s, watch, "spawn_agent")
    s.commit()

    assert outcome.parked is True  # would auto-approve for any other action
    grant = s.query(ApprovalGrant).get(outcome.grant_id)
    assert grant.tool_name == "watch_spawn_agent"
    assert s.query(Agent).filter(Agent.workspace_id == workspace).count() == 0
    s.close()


def test_spawn_agent_full_auto_creates_validates_reassigns(
    workspace, new_session, capture_notifications
):
    """Blueprint rules DEFAULTING guard: an empty rules dict (the
    create_blueprint handler default) must not dead-end the spawn."""
    s = new_session()
    run, _, failed, _ = _seed_failed_mission(s, workspace, with_dependent=False)
    watch = _make_watch(s, workspace, run)
    _seed_blueprint(s, workspace, rules={})
    _set_policy(s, workspace, FULL_AUTO_SETTINGS)

    outcome = _run_action(
        s, watch, "spawn_agent",
        diagnosis="no capable writer on the roster",
        spawn_spec={"system_prompt": "You are a precise market analyst."},
    )
    s.commit()

    assert outcome.executed is True, outcome.error
    spawned = s.query(Agent).filter(Agent.workspace_id == workspace).one()
    assert spawned.tags and "watch-spawned" in spawned.tags

    # The failed task was requeued onto the NEW agent (name pin).
    clone = (
        s.query(OrchestrationTask)
        .filter(
            OrchestrationTask.run_id == run.id,
            OrchestrationTask.title == failed.title,
            OrchestrationTask.id != failed.id,
        )
        .one()
    )
    assert clone.agent_role == spawned.name
    assert clone.state == "queued"
    s.refresh(run)
    assert run.state == "running"
    s.close()


def test_spawn_agent_without_blueprint_escalates(
    workspace, new_session, capture_notifications
):
    s = new_session()
    run, _, _, _ = _seed_failed_mission(s, workspace)
    watch = _make_watch(s, workspace, run)
    _set_policy(s, workspace, FULL_AUTO_SETTINGS)
    # No blueprint seeded: spawn is blueprints-only.

    outcome = _run_action(s, watch, "spawn_agent")
    s.commit()

    assert outcome.escalated is True
    assert "blueprint" in (outcome.error or "")
    assert s.query(Agent).filter(Agent.workspace_id == workspace).count() == 0
    s.refresh(watch)
    assert watch.status == WatchStatus.ESCALATED.value
    s.close()


def test_spawn_strict_blueprint_failure_rolls_back_agent(
    workspace, new_session, capture_notifications
):
    s = new_session()
    run, _, _, _ = _seed_failed_mission(s, workspace)
    watch = _make_watch(s, workspace, run)
    _seed_blueprint(
        s, workspace,
        rules={"min_tools": 5, "enforce_mode": "strict"},  # unmeetable for a fresh spawn
    )
    _set_policy(s, workspace, FULL_AUTO_SETTINGS)

    outcome = _run_action(s, watch, "spawn_agent")
    s.commit()

    assert outcome.escalated is True
    assert "blueprint validation failed" in (outcome.error or "")
    # The spawn was rolled back -- validation stays authoritative.
    assert s.query(Agent).filter(Agent.workspace_id == workspace).count() == 0
    s.close()


# ---------------------------------------------------------------------------
# Budget rail + escalate
# ---------------------------------------------------------------------------


def test_action_budget_zero_goes_straight_to_escalate(
    workspace, new_session, stub_replan, capture_notifications
):
    s = new_session()
    run, _, _, _ = _seed_failed_mission(s, workspace)
    watch = _make_watch(s, workspace, run, action_budget=0)
    _set_policy(s, workspace, FULL_AUTO_SETTINGS)

    outcome = _run_action(s, watch, "replan")
    s.commit()

    assert outcome.escalated is True
    assert "budget" in outcome.detail
    assert stub_replan == []  # never reached the executor
    assert (
        s.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == workspace).count()
        == 0
    )
    s.refresh(watch)
    assert watch.status == WatchStatus.ESCALATED.value
    assert len(_escalation_cards(s, workspace, watch)) == 1
    escalations = [n for n in capture_notifications if n["event_type"] == "watch_escalation"]
    assert len(escalations) == 1
    s.close()


def test_explicit_escalate_action_is_ungated_and_unbudgeted(
    workspace, new_session, capture_notifications
):
    s = new_session()
    run, _, _, _ = _seed_failed_mission(s, workspace)
    watch = _make_watch(s, workspace, run, action_budget=0)  # even at zero budget

    outcome = _run_action(s, watch, "escalate", diagnosis="score below bar twice")
    s.commit()

    assert outcome.escalated is True
    s.refresh(watch)
    assert watch.status == WatchStatus.ESCALATED.value
    assert watch.actions_taken == 0  # escalate never consumes budget
    cards = _escalation_cards(s, workspace, watch)
    assert len(cards) == 1
    assert "score below bar twice" in cards[0].description
    s.close()

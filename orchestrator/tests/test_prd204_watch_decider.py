"""PRD-204 S10 -- the decision step.

Golden-files the policy table; drives the deterministic spine per policy
with the judge and diagnoser STUBBED (no live model anywhere): terminal ->
score -> close/notify (run_and_report), below-threshold -> diagnose -> ONE
tweak+rerun -> rescore -> final (score_and_improve), before/after change
report (watch_change), meaningful-change observation (persistent), judge
failure degrades to close-by-outcome, runaway guard (action_budget=0 ->
straight to escalate), and the new ingest_terminal deferral semantics.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from core.database.database import get_database_url
from core.models.core import BoardTask, RecipeExecution, WorkflowTemplate
from core.models.watches import WatchEvent
from core.models.watch_enums import WatchEventType, WatchStatus
from modules.coordination.run_verdict import ALL_DIMENSIONS, RunVerdict
from services.watch_decider import (
    DECIDED_ACTED,
    DECIDED_ESCALATED,
    DECIDED_FAILED,
    DECIDED_NOOP,
    DECIDED_PASSED,
    DECIDED_RECORDED,
    DEFAULT_ALLOWED_ACTIONS,
    POLICY_TABLE,
    Diagnosis,
    WatchDecider,
)
from services.watch_service import WatchService

FROZEN_NOW = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)
GOLDEN = Path(__file__).parent / "golden" / "prd204_policy_table.json"


# ---------------------------------------------------------------------------
# Golden file: the policy table is a contract, not an implementation detail
# ---------------------------------------------------------------------------


def test_policy_table_matches_golden_file():
    golden = json.loads(GOLDEN.read_text())
    assert golden["policy_table"] == POLICY_TABLE
    assert golden["default_allowed_actions"] == DEFAULT_ALLOWED_ACTIONS


def test_policy_table_covers_every_watch_policy():
    from core.models.watch_enums import WatchPolicy

    assert set(POLICY_TABLE) == {p.value for p in WatchPolicy}
    for flags in POLICY_TABLE.values():
        assert set(flags) == {
            "scores", "acts_on_low_score", "acts_on_failure", "compares", "recurring",
        }


def test_unknown_policy_falls_back_to_run_and_report():
    assert WatchDecider.policy_flags("garbage") == POLICY_TABLE["run_and_report"]


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubVerdicts:
    """RunVerdictService stand-in: queued verdicts, applies like the real one."""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.calls = 0

    async def score_run(self, db, watch, **kwargs):
        self.calls += 1
        return self.verdicts.pop(0)

    @staticmethod
    def apply_verdict(db, watch, verdict):
        from modules.coordination.run_verdict import RunVerdictService

        return RunVerdictService.apply_verdict(db, watch, verdict)


class _StubDiagnoser:
    def __init__(self, diagnosis=None, tweak=None):
        self.diagnosis = diagnosis
        self.tweak = tweak
        self.diagnose_calls = 0
        self.tweak_calls = 0

    async def diagnose(self, db, watch, **kwargs):
        self.diagnose_calls += 1
        return self.diagnosis

    async def draft_tweak(self, db, watch, **kwargs):
        self.tweak_calls += 1
        return self.tweak


def _verdict(score, reasoning="stub reasoning"):
    if score is None:
        return RunVerdict(score=None, reasoning="judge down", judge_failed=True,
                          output_hash="x" * 64)
    return RunVerdict(
        score=score,
        dimension_scores={d: score for d in ALL_DIMENSIONS},
        reasoning=reasoning,
        output_hash=uuid.uuid4().hex * 2,
    )


def _decide(decider, s, watch, state, now=FROZEN_NOW):
    return asyncio.run(decider.decide_terminal(s, watch, state, now))


# ---------------------------------------------------------------------------
# DB fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1 FROM watches LIMIT 1"))
            c.execute(text("SELECT 1 FROM recipe_executions LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"decider suite needs a reachable Postgres with schema: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def workspace(new_session):
    ws_id = str(uuid.uuid4())
    s = new_session()
    s.execute(
        text(
            "INSERT INTO workspaces (id, name) "
            "VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"
        ),
        {"id": ws_id, "n": "prd204-decider"},
    )
    s.commit()
    s.close()

    yield ws_id

    s = new_session.sweep()
    for stmt in (
        "DELETE FROM watch_events WHERE watch_id IN "
        "(SELECT id FROM watches WHERE workspace_id = CAST(:w AS uuid))",
        "DELETE FROM watches WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM approval_grants WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM board_tasks WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM recipe_executions WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM workflow_recipes WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM workspaces WHERE id = CAST(:w AS uuid)",
    ):
        s.execute(text(stmt), {"w": ws_id})
    s.commit()
    s.close()


@pytest.fixture
def notifications(monkeypatch):
    """Capture the SHARED notification seam (decider + ticker + actions)."""
    import services.watch_notifications as wn

    sent = []

    async def _capture(db, watch, *, event_type, title, message, status="ok"):
        sent.append({"event_type": event_type, "title": title,
                     "message": message, "status": status})
        return True

    monkeypatch.setattr(wn, "dispatch_watch_notification", _capture)
    return sent


def _seed_execution(s, ws_id, *, status="completed"):
    recipe = WorkflowTemplate(
        template_id=f"prd204-dc-{uuid.uuid4().hex[:10]}",
        name="decider test playbook",
        description="prd204 decider",
        workspace_id=ws_id,
        template_definition={"steps": []},
        steps=[{"step_id": "s1", "order": 1, "prompt_template": "do it"}],
        created_by="user_test",
    )
    s.add(recipe)
    s.commit()
    execution = RecipeExecution(
        execution_id=f"exec-{uuid.uuid4().hex[:12]}",
        recipe_id=recipe.id,
        workspace_id=ws_id,
        status=status,
        input_data={"topic": "pricing"},
        output_data={"final_output": "the summary"} if status == "completed" else None,
    )
    s.add(execution)
    s.commit()
    return recipe, execution


def _make_watch(s, ws_id, execution, **overrides):
    params = dict(
        workspace_id=ws_id,
        watch_type="playbook_execution",
        target_type="playbook_execution",
        target_id=execution.execution_id,
        title="Watch: decider test",
        success_criteria="a clean summary",
        policy="run_and_report",
        now=FROZEN_NOW,
    )
    params.update(overrides)
    watch = WatchService.create_watch(s, **params)
    s.commit()
    return watch


def _events(s, watch, event_type=None):
    q = s.query(WatchEvent).filter(WatchEvent.watch_id == watch.id)
    if event_type:
        q = q.filter(WatchEvent.event_type == event_type)
    return q.order_by(WatchEvent.created_at).all()


# ---------------------------------------------------------------------------
# ingest_terminal deferral (the S10 seam change)
# ---------------------------------------------------------------------------


def test_ingest_terminal_defers_scorable_and_closes_cancelled(workspace, new_session):
    s = new_session()
    _, execution = _seed_execution(s, workspace)
    watch = _make_watch(s, workspace, execution)

    event = WatchService.ingest_terminal(
        s,
        workspace_id=workspace,
        target_type="playbook_execution",
        target_id=execution.execution_id,
        terminal_state="completed",
        now=FROZEN_NOW,
    )
    s.commit()
    assert event is not None
    s.refresh(watch)
    # Scorable terminal: recorded + pulled forward, NOT closed -- the
    # decision step owns scoring and close.
    assert watch.status == WatchStatus.WATCHING.value
    assert watch.closed_at is None
    assert watch.next_check_at == FROZEN_NOW

    # Cancelled still closes at ingest (nothing to score).
    _, execution2 = _seed_execution(s, workspace, status="cancelled")
    watch2 = _make_watch(s, workspace, execution2)
    WatchService.ingest_terminal(
        s,
        workspace_id=workspace,
        target_type="playbook_execution",
        target_id=execution2.execution_id,
        terminal_state="cancelled",
    )
    s.commit()
    s.refresh(watch2)
    assert watch2.status == WatchStatus.CANCELLED.value
    assert watch2.closed_at is not None

    # Unknown vocabulary still parks for a human.
    _, execution3 = _seed_execution(s, workspace)
    watch3 = _make_watch(s, workspace, execution3)
    WatchService.ingest_terminal(
        s,
        workspace_id=workspace,
        target_type="playbook_execution",
        target_id=execution3.execution_id,
        terminal_state="vanished",
    )
    s.commit()
    s.refresh(watch3)
    assert watch3.status == WatchStatus.NEEDS_ATTENTION.value
    s.close()


def test_ingest_terminal_duplicate_keeps_repoking_decider(workspace, new_session):
    """Self-healing: duplicate deliveries pull the check forward again so a
    crashed decider gets re-poked until the watch closes."""
    s = new_session()
    _, execution = _seed_execution(s, workspace)
    watch = _make_watch(s, workspace, execution)

    first = WatchService.ingest_terminal(
        s, workspace_id=workspace, target_type="playbook_execution",
        target_id=execution.execution_id, terminal_state="completed",
        now=FROZEN_NOW,
    )
    later = FROZEN_NOW + timedelta(minutes=10)
    second = WatchService.ingest_terminal(
        s, workspace_id=workspace, target_type="playbook_execution",
        target_id=execution.execution_id, terminal_state="completed",
        now=later,
    )
    s.commit()
    assert first is not None and second is None  # one event
    s.refresh(watch)
    assert watch.status == WatchStatus.WATCHING.value
    assert watch.next_check_at == later  # re-poked
    assert len(_events(s, watch, WatchEventType.TERMINAL.value)) == 1
    s.close()


# ---------------------------------------------------------------------------
# run_and_report: terminal -> score -> notify -> close (no actions)
# ---------------------------------------------------------------------------


def test_run_and_report_pass_closes_passed_and_notifies(
    workspace, new_session, notifications
):
    s = new_session()
    _, execution = _seed_execution(s, workspace)
    watch = _make_watch(s, workspace, execution)
    decider = WatchDecider(verdict_service=_StubVerdicts([_verdict(0.86)]))

    decision = _decide(decider, s, watch, "completed")
    s.commit()

    assert decision == DECIDED_PASSED
    s.refresh(watch)
    assert watch.status == WatchStatus.PASSED.value
    assert watch.final_score == pytest.approx(0.86)
    assert "stub reasoning" in watch.final_verdict
    assert len(_events(s, watch, WatchEventType.SCORED.value)) == 1

    verdicts = [n for n in notifications if n["event_type"] == "watch_verdict"]
    assert len(verdicts) == 1
    assert verdicts[0]["status"] == "ok"
    assert "8.6/10" in verdicts[0]["message"]  # display edge x10
    s.close()


@pytest.mark.parametrize(
    "score,expected_status",
    [(0.79, WatchStatus.FAILED), (0.80, WatchStatus.PASSED)],
)
def test_run_and_report_threshold_boundary(
    workspace, new_session, notifications, score, expected_status
):
    """0.79 fails the default 0.8 bar; 0.80 passes it."""
    s = new_session()
    _, execution = _seed_execution(s, workspace)
    watch = _make_watch(s, workspace, execution)
    decider = WatchDecider(verdict_service=_StubVerdicts([_verdict(score)]))

    _decide(decider, s, watch, "completed")
    s.commit()
    s.refresh(watch)
    assert watch.status == expected_status.value
    s.close()


def test_run_and_report_failed_run_closes_failed_without_actions(
    workspace, new_session, notifications
):
    s = new_session()
    _, execution = _seed_execution(s, workspace, status="failed")
    watch = _make_watch(s, workspace, execution)
    decider = WatchDecider(
        verdict_service=_StubVerdicts([_verdict(0.1)]),
        diagnoser=_StubDiagnoser(),  # must never be called
    )

    decision = _decide(decider, s, watch, "failed")
    s.commit()

    assert decision == DECIDED_FAILED
    s.refresh(watch)
    assert watch.status == WatchStatus.FAILED.value
    assert watch.actions_taken == 0  # no actions under run_and_report
    assert decider._diagnoser.diagnose_calls == 0
    verdicts = [n for n in notifications if n["event_type"] == "watch_verdict"]
    assert verdicts[0]["status"] == "error"
    s.close()


def test_judge_failure_degrades_to_close_by_outcome(
    workspace, new_session, notifications
):
    """A judge_failed verdict must not wedge the watch: completed closes
    PASSED (v1 outcome semantics), with the degradation noted."""
    s = new_session()
    _, execution = _seed_execution(s, workspace)
    watch = _make_watch(s, workspace, execution)
    decider = WatchDecider(verdict_service=_StubVerdicts([_verdict(None)]))

    decision = _decide(decider, s, watch, "completed")
    s.commit()

    assert decision == DECIDED_PASSED
    s.refresh(watch)
    assert watch.status == WatchStatus.PASSED.value
    assert watch.final_score is None
    assert "scoring unavailable" in watch.final_verdict.lower()
    s.close()


# ---------------------------------------------------------------------------
# score_and_improve: below-threshold -> diagnose -> ONE tweak+rerun -> final
# ---------------------------------------------------------------------------


def _improve_watch(s, workspace, execution, **overrides):
    return _make_watch(
        s, workspace, execution, policy="score_and_improve", **overrides
    )


def test_low_score_diagnoses_and_launches_tweaked_rerun(
    workspace, new_session, notifications, monkeypatch
):
    import services.watch_rerun as wr

    launched = []
    monkeypatch.setattr(wr, "launch_execution", lambda e: launched.append(e.execution_id))

    s = new_session()
    recipe, execution = _seed_execution(s, workspace)
    watch = _improve_watch(s, workspace, execution)
    # full_auto so the rerun gate auto-approves.
    s.execute(
        text("UPDATE workspaces SET settings = CAST(:cfg AS jsonb) WHERE id = CAST(:w AS uuid)"),
        {"cfg": json.dumps({"approval_policy": {"policy": "full_auto"},
                            "autonomy": {"level": "full"}}), "w": workspace},
    )
    s.commit()

    diagnosis = Diagnosis(
        cause="step s1 prompt is too vague",
        proposed_action="tweak_rerun",
        step_overrides={"s1": {"prompt_template": "do it with citations"}},
    )
    decider = WatchDecider(
        verdict_service=_StubVerdicts([_verdict(0.5)]),
        diagnoser=_StubDiagnoser(diagnosis=diagnosis),
    )

    decision = _decide(decider, s, watch, "completed")
    s.commit()

    assert decision == DECIDED_ACTED
    assert len(launched) == 1
    rerun = (
        s.query(RecipeExecution)
        .filter(RecipeExecution.retry_of == execution.execution_id)
        .one()
    )
    assert rerun.execution_metadata["step_overrides"] == {
        "s1": {"prompt_template": "do it with citations"}
    }
    assert rerun.triggered_by == "watch_rerun"

    s.refresh(watch)
    assert watch.target_id == rerun.execution_id  # the watch followed
    assert watch.actions_taken == 1
    assert len(_events(s, watch, WatchEventType.DIAGNOSED.value)) == 1
    actions = [n for n in notifications if n["event_type"] == "watch_action"]
    assert len(actions) == 1
    # Diagnosis was one bounded call; no tweak-draft needed (overrides came
    # with the diagnosis).
    assert decider._diagnoser.diagnose_calls == 1
    assert decider._diagnoser.tweak_calls == 0
    s.close()


def test_second_below_threshold_closes_final_no_second_tweak(
    workspace, new_session, notifications
):
    """The ONE-tweak rule: after a corrective attempt, the rescore is FINAL
    -- below-threshold now closes failed instead of acting again."""
    s = new_session()
    _, execution = _seed_execution(s, workspace)
    watch = _improve_watch(s, workspace, execution)
    watch.actions_taken = 1  # the improve cycle was already spent
    s.commit()

    diagnoser = _StubDiagnoser(diagnosis=Diagnosis(cause="still weak", proposed_action="tweak_rerun"))
    decider = WatchDecider(
        verdict_service=_StubVerdicts([_verdict(0.5)]),
        diagnoser=diagnoser,
    )
    decision = _decide(decider, s, watch, "completed")
    s.commit()

    assert decision == DECIDED_FAILED
    assert diagnoser.diagnose_calls == 0  # no second improve cycle
    s.refresh(watch)
    assert watch.status == WatchStatus.FAILED.value
    assert "corrective attempt was already made" in watch.final_verdict
    s.close()


def test_runaway_guard_budget_zero_straight_to_escalate(
    workspace, new_session, notifications
):
    s = new_session()
    _, execution = _seed_execution(s, workspace, status="failed")
    watch = _improve_watch(s, workspace, execution, action_budget=0)
    decider = WatchDecider(
        verdict_service=_StubVerdicts([_verdict(0.2)]),
        diagnoser=_StubDiagnoser(
            diagnosis=Diagnosis(cause="broken step", proposed_action="rerun")
        ),
    )

    decision = _decide(decider, s, watch, "failed")
    s.commit()

    assert decision == DECIDED_ESCALATED
    # STRAIGHT to escalate: the diagnosis LLM is never spent when no action
    # is possible.
    assert decider._diagnoser.diagnose_calls == 0
    s.refresh(watch)
    assert watch.status == WatchStatus.ESCALATED.value
    # No rerun row was created.
    assert (
        s.query(RecipeExecution)
        .filter(RecipeExecution.retry_of == execution.execution_id)
        .count()
        == 0
    )
    cards = (
        s.query(BoardTask)
        .filter(
            BoardTask.workspace_id == workspace,
            BoardTask.tags.contains(["escalation", f"watch:{watch.id}"]),
        )
        .count()
    )
    assert cards == 1
    escalations = [n for n in notifications if n["event_type"] == "watch_escalation"]
    assert len(escalations) == 1
    s.close()


def test_diagnosis_unavailable_falls_back_deterministically(
    workspace, new_session, notifications, monkeypatch
):
    """LLM down -> Diagnosis None -> deterministic fallback action (rerun
    for playbooks), never a wedge."""
    import services.watch_rerun as wr

    launched = []
    monkeypatch.setattr(wr, "launch_execution", lambda e: launched.append(e.execution_id))

    s = new_session()
    _, execution = _seed_execution(s, workspace, status="failed")
    watch = _improve_watch(s, workspace, execution)
    s.execute(
        text("UPDATE workspaces SET settings = CAST(:cfg AS jsonb) WHERE id = CAST(:w AS uuid)"),
        {"cfg": json.dumps({"approval_policy": {"policy": "full_auto"},
                            "autonomy": {"level": "full"}}), "w": workspace},
    )
    s.commit()

    decider = WatchDecider(
        verdict_service=_StubVerdicts([_verdict(0.2)]),
        diagnoser=_StubDiagnoser(diagnosis=None),
    )
    decision = _decide(decider, s, watch, "failed")
    s.commit()

    assert decision == DECIDED_ACTED
    assert len(launched) == 1  # plain rerun, no overrides
    rerun = (
        s.query(RecipeExecution)
        .filter(RecipeExecution.retry_of == execution.execution_id)
        .one()
    )
    assert "step_overrides" not in (rerun.execution_metadata or {})
    s.close()


# ---------------------------------------------------------------------------
# watch_change: compare against the prior attempt, then report
# ---------------------------------------------------------------------------


def test_watch_change_reports_before_after_delta(
    workspace, new_session, notifications
):
    s = new_session()
    _, execution = _seed_execution(s, workspace)
    watch = _make_watch(s, workspace, execution, policy="watch_change")

    # Simulate the prior attempt: a SCORED event for the original target,
    # then the watch followed a rerun.
    from modules.coordination.run_verdict import RunVerdictService

    prior = _verdict(0.55)
    RunVerdictService.apply_verdict(s, watch, prior)
    s.commit()
    _, rerun_execution = _seed_execution(s, workspace)
    WatchService.follow(
        s, watch,
        new_target_type="playbook_execution",
        new_target_id=rerun_execution.execution_id,
        reason="rerun",
    )
    s.commit()

    decider = WatchDecider(verdict_service=_StubVerdicts([_verdict(0.9)]))
    decision = _decide(decider, s, watch, "completed")
    s.commit()

    assert decision == DECIDED_PASSED
    changes = _events(s, watch, WatchEventType.CHANGE_REPORT.value)
    assert len(changes) == 1
    snap = changes[0].snapshot
    assert snap["before_score"] == pytest.approx(0.55)
    assert snap["after_score"] == pytest.approx(0.9)
    assert snap["delta"] == pytest.approx(0.35)
    s.close()


# ---------------------------------------------------------------------------
# persistent: record, notify on flip, never close on terminal
# ---------------------------------------------------------------------------


def test_persistent_records_and_notifies_only_on_flip(
    workspace, new_session, notifications
):
    s = new_session()
    _, execution = _seed_execution(s, workspace)
    watch = _make_watch(s, workspace, execution, policy="persistent")
    decider = WatchDecider(verdict_service=_StubVerdicts([]))  # never scores

    first = _decide(decider, s, watch, "completed")
    s.commit()
    assert first == DECIDED_RECORDED
    s.refresh(watch)
    assert watch.status == WatchStatus.WATCHING.value  # never closes
    assert notifications == []  # first observation is not a change

    # Same observation again -> dedupe, no noise.
    again = _decide(decider, s, watch, "completed")
    assert again == DECIDED_NOOP

    # A flip (completed -> failed) notifies as degradation.
    flipped = _decide(decider, s, watch, "failed")
    s.commit()
    assert flipped == DECIDED_RECORDED
    escalations = [n for n in notifications if n["event_type"] == "watch_escalation"]
    assert len(escalations) == 1
    assert "'completed' -> 'failed'" in escalations[0]["message"]
    s.refresh(watch)
    assert watch.status == WatchStatus.WATCHING.value  # still recurring
    s.close()


def test_observe_scheduled_flip_detection(workspace, new_session, notifications):
    s = new_session()
    recipe, completed_exec = _seed_execution(s, workspace)  # one completed run
    s.add(
        RecipeExecution(
            execution_id=f"exec-{uuid.uuid4().hex[:12]}",
            recipe_id=recipe.id,
            workspace_id=workspace,
            status="failed",
            input_data={},
        )
    )
    s.commit()
    # Deterministic ordering: the completed run happened first, then the
    # failed one (no wall-clock dependence).
    s.execute(
        text(
            "UPDATE recipe_executions SET started_at = :t "
            "WHERE execution_id = :e"
        ),
        {"t": datetime(2026, 7, 16, 12, 0, 0), "e": completed_exec.execution_id},
    )
    s.execute(
        text(
            "UPDATE recipe_executions SET started_at = :t "
            "WHERE recipe_id = :r AND status = 'failed'"
        ),
        {"t": datetime(2026, 7, 16, 13, 0, 0), "r": recipe.id},
    )
    s.commit()
    watch = WatchService.create_watch(
        s,
        workspace_id=workspace,
        watch_type="scheduled_playbook",
        target_type="scheduled_playbook",
        target_id=str(recipe.id),
        title="Watch: schedule",
        policy="persistent",
        now=FROZEN_NOW,
    )
    s.commit()

    playbook = s.query(WorkflowTemplate).filter(WorkflowTemplate.id == recipe.id).one()
    decider = WatchDecider(verdict_service=_StubVerdicts([]))
    decision = asyncio.run(decider.observe_scheduled(s, watch, playbook, FROZEN_NOW))
    s.commit()

    assert decision == DECIDED_RECORDED
    escalations = [n for n in notifications if n["event_type"] == "watch_escalation"]
    assert len(escalations) == 1
    assert "started failing" in escalations[0]["title"]

    # Idempotent per latest execution.
    again = asyncio.run(decider.observe_scheduled(s, watch, playbook, FROZEN_NOW))
    assert again == DECIDED_NOOP
    s.close()


# ---------------------------------------------------------------------------
# Non-claimable watches are never decided
# ---------------------------------------------------------------------------


def test_decider_noops_on_parked_watch(workspace, new_session, notifications):
    s = new_session()
    _, execution = _seed_execution(s, workspace)
    watch = _make_watch(s, workspace, execution)
    WatchService.transition(s, watch, WatchStatus.NEEDS_ATTENTION)
    s.commit()

    decider = WatchDecider(verdict_service=_StubVerdicts([_verdict(0.9)]))
    assert _decide(decider, s, watch, "completed") == DECIDED_NOOP
    assert notifications == []
    s.close()

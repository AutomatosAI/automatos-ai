"""PRD-164 S4 — bounded replanning: progress ledger + joiner decision point.

Magentic-One pattern: after every coordinator tick the run's task states are
snapshotted into a progress ledger persisted on ``run.config``. Forward
progress (newly done/verified tasks) resets the stall streak; *churn*
(attempts climbing, or verified tasks regressing, with no forward progress)
increments it; pure idleness (nothing changed — e.g. waiting out a stall
threshold) leaves it alone. When the streak reaches the configured limit the
LLMCompiler-style joiner decides: REPLAN through the one existing
``replan_mission`` engine while ``replan_count`` is within
``COORDINATOR_MAX_REPLANS``, else HALT (run FAILED, ``stop_reason='stalled'``).
Every verdict is audited (``run_stall_ledger`` event + ledger history).

AC2: an induced loop replans-or-halts within bounds, with an audit trail.

All tests here are DB-free: the ledger module is pure, and the coordinator
wiring is driven with mocks.
"""
from __future__ import annotations

import asyncio
import copy
import importlib.util as _ilu
import os
import sys as _sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

# Dummy POSTGRES_* satisfies the config chain (blessed pattern, see
# tests/test_harness_self_management.py) — the port points at nothing so the
# modules.tools import chain's fail-soft DB connect refuses instantly instead
# of hanging on a wedged local proxy. CI exports real POSTGRES_* so these
# setdefaults no-op there. Nothing in this file touches a DB.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")


# Lean-venv shim: importing modules.tools.* runs modules/tools/__init__, which
# pulls modules.rag's ingestion chain (camelot at module top). Stub the missing
# *leaf* only when truly absent — never the modules.rag package.
def _camelot_unlocatable() -> bool:  # pragma: no cover - env-dependent
    try:
        return _ilu.find_spec("camelot") is None
    except ValueError:
        return False


if _camelot_unlocatable():  # pragma: no cover - env-dependent
    import types as _types

    _sys.modules.setdefault("camelot", _types.ModuleType("camelot"))

# CI collection-order guard: earlier-collected tests stub modules.*/consumers.*
# in sys.modules (bare ModuleType, no __spec__). On Linux collection order the
# stubs are still live HERE, so the real imports below resolve against them and
# die at collection ("unknown location" ImportError — see PR #434 CI). Purge
# origin-less entries so the real packages import fresh; conftest's autouse
# repair fixture re-binds everything else at test time.
import sys as _sys_guard  # noqa: E402
for _name in [n for n, m in list(_sys_guard.modules.items())
              if (n == "modules" or n.startswith("modules.")
                  or n == "consumers" or n.startswith("consumers."))
              and getattr(m, "__spec__", None) is None]:
    _sys_guard.modules.pop(_name, None)

from modules.coordination import progress_ledger  # noqa: E402
from modules.coordination.progress_ledger import (  # noqa: E402
    HISTORY_LIMIT,
    JoinerDecision,
    advance,
    reset_after_replan,
    snapshot_tasks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STALL_LIMIT = 3
MAX_REPLANS = 2


def _task(state: str, attempts: int = 0) -> SimpleNamespace:
    return SimpleNamespace(state=state, attempt_number=attempts)


def _snap(*, total=2, done=0, verified=0, attempts=0) -> dict:
    return {"total": total, "done": done, "verified": verified,
            "attempts": attempts}


def _advance(ledger, snapshot, *, replan_count=0):
    return advance(
        ledger,
        snapshot,
        stall_limit=STALL_LIMIT,
        replan_count=replan_count,
        max_replans=MAX_REPLANS,
    )


# ---------------------------------------------------------------------------
# snapshot_tasks — counting contract
# ---------------------------------------------------------------------------


class TestSnapshotTasks:
    def test_counts_done_verified_and_attempts(self):
        tasks = [
            _task("verified", 1),
            _task("failed", 3),
            _task("skipped", 0),
            _task("retrying", 2),
            _task("running", 1),
        ]
        snap = snapshot_tasks(tasks)
        assert snap == {"total": 5, "done": 3, "verified": 1, "attempts": 7}

    def test_none_attempts_count_as_zero(self):
        snap = snapshot_tasks([SimpleNamespace(state="queued",
                                               attempt_number=None)])
        assert snap["attempts"] == 0


# ---------------------------------------------------------------------------
# advance — the ledger state machine
# ---------------------------------------------------------------------------


class TestLedgerAdvance:
    def test_first_observation_is_baseline_continue(self):
        ledger, decision = _advance(None, _snap(attempts=1))
        assert decision is JoinerDecision.CONTINUE
        assert ledger["stall_streak"] == 0
        assert ledger["snapshot"]["attempts"] == 1
        assert ledger["history"][-1]["observation"] == "baseline"

    def test_churn_increments_streak(self):
        ledger, _ = _advance(None, _snap(attempts=1))
        ledger, decision = _advance(ledger, _snap(attempts=2))
        assert decision is JoinerDecision.CONTINUE
        assert ledger["stall_streak"] == 1
        assert ledger["history"][-1]["observation"] == "churn"

    def test_progress_resets_streak(self):
        ledger, _ = _advance(None, _snap(attempts=1))
        ledger, _ = _advance(ledger, _snap(attempts=2))
        ledger, _ = _advance(ledger, _snap(attempts=3))
        assert ledger["stall_streak"] == 2
        # A task lands (done up) — streak resets even though attempts also grew
        ledger, decision = _advance(ledger, _snap(done=1, verified=1,
                                                  attempts=4))
        assert decision is JoinerDecision.CONTINUE
        assert ledger["stall_streak"] == 0
        assert ledger["history"][-1]["observation"] == "progress"

    def test_idle_does_not_increment_and_does_not_rewrite(self):
        ledger, _ = _advance(None, _snap(attempts=2))
        ledger, _ = _advance(ledger, _snap(attempts=3))
        before = copy.deepcopy(ledger)
        same, decision = _advance(ledger, _snap(attempts=3))
        assert decision is JoinerDecision.CONTINUE
        # Identical object back — the caller skips the JSONB write on idle
        # ticks (5s cadence would otherwise churn run.config forever).
        assert same is ledger
        assert ledger == before

    def test_verified_regression_counts_as_churn(self):
        # Human rejects a verified task (VERIFIED → RETRYING ping-pong).
        ledger, _ = _advance(None, _snap(verified=2, done=2, attempts=2))
        ledger, _ = _advance(ledger, _snap(verified=1, done=1, attempts=2))
        assert ledger["stall_streak"] == 1
        assert ledger["history"][-1]["observation"] == "churn"

    def test_streak_at_limit_with_replans_left_is_replan(self):
        ledger = None
        snap_attempts = 0
        decision = JoinerDecision.CONTINUE
        ledger, decision = _advance(ledger, _snap(attempts=snap_attempts))
        for _ in range(STALL_LIMIT):
            snap_attempts += 1
            ledger, decision = _advance(ledger, _snap(attempts=snap_attempts),
                                        replan_count=0)
        assert ledger["stall_streak"] == STALL_LIMIT
        assert decision is JoinerDecision.REPLAN
        assert ledger["history"][-1]["decision"] == "replan"

    def test_streak_at_limit_with_replans_exhausted_is_halt(self):
        ledger, _ = _advance(None, _snap(attempts=0))
        decision = JoinerDecision.CONTINUE
        for i in range(STALL_LIMIT):
            ledger, decision = _advance(ledger, _snap(attempts=i + 1),
                                        replan_count=MAX_REPLANS)
        assert decision is JoinerDecision.HALT
        assert ledger["history"][-1]["decision"] == "halt"

    def test_inputs_are_not_mutated(self):
        ledger, _ = _advance(None, _snap(attempts=1))
        frozen = copy.deepcopy(ledger)
        snapshot = _snap(attempts=2)
        frozen_snap = copy.deepcopy(snapshot)
        _advance(ledger, snapshot)
        assert ledger == frozen
        assert snapshot == frozen_snap

    def test_history_is_bounded(self):
        ledger = None
        attempts = 0
        for i in range(HISTORY_LIMIT * 3):
            attempts += 1
            new_ledger, _ = _advance(
                ledger, _snap(done=i % 2, attempts=attempts))
            ledger = new_ledger
        assert len(ledger["history"]) <= HISTORY_LIMIT

    def test_reset_after_replan_rebaselines(self):
        ledger, _ = _advance(None, _snap(attempts=1))
        for i in range(STALL_LIMIT):
            ledger, _ = _advance(ledger, _snap(attempts=i + 2))
        reset = reset_after_replan(ledger)
        assert reset["stall_streak"] == 0
        assert reset["history"][-1]["observation"] == "replan_reset"
        # Next observation is a fresh baseline, not churn against stale counts
        nxt, decision = _advance(reset, _snap(attempts=99))
        assert decision is JoinerDecision.CONTINUE
        assert nxt["stall_streak"] == 0


# ---------------------------------------------------------------------------
# Joiner wiring on the coordinator (AC2) — mocked DB, no I/O
# ---------------------------------------------------------------------------


def _make_run(*, replan_count=0, config=None):
    return SimpleNamespace(
        id=uuid4(),
        state="running",
        config=dict(config or {}),
        replan_count=replan_count,
    )


def _db_returning_tasks(tasks):
    db = MagicMock()
    q = MagicMock()
    q.filter.return_value = q
    q.all.return_value = tasks
    db.query.return_value = q
    return db


@pytest.fixture()
def coordinator(monkeypatch):
    import services.coordinator_service as cs

    svc = cs.CoordinatorService.__new__(cs.CoordinatorService)
    events = []
    transitions = []

    def _emit(db, run_id, event_type, actor_type, actor_id=None,
              task_id=None, payload=None, **kw):
        events.append(SimpleNamespace(event_type=event_type,
                                      actor_id=actor_id, payload=payload))

    def _transition(db, run, new_state, actor_type, actor_id=None,
                    reason=None, stop_reason=None, stop_detail=None, **kw):
        run.state = new_state.value
        transitions.append(SimpleNamespace(new_state=new_state,
                                           stop_reason=stop_reason,
                                           stop_detail=stop_detail,
                                           reason=reason))

    monkeypatch.setattr(cs, "emit_event", _emit)
    monkeypatch.setattr(cs, "transition_run", _transition)
    monkeypatch.setattr(cs, "_store_mission_memory_safe", AsyncMock())
    svc.replan_mission = AsyncMock()
    return SimpleNamespace(svc=svc, events=events, transitions=transitions,
                           cs=cs)


class TestJoinerCheckpointWiring:
    """AC2 — an induced loop triggers replan-or-halt within bounds, with an
    audit trail on the mission (events + run.config ledger)."""

    def _stall_limit(self, coordinator):
        from config import Config
        return Config.COORDINATOR_STALL_LEDGER_LIMIT

    def test_induced_loop_triggers_bounded_replan_with_audit(self, coordinator):
        svc, events = coordinator.svc, coordinator.events
        run = _make_run(replan_count=0)
        limit = self._stall_limit(coordinator)

        async def induce():
            # Tick 0 = baseline, then `limit` churn ticks: one task keeps
            # retrying (attempt_number climbing) while nothing completes.
            for attempts in range(limit + 1):
                tasks = [SimpleNamespace(state="verified", attempt_number=1),
                         SimpleNamespace(state="retrying",
                                         attempt_number=attempts)]
                await svc._joiner_checkpoint(_db_returning_tasks(tasks), run)

        asyncio.run(induce())

        # Replan happened exactly once, through the ONE existing engine
        svc.replan_mission.assert_awaited_once()
        kwargs = svc.replan_mission.await_args.kwargs
        assert kwargs.get("trigger") == "stall_ledger"
        # Audit trail: a run_stall_ledger event carrying the verdict
        verdicts = [e for e in events
                    if getattr(e.event_type, "value", e.event_type)
                    == "run_stall_ledger"]
        assert len(verdicts) == 1
        assert verdicts[0].payload["decision"] == "replan"
        assert verdicts[0].payload["stall_streak"] >= limit
        # Audit trail: the ledger persisted on the mission, reset for the
        # fresh plan so the next window is honestly measured.
        ledger = run.config["progress_ledger"]
        assert ledger["stall_streak"] == 0
        assert any(h["observation"] == "replan_reset"
                   for h in ledger["history"])

    def test_loop_with_replans_exhausted_halts_run(self, coordinator):
        from config import Config

        svc, events, transitions = (coordinator.svc, coordinator.events,
                                    coordinator.transitions)
        run = _make_run(replan_count=Config.COORDINATOR_MAX_REPLANS)
        limit = self._stall_limit(coordinator)

        async def induce():
            for attempts in range(limit + 1):
                tasks = [SimpleNamespace(state="retrying",
                                         attempt_number=attempts)]
                await svc._joiner_checkpoint(_db_returning_tasks(tasks), run)

        asyncio.run(induce())

        svc.replan_mission.assert_not_awaited()
        from core.models.orchestration_enums import RunState
        halts = [t for t in transitions if t.new_state is RunState.FAILED]
        assert len(halts) == 1
        assert halts[0].stop_reason == "stalled"
        verdicts = [e for e in events
                    if getattr(e.event_type, "value", e.event_type)
                    == "run_stall_ledger"]
        assert verdicts and verdicts[-1].payload["decision"] == "halt"
        # Mission memory records the failure for PRD-159 recall
        coordinator.cs._store_mission_memory_safe.assert_awaited_once()

    def test_progressing_mission_never_intervenes(self, coordinator):
        svc, events = coordinator.svc, coordinator.events
        run = _make_run()

        async def progress():
            for done in range(6):
                tasks = [SimpleNamespace(state="verified", attempt_number=1)
                         for _ in range(done)]
                tasks.append(SimpleNamespace(state="running",
                                             attempt_number=done))
                await svc._joiner_checkpoint(_db_returning_tasks(tasks), run)

        asyncio.run(progress())

        svc.replan_mission.assert_not_awaited()
        assert not coordinator.transitions
        assert all(getattr(e.event_type, "value", e.event_type)
                   != "run_stall_ledger" for e in events)

    def test_replan_error_falls_back_to_halt(self, coordinator):
        """A hard replan failure may not strand the loop — joiner halts."""
        svc, transitions = coordinator.svc, coordinator.transitions
        run = _make_run(replan_count=0)
        limit = self._stall_limit(coordinator)
        svc.replan_mission = AsyncMock(side_effect=RuntimeError("planner down"))

        async def induce():
            for attempts in range(limit + 1):
                tasks = [SimpleNamespace(state="retrying",
                                         attempt_number=attempts)]
                await svc._joiner_checkpoint(_db_returning_tasks(tasks), run)

        asyncio.run(induce())

        from core.models.orchestration_enums import RunState
        halts = [t for t in transitions if t.new_state is RunState.FAILED]
        assert len(halts) == 1
        assert halts[0].stop_reason == "stalled"

    def test_total_interventions_bounded_by_max_replans_plus_halt(
            self, coordinator):
        """Drive churn well past every bound: interventions stop at
        MAX_REPLANS replans + exactly one halt — never an unbounded loop."""
        from config import Config

        svc, transitions = coordinator.svc, coordinator.transitions
        run = _make_run(replan_count=0)

        async def fake_replan(db, run_id, actor_id, notes=None, **kwargs):
            run.replan_count += 1   # what the real engine does

        svc.replan_mission = AsyncMock(side_effect=fake_replan)

        async def churn_forever():
            attempts = 0
            for _ in range(40):
                if run.state != "running":
                    break   # _process_run only joins RUNNING runs
                attempts += 1
                tasks = [SimpleNamespace(state="retrying",
                                         attempt_number=attempts)]
                await svc._joiner_checkpoint(_db_returning_tasks(tasks), run)

        asyncio.run(churn_forever())

        assert svc.replan_mission.await_count == Config.COORDINATOR_MAX_REPLANS
        from core.models.orchestration_enums import RunState
        halts = [t for t in transitions if t.new_state is RunState.FAILED]
        assert len(halts) == 1
        assert run.state == RunState.FAILED.value

"""PRD-224 US-003 -- bounded corrective action + escalation for ticket watches.

Pure, LLM-free, DB-free coverage of the board-ticket corrective loop:

- a below-bar board terminal re-runs through the board's OWN run-now machinery
  (api/board_tasks._redispatch_task -- the shared function extracted from the
  Run-Now route, NOT the HTTP route);
- the re-run is budget-railed exactly like mission actions: with action_budget=1
  a below-bar verdict yields exactly ONE re-run, then the next below-bar terminal
  escalates (record_action's hard stop);
- escalation goes through the EXISTING escalate_watch_now → escalation_service
  path and narrates as ``watch_escalation`` through the untouched
  watch_notifications seam to deliver_background_message;
- the watch follows the re-run: lineage is appended on every corrective attempt.

The DB-touching collaborators (record_action budget rail, follow, the escalation
card, the notification dispatcher) are stubbed so the ORCHESTRATION is proven
without Postgres; their own behaviour is locked by the PRD-204 DB-backed suites.
"""
from __future__ import annotations

import asyncio
import inspect
import uuid
from types import SimpleNamespace

import services.watch_actions as wa
import services.watch_service as ws_mod
from modules.coordination.run_verdict import RunVerdict
from services.watch_decider import DECIDED_ACTED, DECIDED_ESCALATED, WatchDecider


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeQuery:
    def __init__(self, result):
        self._result = result

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._result


class _FakeDB:
    def __init__(self, result=None):
        self._result = result

    def query(self, *a, **k):
        return _FakeQuery(self._result)

    def flush(self):
        pass


def _task(**over):
    base = dict(id=4242, assigned_agent_id=7, status="failed", source_type="user")
    base.update(over)
    return SimpleNamespace(**base)


def _watch(**over):
    base = dict(
        id=uuid.uuid4(), workspace_id=uuid.uuid4(), title="Chase Q3 invoices",
        status="watching", target_type="board_task", target_id="4242",
        actions_taken=0, action_budget=2, action_budget_hint=None, lineage=[],
        created_by=None, origin_chat_id=None, quality_threshold=0.8,
    )
    base.update(over)
    return SimpleNamespace(**base)


class _BudgetRail:
    """Emulates WatchService.record_action's rail: allow until taken == budget."""

    def __init__(self, budget):
        self.budget = budget
        self.recorded = []

    def __call__(self, db, watch, *, action, summary=None, snapshot=None):
        taken = watch.actions_taken or 0
        if taken >= self.budget:
            return watch, False
        watch.actions_taken = taken + 1
        self.recorded.append(action)
        return watch, True


def _wire(monkeypatch, *, budget_rail, task_result):
    """Install the DB-touching stubs run_board_task_action leans on and return
    the capture bags."""
    import api.board_tasks as bt

    bag = {"redispatched": [], "escalated": [], "followed": []}

    monkeypatch.setattr(bt, "_redispatch_task",
                        lambda db, t: bag["redispatched"].append(t.id))

    async def _esc(db, w, *, reason):
        bag["escalated"].append(reason)

    monkeypatch.setattr(wa, "escalate_watch_now", _esc)
    monkeypatch.setattr(ws_mod.WatchService, "record_action", staticmethod(budget_rail))

    def _follow(db, w, *, new_target_type, new_target_id, reason=None, **kw):
        w.lineage = [
            *(w.lineage or []),
            {"target_type": new_target_type, "target_id": new_target_id, "reason": reason},
        ]
        bag["followed"].append(new_target_id)
        return w

    monkeypatch.setattr(ws_mod.WatchService, "follow", staticmethod(_follow))
    return bag, _FakeDB(result=task_result)


# ---------------------------------------------------------------------------
# run_board_task_action -- the executor
# ---------------------------------------------------------------------------


def test_rerun_redispatches_through_shared_run_now(monkeypatch):
    watch, task = _watch(), _task()
    bag, db = _wire(monkeypatch, budget_rail=_BudgetRail(2), task_result=task)

    outcome = asyncio.run(wa.run_board_task_action(db, watch, "rerun", diagnosis="below bar"))

    assert outcome.executed is True
    assert bag["redispatched"] == [4242], "the re-run must go through _redispatch_task"
    assert bag["escalated"] == []


def test_rerun_appends_lineage(monkeypatch):
    """AC3: the watch follows the re-run -- lineage grows by one board_task entry."""
    watch, task = _watch(lineage=[{"target_type": "board_task", "target_id": "4242", "reason": "created"}]), _task()
    bag, db = _wire(monkeypatch, budget_rail=_BudgetRail(2), task_result=task)

    asyncio.run(wa.run_board_task_action(db, watch, "rerun", diagnosis="below bar"))

    assert bag["followed"] == ["4242"]
    assert len(watch.lineage) == 2
    assert watch.lineage[-1]["target_type"] == "board_task"


def test_budget_exhausted_escalates_without_rerun(monkeypatch):
    watch, task = _watch(actions_taken=1, action_budget=1), _task()
    bag, db = _wire(monkeypatch, budget_rail=_BudgetRail(1), task_result=task)

    outcome = asyncio.run(wa.run_board_task_action(db, watch, "rerun", diagnosis="still bad"))

    assert outcome.escalated is True
    assert bag["redispatched"] == [], "budget exhausted must NOT re-dispatch"
    assert len(bag["escalated"]) == 1 and "budget exhausted" in bag["escalated"][0].lower()


def test_budget_one_yields_exactly_one_rerun_then_escalation(monkeypatch):
    """AC1: action_budget=1 -> exactly one corrective attempt, then escalation."""
    watch, task = _watch(action_budget=1), _task()
    bag, db = _wire(monkeypatch, budget_rail=_BudgetRail(1), task_result=task)

    first = asyncio.run(wa.run_board_task_action(db, watch, "rerun", diagnosis="below bar"))
    second = asyncio.run(wa.run_board_task_action(db, watch, "rerun", diagnosis="still below bar"))

    assert first.executed is True
    assert second.escalated is True
    assert bag["redispatched"] == [4242], "exactly one re-run attempt"
    assert len(bag["escalated"]) == 1, "then escalation once the budget is spent"


def test_escalate_action_is_the_escape_hatch(monkeypatch):
    watch, task = _watch(), _task()
    bag, db = _wire(monkeypatch, budget_rail=_BudgetRail(2), task_result=task)

    outcome = asyncio.run(wa.run_board_task_action(db, watch, "escalate", diagnosis="give up"))

    assert outcome.escalated is True
    assert bag["redispatched"] == []
    assert bag["escalated"] == ["give up"]


def test_missing_task_escalates(monkeypatch):
    watch = _watch()
    bag, db = _wire(monkeypatch, budget_rail=_BudgetRail(2), task_result=None)

    outcome = asyncio.run(wa.run_board_task_action(db, watch, "rerun"))

    assert outcome.escalated is True
    assert bag["redispatched"] == []


def test_unassigned_task_escalates(monkeypatch):
    watch, task = _watch(), _task(assigned_agent_id=None)
    bag, db = _wire(monkeypatch, budget_rail=_BudgetRail(2), task_result=task)

    outcome = asyncio.run(wa.run_board_task_action(db, watch, "rerun"))

    assert outcome.escalated is True
    assert bag["redispatched"] == []


# ---------------------------------------------------------------------------
# escalation narration -- the EXISTING escalate_watch_now → seam
# ---------------------------------------------------------------------------


def test_escalation_narrates_watch_escalation_through_seam(monkeypatch):
    """AC2: escalation routes through escalation_service.escalate_watch and
    narrates as watch_escalation to deliver_background_message (the untouched
    PRD-205 seam). Exercises the REAL escalate_watch_now → dispatch chain."""
    import services.escalation_service as es
    import core.services.notification_dispatcher as nd
    import services.chat_messenger as cm

    escalate_calls = []
    monkeypatch.setattr(es, "escalate_watch",
                        lambda db, ws, watch, reason: escalate_calls.append(reason))
    monkeypatch.setattr(ws_mod.WatchService, "ingest",
                        staticmethod(lambda db, watch, **kw: None))
    monkeypatch.setattr(ws_mod.WatchService, "transition",
                        staticmethod(lambda db, watch, status, **kw: None))

    class _FakeDispatcher:
        def __init__(self, db, ws):
            pass

        async def dispatch(self, **kw):
            return None

    monkeypatch.setattr(nd, "NotificationDispatcher", _FakeDispatcher)

    delivered = []
    monkeypatch.setattr(cm, "deliver_background_message",
                        lambda db, **kw: delivered.append(kw))

    watch = _watch(actions_taken=1)
    asyncio.run(wa.escalate_watch_now(_FakeDB(), watch, reason="Action budget exhausted (1/1)"))

    assert escalate_calls == ["Action budget exhausted (1/1)"], "goes through escalation_service"
    assert len(delivered) == 1, "narrated once into the originating thread"
    assert delivered[0]["source"]["event"] == "watch_escalation"


# ---------------------------------------------------------------------------
# decider routing -- board terminals take the board corrective path
# ---------------------------------------------------------------------------


def _verdict(score):
    return RunVerdict(score=score, dimension_scores={}, reasoning="weak output",
                      output_hash="x" * 64)


def test_decide_terminal_routes_board_below_bar_to_board_action(monkeypatch):
    """A below-bar board terminal dispatches to _act_board_task -- not the
    mission/playbook _diagnose_and_act path."""
    decider = WatchDecider()

    async def _score(db, watch):
        return _verdict(0.3)

    routed = {}

    async def _act(db, watch, *, terminal_state, verdict, completed):
        routed["called"] = (terminal_state, completed)
        return DECIDED_ACTED

    monkeypatch.setattr(decider, "_score", _score)
    monkeypatch.setattr(decider, "_act_board_task", _act)

    watch = _watch(policy="score_and_improve")
    result = asyncio.run(decider.decide_terminal(_FakeDB(), watch, "completed", None))

    assert result == DECIDED_ACTED
    assert routed["called"] == ("completed", True)


def test_act_board_task_reruns_when_policy_acts(monkeypatch):
    decider = WatchDecider()

    async def _run(db, watch, action, *, diagnosis=None):
        return wa.WatchActionOutcome(action="rerun", executed=True, detail="re-dispatched")

    monkeypatch.setattr(wa, "run_board_task_action", _run)
    notified = []

    async def _notify(db, watch, action, cause):
        notified.append(action)

    monkeypatch.setattr(decider, "_notify_action", _notify)

    watch = _watch(policy="score_and_improve")
    result = asyncio.run(decider._act_board_task(
        _FakeDB(), watch, terminal_state="completed", verdict=_verdict(0.3), completed=True))

    assert result == DECIDED_ACTED
    assert notified == ["rerun"], "an executed re-run narrates watch_action"


def test_act_board_task_escalates_when_action_escalates(monkeypatch):
    decider = WatchDecider()

    async def _run(db, watch, action, *, diagnosis=None):
        return wa.WatchActionOutcome(action="rerun", escalated=True, detail="budget")

    monkeypatch.setattr(wa, "run_board_task_action", _run)

    watch = _watch(policy="score_and_improve")
    result = asyncio.run(decider._act_board_task(
        _FakeDB(), watch, terminal_state="failed", verdict=_verdict(0.0), completed=False))

    assert result == DECIDED_ESCALATED


def test_act_board_task_run_and_report_closes_without_rerun(monkeypatch):
    """run_and_report doesn't act: it reports the failure verdict and closes,
    never re-running (US-005 attaches run_and_report by default)."""
    decider = WatchDecider()

    ran = []

    async def _run(db, watch, action, *, diagnosis=None):
        ran.append(action)
        return wa.WatchActionOutcome(action="rerun", executed=True)

    monkeypatch.setattr(wa, "run_board_task_action", _run)

    closed = {}

    async def _close(db, watch, *, passed, terminal_state, explanation):
        closed.update(passed=passed, terminal_state=terminal_state)
        return "decided:failed"

    monkeypatch.setattr(decider, "_close", _close)

    watch = _watch(policy="run_and_report")
    result = asyncio.run(decider._act_board_task(
        _FakeDB(), watch, terminal_state="completed", verdict=_verdict(0.3), completed=True))

    assert result == "decided:failed"
    assert closed == {"passed": False, "terminal_state": "completed"}
    assert ran == [], "run_and_report must not re-run the ticket"


# ---------------------------------------------------------------------------
# call-site grep guards (AC1 shared fn, AC2 no new escalation path)
# ---------------------------------------------------------------------------


def test_board_rerun_calls_the_shared_run_now_function_not_the_route():
    src = inspect.getsource(wa.run_board_task_action)
    assert "from api.board_tasks import _redispatch_task" in src
    assert "_redispatch_task(db, task)" in src
    # The HTTP route itself is never invoked from the service.
    assert "run_task_now" not in src


def test_board_escalation_uses_existing_escalation_service_only():
    """No new escalation construct: the board action escalates ONLY via
    escalate_watch_now, which routes through escalation_service.escalate_watch."""
    board_src = inspect.getsource(wa.run_board_task_action)
    assert "escalate_watch_now(" in board_src
    esc_src = inspect.getsource(wa.escalate_watch_now)
    assert "from services.escalation_service import escalate_watch" in esc_src
    assert "escalate_watch(db, watch.workspace_id, watch, reason)" in esc_src

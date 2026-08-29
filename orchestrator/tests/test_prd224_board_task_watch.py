"""PRD-224 US-002 -- watches learn target_type ``board_task``.

Pure, LLM-free, DB-free coverage of the board-task supervision wiring:

- the ``platform_create_watch`` schema + handler accept ``board_task`` and
  refuse a non-integer / unknown / cross-workspace task id with a clear error;
- the ticker's terminal detection maps board statuses to the run-shaped state
  the decider consumes (done/failed terminal; review+feedback -> completed;
  everything else running so the deadline sweep owns 'blocked past deadline');
- the SAME run-level scorer (RunVerdictService) composes the board task's
  recorded output -- a new collect branch, never a second scorer;
- a duplicate watch on the same task follows the existing
  ``WatchAlreadyExistsError`` friendliness.

The decider's terminal ROUTING (completed -> score -> pass/fail, failed ->
fail, budget=0 -> escalate) is target-type-agnostic and already locked by
test_prd204_watch_decider.py -- board tasks reuse it unchanged, so what is
board-specific (the mapping + the output fetch) is what this file proves.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest

from modules.coordination.run_verdict import RunOutputBundle, RunVerdictService
from modules.tools.discovery.action_registry import ActionRegistry
from modules.tools.discovery.actions_watches import register_watch_actions
from modules.tools.discovery.handlers_watches import _resolve_target, create_watch
from services.watch_ticker import WatchTicker

_WS = uuid.UUID("00000000-0000-0000-0000-000000000abc")


# ---------------------------------------------------------------------------
# Fakes -- a query chain that ignores its column/filter args and returns a
# seeded row (the ORM class-attribute expressions are genuine; the fake just
# ignores them, same trick as test_board_task_handlers.py).
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


def _task(**over):
    base = dict(
        id=4242, title="Chase Q3 invoices", description="Chase the overdue Q3 invoices.",
        status="done", result=None, review_feedback=None, error_message=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _watch(**over):
    base = dict(target_type="board_task", target_id="4242")
    base.update(over)
    return SimpleNamespace(**base)


def _call(handler, db, params):
    return asyncio.run(handler(db, _WS, params))


# ---------------------------------------------------------------------------
# Schema: the tool advertises board_task (enum-in-code)
# ---------------------------------------------------------------------------


def test_create_watch_schema_offers_board_task():
    registry = ActionRegistry()
    register_watch_actions(registry)
    enum = registry.get("platform_create_watch").parameters["properties"]["target_type"]["enum"]
    assert "board_task" in enum
    # The three originals are untouched (no enum churn).
    assert {"mission", "playbook_execution", "scheduled_playbook"} <= set(enum)


def test_target_type_enum_has_board_task():
    from core.models.watch_enums import WatchTargetType

    assert WatchTargetType.BOARD_TASK.value == "board_task"


# ---------------------------------------------------------------------------
# Terminal detection -- ticker maps board status -> run-shaped state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status, review_feedback, expected, downstream",
    [
        # done/failed are the plain terminals the decider scores/fails.
        ("done", None, "completed", "scored"),
        ("failed", None, "failed", "failure verdict"),
        # review WITH recorded feedback = scorable output.
        ("review", "tighten the headline", "completed", "scored"),
        # bare review (awaiting a human) is NOT terminal yet.
        ("review", None, "running", "no verdict"),
        # blocked stays live -> the generic deadline sweep escalates it.
        ("blocked", None, "running", "deadline -> escalate"),
        ("in_progress", None, "running", "no verdict"),
        ("assigned", None, "running", "no verdict"),
        ("inbox", None, "running", "no verdict"),
    ],
)
def test_read_target_state_board_mapping(status, review_feedback, expected, downstream):
    ticker = WatchTicker()
    db = _FakeDB(result=(status, review_feedback))
    assert ticker._read_target_state(db, _watch()) == expected, downstream


def test_read_target_state_missing_task_is_none():
    """A deleted/archived task row -> None, so the ticker parks the watch."""
    assert WatchTicker()._read_target_state(_FakeDB(result=None), _watch()) is None


def test_read_target_state_non_integer_target_id_is_none():
    assert WatchTicker()._read_target_state(_FakeDB(), _watch(target_id="not-an-int")) is None


# ---------------------------------------------------------------------------
# Scoring -- the SAME RunVerdictService composes board output (no 2nd scorer)
# ---------------------------------------------------------------------------


def test_collect_board_done_composes_result_bundle():
    db = _FakeDB(result=_task(status="done", result="Collected 4 of 5 invoices; £12k recovered."))
    bundle = RunVerdictService.collect_run_output(db, _watch())
    assert isinstance(bundle, RunOutputBundle)
    assert bundle.kind == "board_task"
    assert bundle.terminal_state == "done"
    assert "12k recovered" in bundle.text
    assert bundle.mechanics_reliability == 1.0
    assert bundle.empty is False


def test_collect_board_failed_uses_error_and_zero_mechanics():
    db = _FakeDB(result=_task(status="failed", result=None, error_message="API auth expired"))
    bundle = RunVerdictService.collect_run_output(db, _watch())
    assert "API auth expired" in bundle.text
    assert bundle.mechanics_reliability == 0.0
    assert bundle.terminal_state == "failed"


def test_collect_board_review_includes_feedback_and_neutral_mechanics():
    db = _FakeDB(result=_task(
        status="review", result="Draft summary attached.", review_feedback="add the totals row",
    ))
    bundle = RunVerdictService.collect_run_output(db, _watch())
    assert "Draft summary attached." in bundle.text
    assert "add the totals row" in bundle.text
    assert bundle.mechanics_reliability == 0.5


def test_collect_board_no_output_is_empty():
    """A done task with no recorded result scores the deterministic floor, not
    an LLM call -- bundle.empty drives score_run's no-output branch."""
    db = _FakeDB(result=_task(status="done", result=None))
    bundle = RunVerdictService.collect_run_output(db, _watch())
    assert bundle.empty is True


def test_collect_board_missing_task_is_none():
    assert RunVerdictService.collect_run_output(_FakeDB(result=None), _watch()) is None


def test_scoring_reuses_run_verdict_service_no_second_scorer():
    """The decider scores board tasks through the SAME seam missions/playbooks
    use: WatchDecider._score -> self._verdicts().score_run, and _verdicts() is a
    RunVerdictService. No board-specific judge/scorer class is introduced."""
    import inspect

    from services.watch_decider import WatchDecider

    score_src = inspect.getsource(WatchDecider._score)
    assert "self._verdicts()" in score_src and ".score_run(" in score_src
    verdicts_src = inspect.getsource(WatchDecider._verdicts)
    assert "RunVerdictService" in verdicts_src
    # The dispatch adds a branch to the existing collector, not a new scorer.
    collect_src = inspect.getsource(RunVerdictService.collect_run_output)
    assert "_collect_board_task" in collect_src


# ---------------------------------------------------------------------------
# create_watch handler -- accept board_task, refuse phantom targets
# ---------------------------------------------------------------------------


def test_resolve_target_board_task_returns_title_and_criteria():
    resolved = _resolve_target(_FakeDB(result=_task()), _WS, "board_task", "4242")
    assert resolved is not None
    assert resolved["title"] == "Watch: Chase Q3 invoices"
    assert resolved["criteria"] == "Chase the overdue Q3 invoices."


def test_resolve_target_board_task_unknown_is_none():
    assert _resolve_target(_FakeDB(result=None), _WS, "board_task", "9999") is None


def test_resolve_target_board_task_non_integer_is_none():
    assert _resolve_target(_FakeDB(result=_task()), _WS, "board_task", "abc") is None


def test_create_watch_rejects_unknown_board_task():
    """An unknown/cross-workspace task id is refused at creation with a clear
    error -- never a watch on a phantom target."""
    result = _call(create_watch, _FakeDB(result=None),
                   {"target_type": "board_task", "target_id": "9999"})
    assert result["success"] is False
    assert "board_task" in result["error"] and "9999" in result["error"]


def test_create_watch_rejects_non_integer_board_task_id():
    result = _call(create_watch, _FakeDB(result=_task()),
                   {"target_type": "board_task", "target_id": "not-an-int"})
    assert result["success"] is False
    assert "board_task" in result["error"]


def _stub_watch(**over):
    base = dict(
        id=uuid.uuid4(), title="Watch: Chase Q3 invoices", watch_type="board_task",
        target_type="board_task", target_id="4242", status="watching",
        policy="run_and_report", success_criteria="Chase the overdue Q3 invoices.",
        quality_threshold=0.8, final_score=None, final_verdict=None, actions_taken=0,
        action_budget=2, last_checked_at=None, next_check_at=None, deadline_at=None,
        created_at=None, closed_at=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_create_watch_accepts_board_task(monkeypatch):
    """Full handler acceptance: a valid integer task id creates a board_task
    watch and confirms it (WatchService.create_watch stubbed -- no DB)."""
    import services.watch_service as ws_mod

    captured = {}

    def _fake_create(db, **kwargs):
        captured.update(kwargs)
        return _stub_watch(target_id=kwargs["target_id"], target_type=kwargs["target_type"])

    monkeypatch.setattr(ws_mod.WatchService, "create_watch", staticmethod(_fake_create))

    result = _call(create_watch, _FakeDB(result=_task()),
                   {"target_type": "board_task", "target_id": "4242"})

    assert result["success"] is True
    assert result["existing"] is False
    assert result["watch"]["target_type"] == "board_task"
    assert result["watch"]["target_id"] == "4242"
    assert "board_task" in result["message"]
    # watch_type mirrors target_type for the board lane (the ticker dispatches
    # non-scheduled targets through _check_run_target regardless).
    assert captured["watch_type"] == "board_task"
    assert captured["target_type"] == "board_task"


def test_duplicate_board_task_watch_is_friendly(monkeypatch):
    """A duplicate follows the existing uniqueness semantics: the handler catches
    WatchAlreadyExistsError and returns the live watch with existing=True (the
    partial unique index is the race backstop) -- identical to mission/playbook."""
    import services.watch_service as ws_mod

    def _raise(db, **kwargs):
        raise ws_mod.WatchAlreadyExistsError(_WS, "board_task", "4242")

    monkeypatch.setattr(ws_mod.WatchService, "create_watch", staticmethod(_raise))
    monkeypatch.setattr(
        ws_mod.WatchService, "find_live_watch",
        staticmethod(lambda db, **kwargs: _stub_watch()),
    )

    result = _call(create_watch, _FakeDB(result=_task()),
                   {"target_type": "board_task", "target_id": "4242"})

    assert result["success"] is True
    assert result["existing"] is True
    assert "already being watched" in result["message"]

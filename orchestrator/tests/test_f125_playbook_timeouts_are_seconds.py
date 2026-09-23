"""F125 — a playbook's timeouts are seconds, never guessed from their size.

Run 4: an owner asked for a 4-hour limit, and Auto stored recipe 93 with
{total_timeout: 14400, per_step_timeout: 3600}. The executor took anything of
10,000 or more for milliseconds, so the budget became 14.4 s, then the 900 s
floor. Execution 165 stopped "at step 2" after 1002 s with step 1 finished.
Another playbook's `timeout_minutes: 240` was read by nothing, so it ran on the
1800 s default. The quality score guessed the unit the same way. Now every value
is seconds, `timeout_minutes` is the total when `total_timeout` is absent, and
a legacy millisecond row is normalised once by the migration
f125_playbook_timeouts_seconds.
"""
from __future__ import annotations

import importlib.util
import logging
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.helpers_playbook_run import done, run_playbook

MIGRATION = Path(__file__).resolve().parents[1] / "alembic" / "versions" / "f125_playbook_timeouts_seconds.py"


def _timeouts_line(caplog):
    return next(r.getMessage() for r in caplog.records if "[recipe_direct] Timeouts:" in r.getMessage())


# ── the executor: the budget is what the owner configured ───────────────────

def test_a_four_hour_budget_is_four_hours(monkeypatch, caplog):
    """Execution 165's config and pace: step 1 took 1002 s. The run finishes."""
    with caplog.at_level(logging.INFO, logger="api.recipe_executor"):
        execution, card = run_playbook(
            monkeypatch, outcomes=[done("the list"), done("the sheet")], step_seconds=1002,
            exec_config={"total_timeout": 14400, "per_step_timeout": 3600},
        )
    assert execution.status == "completed" and card.status == "done"
    assert _timeouts_line(caplog).endswith(
        "Timeouts: step=3600s, total=14400s (configured: step=3600s, total=14400s)")


def test_a_step_timeout_of_12000_seconds_is_12000_seconds(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger="api.recipe_executor"):
        run_playbook(monkeypatch, outcomes=[done("a"), done("b")], step_seconds=5,
                     exec_config={"per_step_timeout": 12000, "total_timeout": 36000})
    assert "Timeouts: step=12000s, total=36000s" in _timeouts_line(caplog)


def test_timeout_minutes_is_the_total_when_total_timeout_is_absent(monkeypatch, caplog):
    """240 minutes is 14,400 s. The 1800 s default would stop this run before step 2."""
    with caplog.at_level(logging.INFO, logger="api.recipe_executor"):
        execution, card = run_playbook(monkeypatch, outcomes=[done("a"), done("b")], step_seconds=1900,
                                       exec_config={"timeout_minutes": 240})
    assert execution.status == "completed" and card.status == "done"
    assert "total=14400s" in _timeouts_line(caplog)


def test_an_explicit_total_timeout_wins_over_timeout_minutes(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger="api.recipe_executor"):
        run_playbook(monkeypatch, outcomes=[done("a"), done("b")], step_seconds=5,
                     exec_config={"total_timeout": 3600, "timeout_minutes": 240})
    assert "total=3600s" in _timeouts_line(caplog)


# ── the quality score reads the same seconds ────────────────────────────────

def _efficiency(execution_config):
    from core.services.playbook_quality_service import PlaybookQualityService

    started = datetime(2026, 9, 23, 15, 40)
    execution = SimpleNamespace(
        started_at=started, completed_at=started + timedelta(seconds=1002),
        step_results=[{"duration_ms": 1_001_661, "retries": 0}],
    )
    return PlaybookQualityService(db=MagicMock())._assess_efficiency(execution, SimpleNamespace(execution_config=execution_config))


def test_a_four_hour_budget_scores_a_17_minute_run_like_a_two_hour_one():
    """Both budgets hold the run with room to spare. Read as 14.4 s, it scored as blown."""
    four_hours = _efficiency({"total_timeout": 14400, "per_step_timeout": 3600})
    assert four_hours == _efficiency({"total_timeout": 7200, "per_step_timeout": 3600})


# ── the migration: a legacy millisecond row is normalised once ──────────────

def _migration():
    spec = importlib.util.spec_from_file_location("f125_playbook_timeouts_seconds", MIGRATION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FLOORS = {"timeout_per_step": 300, "per_step_timeout": 300, "total_timeout": 900}


@pytest.mark.parametrize("before, after", [
    ({"timeout_per_step": 120000, "total_timeout": 600000}, {"timeout_per_step": 300, "total_timeout": 900}),
    ({"total_timeout": 3600000}, {"total_timeout": 3600}),
    ({"per_step_timeout": 100000}, {"per_step_timeout": 300}),      # the boundary converts, onto the floor
    ({"total_timeout": 99999}, {"total_timeout": 99999}),           # below it stays seconds
    ({"total_timeout": 14400, "timeout_minutes": 240}, {"total_timeout": 14400, "timeout_minutes": 240}),
    ({"total_timeout": "600000", "per_step_timeout": True}, {"total_timeout": "600000", "per_step_timeout": True}),
])
def test_the_migration_normalises_only_legacy_milliseconds(before, after):
    normalised, changes = _migration().normalise(before, FLOORS)
    assert normalised == after
    assert len(changes) == sum(1 for k in before if before[k] != after[k])


def test_the_migration_is_idempotent_and_names_every_change():
    mod = _migration()
    once, changes = mod.normalise({"timeout_per_step": 120000, "total_timeout": 1800000}, FLOORS)
    assert changes == ["timeout_per_step 120000 -> 300", "total_timeout 1800000 -> 1800"]
    assert mod.normalise(once, FLOORS) == (once, [])


def test_the_migration_chains_onto_the_night_fix_head():
    mod = _migration()
    assert (mod.revision, mod.down_revision) == ("f125_playbook_timeouts_seconds", "llm_usage_agent_name")

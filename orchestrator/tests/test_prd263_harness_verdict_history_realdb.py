"""PRD-263 US-014b — verdicts use a clean reference and the week-by-week history.

A first flag is WATCH; REGRESSION needs flags in two distinct, consecutive ISO weeks,
or one flag below HARNESS_STRONG_P. The history is read from the workspace's
completed ``harness_runs.verdicts``, so a rerun of a week already judged (Auto can call
``platform_harness_trigger`` from chat) keeps that week's verdict and is never a second
flag. The reference is the weeks of HARNESS_REF_DAYS before the judged one, less weeks
already flagged for that agent and metric (a bad week is never absorbed into it) and
less the weeks from a change HARNESS applied to that agent (``harness_task_ledger``).

Each case runs the real pipeline on the transactional ``db_session``
(``helpers_harness_diagnose.judge``), one harness_runs row per judged week. The
baseline-file rollback check (``_detect_auto_applied_regressions``) is gone; US-056
rebuilds rollback on the ledger.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests import helpers_harness_diagnose as hdx
from tests import helpers_harness_runs as hr

NEXT_WEEK = "2026-W43"


@pytest.fixture
def ws(db_session, seed_workspace, monkeypatch, tmp_path):
    hr.fake_platform(monkeypatch, tmp_path)
    a = seed_workspace(name="harness-history")
    hdx.auto(db_session, a)
    return a


def _steady_then(db, ws, agent_id, *, judged_weeks):
    """16 of 20 a week in W38-W41, then 12 of 20 in each of ``judged_weeks`` (offsets)."""
    for week in (-4, -3, -2, -1):
        hdx.tickets(db, ws, agent_id, week, 16, 4)
    for week in judged_weeks:
        hdx.tickets(db, ws, agent_id, week, 12, 8)


def test_a_drop_is_watch_in_week_one_and_regression_in_week_two(db_session, ws):
    barista = hdx.agent(db_session, ws, "Barista")
    _steady_then(db_session, ws, barista, judged_weeks=(0, 1))

    first = hdx.metric(hdx.judge(db_session, ws, hdx.WEEK), barista)
    second_run = hdx.judge(db_session, ws, NEXT_WEEK)
    second = hdx.metric(second_run, barista)

    assert (first["verdict"], first["reason"]) == ("WATCH", None)
    assert (second["verdict"], second["reason"]) == ("REGRESSION", "second_week")
    assert 0.001 < second["p_value"] < 0.05   # not strong on its own: the previous week makes it
    assert "Task success fell from 80% (48 of 60) to 60% (12 of 20)" in second_run["report_md"]
    assert "the second week in a row" in second_run["report_md"]


def test_a_rerun_of_the_same_week_stays_watch(db_session, ws):
    barista = hdx.agent(db_session, ws, "Barista")
    _steady_then(db_session, ws, barista, judged_weeks=(0,))

    first = hdx.metric(hdx.judge(db_session, ws, hdx.WEEK), barista)
    rerun_row = hdx.judge(db_session, ws, hdx.WEEK)
    rerun = hdx.metric(rerun_row, barista)

    assert first["verdict"] == "WATCH"
    assert (rerun["verdict"], rerun["reason"]) == ("WATCH", "kept")
    assert rerun_row["verdicts"]["convergence"] == "watching"


def test_a_flagged_week_is_left_out_of_the_next_reference(db_session, ws):
    barista = hdx.agent(db_session, ws, "Barista")
    _steady_then(db_session, ws, barista, judged_weeks=(0, 1))

    hdx.judge(db_session, ws, hdx.WEEK)
    second = hdx.metric(hdx.judge(db_session, ws, NEXT_WEEK), barista)

    assert second["ref_weeks"] == ["2026-W39", "2026-W40", "2026-W41"]   # W42, flagged, is out
    assert (second["x_ref"], second["n_ref"]) == (48, 60)


def test_one_flag_below_the_strong_p_is_regression_at_once(db_session, ws):
    barista = hdx.agent(db_session, ws, "Barista")
    hdx.tickets(db_session, ws, barista, -1, 30, 2)
    hdx.tickets(db_session, ws, barista, 0, 10, 22)

    task = hdx.metric(hdx.judge(db_session, ws), barista)

    assert (task["verdict"], task["reason"]) == ("REGRESSION", "strong_p")
    assert task["p_value"] < 0.001


def test_weeks_after_a_ledger_applied_change_are_left_out_of_that_agents_reference(db_session, ws):
    changed = hdx.agent(db_session, ws, "Changed")
    untouched = hdx.agent(db_session, ws, "Untouched")
    for agent_id in (changed, untouched):
        _steady_then(db_session, ws, agent_id, judged_weeks=(0,))
    hdx.applied_change(db_session, ws, changed, hdx.at(-2))   # applied in W40

    run = hdx.judge(db_session, ws)

    assert hdx.metric(run, changed)["ref_weeks"] == ["2026-W38", "2026-W39"]
    assert hdx.metric(run, untouched)["ref_weeks"] == ["2026-W38", "2026-W39", "2026-W40", "2026-W41"]


def test_a_change_older_than_the_reference_leaves_it_whole(db_session, ws):
    """Every reference week is then the agent as it now stands."""
    changed = hdx.agent(db_session, ws, "Changed long ago")
    _steady_then(db_session, ws, changed, judged_weeks=(0,))
    hdx.applied_change(db_session, ws, changed, hdx.at(-6))   # W36, before the reference

    run = hdx.judge(db_session, ws)

    assert hdx.metric(run, changed)["ref_weeks"] == ["2026-W38", "2026-W39", "2026-W40", "2026-W41"]


def test_the_baseline_rollback_check_is_gone():
    src = (Path(__file__).resolve().parents[1] / "services" / "harness_service.py").read_text()
    for name in ("_detect_auto_applied_regressions", "auto_applied_regression", "rollback_spec",
                 "_read_baseline", "baseline_latest"):
        assert name not in src, name

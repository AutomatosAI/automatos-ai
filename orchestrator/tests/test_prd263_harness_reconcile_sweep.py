"""PRD-263 US-053 — every proposal ends in a recorded state, and lapsed cards close within a day.

services.harness_grants.reconcile moves each open harness_task_ledger row by its approval
grant, at the start of a HARNESS tick (ORG-2) and in the daily sweep, never on a read:
grant DENIED → ``denied``; REVOKED with no ``executed_result`` → ``withdrawn``; PENDING
past ``expires_at`` → the grant is closed (``expire_pending_grants``) and the row
``expired``; PENDING over a setting that no longer reads its prior value → closed and
``drifted``; GRANTED → ``failed`` once the click's resume recorded a result and the row is
still pending ("evidence card was blocked; nothing changed" when the Approve re-queued the
card, otherwise "no outcome"), or once ``granted_at`` is older than
HARNESS_RECONCILE_GRACE_MINUTES; a GRANTED grant inside the grace with no result stays
``pending`` (its click may still be running). An ``applying`` row older than the grace →
``failed`` ("interrupted"). Final states are never left, and a revoke after the change
applied leaves the row ``applied``.

The status read (``ledger_counts``) derives each pending row's effective state from its
grant without writing. The daily sweep runs at HARNESS_GRANT_SWEEP_CRON (default
``0 9 * * *``) in UTC on the shared scheduler. Both settings are config.py settings listed
in the config surface.

Rows go into the real database (``db_session``, rolled back); the tick case stubs its
session and helpers.
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from services import harness_grants as hg
from tests import helpers_harness_grants as hgx

ORCH = Path(__file__).resolve().parents[1]
RX = "rx-1-001"
LONG_AGO = timedelta(hours=2)
SUNDAY_NOON = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
FINAL = ("failed", "expired", "drifted", "denied", "withdrawn", "raise_failed", "rolled_back")


@pytest.fixture(autouse=True)
def _grace(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "HARNESS_RECONCILE_GRACE_MINUTES", 60)


@pytest.fixture
def crew(db_session, seed_workspace):
    return hgx.team(db_session, seed_workspace(name="harness-reconcile"))


def _reconciled(db, crew, ledger_id):
    """Reconcile the crew's workspace; the row afterwards and what moved."""
    moved = hg.reconcile(db, crew.ws)
    return hgx.row(db, ledger_id), moved


# ── each transition ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("status,state", [("denied", "denied"), ("revoked", "withdrawn"), ("expired", "expired")])
def test_a_decided_grant_ends_its_row(db_session, crew, status, state):
    ledger_id, grant_id = hgx.proposal(db_session, crew, status=status)

    row, moved = _reconciled(db_session, crew, ledger_id)

    assert (row["state"], moved) == (state, {state: 1})
    assert hgx.grant_status(db_session, grant_id) == status


def test_a_lapsed_pending_grant_is_closed_and_its_row_expires(db_session, crew):
    ledger_id, grant_id = hgx.proposal(db_session, crew, status="pending", expires_in=hgx.LAPSED)

    row, moved = _reconciled(db_session, crew, ledger_id)

    assert (row["state"], moved) == ("expired", {"expired": 1})
    assert hgx.grant_status(db_session, grant_id) == "expired"


@pytest.mark.parametrize("executed,reason", [
    ({"resumed_via": "board_task_requeue", "board_task_id": 7}, "evidence card was blocked; nothing changed"),
    ({"success": False, "error": "Action failed"}, "no outcome"),
], ids=["card-blocked", "no-outcome"])
def test_a_click_that_recorded_no_outcome_ends_failed(db_session, crew, executed, reason):
    details = {"action": hgx.APPLY, "params": {"rx_id": RX}, "executed_result": executed}
    ledger_id, _ = hgx.proposal(db_session, crew, details=details)

    row, moved = _reconciled(db_session, crew, ledger_id)

    assert (row["state"], row["entry"], moved) == ("failed", {"reason": reason}, {"failed": 1})


def test_a_grant_flipped_to_granted_with_no_result_stays_pending_inside_the_grace(db_session, crew):
    ledger_id, grant_id = hgx.proposal(db_session, crew, granted_ago=timedelta(minutes=1))

    row, moved = _reconciled(db_session, crew, ledger_id)

    assert (row["state"], moved) == ("pending", {})
    assert hgx.grant_status(db_session, grant_id) == "granted"


def test_a_grant_granted_longer_than_the_grace_ago_ends_failed(db_session, crew):
    ledger_id, _ = hgx.proposal(db_session, crew, granted_ago=LONG_AGO)

    row, moved = _reconciled(db_session, crew, ledger_id)

    assert (row["state"], row["entry"], moved) == ("failed", {"reason": "no outcome"}, {"failed": 1})


def test_a_stale_applying_row_closes_as_interrupted_and_a_fresh_one_stays(db_session, crew):
    stale = hgx.ledger(db_session, crew.ws, RX, crew.scout, hgx.grant(db_session, crew.ws, RX),
                       state="applying", updated_at=hgx.now() - LONG_AGO)
    fresh = hgx.ledger(db_session, crew.ws, "rx-1-002", crew.scout, hgx.grant(db_session, crew.ws, "rx-1-002"),
                       state="applying", updated_at=hgx.now())

    moved = hg.reconcile(db_session, crew.ws)

    assert hgx.row(db_session, stale)["state"] == "failed"
    assert hgx.row(db_session, stale)["entry"] == {"reason": "interrupted"}
    assert hgx.row(db_session, fresh)["state"] == "applying"
    assert moved == {"failed": 1}


def test_a_row_without_its_grant_ends_failed(db_session, crew):
    ledger_id = hgx.ledger(db_session, crew.ws, RX, crew.scout, None)

    row, moved = _reconciled(db_session, crew, ledger_id)

    assert (row["state"], row["entry"], moved) == ("failed", {"reason": "no approval grant"}, {"failed": 1})


def test_a_revoke_after_apply_stays_applied(db_session, crew):
    details = {"action": hgx.APPLY, "params": {"rx_id": RX}, "executed_result": {"success": True}}
    grant_id = hgx.grant(db_session, crew.ws, RX, "revoked", details=details)
    ledger_id = hgx.ledger(db_session, crew.ws, RX, crew.scout, grant_id, state="applied")

    row, moved = _reconciled(db_session, crew, ledger_id)

    assert (row["state"], moved) == ("applied", {})


def test_the_sweep_closes_a_waiting_card_whose_setting_changed(db_session, crew):
    ledger_id, grant_id = hgx.proposal(db_session, crew, status="pending")
    hgx.set_interval(db_session, crew.scout, 30)

    moved = hg.sweep(db_session)

    assert hgx.row(db_session, ledger_id)["state"] == "drifted" and moved["drifted"] >= 1
    assert hgx.grant_status(db_session, grant_id) == "expired"
    assert hgx.interval(db_session, crew.scout) == 30


def test_a_waiting_card_over_an_unchanged_setting_stays(db_session, crew):
    ledger_id, grant_id = hgx.proposal(db_session, crew, status="pending")

    row, moved = _reconciled(db_session, crew, ledger_id)

    assert (row["state"], moved) == ("pending", {})
    assert hgx.grant_status(db_session, grant_id) == "pending"


@pytest.mark.parametrize("state", FINAL + ("applied",))
def test_a_final_state_is_never_left(db_session, crew, state):
    """A lapsed pending grant would expire an open row; a closed row keeps its state."""
    from core.models.harness import HarnessTaskLedger

    grant_id = hgx.grant(db_session, crew.ws, RX, "pending", expires_in=hgx.LAPSED)
    ledger_id = hgx.ledger(db_session, crew.ws, RX, crew.scout, grant_id, state=state)

    row, moved = _reconciled(db_session, crew, ledger_id)

    assert (row["state"], moved) == (state, {})
    assert hgx.grant_status(db_session, grant_id) == "pending"
    assert hg.move(db_session, db_session.get(HarnessTaskLedger, ledger_id), "expired") is False


def test_another_workspaces_rows_are_never_touched(db_session, crew, seed_workspace):
    elsewhere = hgx.team(db_session, seed_workspace(name="harness-reconcile-elsewhere"))
    mine, _ = hgx.proposal(db_session, crew, status="pending", expires_in=hgx.LAPSED)
    theirs, their_grant = hgx.proposal(db_session, elsewhere, status="pending", expires_in=hgx.LAPSED)

    assert hg.reconcile(db_session, crew.ws) == {"expired": 1}
    assert hgx.row(db_session, mine)["state"] == "expired"
    assert hgx.row(db_session, theirs)["state"] == "pending"
    assert hgx.grant_status(db_session, their_grant) == "pending"


def test_the_daily_sweep_reconciles_every_workspace_with_an_open_row(db_session, crew, seed_workspace):
    elsewhere = hgx.team(db_session, seed_workspace(name="harness-reconcile-sweep"))
    mine, _ = hgx.proposal(db_session, crew, status="denied")
    theirs, _ = hgx.proposal(db_session, elsewhere, status="pending", expires_in=hgx.LAPSED)

    moved = hg.sweep(db_session)

    assert moved["denied"] >= 1 and moved["expired"] >= 1
    assert (hgx.row(db_session, mine)["state"], hgx.row(db_session, theirs)["state"]) == ("denied", "expired")


# ── the status read writes nothing ───────────────────────────────────────────


def test_the_status_read_derives_effective_states_and_writes_nothing(db_session, crew):
    lapsed, lapsed_grant = hgx.proposal(db_session, crew, "rx-1-001", "pending", expires_in=hgx.LAPSED)
    declined, declined_grant = hgx.proposal(db_session, crew, "rx-1-002", "denied")
    late_click, late_grant = hgx.proposal(db_session, crew, "rx-1-003", expires_in=hgx.LAPSED)
    waiting, waiting_grant = hgx.proposal(db_session, crew, "rx-1-004", "pending")
    applied = hgx.ledger(db_session, crew.ws, "rx-1-005", crew.scout, None, state="applied")
    db_session.flush()

    counts = hg.ledger_counts(db_session, crew.ws)

    assert counts == {"expired": 2, "denied": 1, "pending": 1, "applied": 1}
    assert [hgx.row(db_session, i)["state"] for i in (lapsed, declined, late_click, waiting, applied)] == [
        "pending", "pending", "pending", "pending", "applied"]
    assert [hgx.grant_status(db_session, g) for g in (lapsed_grant, declined_grant, late_grant, waiting_grant)] == [
        "pending", "denied", "granted", "pending"]
    assert not (db_session.new or db_session.dirty or db_session.deleted)


def test_the_status_read_shows_a_withdrawn_grant_and_reads_one_workspace(db_session, crew, seed_workspace):
    elsewhere = hgx.team(db_session, seed_workspace(name="harness-reconcile-read"))
    hgx.proposal(db_session, crew, status="revoked")
    hgx.proposal(db_session, elsewhere, status="denied")

    assert hg.ledger_counts(db_session, crew.ws) == {"withdrawn": 1}


# ── the daily job and the tick ───────────────────────────────────────────────


def _shipped(name: str) -> str:
    (default,) = re.findall(rf'{name}: \w+ = \w*\(?os\.getenv\("{name}", "([^"]+)"\)', (ORCH / "config.py").read_text())
    return default


def test_the_settings_are_config_settings_listed_in_the_surface():
    assert _shipped("HARNESS_GRANT_SWEEP_CRON") == "0 9 * * *"
    assert _shipped("HARNESS_RECONCILE_GRACE_MINUTES") == "60"
    settings = json.loads((ORCH / "reports" / "config-surface.json").read_text())["settings"]
    assert {"HARNESS_GRANT_SWEEP_CRON", "HARNESS_RECONCILE_GRACE_MINUTES"} <= set(settings)
    assert settings == sorted(settings)


def test_the_daily_sweep_fires_every_day_at_nine_utc(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "HARNESS_GRANT_SWEEP_CRON", _shipped("HARNESS_GRANT_SWEEP_CRON"))
    trigger = hg.sweep_trigger()
    first = trigger.get_next_fire_time(None, SUNDAY_NOON)
    second = trigger.get_next_fire_time(first, first + timedelta(seconds=1))
    assert (first, second) == (datetime(2026, 10, 19, 9, 0, tzinfo=timezone.utc),
                               datetime(2026, 10, 20, 9, 0, tzinfo=timezone.utc))
    assert trigger.timezone.utcoffset(SUNDAY_NOON) == timedelta(0)


def test_the_service_registers_the_daily_sweep_on_its_scheduler():
    from services.harness_service import HarnessService

    class _Scheduler:
        def __init__(self):
            self.jobs = {}

        def add_job(self, func, **kwargs):
            self.jobs[kwargs["id"]] = (func, kwargs)

    scheduler = _Scheduler()
    asyncio.run(HarnessService().start(scheduler=scheduler))

    func, kwargs = scheduler.jobs[hg.SWEEP_JOB_ID]
    assert func is hg.run_daily_sweep and kwargs["max_instances"] == 1
    assert "harness_sweep" in scheduler.jobs


def test_a_tick_reconciles_before_its_weekly_skip(monkeypatch):
    """ORG-2: under the lock, the ledger is reconciled before the week-settled check, so a
    week already judged still closes lapsed cards."""
    from core.database import database as dbmod
    from services import harness_runs
    from services.harness_service import HarnessService

    calls = []
    workspace = uuid4()
    monkeypatch.setattr(dbmod, "SessionLocal", lambda: SimpleNamespace(close=lambda: calls.append("close")))
    monkeypatch.setattr(HarnessService, "workspace_enabled", lambda self, db, ws: True)
    monkeypatch.setattr(harness_runs, "last_complete_iso_week", lambda now=None: "2026-W42")
    monkeypatch.setattr(harness_runs, "close_interrupted", lambda db, ws: calls.append("close_interrupted"))
    monkeypatch.setattr(hg, "reconcile", lambda db, ws: calls.append(("reconcile", ws)))
    monkeypatch.setattr(harness_runs, "week_is_settled", lambda db, ws, window: calls.append("settled") or True)

    asyncio.run(HarnessService()._run_locked(workspace, sweep=True))

    assert calls == ["close_interrupted", ("reconcile", workspace), "settled", "close"]


def test_reconcile_reports_a_failure_and_rolls_back():
    class _Broken:
        def __init__(self):
            self.rolled_back = False

        def query(self, *a, **k):
            raise RuntimeError("database gone")

        def rollback(self):
            self.rolled_back = True

    db = _Broken()
    assert hg.reconcile(db, UUID(int=1)) is None and db.rolled_back

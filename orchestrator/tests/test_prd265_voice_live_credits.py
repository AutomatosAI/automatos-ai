"""PRD-265 D-6 (Gerard, 30 Oct 2026) — an Auto Live call is paid from credits, like chat.

``modules/voice/live_billing.py``:

- a call costs what Retell charges us (``call_cost.combined_cost``, cents) x the markup, by
  ``routing.credits_for``; an event with no cost is charged by its minutes at the estimate rate;
- the debit is keyed on the call id; a redelivered event charges nothing and reports 0;
- a workspace not paying in credits (local, an allowance plan, a free owner) is never charged;
- a call that never connected and carries no cost is free;
- ``call_events.apply_call_event`` charges on ``call_ended`` only, then sends the balance alerts;
- the Billing tab names the movement "Auto Live call".

DB-free: the funding and the ledger are stand-ins (the ledger itself is proven in
``test_prd265_credit_ledger_realdb.py``).
"""
from __future__ import annotations

import logging
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

from config import config
from modules.voice import call_events, live_billing
from services.credits.routing import (
    FUNDED_BY_ALLOWANCE,
    FUNDED_BY_CREDITS,
    FUNDED_BY_LOCAL,
    Funding,
    credits_for,
)

WS = "6a0f3c1e-2b4d-4e8f-9a1b-265d60000001"
BUCKETS = ("trial", "monthly", "topup")


def _row(**overrides):
    base = dict(call_id="call_1", workspace_id=WS, status="ended", duration_seconds=90,
                started_at=None, ended_at=None, disconnect_reason=None, vendor_latency=None)
    return NS(**{**base, **overrides})


def _cost(cents):
    return {"call_id": "call_1", "call_cost": {"combined_cost": cents}}


@pytest.fixture
def ledger(monkeypatch):
    """Credits-funded unless a test says otherwise; every record() call kept, the first new."""
    state = {"funding": Funding(FUNDED_BY_CREDITS, BUCKETS, 500), "entries": [], "seen": set()}
    monkeypatch.setattr(live_billing, "workspace_funding", lambda db, workspace_id: state["funding"])

    def record(db, workspace_id, entry, *, spend_buckets):
        state["entries"].append((workspace_id, entry, tuple(spend_buckets)))
        fresh = entry.source_ref not in state["seen"]
        state["seen"].add(entry.source_ref)
        return fresh

    monkeypatch.setattr(live_billing, "record", record)
    return state


@pytest.mark.parametrize("call, expected", [
    (_cost(31), 0.31),
    (_cost(0), 0.0),
    ({"call_cost": {}}, None),
    ({}, None),
    (_cost(True), None),
    (_cost(-5), None),
    (_cost(float("nan")), None),
    ({"call_cost": "31"}, None),
    (None, None),
])
def test_retells_cost_is_read_in_cents_and_anything_else_is_no_cost(call, expected):
    assert live_billing.call_cost_usd(call) == expected


def test_a_call_is_charged_retells_cost_times_the_markup(ledger):
    charged = live_billing.charge_call(MagicMock(), _row(), _cost(31))

    assert charged == credits_for(0.31) == 38  # ceil(31 x 1.2)
    ((workspace, entry, buckets),) = ledger["entries"]
    assert workspace == WS and buckets == BUCKETS
    assert (entry.kind, entry.source_type, entry.source_ref) == ("debit", "voice_live_call", "call_1")
    assert (entry.credits, entry.charged_credits, entry.actual_cost_usd) == (-38, 38, 0.31)
    assert entry.reason == "Auto Live call, 2 min" and entry.band == "metered"


def test_a_redelivered_event_charges_nothing_more(ledger):
    assert live_billing.charge_call(MagicMock(), _row(), _cost(31)) == 38
    assert live_billing.charge_call(MagicMock(), _row(), _cost(31)) == 0


def test_an_event_with_no_cost_is_charged_by_its_minutes_at_the_estimate(ledger, monkeypatch, caplog):
    monkeypatch.setattr(config, "VOICE_LIVE_ESTIMATE_USD_PER_MINUTE", 0.155)
    with caplog.at_level(logging.WARNING, logger="modules.voice.live_billing"):
        charged = live_billing.charge_call(MagicMock(), _row(duration_seconds=120), {"call_id": "call_1"})

    assert charged == credits_for(0.31)
    assert "call_1 carried no cost" in caplog.text


def test_a_call_that_never_connected_and_carries_no_cost_is_free(ledger):
    assert live_billing.charge_call(MagicMock(), _row(status="failed", duration_seconds=0), {}) == 0
    assert ledger["entries"] == []


@pytest.mark.parametrize("funded_by", [FUNDED_BY_ALLOWANCE, FUNDED_BY_LOCAL])
def test_a_workspace_not_paying_in_credits_is_never_charged(ledger, funded_by):
    ledger["funding"] = Funding(funded_by)

    assert live_billing.charge_call(MagicMock(), _row(), _cost(31)) == 0
    assert ledger["entries"] == []


def test_a_call_with_no_workspace_is_never_charged(ledger):
    assert live_billing.charge_call(MagicMock(), _row(workspace_id=None), _cost(31)) == 0
    assert ledger["entries"] == []


def test_a_call_may_start_only_with_credits_left_when_paying_in_credits(monkeypatch):
    cases = [(Funding(FUNDED_BY_CREDITS, BUCKETS, 0), False), (Funding(FUNDED_BY_CREDITS, BUCKETS, 1), True),
             (Funding(FUNDED_BY_ALLOWANCE), True), (Funding(FUNDED_BY_LOCAL), True)]
    for funding, allowed in cases:
        monkeypatch.setattr(live_billing, "workspace_funding", lambda db, workspace_id, f=funding: f)
        assert live_billing.has_credits(MagicMock(), WS) is allowed


@pytest.fixture
def events(monkeypatch):
    """apply_call_event over a stand-in row, with the charge and the alerts recorded."""
    state = {"row": _row(status="started"), "charges": [], "alerts": []}
    monkeypatch.setattr(call_events, "_call_row", lambda db, call_id, event: (state["row"], "updated"))

    def charge(db, row, call):
        state["charges"].append(row.call_id)
        return 38

    monkeypatch.setattr(live_billing, "charge_call", charge)
    monkeypatch.setattr(live_billing, "after_charge", lambda db, workspace_id: state["alerts"].append(workspace_id))
    return state


def test_call_ended_charges_the_call_then_sends_the_alerts(events):
    db = MagicMock()
    call = {"call_id": "call_1", "start_timestamp": 1_000, "end_timestamp": 121_000, **_cost(31)}

    call_events.apply_call_event(db, call_events.EVENT_CALL_ENDED, call)

    assert events["charges"] == ["call_1"] and events["alerts"] == [WS]
    db.commit.assert_called_once()


@pytest.mark.parametrize("event", [call_events.EVENT_CALL_STARTED, call_events.EVENT_CALL_ANALYZED])
def test_other_events_charge_nothing(events, event):
    call_events.apply_call_event(MagicMock(), event, {"call_id": "call_1", "start_timestamp": 1_000})

    assert events["charges"] == [] and events["alerts"] == []


def test_the_billing_tab_names_the_movement():
    from services.credits.summary import _SOURCE_LABELS

    assert _SOURCE_LABELS[live_billing.SOURCE_VOICE_LIVE] == "Auto Live call"

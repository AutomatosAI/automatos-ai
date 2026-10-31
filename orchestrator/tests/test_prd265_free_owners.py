"""PRD-265 — the people who never pay (the platform's own owners: "Super Admin doesn't have to pay").

A workspace whose owner's email is in ``BILLING_FREE_OWNER_EMAILS`` runs on an
allowance whatever its plan, like Enterprise: never charged, never paused, still
recorded. With nobody listed, nothing is looked up at all.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from config import config
from services.credits.ledger import Balance
from services.credits.routing import (
    FUNDED_BY_ALLOWANCE,
    FUNDED_BY_CREDITS,
    FUNDED_BY_LOCAL,
    can_pay,
    free_workspace_ids,
    funding_for,
)


def test_a_free_owners_workspace_is_an_allowance_on_any_plan():
    for plan in ("basic", "pro", "business"):
        funding = funding_for(plan, Balance(), edition="saas", free=True)
        assert funding.funded_by == FUNDED_BY_ALLOWANCE and can_pay(funding, 1_000_000)


def test_everyone_else_still_pays_and_the_local_edition_still_never_meters():
    assert funding_for("pro", Balance(), edition="saas").funded_by == FUNDED_BY_CREDITS
    assert funding_for("pro", Balance(), edition="local", free=True).funded_by == FUNDED_BY_LOCAL


def test_with_nobody_listed_nothing_is_looked_up(monkeypatch):
    monkeypatch.setattr(config, "BILLING_FREE_OWNER_EMAILS", frozenset())
    db = MagicMock()
    assert free_workspace_ids(db, ["ws-1", "ws-2"]) == set()
    db.execute.assert_not_called()


def test_the_lookup_matches_owner_emails_case_blind(monkeypatch):
    monkeypatch.setattr(config, "BILLING_FREE_OWNER_EMAILS", frozenset({"owner@example.test"}))
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [("ws-1",)]
    assert free_workspace_ids(db, ["ws-1", "ws-2", None]) == {"ws-1"}
    sql, params = db.execute.call_args[0]
    assert "lower(u.email) = ANY(:emails)" in str(sql)
    assert params == {"ids": ["ws-1", "ws-2"], "emails": ["owner@example.test"]}


def test_the_meter_waves_through_only_the_free_owners_calls(monkeypatch):
    from services.credits import meter

    def call(usage_id, workspace_id):
        return meter.PendingCall(usage_id=usage_id, workspace_id=workspace_id, model_id="m", provider="p",
                                 tier=None, request_type="chat", input_tokens=1, output_tokens=1,
                                 total_cost=0.5, is_byok=False)

    seen = {}

    def fake_meter_call(db, pending, plan, *, free):
        seen[pending.usage_id] = free
        return not free

    monkeypatch.setattr(meter, "_plans", lambda db, ids: {"ws-free": "pro", "ws-paying": "pro"})
    monkeypatch.setattr(meter, "free_workspace_ids", lambda db, ids: {"ws-free"} & set(ids))
    monkeypatch.setattr(meter, "meter_call", fake_meter_call)
    debited, touched = meter._meter_batch(MagicMock(), [call(1, "ws-free"), call(2, "ws-paying")])
    assert seen == {1: True, 2: False}
    assert (debited, touched) == (1, {"ws-paying"})

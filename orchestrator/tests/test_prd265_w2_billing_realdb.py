"""PRD-265 W2 against real Postgres (``@integration``): Stripe events → plans and credits.

Whole webhook events go through ``services.billing.events.receive`` (stored, then
applied, one transaction) with Stripe's own translator; only the signature check
is faked. Pinned:

- a paid Pro invoice puts the workspace on Pro and grants the month (2,000),
  expiring at the period end; the same event again changes nothing;
- a renewal expires what is left of the month, then grants a fresh one;
- an upgrade mid-period grants only the difference; a downgrade grants nothing;
- a top-up grants 1,000 credits for a year; refunds take back the refunded share
  (the running total), to a negative bucket once spent; a dispute takes back and
  a dispute won gives back;
- a payment found through ``invoice_payment.paid`` can be refunded;
- a subscription that ends, or goes unpaid, moves the workspace to Own keys and
  expires the month; a failed payment marks it past due; both tell the owner;
- an invoice for a price we don't sell is kept for review and grants nothing;
- a processing failure marks the event failed and the next delivery settles it;
- the hourly sweep expires a month past its grace (not before) and the part of
  an expired top-up pack that is still left, once.

Skips cleanly when no Postgres with the ``prd265_billing`` schema is reachable
(CI runs it).
"""
from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from config import config  # noqa: E402
from core.database.database import get_database_url  # noqa: E402
from services.billing import events  # noqa: E402
from services.billing.accounts import link_account  # noqa: E402
from services.billing.facts import WebhookEvent  # noqa: E402
from services.billing.stripe_events import translate  # noqa: E402
from services.credits import ledger  # noqa: E402
from services.credits.expiry import expire_due  # noqa: E402
from services.credits.ledger import Entry, read_balance  # noqa: E402

pytestmark = pytest.mark.integration

DAY = 86400


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            for tbl in ("workspaces", "credit_ledger", "credit_balances", "billing_accounts", "billing_events"):
                c.execute(text(f"SELECT 1 FROM {tbl} LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"billing suite needs Postgres with prd265_billing applied: {exc}")
    yield eng
    eng.dispose()


class FakeStripe:
    """Stripe without the signature check: the payload is the event itself."""

    name = "stripe"

    def parse_webhook(self, payload, signature):
        return WebhookEvent(provider="stripe", event_id=payload["id"], event_type=payload["type"],
                            livemode=False, data=payload["data"]["object"])

    def translate(self, event):
        return translate(event)


class Notices:
    sent: list = []

    def __init__(self, db, workspace_id):
        self.workspace_id = workspace_id

    async def dispatch(self, *, event_type, title, message, status):
        Notices.sent.append((self.workspace_id, event_type))


@pytest.fixture(autouse=True)
def stripe_config(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "STRIPE_PRICE_BASIC", "price_basic")
    monkeypatch.setattr(config, "STRIPE_PRICE_PRO", "price_pro")
    monkeypatch.setattr(config, "STRIPE_PRICE_BUSINESS", "price_business")
    monkeypatch.setattr(config, "STRIPE_PRICE_TOPUP", "price_topup")
    monkeypatch.setattr(config, "TOPUP_CREDITS", 1000)
    monkeypatch.setattr(config, "TOPUP_VALID_DAYS", 365)
    monkeypatch.setattr(config, "BILLING_MONTHLY_EXPIRY_GRACE_HOURS", 24)
    monkeypatch.setattr("core.services.notification_dispatcher.NotificationDispatcher", Notices)
    Notices.sent = []


@pytest.fixture
def billing(engine, new_session):
    """A workspace linked to its Stripe customer; events carry a per-test prefix."""
    tag = uuid.uuid4().hex[:10]
    made: list[str] = []

    def make(plan: str = "basic") -> dict:
        ws = str(uuid.uuid4())
        s = new_session()
        s.execute(text("INSERT INTO workspaces (id, name, slug, plan, onboarding) "
                       "VALUES (CAST(:id AS uuid), :n, :n, :plan, CAST(:ob AS jsonb))"),
                  {"id": ws, "n": f"billing-{ws[:8]}", "plan": plan, "ob": '{"stage": "not_started"}'})
        s.commit()
        customer = f"cus_{tag}{len(made)}"
        link_account(s, ws, customer, provider="stripe")
        s.commit()
        made.append(ws)
        return {"ws": ws, "customer": customer, "tag": f"{tag}{len(made)}"}

    yield make
    s = new_session.sweep()
    s.execute(text("DELETE FROM billing_events WHERE event_id LIKE :p"), {"p": f"evt_{tag}%"})
    for ws in made:
        for table in ("billing_accounts", "credit_ledger", "credit_balances"):
            s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})
        s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()


def _event(kind: str, obj: dict, ref: str) -> dict:
    return {"id": f"evt_{ref}", "type": kind, "data": {"object": obj}}


def _invoice(inv: str, customer: str, lines: list, *, sub: str = "sub_1", pi=None) -> dict:
    return {
        "id": inv, "object": "invoice", "customer": customer, "amount_paid": sum(max(0, a) for a, _, _ in lines),
        "parent": {"type": "subscription_details", "subscription_details": {"subscription": sub, "metadata": {}}},
        "lines": {"data": [{"amount": amount, "pricing": {"price_details": {"price": price}},
                            "period": {"start": end - 30 * DAY, "end": end}} for amount, price, end in lines]},
        "payments": {"data": [{"payment": {"type": "payment_intent", "payment_intent": pi}}] if pi else []},
    }


def _topup_session(session_id: str, b: dict, pi: str) -> dict:
    return {"id": session_id, "object": "checkout.session", "mode": "payment", "payment_status": "paid",
            "customer": b["customer"], "client_reference_id": b["ws"], "payment_intent": pi, "amount_total": 1000,
            "metadata": {"workspace_id": b["ws"], "kind": "topup", "quantity": "1"}}


def _charge(charge_id: str, pi: str, amount: int, refunded: int) -> dict:
    return {"id": charge_id, "object": "charge", "payment_intent": pi, "amount": amount, "amount_refunded": refunded}


def _send(new_session, payload: dict) -> str:
    return events.receive(new_session(), FakeStripe(), payload, "")


def _plan(s, ws: str) -> str:
    return s.execute(text("SELECT plan FROM workspaces WHERE id = CAST(:ws AS uuid)"), {"ws": ws}).scalar()


def _account(s, ws: str):
    return s.execute(text(
        "SELECT plan, subscription_id, subscription_status, period_credits, credited_period_end "
        "FROM billing_accounts WHERE workspace_id = CAST(:ws AS uuid)"), {"ws": ws}).fetchone()


def _rows(s, ws: str, source_type: str) -> list:
    return s.execute(text(
        "SELECT kind, bucket, credits FROM credit_ledger WHERE workspace_id = CAST(:ws AS uuid) "
        "AND source_type = :t ORDER BY id"), {"ws": ws, "t": source_type}).fetchall()


def _event_row(s, ref: str):
    return s.execute(text("SELECT status, attempts, payment_ref, grant_source FROM billing_events "
                          "WHERE event_id = :e"), {"e": f"evt_{ref}"}).fetchone()


def _spend(s, ws: str, credits: int) -> None:
    ledger.record(s, ws, Entry(kind=ledger.KIND_DEBIT, source_type="test_spend", source_ref=uuid.uuid4().hex,
                               credits=-credits))
    s.commit()


def test_a_paid_pro_invoice_puts_the_workspace_on_pro_with_its_month(billing, new_session):
    b = billing()
    end = int(time.time()) + 30 * DAY
    paid = _event("invoice.paid", _invoice("in_" + b["tag"], b["customer"], [(2900, "price_pro", end)]), b["tag"])
    assert _send(new_session, paid) == "processed"
    s = new_session()
    balance = read_balance(s, b["ws"])
    assert balance.monthly == 2000 and int(balance.period_end.timestamp()) == end
    assert _plan(s, b["ws"]) == "pro"
    plan, sub, status, period_credits, credited = _account(s, b["ws"])
    assert (plan, sub, status, period_credits) == ("pro", "sub_1", "active", 2000)
    assert int(credited.timestamp()) == end
    assert _send(new_session, paid) == "processed"  # redelivered: settled already
    assert read_balance(new_session(), b["ws"]).monthly == 2000
    assert _rows(new_session(), b["ws"], "stripe_invoice") == [("grant", "monthly", 2000)]
    assert _event_row(new_session(), b["tag"])[1] == 1  # processed once


def test_a_renewal_expires_the_old_month_then_grants_a_new_one(billing, new_session):
    b = billing()
    end1 = int(time.time()) + 30 * DAY
    end2 = end1 + 30 * DAY
    _send(new_session, _event("invoice.paid", _invoice("in_a" + b["tag"], b["customer"], [(2900, "price_pro", end1)]),
                              b["tag"] + "a"))
    _spend(new_session(), b["ws"], 500)
    _send(new_session, _event("invoice.paid", _invoice("in_b" + b["tag"], b["customer"], [(2900, "price_pro", end2)]),
                              b["tag"] + "b"))
    s = new_session()
    assert _rows(s, b["ws"], "monthly_expire") == [("expire", "monthly", -1500)]
    balance = read_balance(s, b["ws"])
    assert balance.monthly == 2000 and int(balance.period_end.timestamp()) == end2


def test_an_upgrade_mid_period_grants_only_the_difference(billing, new_session):
    b = billing()
    end = int(time.time()) + 20 * DAY
    _send(new_session, _event("invoice.paid", _invoice("in_o" + b["tag"], b["customer"], [(900, "price_basic", end)]),
                              b["tag"] + "o"))
    assert read_balance(new_session(), b["ws"]).monthly == 0 and _plan(new_session(), b["ws"]) == "basic"
    up = _invoice("in_p" + b["tag"], b["customer"], [(1300, "price_pro", end), (-400, "price_basic", end)])
    _send(new_session, _event("invoice.paid", up, b["tag"] + "p"))
    assert read_balance(new_session(), b["ws"]).monthly == 2000 and _plan(new_session(), b["ws"]) == "pro"
    biz = _invoice("in_b" + b["tag"], b["customer"], [(4500, "price_business", end), (-1300, "price_pro", end)])
    _send(new_session, _event("invoice.paid", biz, b["tag"] + "b"))
    assert read_balance(new_session(), b["ws"]).monthly == 8000 and _plan(new_session(), b["ws"]) == "business"
    down = _invoice("in_d" + b["tag"], b["customer"], [(800, "price_pro", end), (-3000, "price_business", end)])
    _send(new_session, _event("invoice.paid", down, b["tag"] + "d"))
    s = new_session()
    assert read_balance(s, b["ws"]).monthly == 8000  # a downgrade never mints, never takes
    assert _plan(s, b["ws"]) == "pro" and _account(s, b["ws"])[3] == 8000


def test_a_topup_then_refunds_take_back_the_refunded_share(billing, new_session):
    b = billing()
    pi = "pi_t" + b["tag"]
    _send(new_session, _event("checkout.session.completed", _topup_session("cs_" + b["tag"], b, pi), b["tag"] + "t"))
    s = new_session()
    assert read_balance(s, b["ws"]).topup == 1000
    expires = s.execute(text("SELECT expires_at FROM credit_ledger WHERE source_type = 'stripe_topup' "
                             "AND workspace_id = CAST(:ws AS uuid)"), {"ws": b["ws"]}).scalar()
    assert abs((expires - datetime.now(timezone.utc)) - timedelta(days=365)) < timedelta(minutes=5)
    half = _event("charge.refunded", _charge("ch_" + b["tag"], pi, 1000, 500), b["tag"] + "r1")
    _send(new_session, half)
    assert read_balance(new_session(), b["ws"]).topup == 500
    full = _event("charge.refunded", _charge("ch_" + b["tag"], pi, 1000, 1000), b["tag"] + "r2")
    _send(new_session, full)
    _send(new_session, full)
    s = new_session()
    assert read_balance(s, b["ws"]).topup == 0
    assert _rows(s, b["ws"], "stripe_refund") == [("refund", "topup", -500), ("refund", "topup", -500)]


def test_a_refund_after_the_credits_were_spent_goes_negative(billing, new_session):
    b = billing()
    pi = "pi_s" + b["tag"]
    _send(new_session, _event("checkout.session.completed", _topup_session("cs_" + b["tag"], b, pi), b["tag"] + "t"))
    _spend(new_session(), b["ws"], 800)
    _send(new_session, _event("charge.refunded", _charge("ch_" + b["tag"], pi, 1000, 1000), b["tag"] + "r"))
    balance = read_balance(new_session(), b["ws"])
    assert balance.topup == -800 and balance.spendable() == 0


def test_a_dispute_takes_back_and_a_dispute_won_gives_back(billing, new_session):
    b = billing()
    pi = "pi_d" + b["tag"]
    _send(new_session, _event("checkout.session.completed", _topup_session("cs_" + b["tag"], b, pi), b["tag"] + "t"))
    dispute = {"id": "dp_" + b["tag"], "object": "dispute", "payment_intent": pi, "amount": 1000,
               "status": "needs_response"}
    _send(new_session, _event("charge.dispute.created", dispute, b["tag"] + "d1"))
    assert read_balance(new_session(), b["ws"]).topup == 0
    _send(new_session, _event("charge.dispute.closed", {**dispute, "status": "won"}, b["tag"] + "d2"))
    assert read_balance(new_session(), b["ws"]).topup == 1000


def test_an_invoice_payment_links_the_payment_so_its_refund_takes_the_month_back(billing, new_session):
    b = billing()
    end = int(time.time()) + 30 * DAY
    inv = "in_" + b["tag"]
    _send(new_session, _event("invoice.paid", _invoice(inv, b["customer"], [(2900, "price_pro", end)]), b["tag"] + "i"))
    link = {"id": "inpay_" + b["tag"], "object": "invoice_payment", "invoice": inv, "amount_paid": 2900,
            "payment": {"type": "payment_intent", "payment_intent": "pi_i" + b["tag"]}}
    _send(new_session, _event("invoice_payment.paid", link, b["tag"] + "l"))
    assert _event_row(new_session(), b["tag"] + "l")[3] == f"stripe_invoice:{inv}"
    _send(new_session, _event("charge.refunded", _charge("ch_" + b["tag"], "pi_i" + b["tag"], 2900, 2900),
                              b["tag"] + "r"))
    assert read_balance(new_session(), b["ws"]).monthly == 0


def test_a_subscription_that_ends_moves_to_own_keys_and_expires_the_month(billing, new_session):
    b = billing()
    end = int(time.time()) + 30 * DAY
    _send(new_session, _event("invoice.paid", _invoice("in_" + b["tag"], b["customer"], [(2900, "price_pro", end)]),
                              b["tag"] + "i"))
    sub = {"id": "sub_1", "object": "subscription", "customer": b["customer"], "status": "canceled",
           "items": {"data": []}}
    _send(new_session, _event("customer.subscription.deleted", sub, b["tag"] + "x"))
    s = new_session()
    assert _plan(s, b["ws"]) == "basic" and read_balance(s, b["ws"]).monthly == 0
    assert _account(s, b["ws"])[2] == "canceled"
    assert (b["ws"], "billing_subscription_ended") in Notices.sent


def test_an_unpaid_subscription_ends_once_and_tells_the_owner_once(billing, new_session):
    b = billing()
    end = int(time.time()) + 30 * DAY
    _send(new_session, _event("invoice.paid", _invoice("in_" + b["tag"], b["customer"], [(2900, "price_pro", end)]),
                              b["tag"] + "i"))
    failed = {"id": "in_f" + b["tag"], "object": "invoice", "customer": b["customer"], "attempt_count": 1,
              "next_payment_attempt": end + 3 * DAY, "parent": {"subscription_details": {"subscription": "sub_1"}}}
    _send(new_session, _event("invoice.payment_failed", failed, b["tag"] + "f"))
    assert _account(new_session(), b["ws"])[2] == "past_due"
    unpaid = {"id": "sub_1", "object": "subscription", "customer": b["customer"], "status": "unpaid",
              "items": {"data": [{"price": {"id": "price_pro"}, "current_period_end": end}]}}
    _send(new_session, _event("customer.subscription.updated", unpaid, b["tag"] + "u1"))
    _send(new_session, _event("customer.subscription.updated", unpaid, b["tag"] + "u2"))
    assert _plan(new_session(), b["ws"]) == "basic"
    sent = [kind for ws, kind in Notices.sent if ws == b["ws"]]
    assert sent.count("billing_payment_failed") == 1 and sent.count("billing_subscription_ended") == 1


def test_an_invoice_for_a_price_we_dont_sell_is_kept_for_review(billing, new_session):
    b = billing()
    end = int(time.time()) + 30 * DAY
    odd = _event("invoice.paid", _invoice("in_" + b["tag"], b["customer"], [(500, "price_other", end)]), b["tag"])
    assert _send(new_session, odd) == "needs_review"
    s = new_session()
    assert _event_row(s, b["tag"])[0] == "needs_review"
    assert _rows(s, b["ws"], "stripe_invoice") == [] and _plan(s, b["ws"]) == "basic"


def test_a_failed_processing_is_kept_and_the_next_delivery_settles_it(billing, new_session, monkeypatch):
    b = billing()
    end = int(time.time()) + 30 * DAY
    paid = _event("invoice.paid", _invoice("in_" + b["tag"], b["customer"], [(2900, "price_pro", end)]), b["tag"])
    real = events.apply_fact

    def broken(*args, **kwargs):
        raise RuntimeError("database hiccup")

    monkeypatch.setattr(events, "apply_fact", broken)
    with pytest.raises(RuntimeError):
        _send(new_session, paid)
    assert _event_row(new_session(), b["tag"])[:2] == ("failed", 1)
    assert read_balance(new_session(), b["ws"]).monthly == 0
    monkeypatch.setattr(events, "apply_fact", real)
    assert _send(new_session, paid) == "processed"
    assert _event_row(new_session(), b["tag"])[:2] == ("processed", 2)
    assert read_balance(new_session(), b["ws"]).monthly == 2000


def _grant(s, ws: str, bucket: str, credits: int, *, kind=ledger.KIND_GRANT, expires_at=None) -> None:
    ledger.record(s, ws, Entry(kind=kind, source_type="test_grant", source_ref=uuid.uuid4().hex, credits=credits,
                               bucket=bucket, expires_at=expires_at))
    s.commit()


def test_the_sweep_expires_a_month_past_its_grace_and_not_before(billing, new_session):
    past, within = billing(), billing()
    now = datetime.now(timezone.utc)
    for b, ended in ((past, now - timedelta(hours=25)), (within, now - timedelta(hours=23))):
        s = new_session()
        _grant(s, b["ws"], ledger.BUCKET_MONTHLY, 700)
        s.execute(text("UPDATE credit_balances SET period_end = :end WHERE workspace_id = CAST(:ws AS uuid)"),
                  {"end": ended, "ws": b["ws"]})
        s.commit()
    expire_due(new_session(), now=now)
    expire_due(new_session(), now=now)  # a second sweep changes nothing
    assert read_balance(new_session(), past["ws"]).monthly == 0
    assert _rows(new_session(), past["ws"], "monthly_expire") == [("expire", "monthly", -700)]
    assert read_balance(new_session(), within["ws"]).monthly == 700


def test_the_sweep_expires_what_is_left_of_an_old_topup_pack_once(billing, new_session):
    b = billing()
    now = datetime.now(timezone.utc)
    s = new_session()
    _grant(s, b["ws"], ledger.BUCKET_TOPUP, 1000, kind=ledger.KIND_TOPUP, expires_at=now - timedelta(hours=1))
    _grant(s, b["ws"], ledger.BUCKET_TOPUP, 1000, kind=ledger.KIND_TOPUP, expires_at=now + timedelta(days=100))
    _spend(s, b["ws"], 300)  # spent from the old pack first: it holds 700, the new one 1,000
    expire_due(new_session(), now=now)
    expire_due(new_session(), now=now)
    s = new_session()
    assert read_balance(s, b["ws"]).topup == 1000
    assert _rows(s, b["ws"], "topup_expire") == [("expire", "topup", -700)]

"""PRD-265 W2 — an upgrade mid-period gets credits for the share of the period left.

Stripe prorates the price of an upgrade by the time left in the period; the credits
follow the same share, so an upgrade on the last day can't buy a whole month of
credits for a day's difference (``services/billing/grants.share_left``). The prorated
line's start is the moment of the upgrade (``stripe_events.invoice_paid``).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from services.billing.facts import PeriodPaid
from services.billing.grants import share_left
from services.billing.stripe_events import invoice_paid

START = datetime(2026, 11, 1, tzinfo=timezone.utc)
END = START + timedelta(days=30)


def test_half_way_through_the_period_half_is_left():
    assert share_left(START, START + timedelta(days=15), END) == 0.5


def test_on_the_last_day_almost_nothing_is_left():
    assert share_left(START, END - timedelta(days=1), END) == 1 / 30


def test_the_share_stays_between_none_and_all():
    assert share_left(START, START - timedelta(days=5), END) == 1.0
    assert share_left(START, END + timedelta(days=5), END) == 0.0


def test_unknown_bounds_count_as_the_whole_period():
    assert share_left(None, START, END) == 1.0
    assert share_left(START, None, END) == 1.0
    assert share_left(END, START, END) == 1.0  # a zero-length period


def test_the_paid_line_start_is_the_upgrade_moment():
    upgraded = int((START + timedelta(days=20)).timestamp())
    end = int(END.timestamp())
    fact = invoice_paid({
        "id": "in_up", "customer": "cus_1", "amount_paid": 700,
        "parent": {"subscription_details": {"subscription": "sub_1"}},
        "lines": {"data": [
            {"amount": -300, "pricing": {"price_details": {"price": "price_basic"}},
             "period": {"start": upgraded, "end": end}},
            {"amount": 1000, "pricing": {"price_details": {"price": "price_pro"}},
             "period": {"start": upgraded, "end": end}},
        ]},
    })
    assert isinstance(fact, PeriodPaid)
    assert fact.price_ids == ("price_pro", "price_basic")
    assert fact.period_start == START + timedelta(days=20) and fact.period_end == END

"""PRD-265 W2 — ``_left_per_pack`` (``services/credits/expiry.py``): pure FIFO math.

Top-ups are spent oldest first, so what an expiring pack still holds is the
top-up balance less what the younger packs still hold.
"""
from __future__ import annotations

from datetime import datetime, timezone

from services.credits.expiry import _Pack, _left_per_pack

EXPIRES = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _packs(*credits: int) -> list[_Pack]:
    return [_Pack(source_type="topup", source_ref=f"pack-{i}", credits=c, expires_at=EXPIRES)
            for i, c in enumerate(credits)]


def test_three_packs_oldest_first_split_a_partially_spent_balance():
    packs = _packs(1000, 1000, 500)
    assert _left_per_pack(packs, 1800) == [300, 1000, 500]


def test_a_zero_balance_leaves_every_pack_with_nothing():
    packs = _packs(1000, 1000, 500)
    assert _left_per_pack(packs, 0) == [0, 0, 0]


def test_a_negative_balance_leaves_every_pack_with_nothing():
    packs = _packs(1000, 1000, 500)
    assert _left_per_pack(packs, -500) == [0, 0, 0]


def test_a_balance_above_the_total_leaves_every_pack_full():
    packs = _packs(1000, 1000, 500)
    assert _left_per_pack(packs, 10_000) == [1000, 1000, 500]


def test_no_packs_is_an_empty_list_regardless_of_balance():
    assert _left_per_pack([], 500) == []


def test_a_single_pack_keeps_the_balance_up_to_its_own_credits():
    packs = _packs(1000)
    assert _left_per_pack(packs, 400) == [400]
    assert _left_per_pack(packs, 1500) == [1000]

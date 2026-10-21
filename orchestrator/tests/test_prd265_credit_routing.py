"""PRD-265 US-004 — who pays for a platform-key call (pure parts of the gate).

The rules pinned here:

- the local edition never meters; an allowance plan (enterprise) is never gated;
- trial credits pay on any plan; monthly and top-up credits only on a plan with
  ``platform_ai`` — Own keys (``basic``) has no platform lane once its trial is spent;
- background work is never paid from the trial (PRD-222 FR-17);
- the paused error names the way out: add a key on Own keys, top up or add a key on Pro;
- a credit stop is recognised by its ``error_code``, so it is never retried or folded
  into a reply.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from services.credits.ledger import BUCKET_MONTHLY, BUCKET_TOPUP, BUCKET_TRIAL, Balance
from services.credits.routing import (
    CREDITS_EXHAUSTED_CODE,
    FUNDED_BY_ALLOWANCE,
    FUNDED_BY_CREDITS,
    FUNDED_BY_LOCAL,
    CreditsExhaustedError,
    can_pay,
    funding_for,
    has_own_keys,
    is_credit_stop,
    paused_error,
    spend_buckets_for,
)


def test_the_local_edition_never_meters():
    funding = funding_for("basic", Balance(), edition="local")
    assert funding.funded_by == FUNDED_BY_LOCAL
    assert can_pay(funding, 10_000)


def test_enterprise_is_an_allowance_never_gated():
    funding = funding_for("enterprise", Balance(), edition="saas")
    assert funding.funded_by == FUNDED_BY_ALLOWANCE
    assert can_pay(funding, 10_000)


def test_own_keys_spends_only_its_trial():
    assert spend_buckets_for("basic") == (BUCKET_TRIAL,)
    funding = funding_for("basic", Balance(trial=0, monthly=900, topup=900), edition="saas")
    assert funding.funded_by == FUNDED_BY_CREDITS
    assert funding.spendable == 0  # monthly/top-up can't pay for platform AI on Own keys
    assert funding.own_keys_plan is True


def test_pro_spends_trial_then_monthly_then_topup():
    assert spend_buckets_for("pro") == (BUCKET_TRIAL, BUCKET_MONTHLY, BUCKET_TOPUP)
    funding = funding_for("pro", Balance(trial=2, monthly=10, topup=5), edition="saas")
    assert funding.spendable == 17
    assert funding.own_keys_plan is False


def test_an_unknown_plan_is_treated_as_own_keys():
    assert spend_buckets_for("starter") == (BUCKET_TRIAL,)


@pytest.mark.parametrize("plan", ["basic", "pro", "business"])
def test_background_work_never_spends_the_trial(plan):
    assert BUCKET_TRIAL not in spend_buckets_for(plan, background=True)
    funding = funding_for(plan, Balance(trial=500), edition="saas", background=True)
    assert funding.spendable == 0


def test_any_credit_left_can_pay():
    assert can_pay(funding_for("pro", Balance(trial=1), edition="saas"), 1)
    assert not can_pay(funding_for("pro", Balance(trial=0, monthly=-3), edition="saas"), 1)


def test_the_paused_error_names_the_way_out():
    own_keys = paused_error(funding_for("basic", Balance(), edition="saas"))
    pro = paused_error(funding_for("pro", Balance(), edition="saas"))
    assert isinstance(own_keys, CreditsExhaustedError) and own_keys.error_code == CREDITS_EXHAUSTED_CODE
    assert "own keys" in own_keys.message.lower() and "add your" in own_keys.message.lower()
    assert "top up" in pro.message.lower()


def test_credit_stops_are_recognised_by_their_code():
    assert is_credit_stop(CreditsExhaustedError())
    assert not is_credit_stop(ValueError("credits"))


def test_own_keys_are_the_switched_on_byok_overrides():
    assert has_own_keys(SimpleNamespace(settings={"byok_overrides": {"openrouter": True}}))
    assert not has_own_keys(SimpleNamespace(settings={"byok_overrides": {"openrouter": False}}))
    assert not has_own_keys(SimpleNamespace(settings=None))

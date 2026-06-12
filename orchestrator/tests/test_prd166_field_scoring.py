"""PRD-166 S2 — pure field scoring (three-factor + adaptive half-life).

No Qdrant, no env: imports only the pure scoring module.
"""
from __future__ import annotations

import math

from modules.context.field_scoring import (
    ScoringParams,
    budget_results,
    decayed_strength,
    estimate_tokens,
    is_prunable,
    recency_factor,
    resonance,
    stability_factor,
)

P = ScoringParams(
    decay_rate=0.1,
    reinforce_bonus=0.05,
    reinforce_cap=2.0,
    archival_threshold=0.05,
    half_life_access_scale=0.5,
)


def test_stability_rises_with_access_and_caps():
    base = stability_factor(1.0, 0, reinforce_bonus=0.05, reinforce_cap=2.0)
    more = stability_factor(1.0, 10, reinforce_bonus=0.05, reinforce_cap=2.0)
    capped = stability_factor(1.0, 1000, reinforce_bonus=0.05, reinforce_cap=2.0)
    assert base == 1.0
    assert more > base
    assert capped == 2.0  # 1 + 1000*0.05 clamped to cap


def test_recency_decays_with_age():
    fresh = recency_factor(0.0, 0, decay_rate=0.1, half_life_access_scale=0.5)
    old = recency_factor(48.0, 0, decay_rate=0.1, half_life_access_scale=0.5)
    assert fresh == 1.0
    assert 0.0 < old < fresh


def test_adaptive_half_life_slows_decay_for_reused_patterns():
    """The PRD's adaptive half-life: at the SAME age, a frequently-accessed
    pattern retains more recency than a one-off one."""
    age = 24.0
    one_off = recency_factor(age, 0, decay_rate=0.1, half_life_access_scale=0.5)
    reused = recency_factor(age, 8, decay_rate=0.1, half_life_access_scale=0.5)
    assert reused > one_off
    # effective half-life of the reused pattern is materially longer
    assert reused > one_off * 1.5


def test_resonance_is_three_factors():
    # cosine² × stability × recency, all independently monotonic
    hi = resonance(0.9, 1.0, 1.0, 5, P)
    lo_sim = resonance(0.3, 1.0, 1.0, 5, P)
    lo_str = resonance(0.9, 0.2, 1.0, 5, P)
    lo_rec = resonance(0.9, 1.0, 200.0, 0, P)
    assert hi > lo_sim and hi > lo_str and hi > lo_rec
    # explicit value: 0.9²×(1.0×(1+5*0.05))×e^(-(0.1/(1+0.5*5))*1)
    sim = 0.9 ** 2
    stab = 1.0 * (1 + 5 * 0.05)
    rec = math.exp(-(0.1 / (1 + 0.5 * 5)) * 1.0)
    assert abs(hi - sim * stab * rec) < 1e-9


def test_golden_ranking_recent_strong_relevant_wins():
    """A relevant+strong+fresh pattern outranks a stale or weak or off-topic one."""
    winner = resonance(cosine=0.85, strength=1.0, age_hours=2.0, access_count=4, params=P)
    stale = resonance(cosine=0.85, strength=1.0, age_hours=300.0, access_count=0, params=P)
    weak = resonance(cosine=0.85, strength=0.1, age_hours=2.0, access_count=0, params=P)
    offtopic = resonance(cosine=0.2, strength=1.0, age_hours=2.0, access_count=4, params=P)
    ranked = sorted(
        [("winner", winner), ("stale", stale), ("weak", weak), ("offtopic", offtopic)],
        key=lambda x: x[1], reverse=True,
    )
    assert ranked[0][0] == "winner"


def test_negative_cosine_floored():
    assert resonance(-0.5, 1.0, 0.0, 0, P) == 0.0


def test_is_prunable_below_hard_threshold():
    # a near-dead one-off pattern prunes; a reused fresh one does not
    assert is_prunable(0.1, 500.0, 0, P, prune_threshold=0.01) is True
    assert is_prunable(1.0, 1.0, 5, P, prune_threshold=0.01) is False


# ── S2 budgeted query (no silent caps) ─────────────────────────

def test_budget_keeps_all_under_budget():
    rows = [{"value": "x" * 40} for _ in range(3)]  # ~10 tokens each
    kept, truncated = budget_results(rows, token_budget=1000)
    assert len(kept) == 3 and truncated is False


def test_budget_truncates_and_flags():
    rows = [{"value": "x" * 400} for _ in range(10)]  # ~100 tokens each
    kept, truncated = budget_results(rows, token_budget=250)
    assert truncated is True
    assert 0 < len(kept) < 10


def test_budget_always_keeps_top_result_even_if_oversized():
    rows = [{"value": "x" * 8000}, {"value": "y" * 8000}]
    kept, truncated = budget_results(rows, token_budget=100)
    assert len(kept) == 1 and truncated is True


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100

"""Wave 4a: ``evals.memory_profiles`` scores what a profile's distillation keeps (offline in CI).

The live run distils each sample exchange with ``general`` and with its domain profile on a
model; CI checks the fixtures (every profile seeded, every profile prompt carries it) and the
scoring the live run relies on.
"""
from __future__ import annotations

import pytest

from evals import memory_profiles as ev


def test_every_exchanges_profile_is_seeded_and_its_prompt_carries_it():
    from core.seeds.seed_memory_profiles import SEEDED_PROFILES

    seeded = {p["slug"] for p in SEEDED_PROFILES}
    assert {e.profile for e in ev.EXCHANGES} == seeded - {"general"}
    sizes = ev.check_fixtures()["prompt_chars"]
    assert all(arms[domain] > arms["general"] for domain, arms in sizes.items())


def test_a_need_is_kept_only_when_one_fact_holds_all_its_words():
    incident = next(e for e in ev.EXCHANGES if e.profile == "incident")
    facts = [{"fact": "A postmortem is owed in the incident channel by 12 November."},
             {"fact": "Checkout began failing with 502s at 14:05 UTC."},
             {"fact": "It was not DNS."},
             {"fact": "The pool fix is pending."}]
    scored = ev.score(incident, facts)
    assert "postmortem owed by 12 November" in scored["kept"] and "502s from 14:05 UTC" in scored["kept"]
    assert "cause: the pool dropped to 5" not in scored["kept"]  # "pool" without the "5" is not the cause
    assert scored["unwanted"] == ["the ruled-out DNS check"] and scored["facts"] == 4
    assert scored["kept_share"] == pytest.approx(2 / len(incident.needed))


def test_alternatives_count_and_case_is_ignored():
    case = next(e for e in ev.EXCHANGES if e.profile == "case")
    kept = ev.score(case, [{"fact": "The lease began in March 2024 on a 5-year term."},
                           {"fact": "COPPERFIELD CLAIMS £38,000."}])["kept"]
    assert {"the lease: 1 March 2024, five years", "the £38,000 claim"} <= set(kept)


def test_the_offline_run_needs_no_model_and_live_needs_one():
    assert ev.main([]) == 0
    with pytest.raises(SystemExit):
        ev.main(["--live"])

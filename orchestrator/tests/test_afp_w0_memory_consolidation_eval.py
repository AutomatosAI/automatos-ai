"""#1430 Wave 0: the memory-consolidation eval's self-tests (evals/memory_consolidation.py).

The eval files facts and their restatements in a workspace, runs the real
``run_sleep_time_consolidation`` once and counts, per fact, the rows left. These tests
hold the counting honest with a consolidation that does nothing and one that deletes a
known row; what consolidation itself keeps or loses is the eval's published number,
not a test's to pin.
"""
from __future__ import annotations

import asyncio
import json

import evals.memory_consolidation as mc
import evals.memory_pipeline as mp
from evals.memory_recall import load_corpus


def _facts():
    corpus = load_corpus()
    return [d for d in corpus if d.workspace_id == "tenant-acme"][:3]


def _run(service, facts):
    restated = [mc.restatement(d, k) for d in facts for k in mc.KINDS]
    return asyncio.run(mc._consolidate(service, facts, restated))


def test_restatements_say_the_same_fact_three_ways():
    fact = next(d for d in load_corpus() if d.memory_id == "acme-01")
    exact, light, reworded = (mc.restatement(fact, k) for k in mc.KINDS)
    assert exact.text == fact.text and exact.memory_id == "acme-01#exact"
    assert light.text.startswith("As noted, ") and light.text.endswith(fact.text[1:])
    assert reworded.text != fact.text and "each year" in reworded.text      # every → each


def test_with_no_consolidation_every_restatement_is_a_duplicate_and_nothing_is_lost():
    service = mp.offline_service()

    async def no_consolidation(workspace_id=None):
        return {}

    service.run_sleep_time_consolidation = no_consolidation
    facts = _facts()
    result = _run(service, facts)
    reworded_rows = sum(mc.restatement(d, "reworded").text != d.text for d in facts)
    assert result["rows_written"] == result["rows_left"] == len(facts) * 2 + reworded_rows  # exact folds at write
    assert result["duplicates_left"] == len(facts) + reworded_rows
    assert result["restatements_left"] == {"exact": 0, "light": len(facts), "reworded": reworded_rows}
    assert result["facts_lost"] == []


def test_a_fact_whose_every_row_is_deleted_is_counted_lost():
    service = mp.offline_service()
    real = service.run_sleep_time_consolidation

    async def delete_acme_01(workspace_id=None):
        await real(workspace_id=workspace_id)
        for m in await service.get_all_memories(workspace_id, limit=1000):
            if str((m.get("metadata") or {}).get("memory_id")).startswith("acme-01"):
                await service.delete_memory(m["id"], workspace_id)
        return {}

    service.run_sleep_time_consolidation = delete_acme_01
    assert _run(service, _facts())["facts_lost"] == ["acme-01"]


def test_the_scenarios_cover_tenants_lookalikes_and_the_200_row_window():
    corpus = load_corpus()
    (_, per_tenant), (_, with_twins), (label, one) = mc._scenarios(corpus)
    assert sum(map(len, per_tenant)) == len(corpus) and len(per_tenant) == 4
    assert sum(map(len, with_twins)) == len(mp.with_lookalikes(corpus, mc.LOOKALIKES))
    assert len(one) == 1 and len(one[0]) > mc.ROWS_READ and str(mc.ROWS_READ) in label


def test_the_command_publishes_every_scenario_and_exits_zero(capsys):
    assert mc.main(["--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert [s["facts"] for s in result["scenarios"]] == [54, 414, 414]
    assert all(s["rows_left"] <= s["rows_written"] for s in result["scenarios"])

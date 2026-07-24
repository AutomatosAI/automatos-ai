"""PRD-206 S10 — continuity becomes a number.

The memory gold set gains a ``continuity`` slice (resume / decision-recall /
open-loop queries against a tenant whose corpus carries the S1 continuity
types plus same-topic distractors), and the harness reports recall per
slice. NO passing-recall assertion on the bundled fixture — a sub-threshold
continuity number is a valid, honest outcome (the house discipline); the
number existing and being computed correctly is what's tested.
"""
import sys
from pathlib import Path

_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

from evals.memory_recall import (  # noqa: E402
    GoldQuery,
    MemoryDoc,
    load_corpus,
    load_gold_set,
    run_memory_recall_eval,
)


# ---------------------------------------------------------------------------
# Schema: the slice field
# ---------------------------------------------------------------------------

def test_gold_set_schema_carries_the_continuity_slice():
    gold = load_gold_set()
    slices = {q.slice for q in gold}
    assert "continuity" in slices
    # Legacy rows (no slice key) default to "core" — nothing reclassified.
    assert "core" in slices
    continuity = [q for q in gold if q.slice == "continuity"]
    assert len(continuity) >= 8
    for q in continuity:
        assert q.workspace_id == "tenant-continuity"
        assert q.memory_dependent, "continuity queries are memory-dependent by nature"


def test_continuity_corpus_carries_the_s1_types_and_distractors():
    corpus = [d for d in load_corpus() if d.workspace_id == "tenant-continuity"]
    types = {d.content_type for d in corpus}
    assert {"decision", "open_loop", "thread_summary"} <= types
    # Honesty: the slice carries same-topic distractors, so the number
    # reflects discrimination, not a rigged fixture.
    assert "business_fact" in types


# ---------------------------------------------------------------------------
# Harness: per-slice scoring (exact numbers on a synthetic fixture)
# ---------------------------------------------------------------------------

def _doc(mid, ws, text, ctype="decision", cat="c"):
    return MemoryDoc(mid, ws, text, ctype, cat)


def _q(qid, ws, query, relevant, slice_name):
    return GoldQuery(qid, ws, query, frozenset(relevant), "c", "easy", True, slice_name)


def test_harness_scores_per_slice():
    corpus = [
        _doc("d1", "t1", "we decided to ship the beta on friday"),
        _doc("d2", "t1", "the office plant needs watering on mondays"),
    ]
    gold = [
        # continuity query → hits d1 at rank 1.
        _q("g1", "t1", "what did we decide about the beta ship date", {"d1"}, "continuity"),
        # core query → labelled relevant is d1, but the query is about the
        # plant → d2 outranks d1 → recall@5 still 1.0 (both in window),
        # kept simple: assert per-slice split, not ranking subtleties.
        _q("g2", "t1", "when does the office plant need watering", {"d2"}, "core"),
        # continuity miss: labels a memory that scores zero overlap.
        _q("g3", "t1", "what did we decide about the beta ship date", {"d2"}, "continuity"),
    ]
    report = run_memory_recall_eval(corpus, gold)

    assert set(report.slices) == {"continuity", "core"}
    cont = report.slices["continuity"]
    core = report.slices["core"]
    assert cont["n"] == 2 and core["n"] == 1
    # Two-doc corpus: everything sits in the top-5 window, so recall@5 is 1.0
    # for every query — the slice split is what's under test here.
    assert cont["recall_at_5"] == 1.0 and core["recall_at_5"] == 1.0
    # MRR separates them: g1 hits at rank 1 (1.0), g3's label ranks second
    # (0.5) → continuity MRR 0.75; core's d2 ranks first → 1.0.
    assert cont["mrr"] == 0.75
    assert core["mrr"] == 1.0


def test_report_dict_and_render_include_slices():
    from evals.memory_recall import render_report

    report = run_memory_recall_eval(load_corpus(), load_gold_set())
    d = report.to_dict()
    assert "slices" in d
    assert d["slices"]["continuity"]["n"] == 8
    assert 0.0 <= d["slices"]["continuity"]["recall_at_5"] <= 1.0
    assert d["slices"]["core"]["n"] == 51

    rendered = render_report(report)
    assert "| slice |" in rendered
    assert "continuity" in rendered

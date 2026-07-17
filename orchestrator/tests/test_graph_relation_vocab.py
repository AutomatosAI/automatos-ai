"""Knowledge-graph relation vocabulary — the LLM extraction path snaps every
free-text relation to a bounded canonical set while preserving the original
phrasing as a label.

Why this exists: LLM extraction used to emit an open relation vocabulary, so a
workspace graph accrued thousands of singleton relation strings — the legend
flooded and the graph-neighbors ``relation_filter`` tool became unusable. These
tests pin the deterministic (LLM-free) canonicaliser that fixes it.

Pure — no DB, no network, no LLM.
"""
from __future__ import annotations

import pytest

from modules.knowledge.graph_extraction import (
    CANONICAL_RELATIONS,
    canonicalize_relation,
    _normalise_extraction,
)


def test_exact_canonical_passes_through():
    for rel in CANONICAL_RELATIONS:
        canon, label = canonicalize_relation(rel)
        assert canon == rel
        assert label == rel


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("used as", "uses"),
        ("Used By", "uses"),          # case-insensitive
        ("belongs to", "part_of"),
        ("is part of", "part_of"),
        ("requires", "depends_on"),
        ("is caused by", "causes"),
        ("results in", "causes"),
        ("prevents", "blocks"),
        ("is unavailable", "has_property"),
        ("has description", "has_property"),
        ("tracks metric", "measures"),
        ("stopped after", "precedes"),
        ("triggered by", "triggers"),
        ("contrasts with", "references"),
    ],
)
def test_synonyms_map_to_canonical(raw, expected):
    canon, label = canonicalize_relation(raw)
    assert canon == expected
    assert canon in CANONICAL_RELATIONS
    # original phrasing preserved verbatim for display
    assert label == raw


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("depends heavily on", "depends_on"),   # 'depend' stem
        ("directly causes", "causes"),          # 'caus' stem
        ("generated the", "produces"),          # 'generat' stem
        ("governs the policy", "governed_by"),  # 'govern' stem
        ("mentioned briefly", "references"),    # 'mention' stem
    ],
)
def test_keyword_heuristic_catches_unlisted_phrases(raw, expected):
    canon, _ = canonicalize_relation(raw)
    assert canon == expected


@pytest.mark.parametrize("raw", ["", "   ", None, "wibble frobnicate", "qux"])
def test_unknown_falls_back_to_related_to(raw):
    canon, label = canonicalize_relation(raw)
    assert canon == "related_to"
    assert canon in CANONICAL_RELATIONS
    # a blank/None input still yields a usable label
    assert isinstance(label, str) and label


def test_every_output_is_in_the_controlled_set():
    noisy = [
        "used as", "has not produced", "is unavailable", "stopped after",
        "does not exist in", "implies no", "was affected by", "is blind without",
        "reassigns", "resulted in issue", "uses primary model", "due to error",
        "anything at all", "", None,
    ]
    for raw in noisy:
        canon, _ = canonicalize_relation(raw)
        assert canon in CANONICAL_RELATIONS, f"{raw!r} -> {canon} escaped the vocab"


def test_normalise_extraction_canonicalises_edges_and_keeps_label():
    raw = {
        "nodes": [
            {"id": "a", "label": "A", "file_type": "entity"},
            {"id": "b", "label": "B", "file_type": "entity"},
        ],
        "edges": [
            {"source": "a", "target": "b", "relation": "is unavailable"},
        ],
    }
    out = _normalise_extraction(raw, source_file="doc.md")
    edge = out["edges"][0]
    assert edge["relation"] == "has_property"           # snapped to vocab
    assert edge["relation_label"] == "is unavailable"   # original preserved


def test_normalise_extraction_prefers_explicit_relation_label():
    raw = {
        "edges": [
            {"source": "a", "target": "b",
             "relation": "produces", "relation_label": "ships with every order"},
        ],
    }
    edge = _normalise_extraction(raw, source_file="doc.md")["edges"][0]
    assert edge["relation"] == "produces"
    assert edge["relation_label"] == "ships with every order"

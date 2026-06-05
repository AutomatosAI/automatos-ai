"""PRD-142 Wave 2 · WS-I / W2-S10 — RAG primitive tests.

The full RAG round-trip (embed → vector store → retrieve) can only be proven
honestly against real infrastructure, and faking the embedder + vector store
would just test the fakes. So instead of mock-theater, this pins the *real,
deterministic* building blocks of ingest and retrieval — pure functions that
need no embeddings, no vector DB, and no network:

* ``_basic_chunk`` — the fallback ingest-chunking primitive (fixed-size windows,
  drops slivers).
* ``_calculate_content_quality`` — the per-chunk quality score that down-weights
  ASCII-art / boilerplate before the knapsack picks a context budget.
* ``_knapsack_dp`` — the 0/1 knapsack that selects the highest-value chunks that
  fit the token budget. **A real suboptimality is documented below** (see the
  xfail): when ``max_items`` binds before the token budget, it can pick a
  lower-value set. Flagged for fix — the test net surfaced it.
* ``_format_context`` — assembles selected chunks into the prompt context block.

All four are exercised on a bare instance (``__new__`` bypasses the DB-reading
``RAGConfig`` in ``__init__``) since none of them touch instance state.
"""

from __future__ import annotations

import os
import sys
import types

for _k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
    os.environ.setdefault(_k, "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")

# modules/rag/__init__ pulls camelot (optional PDF dep, absent in test env).
sys.modules.setdefault("camelot", types.ModuleType("camelot"))

import pytest  # noqa: E402

from modules.rag.service import RAGService  # noqa: E402


def _svc() -> RAGService:
    """Bare RAGService — bypasses __init__ (which constructs RAGConfig and
    reads system_settings from the DB). The methods under test are pure and
    never touch instance state."""
    return RAGService.__new__(RAGService)


# =========================================================== _basic_chunk
# Fixed-size ingest chunking; slivers (≤50 stripped chars) are dropped.


def test_basic_chunk_splits_into_fixed_windows():
    chunks = _svc()._basic_chunk("x" * 1200)  # default chunk_size=500
    assert len(chunks) == 3
    assert chunks[0]["content"] == "x" * 500
    assert chunks[-1]["content"] == "x" * 200
    assert all(c["metadata"] == {} for c in chunks)


def test_basic_chunk_drops_short_sliver_tail():
    # 540 chars → [0:500] kept, [500:540] is a 40-char sliver (≤50) → dropped.
    assert len(_svc()._basic_chunk("x" * 540)) == 1
    # 560 chars → tail is 60 chars (>50) → kept.
    assert len(_svc()._basic_chunk("x" * 560)) == 2


def test_basic_chunk_drops_entirely_short_input():
    assert _svc()._basic_chunk("x" * 30) == []


# ============================================= _calculate_content_quality
# 0.0–1.0 score. ASCII-art penalised; short valid content scored fairly.


@pytest.mark.parametrize(
    "text,expected",
    [
        ("   ", 0.1),                       # empty / whitespace
        (" ".join(["w"] * 3), 0.5),         # <5 words
        (" ".join(["w"] * 15), 0.7),        # <20 words
        (" ".join(["w"] * 30), 0.85),       # <50 words
        (" ".join(["w"] * 60), 1.0),        # ≥50 words
    ],
)
def test_content_quality_word_count_bands(text, expected):
    assert _svc()._calculate_content_quality(text) == expected


def test_content_quality_penalises_ascii_art_heavily():
    art = "│" * 20 + "x" * 60  # >15% box-drawing chars
    assert _svc()._calculate_content_quality(art) == 0.2


# ================================================================ _knapsack_dp
# 0/1 knapsack: maximise value within token capacity and max_items.


def test_knapsack_rejects_degenerate_params():
    s = _svc()
    assert s._knapsack_dp([], [], 10, 5) == []
    assert s._knapsack_dp([1.0], [1], 0, 5) == []      # zero capacity
    assert s._knapsack_dp([1.0], [1], 10, 0) == []     # zero max_items


def test_knapsack_picks_optimal_set_within_capacity():
    # items (value, weight): A(10,5) B(40,4) C(30,6) D(50,3); cap=10.
    # Optimal = B+D (value 90, weight 7). A+B+D would weigh 12 > 10.
    selected = _svc()._knapsack_dp([10, 40, 30, 50], [5, 4, 6, 3], 10, 10)
    assert selected == [1, 3]


def test_knapsack_respects_capacity_bound():
    # Total weight of all items (18) exceeds capacity (10): result must fit.
    weights = [5, 4, 6, 3]
    selected = _svc()._knapsack_dp([10, 40, 30, 50], weights, 10, 10)
    assert sum(weights[i] for i in selected) <= 10


def test_knapsack_honours_max_items_count():
    # With max_items=2, never return more than 2 indices.
    selected = _svc()._knapsack_dp([10, 40, 30, 50], [5, 4, 6, 3], 10, 2)
    assert len(selected) <= 2


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN DEFECT (PRD-142 W2-S10): _knapsack_dp is suboptimal when "
        "max_items binds before the token budget. For values=[10,40,30,50] "
        "weights=[5,4,6,3] cap=10 max_items=1 it returns index 0 (value 10) "
        "instead of index 3 (value 50). The 2D DP + greedy item_count tracking "
        "does not solve the 3-constraint knapsack its docstring promises "
        "(dp[i][w][k]). Flagged for fix; remove this marker once corrected."
    ),
)
def test_knapsack_is_optimal_when_max_items_binds():
    # The single most valuable item that fits cap=10 is D (index 3, value 50).
    selected = _svc()._knapsack_dp([10, 40, 30, 50], [5, 4, 6, 3], 10, 1)
    assert selected == [3]


# ============================================================= _format_context


def test_format_context_empty_is_explicit_sentinel():
    assert _svc()._format_context([], "anything") == "No relevant context found."


def test_format_context_renders_sources_with_relevance():
    out = _svc()._format_context(
        [{"source_file": "guide.md", "similarity": 0.9, "content": "the answer"}],
        "my query",
    )
    assert "## Retrieved Context for: my query" in out
    assert "### Source 1: guide.md (relevance: 90%)" in out
    assert "the answer" in out

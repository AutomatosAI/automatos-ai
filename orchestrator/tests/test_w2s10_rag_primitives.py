"""PRD-142 Wave 2 · WS-I / W2-S10 — RAG primitive tests.

The full RAG round-trip (embed → vector store → retrieve) can only be proven
honestly against real infrastructure, and faking the embedder + vector store
would just test the fakes. So instead of mock-theater, this pins the *real,
deterministic* building blocks of ingest and retrieval — pure functions that
need no embeddings, no vector DB, and no network:

* ``_basic_chunk`` — the fallback ingest-chunking primitive (fixed-size windows,
  drops slivers).
* ``_calculate_content_quality`` — the per-chunk quality score that down-weights
  ASCII-art / boilerplate before the budgeter picks a context budget.
* ``_format_context`` — assembles selected chunks into the numbered-citation
  prompt context block (PRD-157 S3).

PRD-157 S3/S4 replaced the pure-Python knapsack DP with the whole-chunk token
budgeter (``modules/rag/budget.py``); its selection behaviour — including the
case the old knapsack got wrong — is covered by ``tests/test_context_budget.py``.

These are exercised on a bare instance (``__new__`` bypasses the DB-reading
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


# ================================================================ token budgeter
# PRD-157 S3/S4: the pure-Python knapsack DP was replaced by the whole-chunk
# token budgeter (modules/rag/budget.py). Selection/boundary/ordering behaviour —
# including the previously-xfailed "optimal when max_items binds" case, which the
# budgeter handles by score-ordered greedy selection — is covered by
# tests/test_context_budget.py. The knapsack primitive no longer exists.


# ============================================================= _format_context


def test_format_context_empty_is_explicit_sentinel():
    assert _svc()._format_context([], "anything") == "No relevant context found."


def test_format_context_renders_numbered_citations():
    # PRD-157 S3: numbered citations [1]..[n] + a source map replace the old
    # "### Source i: file (relevance: NN%)" headers.
    out = _svc()._format_context(
        [{"source_file": "guide.md", "similarity": 0.9, "content": "the answer", "document_id": 7}],
        "my query",
    )
    assert "## Retrieved context for: my query" in out
    assert "[1] (source: guide.md)" in out
    assert "the answer" in out
    assert "Sources (cite as [n]):" in out

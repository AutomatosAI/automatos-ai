"""PRD-157 S3 — token-budgeted context assembly + numbered citations.

Pure unit suite (no DB): whole-chunk boundaries, ordering by score, max_chunks,
oversized-first-chunk guarantee, citation numbering + source map, and
token-boundary truncation.
"""

from __future__ import annotations

from modules.rag.budget import (
    BudgetedSelection,
    assemble_with_citations,
    count_tokens,
    resolve_budget,
    select_within_budget,
    truncate_to_token_budget,
)


def _chunk(content, score, source="a.txt", doc_id=1, tokens=None):
    c = {"content": content, "similarity": score, "source_file": source, "document_id": doc_id}
    if tokens is not None:
        c["tokens"] = tokens
    return c


class TestSelectWithinBudget:
    def test_orders_by_score_desc(self):
        cands = [_chunk("low", 0.1, tokens=10), _chunk("high", 0.9, tokens=10), _chunk("mid", 0.5, tokens=10)]
        sel = select_within_budget(cands, max_tokens=1000)
        assert [c["content"] for c in sel.chunks] == ["high", "mid", "low"]

    def test_whole_chunk_boundary_stops_before_overflow(self):
        # budget 25; chunks 10+10+10 -> first two fit (20), third would hit 30 -> dropped
        cands = [_chunk("a", 0.9, tokens=10), _chunk("b", 0.8, tokens=10), _chunk("c", 0.7, tokens=10)]
        sel = select_within_budget(cands, max_tokens=25)
        assert [c["content"] for c in sel.chunks] == ["a", "b"]
        assert sel.total_tokens == 20
        assert sel.dropped == 1

    def test_never_splits_a_chunk(self):
        cands = [_chunk("a", 0.9, tokens=10), _chunk("big", 0.8, tokens=100)]
        sel = select_within_budget(cands, max_tokens=15)
        # 'a' fits (10); 'big' would overflow -> excluded whole, not sliced
        assert [c["content"] for c in sel.chunks] == ["a"]

    def test_always_includes_top_chunk_even_if_oversized(self):
        cands = [_chunk("huge", 0.9, tokens=100), _chunk("small", 0.8, tokens=5)]
        sel = select_within_budget(cands, max_tokens=10)
        # top chunk always surfaces; the smaller one then overflows
        assert sel.chunks[0]["content"] == "huge"
        assert len(sel.chunks) == 1

    def test_max_chunks_cap(self):
        cands = [_chunk(str(i), 1.0 - i / 100, tokens=1) for i in range(10)]
        sel = select_within_budget(cands, max_tokens=10_000, max_chunks=3)
        assert len(sel.chunks) == 3
        assert sel.dropped == 7

    def test_counts_tokens_when_not_cached(self):
        cands = [_chunk("some real words here", 0.9)]  # no tokens key
        sel = select_within_budget(cands, max_tokens=10_000)
        assert sel.total_tokens == count_tokens("some real words here")

    def test_empty(self):
        sel = select_within_budget([], max_tokens=100)
        assert isinstance(sel, BudgetedSelection)
        assert sel.chunks == [] and sel.total_tokens == 0


class TestAssembleWithCitations:
    def test_numbered_sources_and_map(self):
        chunks = [
            _chunk("first", 0.9, source="alpha.pdf", doc_id=11),
            _chunk("second", 0.8, source="beta.pdf", doc_id=22),
        ]
        text, smap = assemble_with_citations(chunks, query="q")
        assert "[1] (source: alpha.pdf)" in text
        assert "[2] (source: beta.pdf)" in text
        assert "Sources (cite as [n]):" in text
        assert "[1] alpha.pdf (doc 11)" in text
        assert smap[0] == {"citation": 1, "source_file": "alpha.pdf", "document_id": 11, "score": 0.9}
        assert smap[1]["document_id"] == 22

    def test_maps_real_doc_ids_in_order(self):
        chunks = [_chunk("c", 0.5, doc_id=99)]
        _, smap = assemble_with_citations(chunks)
        assert smap[0]["citation"] == 1
        assert smap[0]["document_id"] == 99

    def test_empty_is_honest(self):
        text, smap = assemble_with_citations([])
        assert text == "No relevant context found."
        assert smap == []


class TestTruncateToTokenBudget:
    def test_under_budget_unchanged(self):
        text = "short text"
        assert truncate_to_token_budget(text, 1000) == text

    def test_over_budget_truncated_on_token_boundary(self):
        text = "word " * 500  # ~500+ tokens
        out = truncate_to_token_budget(text, 50)
        assert out.endswith("... (truncated)")
        assert count_tokens(out) <= 50 + count_tokens("\n... (truncated)") + 2

    def test_zero_budget_returns_input(self):
        assert truncate_to_token_budget("abc", 0) == "abc"


class TestResolveBudget:
    def test_explicit_wins(self):
        assert resolve_budget(1234) == 1234

    def test_model_window_fraction_with_floor(self):
        # no DB -> get_context_window returns the 128k default; fraction applied
        b = resolve_budget(None, model="gpt-4o")
        assert b >= 1000  # floor
        assert b == int(128_000 * 0.35)

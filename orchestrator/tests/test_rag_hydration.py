"""PRD-154 S1 — RAG full-chunk hydration is batched (one SQL round-trip) and
consumed by the retrieval/format primitives.

Root cause (reports/PLATFORM_DEEP_REVIEW_2026-06.md §2.1): candidates carry only
the 500-char S3 metadata preview; the full chunk text lives in
``document_chunks.content``. ``_expand_to_parent_context`` hydrated it — but via
one asyncpg query PER candidate (N+1) into ``expanded_content`` that NOTHING read,
because the retrieval primitives (``_optimize_with_context_optimizer`` :544 and
``_basic_retrieval`` :754) kept reading the 500-char ``content`` preview. So the
formatter's 4000-char/doc budget never filled.

These tests pin the fix with NO DB and NO network: ``asyncpg.connect`` is faked so
we can both (a) count the hydration round-trips — exactly one, regardless of
candidate count — and (b) prove the read-sites consume the hydrated full text,
not the preview.
"""
from __future__ import annotations

import os
import sys
import types

# --- collection-order guard (Ralph mandate for module-level modules.* import) -
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")
# modules/rag/__init__ pulls camelot (optional PDF dep, absent in test env).
sys.modules.setdefault("camelot", types.ModuleType("camelot"))

import tests.conftest as _conftest  # noqa: E402
_conftest._restore_real_app_modules()

import asyncio  # noqa: E402
from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402

import pytest  # noqa: E402

from modules.rag.service import RAGConfig, RAGService  # noqa: E402

PREVIEW_LEN = 500
PREVIEW = "p" * PREVIEW_LEN


def _expander() -> RAGService:
    """Bare service with parent-child expansion ON and a window of 1 — bypasses
    the DB-reading ``__init__``/``RAGConfig.__post_init__``."""
    svc = RAGService.__new__(RAGService)
    cfg = RAGConfig.__new__(RAGConfig)
    cfg.parent_child_expansion = True
    cfg.expansion_window = 1
    svc.config = cfg
    return svc


def _fake_conn(rows):
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=rows)
    conn.close = AsyncMock()
    return conn


# ====================================================== batched single query ===


def test_hydration_is_a_single_batched_query():
    """Three candidates across two documents → exactly one SQL round-trip, and
    every candidate is hydrated with full chunk text well beyond the preview."""
    svc = _expander()
    candidates = [
        {"document_id": "10", "content": PREVIEW, "metadata": {"chunk_index": 5}},
        {"document_id": "10", "content": PREVIEW, "metadata": {"chunk_index": 6}},
        {"document_id": "20", "content": PREVIEW, "metadata": {"chunk_index": 0}},
    ]
    # Full chunk text in PG — each well past the 500-char preview.
    full = {
        (10, 4): "A" * 800, (10, 5): "B" * 800, (10, 6): "C" * 800, (10, 7): "D" * 800,
        (20, 0): "E" * 800, (20, 1): "F" * 800,
    }
    rows = [{"document_id": d, "chunk_index": c, "content": t} for (d, c), t in full.items()]
    conn = _fake_conn(rows)

    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
        out = asyncio.run(svc._expand_to_parent_context(candidates, expand_window=1))

    # The whole point: ONE hydration round-trip, not one-per-candidate.
    assert conn.fetch.await_count == 1

    for cand in out:
        body = cand["expanded_content"]
        assert len(body) > PREVIEW_LEN          # hydrated, not the 500-char cap
        assert PREVIEW not in body              # not the preview text


def test_hydration_falls_back_to_preview_when_parent_absent():
    """A candidate whose parent chunk is not in PG keeps its preview (and the
    single round-trip still holds)."""
    svc = _expander()
    candidates = [{"document_id": "99", "content": PREVIEW, "metadata": {"chunk_index": 3}}]
    conn = _fake_conn([])  # nothing comes back from PG

    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
        out = asyncio.run(svc._expand_to_parent_context(candidates, expand_window=1))

    assert conn.fetch.await_count == 1
    assert out[0]["expanded_content"] == PREVIEW


# ===================================================== read-sites consume it ===


def test_basic_retrieval_consumes_hydrated_text():
    """``_basic_retrieval`` (:754) must surface ``expanded_content``, not the
    500-char preview, into the returned chunk."""
    svc = RAGService.__new__(RAGService)
    candidates = [{
        "document_id": "10",
        "content": PREVIEW,
        "expanded_content": "F" * 3000,
        "similarity": 0.9,
        "source_file": "doc.md",
        "metadata": {"chunk_index": 5},
    }]
    result = svc._basic_retrieval("q", candidates, max_chunks=5, max_tokens=100_000)
    assert result.chunks
    body = result.chunks[0]["content"]
    assert body == "F" * 3000
    assert len(body) > PREVIEW_LEN


def test_optimizer_consumes_hydrated_text():
    """``_optimize_with_context_optimizer`` (:544) must surface
    ``expanded_content`` into the selected chunk."""
    from modules.search import ContextItem

    svc = RAGService.__new__(RAGService)
    svc._ContextItem = ContextItem  # the only instance attr the method needs
    candidates = [{
        "document_id": "10",
        "content": PREVIEW,
        "expanded_content": "G" * 3000,
        "similarity": 0.9,
        "source_file": "doc.md",
        "metadata": {"chunk_index": 5},
    }]
    result = asyncio.run(
        svc._optimize_with_context_optimizer(
            "q", candidates, max_chunks=5, max_tokens=100_000, diversity=0.3
        )
    )
    assert result.chunks
    assert result.chunks[0]["content"] == "G" * 3000

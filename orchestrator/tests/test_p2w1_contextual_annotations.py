"""PRD-188 S2: contextual chunk annotations at ingestion (flag-OFF by default).

The chunker emits ~500-char chunks with no situating context — the exact gap
Anthropic's contextual retrieval attacks. These tests pin the annotator's
contract: annotation is prepended to the content that gets embedded (and
stored — so the tsvector trigger indexes it for the BM25 leg), the annotation
is inspectable in chunk metadata, failure is LOUD and falls back to raw text
without failing the ingest, re-annotation is idempotent, and the originals are
never mutated (new objects only).

The LLM boundary is mocked — no network, no DB. The manager-side flag gate
(default OFF until the corpus reprocess runs) is asserted at source level:
annotate_chunks must be reachable only under RAG_CONTEXTUAL_ANNOTATIONS_ENABLED
and must run BEFORE chunk_texts is built for embedding.
"""
import asyncio
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

import modules.rag.ingestion.contextual_annotator as annotator_mod  # noqa: E402
from modules.rag.ingestion.contextual_annotator import (  # noqa: E402
    ANNOTATION_ERROR_KEY,
    ANNOTATION_METADATA_KEY,
    MAX_PARENT_CHARS,
    annotate_chunks,
)
from modules.rag.ingestion.manager import DocumentChunk  # noqa: E402


class FakeLLM:
    """Captures prompts; returns a fixed annotation (or raises)."""

    def __init__(self, annotation="This chunk covers the billing renewal terms.", error=None):
        self.annotation = annotation
        self.error = error
        self.prompts = []

    async def generate_response(self, messages, tools=None):
        self.prompts.append(messages[0]["content"])
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.annotation)


def _chunk(content="The subscription renews every March 1st.", metadata=None, idx=0):
    return DocumentChunk(document_id=1, chunk_index=idx, content=content, metadata=metadata)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# The annotation contract
# ---------------------------------------------------------------------------

def test_annotation_prepended_and_stored_in_metadata():
    llm = FakeLLM(annotation="  Situates the chunk in the Acme contract.  ")
    original = _chunk()
    out = _run(annotate_chunks([original], "full parent document text", llm_provider=llm))

    assert len(out) == 1
    annotated = out[0]
    # Prepended before the raw text — this is the content that gets embedded
    # AND persisted (the tsvector trigger indexes the preface too).
    assert annotated.content == (
        "Situates the chunk in the Acme contract.\n\nThe subscription renews every March 1st."
    )
    # Inspectable + re-annotation detectable.
    assert annotated.metadata[ANNOTATION_METADATA_KEY] == "Situates the chunk in the Acme contract."
    # The prompt saw both the parent document and the chunk.
    assert "full parent document text" in llm.prompts[0]
    assert "renews every March 1st" in llm.prompts[0]


def test_originals_are_never_mutated():
    llm = FakeLLM()
    original = _chunk(metadata={"page_start": 3})
    before_content = original.content
    before_meta = dict(original.metadata)

    out = _run(annotate_chunks([original], "parent", llm_provider=llm))

    assert original.content == before_content
    assert original.metadata == before_meta
    assert out[0] is not original
    assert out[0].metadata["page_start"] == 3  # existing metadata preserved on the copy


def test_annotation_failure_is_loud_not_silent(caplog):
    llm = FakeLLM(error=RuntimeError("annotation model down"))
    original = _chunk()
    with caplog.at_level(logging.WARNING):
        out = _run(annotate_chunks([original], "parent", llm_provider=llm))

    assert any("falling back to raw text" in r.message for r in caplog.records)
    fallen_back = out[0]
    assert fallen_back.content == original.content  # raw text embeds — ingest never fails
    assert "annotation model down" in fallen_back.metadata[ANNOTATION_ERROR_KEY]
    assert ANNOTATION_METADATA_KEY not in fallen_back.metadata


def test_empty_annotation_falls_back_to_raw(caplog):
    llm = FakeLLM(annotation="   ")
    original = _chunk()
    with caplog.at_level(logging.WARNING):
        out = _run(annotate_chunks([original], "parent", llm_provider=llm))
    assert out[0].content == original.content
    assert out[0].metadata[ANNOTATION_ERROR_KEY] == "empty annotation"


def test_already_annotated_chunk_is_idempotent_passthrough():
    llm = FakeLLM()
    already = _chunk(
        content="Old preface.\n\nThe subscription renews.",
        metadata={ANNOTATION_METADATA_KEY: "Old preface."},
    )
    out = _run(annotate_chunks([already], "parent", llm_provider=llm))
    assert out[0] is already  # untouched — no double-prepend
    assert llm.prompts == []  # and no spend


def test_no_llm_available_passes_through_loudly(monkeypatch, caplog):
    monkeypatch.setattr(annotator_mod, "_default_llm", lambda: None)
    original = _chunk()
    with caplog.at_level(logging.WARNING):
        out = _run(annotate_chunks([original], "parent"))
    assert out == [original]
    assert any("no LLM available" in r.message for r in caplog.records)


def test_parent_document_is_capped_in_prompt():
    llm = FakeLLM()
    parent = ("A" * MAX_PARENT_CHARS) + "TAIL-SENTINEL"
    _run(annotate_chunks([_chunk()], parent, llm_provider=llm))
    assert "TAIL-SENTINEL" not in llm.prompts[0]


def test_order_preserved_across_mixed_outcomes():
    class FlakyLLM(FakeLLM):
        async def generate_response(self, messages, tools=None):
            self.prompts.append(messages[0]["content"])
            if "second" in messages[0]["content"]:
                raise RuntimeError("boom")
            return SimpleNamespace(content="ctx")

    chunks = [_chunk("first chunk text here.", idx=0), _chunk("second chunk text here.", idx=1)]
    out = _run(annotate_chunks(chunks, "parent", llm_provider=FlakyLLM()))
    assert [c.chunk_index for c in out] == [0, 1]
    assert out[0].content.startswith("ctx\n\n")
    assert out[1].content == "second chunk text here."  # failed one embeds raw


# ---------------------------------------------------------------------------
# The manager-side gate (source-level: no DB path to drive purely)
# ---------------------------------------------------------------------------

def test_manager_gates_on_flag_and_annotates_before_embedding():
    """The hook must be (a) gated by RAG_CONTEXTUAL_ANNOTATIONS_ENABLED — so
    flag-off embeds raw chunks byte-identically — and (b) placed BEFORE
    chunk_texts is built, so the annotated content is what gets embedded."""
    manager_src = (
        Path(_orchestrator_root) / "modules" / "rag" / "ingestion" / "manager.py"
    ).read_text()

    gate = manager_src.find("RAG_CONTEXTUAL_ANNOTATIONS_ENABLED")
    call = manager_src.find("annotate_chunks(filtered_chunks")
    embed_input = manager_src.find("chunk_texts = [c.content for c in filtered_chunks]")

    assert gate != -1, "flag gate missing from ingestion manager"
    assert call != -1, "annotator is not wired into ingestion"
    assert gate < call < embed_input, (
        "annotation must be flag-gated and must run before chunk_texts is "
        "built for embedding"
    )


def test_flag_defaults_off(monkeypatch):
    """Default OFF until the corpus reprocess has run (a half-annotated store
    must never serve live retrieval)."""
    import core.llm.manager as llm_manager_mod
    from config import config as app_config

    monkeypatch.setattr(
        llm_manager_mod, "get_system_setting", lambda group, key, default=None: default
    )
    monkeypatch.delenv("RAG_CONTEXTUAL_ANNOTATIONS_ENABLED", raising=False)
    assert app_config.RAG_CONTEXTUAL_ANNOTATIONS_ENABLED is False

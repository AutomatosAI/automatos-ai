"""F086 (night 3) — ingestion keeps every line of a document, and says when it did not.

The chunker dropped any segment under 100 characters at a topic shift: a price,
a date, a sign-off, a banned word alone on its line vanished while the document
read "completed" (brand-voice.md kept 461 of 1,304 characters; the same guide in
full sentences kept 92%). CSVs lost rows and their headers. Now nothing is
dropped, a spreadsheet is chunked by rows with its header on every chunk, and
the share kept is measured and shown to the owner when it falls short.
"""
from __future__ import annotations

import asyncio
import csv
from types import SimpleNamespace as NS

import pytest

from modules.rag.ingestion import manager as mgr
from modules.rag.ingestion.coverage import kept_pct
from modules.rag.ingestion.manager import DocumentChunk, DocumentManager, DocumentProcessor, DocumentType

BRAND_VOICE = """# Harbourline brand voice

## Words we like
roast day · the club · brew · lot · the crew · cheers · proper · have a go · tasting notes

## Words we don't use
artisanal · curated · elevate · premium · journey · indulge · passion

Never write: arch

Club price: £21 a month, £30 for the Christmas box

- Say what the coffee tastes like in plain words ("stewed plums", "brown sugar") — no wine-snob lists of 9 notes.

Sign off every email:
Cheers, the Harbourline crew
(Not "Best wishes", not "Warmly", not "Stay caffeinated" — someone actually wrote that once.)
"""


def _normal(text):
    return " ".join(text.split())


def test_a_price_a_sign_off_and_a_banned_word_alone_on_their_lines_all_survive():
    chunks = [c.content for c in DocumentProcessor().chunk_document(BRAND_VOICE, DocumentType.MARKDOWN, {"document_id": 1})]
    assert kept_pct(BRAND_VOICE, chunks) == 100
    held = _normal(" ".join(chunks))
    for line in ("Never write: arch", "Club price: £21 a month, £30 for the Christmas box",
                 "Cheers, the Harbourline crew"):
        assert _normal(line) in held, line
    for line in (l for l in BRAND_VOICE.splitlines() if l.strip()):
        assert all(word in held for word in line.split()), line


def test_no_chunking_strategy_drops_a_short_segment():
    from modules.rag.chunking.semantic_chunker import ChunkingStrategy, SemanticChunker

    for strategy in (ChunkingStrategy.TOPIC_COHERENCE, ChunkingStrategy.SEMANTIC_SIMILARITY,
                     ChunkingStrategy.INFORMATION_DENSITY):
        chunker = SemanticChunker(strategy=strategy, target_chunk_size=500, min_chunk_size=100, max_chunk_size=1500)
        chunks = [c.content for c in chunker.chunk_text(BRAND_VOICE, document_id="1")]
        assert kept_pct(BRAND_VOICE, chunks) == 100, strategy


def test_a_30_row_csv_keeps_every_row_with_its_header(tmp_path):
    path = tmp_path / "cafe.csv"
    rows = [["cafe", "price", "notes"]] + [
        [f"cafe {i}", f"£{i}.50", f"Delivers on Tuesday. Pays in {i} days."] for i in range(1, 31)
    ]
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    processor = DocumentProcessor()
    text = processor._extract_spreadsheet_csv(str(path))
    chunks = [c.content for c in processor.chunk_document(text, DocumentType.CSV, {"document_id": 2})]
    header = "| cafe | price | notes |"
    assert chunks and all(chunk.startswith(header + "\n| --- | --- | --- |") for chunk in chunks)
    for i in range(1, 31):
        row = f"| cafe {i} | £{i}.50 | Delivers on Tuesday. Pays in {i} days. |"
        assert sum(row in chunk.splitlines() for chunk in chunks) == 1, row   # whole, both sentences, once
    assert kept_pct(text, chunks) == 100


def test_kept_is_measured_by_words_and_overlap_never_counts_twice():
    assert kept_pct("a b c d", ["a b"]) == 50
    assert kept_pct("a b c", ["a b", "b c"]) == 100
    assert kept_pct("a a b", ["a b"]) == 66
    assert kept_pct("", []) == 100


# ── ingestion records it ────────────────────────────────────────────────────

class _Cursor:
    def __init__(self, log):
        self.log = log

    def execute(self, sql, params=None):
        sql = " ".join(sql.split())
        self.log.append((sql, params))
        self._row = ("ws-1", []) if sql.startswith("SELECT workspace_id, team_access FROM documents") else None

    def fetchone(self):
        return self._row

    def close(self):
        pass


class _Conn:
    def __init__(self, log):
        self.log = log

    def cursor(self):
        return _Cursor(self.log)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


def _ingest(monkeypatch, text, chunk_texts):
    import modules.rag.ingestion.multimodal as multimodal
    from config import config

    log = []
    manager = DocumentManager.__new__(DocumentManager)
    manager.db_config, manager.workspace_id = {}, None
    manager.enable_multimodal, manager.use_s3_vectors, manager._s3_backend = False, False, None
    manager.processor = NS(
        extract_text_from_file=lambda _path: text,
        chunk_document=lambda _t, _type, _meta: [
            DocumentChunk(document_id=9, chunk_index=i, content=c) for i, c in enumerate(chunk_texts)
        ],
    )
    monkeypatch.setattr(manager, "_ensure_database_initialized", lambda: None)
    monkeypatch.setattr(manager, "_emit_ingest_heartbeat", lambda **_kw: None)

    async def embed(texts):
        return [[0.1]] * len(texts)

    async def persist(conn, cursor, **kw):
        return []

    monkeypatch.setattr(manager, "_generate_embeddings_batch", embed)
    monkeypatch.setattr(manager, "_persist_chunks_and_vectors", persist)
    monkeypatch.setattr(mgr.psycopg2, "connect", lambda **_kw: _Conn(log))
    monkeypatch.setattr(mgr, "_KB_ENTITIES_EXISTS", False)
    monkeypatch.setattr(type(config), "RAG_CONTEXTUAL_ANNOTATIONS_ENABLED", property(lambda _self: False))
    monkeypatch.setattr(multimodal, "FormulaProcessor", lambda: NS(extract_formulas_from_text=lambda _t: []))
    asyncio.run(manager._process_document(9, "/tmp/plan.md", DocumentType.MARKDOWN, filename="plan.md"))
    [(sql, params)] = [(s, p) for s, p in log if s.startswith("UPDATE documents SET status")]
    return sql, params


def test_ingestion_records_what_it_kept_and_warns_when_it_is_short(monkeypatch, caplog):
    text = "The box costs thirty four pounds. " * 3 + "Orders open on Monday the second of November. " * 3
    sql, params = _ingest(monkeypatch, text, ["The box costs thirty four pounds. " * 3])
    assert "doc_metadata" in sql and params[0] == "completed"
    assert '"kept_pct": 42' in params[3]
    assert "keeps 42% of its text" in caplog.text

    sql, params = _ingest(monkeypatch, text, [text])
    assert '"kept_pct": 100' in params[3]


# ── the owner sees it ───────────────────────────────────────────────────────

def test_the_list_and_the_page_say_partial_under_the_threshold():
    from api.documents import _coverage_fields

    assert _coverage_fields(NS(doc_metadata={"kept_pct": 61})) == {"kept_pct": 61, "partial": True}
    assert _coverage_fields(NS(doc_metadata={"kept_pct": 100})) == {"kept_pct": 100, "partial": False}
    assert _coverage_fields(NS(doc_metadata=None)) == {}         # ingested before the measure existed


@pytest.mark.parametrize("extension, listed, processed", [
    (".csv", "csv", DocumentType.CSV),
    (".xlsx", "spreadsheet", DocumentType.XLSX),
    (".docx", "document", DocumentType.DOCX),
    (".md", "markdown", DocumentType.MARKDOWN),
])
def test_an_upload_is_listed_and_chunked_as_what_it_is(extension, listed, processed):
    from api.documents import UPLOAD_FILE_TYPES, processing_type

    assert UPLOAD_FILE_TYPES[extension] == listed and processing_type(listed) == processed

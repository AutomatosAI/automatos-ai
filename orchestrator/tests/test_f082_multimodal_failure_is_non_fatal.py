"""F082 — a failed multimodal step never takes the rest of the document with it.

The ingestion manager runs table, formula and entity extraction on the same
psycopg2 connection that later writes the chunks and the COMPLETED status. The
fake below behaves as psycopg2 did on Postgres 16 when measured for this fix: a
failed statement aborts the transaction, every later statement fails, commit()
silently rolls back, rollback() recovers.

Before the fix:
* a document with two tables (kb_tables allows one per catalog item) aborted the
  connection, its formulas then failed too, and the next commit silently threw
  both away;
* a failure that escaped the block (the catalog insert itself) left the chunk
  writes and the status update on a dead connection: the document was marked
  failed and its vectors deleted.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

from modules.rag.ingestion import manager as mgr
from modules.rag.ingestion.manager import DocumentChunk, DocumentManager, DocumentStatus, DocumentType

# Single-column tables: the manager's markdown regex matches no other kind (a
# multi-column separator, |---|---|, never matches — flagged, not changed here).
ONE_TABLE = "# Prices\n\n| sku |\n|---|\n| A1 |\n| B2 |\n\nMargins are thin this quarter.\n"
TWO_TABLES = ONE_TABLE + "\n| region |\n|---|\n| north |\n| south |\n"


class Aborted(Exception):
    """psycopg2.errors.InFailedSqlTransaction"""


class UniqueViolation(Exception):
    """psycopg2.errors.UniqueViolation"""


class FakeDb:
    def __init__(self, fail_on=lambda sql, db: None):
        self.fail_on = fail_on
        self.statements, self.pending, self.committed = [], [], []
        self.aborted = False

    def connect(self, **_kw):
        return FakeConn(self)


class FakeConn:
    def __init__(self, db):
        self.db = db

    def cursor(self):
        return FakeCursor(self.db)

    def commit(self):
        if self.db.aborted:             # COMMIT of an aborted transaction is a silent ROLLBACK
            self.db.pending.clear()
            self.db.aborted = False
            return
        self.db.committed.extend(self.db.pending)
        self.db.pending.clear()

    def rollback(self):
        self.db.pending.clear()
        self.db.aborted = False

    def close(self):
        pass


class FakeCursor:
    def __init__(self, db):
        self.db, self._row = db, None

    def execute(self, sql, params=None):
        sql = " ".join(sql.split())
        if self.db.aborted:
            raise Aborted("current transaction is aborted, commands ignored until end of transaction block")
        failure = self.db.fail_on(sql, self.db)
        if failure is not None:
            self.db.aborted = True
            raise failure
        self.db.statements.append(sql)
        self._row = ("ws-1", []) if sql.startswith("SELECT workspace_id, team_access FROM documents") else None
        if not sql.startswith("SELECT"):
            self.db.pending.append((sql, params))

    def fetchone(self):
        return self._row

    def close(self):
        pass


def _second_table_violates_unique(sql, db):
    if sql.startswith("INSERT INTO kb_tables") and any(s.startswith("INSERT INTO kb_tables") for s in db.statements):
        return UniqueViolation('duplicate key value violates unique constraint "kb_tables_knowledge_item_id_key"')
    return None


def _catalog_insert_fails(sql, db):
    if sql.startswith("INSERT INTO knowledge_items"):
        return UniqueViolation("the catalog insert failed on the server")
    return None


def _process(monkeypatch, db, text):
    import modules.rag.ingestion.multimodal as multimodal
    from config import config

    manager = DocumentManager.__new__(DocumentManager)
    manager.db_config, manager.workspace_id = {}, None
    manager.enable_multimodal, manager.use_s3_vectors, manager._s3_backend = False, False, None
    manager.processor = NS(
        extract_text_from_file=lambda _path: text,
        chunk_document=lambda _text, _type, _meta: [
            DocumentChunk(document_id=7, chunk_index=0, content="Margins are thin this quarter across both regions.")
        ],
    )
    monkeypatch.setattr(manager, "_ensure_database_initialized", lambda: None)
    monkeypatch.setattr(manager, "_emit_ingest_heartbeat", lambda **_kw: None)

    async def embed(texts):
        return [[0.1, 0.2]] * len(texts)

    persisted = {}

    async def persist(conn, cursor, **kw):
        if db.aborted:                  # the chunk INSERTs would fail exactly like this
            raise Aborted("chunk INSERT on an aborted transaction")
        persisted["chunks"] = len(kw["filtered_chunks"])
        return []

    monkeypatch.setattr(manager, "_generate_embeddings_batch", embed)
    monkeypatch.setattr(manager, "_persist_chunks_and_vectors", persist)
    monkeypatch.setattr(mgr.psycopg2, "connect", db.connect)
    monkeypatch.setattr(mgr, "_KB_ENTITIES_EXISTS", False)
    monkeypatch.setattr(type(config), "RAG_CONTEXTUAL_ANNOTATIONS_ENABLED", property(lambda _self: False))
    monkeypatch.setattr(multimodal, "FormulaProcessor",
                        lambda: NS(extract_formulas_from_text=lambda _t: [{"latex": "m = p - c", "mathml": None}]))

    asyncio.run(manager._process_document(7, "/tmp/prices.md", DocumentType.MARKDOWN, filename="prices.md"))
    return persisted


def _committed(db, prefix):
    return [params for sql, params in db.committed if sql.startswith(prefix)]


def _status_writes(db):
    return [params[0] for sql, params in db.committed if sql.startswith("UPDATE documents SET status")]


def test_a_second_table_no_longer_costs_the_documents_formulas(monkeypatch):
    db = FakeDb(_second_table_violates_unique)
    persisted = _process(monkeypatch, db, TWO_TABLES)
    assert _committed(db, "INSERT INTO kb_formulas"), "the formula step ran on a dead connection"
    assert _committed(db, "INSERT INTO kb_tables") == []       # the failed batch is rolled back, not half-kept
    assert persisted == {"chunks": 1}
    assert _status_writes(db) == [DocumentStatus.COMPLETED.value]


def test_a_failure_that_escapes_the_block_does_not_fail_the_document(monkeypatch):
    db = FakeDb(_catalog_insert_fails)
    persisted = _process(monkeypatch, db, ONE_TABLE)            # raised before the fix: marked FAILED
    assert persisted == {"chunks": 1}
    assert _status_writes(db) == [DocumentStatus.COMPLETED.value]
    assert DocumentStatus.FAILED.value not in _status_writes(db)


def test_the_happy_path_is_unchanged(monkeypatch):
    db = FakeDb()
    persisted = _process(monkeypatch, db, ONE_TABLE)
    assert _committed(db, "INSERT INTO knowledge_items") and _committed(db, "INSERT INTO kb_tables")
    assert _committed(db, "INSERT INTO kb_formulas")
    assert persisted == {"chunks": 1}
    assert _status_writes(db) == [DocumentStatus.COMPLETED.value]

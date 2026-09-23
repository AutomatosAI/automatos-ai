"""F081 — the one-shot that types document catalog rows left with no kb_type.

Runs on real Postgres against TEMP tables, which shadow the real ones for this
connection (pg_temp is searched first), so the test owns every row it reads.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from scripts.repair_catalog_kb_types import repair

DOCUMENT, TABLE = 1, 3


_DROP_TEMP = "DROP TABLE IF EXISTS pg_temp.kb_types, pg_temp.documents, pg_temp.knowledge_items"


@pytest.fixture
def db(test_engine):
    # Temp tables live as long as the pooled connection, not the test: drop them
    # on the way in and out — schema-qualified, so a real table is never touched.
    with test_engine.connect() as conn:
        session = Session(bind=conn)
        session.execute(text(_DROP_TEMP))
        for ddl in ("CREATE TEMP TABLE kb_types (id int, type_name varchar)",
                    "CREATE TEMP TABLE documents (id int)",
                    "CREATE TEMP TABLE knowledge_items (id int, kb_type_id int)"):
            session.execute(text(ddl))
        session.execute(text("INSERT INTO kb_types VALUES (1, 'document'), (3, 'table')"))
        session.execute(text("INSERT INTO documents VALUES (10), (11)"))
        # 10: a document's catalog row left untyped · 11: already typed
        # 12: untyped, its document gone (an orphan) · 13: an extracted table
        session.execute(text("INSERT INTO knowledge_items VALUES (10, NULL), (11, 1), (12, NULL), (13, 3)"))
        yield session
        session.rollback()
        session.execute(text(_DROP_TEMP))
        session.commit()
        session.close()


def _types(db):
    return dict(db.execute(text("SELECT id, kb_type_id FROM knowledge_items ORDER BY id")).all())


def test_a_dry_run_counts_and_changes_nothing(db):
    before = _types(db)
    assert repair(db, dry_run=True) == {"untyped_documents": 1, "orphans_left_alone": 1, "typed": 0}
    assert _types(db) == before


def test_only_a_documents_own_catalog_row_is_typed_and_a_rerun_types_nothing(db):
    assert repair(db, dry_run=False) == {"untyped_documents": 1, "orphans_left_alone": 1, "typed": 1}
    assert _types(db) == {10: DOCUMENT, 11: DOCUMENT, 12: None, 13: TABLE}
    assert repair(db, dry_run=False) == {"untyped_documents": 0, "orphans_left_alone": 1, "typed": 0}


def test_without_a_document_type_it_refuses_and_changes_nothing(db):
    db.execute(text("DELETE FROM kb_types WHERE type_name = 'document'"))
    before = _types(db)
    with pytest.raises(RuntimeError, match="run the migrations"):
        repair(db, dry_run=False)
    assert _types(db) == before

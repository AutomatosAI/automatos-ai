"""
F081: type the document catalog rows written while kb_types was unseeded
=========================================================================

``modules/rag/ingestion/manager.py`` gives every ingested document a catalog
row in ``knowledge_items`` typed ``(SELECT id FROM kb_types WHERE type_name =
'document')``. Before kb_types was seeded (locally: until 2026-09-18 08:42) that
subquery was NULL, so those rows carry no type, and every typed read — the
Knowledge stats by type, search_multimodal — passes them by.

A NULL-typed row is typed ``document`` only when a documents row with the same
id exists: the catalog row IS that document's. A row with no document behind it
(the document was deleted; nothing cascades) is counted, never touched.
Idempotent: a second run types nothing.

Usage::

    cd orchestrator
    python scripts/repair_catalog_kb_types.py --dry-run
    python scripts/repair_catalog_kb_types.py
"""

from __future__ import annotations

import argparse
import os
import sys

# Make orchestrator package imports work when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

_UNTYPED = """
    SELECT
        count(*) FILTER (WHERE EXISTS (SELECT 1 FROM documents d WHERE d.id = ki.id)),
        count(*) FILTER (WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.id = ki.id))
    FROM knowledge_items ki
    WHERE ki.kb_type_id IS NULL
"""

_TYPE_THEM = """
    UPDATE knowledge_items ki
       SET kb_type_id = kt.id
      FROM kb_types kt
     WHERE kt.type_name = 'document'
       AND ki.kb_type_id IS NULL
       AND EXISTS (SELECT 1 FROM documents d WHERE d.id = ki.id)
"""


def repair(db: Session, *, dry_run: bool) -> dict:
    """``{untyped_documents, orphans_left_alone, typed}`` — typed stays 0 on a dry run."""
    untyped, orphans = db.execute(text(_UNTYPED)).one()
    typed = 0
    if untyped and not dry_run:
        if db.execute(text("SELECT 1 FROM kb_types WHERE type_name = 'document'")).first() is None:
            raise RuntimeError("kb_types has no 'document' row — run the migrations first")
        typed = db.execute(text(_TYPE_THEM)).rowcount
        db.commit()
    return {"untyped_documents": untyped, "orphans_left_alone": orphans, "typed": typed}


def main() -> int:
    parser = argparse.ArgumentParser(description="Type document catalog rows left with no kb_type (F081)")
    parser.add_argument("--dry-run", action="store_true", help="Count, change nothing")
    args = parser.parse_args()

    from config import config

    with Session(create_engine(config.DATABASE_URL)) as db:
        summary = repair(db, dry_run=args.dry_run)
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())

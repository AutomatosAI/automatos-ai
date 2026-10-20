"""F443 (a), P256-FIX-N17-11 (night 16): the prompt's absence rule, and the section still fits.

The November price letter quoted the October terms and said the documents held no November
change, after three platform_search_documents calls; the prompt's documents sentence named
search_knowledge and said nothing about saying a thing is absent. It now ends: never tell the
owner a document does not give something until search_knowledge, in the thing's own words, finds
nothing either; platform_search_documents calls do not count.

The section is cut from the end past ``max_tokens`` (head kept, base.py ``truncate``), and the
database sentence (F302) comes last: in a workspace shaped like the night's (ten long file names,
a spreadsheet, one database) the whole section, that sentence included, fits the raised budget.
"""
from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from core.context_guard import count_tokens
from modules.context.sections.documents_inventory import DocumentsInventorySection, documents_summary

TABLES = ("documents", "database_knowledge_sources")
ABSENCE = ("Never tell the owner a document does not give something until search_knowledge, in the thing's own "
           "words, finds nothing either; platform_search_documents calls do not count.")
ONE_DATABASE = "With one database no name is needed."
SPREADSHEET = "A spreadsheet (CSV or Excel) is counted or totalled with code, never searched"


@pytest.fixture
def db(test_engine):
    with test_engine.connect() as conn:
        session = Session(bind=conn)
        for table in TABLES:
            session.execute(text(f"DROP TABLE IF EXISTS pg_temp.{table}"))
            session.execute(text(f"CREATE TEMP TABLE {table} (LIKE public.{table} INCLUDING DEFAULTS)"))
        yield session
        session.rollback()
        for table in TABLES:
            session.execute(text(f"DROP TABLE IF EXISTS pg_temp.{table}"))
        session.commit()
        session.close()


def _night16_workspace(db):
    """Ten long file names (the newest a CSV) and one connected database."""
    ws = str(uuid4())
    for day in range(1, 11):
        name, kind = (f"harbourline-wholesale-price-list-2026-{day:02d}.md", "markdown")
        if day == 10:
            name, kind = "harbourline-green-coffee-stock-autumn-2026.csv", "csv"
        db.execute(text("INSERT INTO documents (id, workspace_id, filename, status, file_type, upload_date) "
                        "VALUES (:id, CAST(:ws AS uuid), :name, 'completed', :kind, "
                        "make_timestamp(2026, 10, :day, 12, 0, 0))"),
                   {"id": day, "ws": ws, "name": name, "kind": kind, "day": day})
    db.execute(text("INSERT INTO database_knowledge_sources "
                    "(id, workspace_id, tenant_id, name, credential_id, dialect, is_active, created_at) "
                    "VALUES (36, CAST(:ws AS uuid), 1, 'harbourline_shop', 1, 'postgresql', TRUE, now())"), {"ws": ws})
    return ws


def test_the_documents_section_fits(db):
    summary = documents_summary(db, _night16_workspace(db))

    assert summary.endswith(ONE_DATABASE)                                # the F302 sentence, whole, last
    assert ABSENCE in summary and SPREADSHEET in summary
    assert summary.index(ABSENCE) < summary.index(SPREADSHEET)            # the rule before the spreadsheet one
    assert count_tokens(summary) <= DocumentsInventorySection.max_tokens
    assert DocumentsInventorySection().truncate(summary, DocumentsInventorySection.max_tokens) == summary


def test_without_a_database_the_rule_ends_the_documents_sentence(db):
    ws = str(uuid4())
    db.execute(text("INSERT INTO documents (id, workspace_id, filename, status) "
                    "VALUES (1, CAST(:ws AS uuid), 'wholesale-terms-2026-10.md', 'completed')"), {"ws": ws})

    assert documents_summary(db, ws).endswith(f"name the file you used. {ABSENCE}")

"""F138 (night 4) — platform_list_documents honours `search` and pages.

B6: a filtered call came back identical to the unfiltered one, the newest page
first. B71: after reading the newest 20, Auto told the owner a document "might
have been deleted". The action had no `search` or `offset`, and the handler read
neither. Now `search` keeps the documents whose name or description contains it
(literally: `_` and `%` in a filename are not wildcards), `offset` pages, and
`total` says how many match.
"""
from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from modules.tools.discovery.handlers_documents import list_documents


@pytest.fixture
def db(test_engine):
    with test_engine.connect() as conn:
        session = Session(bind=conn)
        session.execute(text("DROP TABLE IF EXISTS pg_temp.documents"))
        session.execute(text("CREATE TEMP TABLE documents (LIKE public.documents INCLUDING DEFAULTS)"))
        yield session
        session.rollback()
        session.execute(text("DROP TABLE IF EXISTS pg_temp.documents"))
        session.commit()
        session.close()


def _doc(db, doc_id, ws, name, *, day, description=None):
    db.execute(text("INSERT INTO documents (id, workspace_id, filename, original_filename, description, status, "
                    "upload_date) VALUES (:id, CAST(:ws AS uuid), :name, :name, :description, 'completed', "
                    "make_timestamp(2026, 9, :day, 12, 0, 0))"),
               {"id": doc_id, "ws": ws, "name": name, "description": description, "day": day})


def _names(reply):
    return [d["filename"] for d in reply["documents"]]


@pytest.fixture
def shelf(db):
    ws = str(uuid4())
    _doc(db, 1, ws, "roast-log-2026-09-17-thursday.md", day=17)
    _doc(db, 2, ws, "green-coffee-list-autumn-2026.csv", day=18)
    _doc(db, 3, ws, "brand_voice.md", day=19)
    _doc(db, 4, ws, "brand-voice-old.md", day=20)
    _doc(db, 5, ws, "wholesale-terms.pdf", day=21, description="Minimum order and the roast log schedule")
    return ws


def test_search_keeps_what_the_name_or_description_contains(db, shelf):
    reply = asyncio.run(list_documents(db, UUID(shelf), {"search": "roast log"}))
    assert _names(reply) == ["wholesale-terms.pdf"]
    reply = asyncio.run(list_documents(db, UUID(shelf), {"search": "ROAST-LOG"}))
    assert _names(reply) == ["roast-log-2026-09-17-thursday.md"] and reply["total"] == 1


def test_an_underscore_in_the_search_is_a_character_not_a_wildcard(db, shelf):
    assert _names(asyncio.run(list_documents(db, UUID(shelf), {"search": "brand_voice"}))) == ["brand_voice.md"]


def test_offset_pages_past_the_newest_and_total_counts_every_match(db, shelf):
    reply = asyncio.run(list_documents(db, UUID(shelf), {"limit": 2, "offset": 2}))
    assert _names(reply) == ["brand_voice.md", "green-coffee-list-autumn-2026.csv"]
    assert (reply["count"], reply["total"], reply["offset"]) == (2, 5, 2)


def test_a_limit_or_offset_sent_as_text_still_works(db, shelf):
    reply = asyncio.run(list_documents(db, UUID(shelf), {"limit": "1", "offset": "1"}))
    assert _names(reply) == ["brand-voice-old.md"]

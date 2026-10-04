"""F327 (night 9b) — platform_read_document reads exactly the document asked for.

Chat ca9d92d2, 4 Oct 15:48Z: Auto wanted club-box-october-2026.md (document
1518). Its first call sent the filename as ``document_id`` and was refused
("document_id must be an integer"); then ``document_name``, refused; then it
sent 1523 and was handed roast-rules.md, then 1522 and was handed
margin-sheet-sep-2026.csv. The handler read ``documents.id`` exactly — those
ids ARE those documents — and nothing told Auto that the id it sent named
another document than the one it meant.

Now a document can be named by its filename (``filename``, ``document_name``,
or the filename sent as ``document_id``), an id and a filename that name two
different documents are refused with which document each one is, and an id or
name this workspace has no readable document under is refused in plain words
with how to find the right one. Never a fall-back to another document, and
never another workspace's.
"""
from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from modules.tools.discovery.handlers_documents import read_document

CLUB_BOX = "club-box-october-2026.md"


@pytest.fixture
def db(test_engine):
    with test_engine.connect() as conn:
        session = Session(bind=conn)
        for table in ("documents", "document_chunks"):
            session.execute(text(f"DROP TABLE IF EXISTS pg_temp.{table}"))
            session.execute(text(f"CREATE TEMP TABLE {table} (LIKE public.{table} INCLUDING DEFAULTS)"))
        yield session
        session.rollback()
        for table in ("documents", "document_chunks"):
            session.execute(text(f"DROP TABLE IF EXISTS pg_temp.{table}"))
        session.commit()
        session.close()


def _doc(db, doc_id, ws, name, content):
    db.execute(text("INSERT INTO documents (id, workspace_id, filename, original_filename, status, file_type, "
                    "team_access, upload_date) VALUES (:id, CAST(:ws AS uuid), :name, :name, 'completed', "
                    "'markdown', '{}', make_timestamp(2026, 10, 4, 10, 33, 0))"),
               {"id": doc_id, "ws": ws, "name": name})
    db.execute(text("INSERT INTO document_chunks (document_id, chunk_index, content) VALUES (:id, 0, :content)"),
               {"id": doc_id, "content": content})


@pytest.fixture
def shelf(db):
    """Night 9b's shelf, by its real ids, and another workspace's club box."""
    ws, other = str(uuid4()), str(uuid4())
    _doc(db, 1515, ws, "brand-voice.md", "Brand voice: warm, plain, no exclamation marks.")
    _doc(db, 1518, ws, CLUB_BOX, "October club box: Guji Shakiso and Kirinyaga AA.")
    _doc(db, 1522, ws, "margin-sheet-sep-2026.csv", "coffee,bag_g,price_gbp,margin_pct")
    _doc(db, 1523, ws, "roast-rules.md", "Roast rules: never roast on a Sunday.")
    _doc(db, 1600, ws, "1515-price-list.md", "Price list 1515: the old one.")
    _doc(db, 1700, other, CLUB_BOX, "Another shop's October box: Brazil Cerrado.")
    return UUID(ws)


def _read(db, ws, **params):
    return asyncio.run(read_document(db, ws, params))


def test_the_number_in_a_title_is_never_read_as_its_id(db, shelf):
    """1515 is brand-voice.md; "1515-price-list.md" is document 1600."""
    assert _read(db, shelf, document_id=1515)["filename"] == "brand-voice.md"
    reply = _read(db, shelf, document_id=1515, filename="1515-price-list.md")
    assert reply["success"] is False and "content" not in reply
    assert reply["error"].endswith("'1515-price-list.md' is document 1600.")


@pytest.mark.parametrize("params", [
    {"filename": CLUB_BOX},
    {"document_name": CLUB_BOX},
    {"document_id": CLUB_BOX},             # night 9b's first call
    {"filename": "Club-Box-October-2026.MD"},
    {"document_id": 1518, "filename": CLUB_BOX},
])
def test_its_filename_reads_that_document_and_never_another_workspaces(db, shelf, params):
    reply = _read(db, shelf, **params)
    assert reply["success"] is True and reply["document_id"] == 1518
    assert "Guji Shakiso" in reply["content"] and "Brazil Cerrado" not in reply["content"]


def test_an_id_naming_another_document_than_the_filename_reads_nothing(db, shelf):
    """Night 9b: Auto meant the club box and sent 1523 (roast-rules.md)."""
    reply = _read(db, shelf, document_id=1523, filename=CLUB_BOX)
    assert reply["success"] is False and "content" not in reply
    assert reply["error"] == (f"Document 1523 is 'roast-rules.md', not '{CLUB_BOX}', so nothing was read. "
                              f"'{CLUB_BOX}' is document 1518.")


def test_another_workspaces_id_is_refused_and_nothing_of_it_is_shown(db, shelf):
    reply = _read(db, shelf, document_id=1700)
    assert reply["success"] is False and "content" not in reply
    assert reply["error"].startswith("There is no document 1700 you can read in this workspace, so nothing was read.")
    assert "Brazil Cerrado" not in str(reply)


def test_an_id_that_does_not_exist_says_how_to_find_the_right_one(db, shelf):
    reply = _read(db, shelf, document_id=9999)
    assert reply["success"] is False and "content" not in reply
    assert "There is no document 9999" in reply["error"] and "platform_list_documents" in reply["error"]


def test_a_name_no_document_has_falls_back_to_none(db, shelf):
    reply = _read(db, shelf, filename="club-box-november-2026.md")
    assert reply["success"] is False and "content" not in reply
    assert reply["error"].startswith("No document named 'club-box-november-2026.md' in this workspace")
    assert "platform_list_documents" in reply["error"]


def test_two_documents_with_one_name_are_both_named_and_neither_read(db, shelf):
    _doc(db, 1561, str(shelf), CLUB_BOX, "A second upload of the October box.")
    reply = _read(db, shelf, filename=CLUB_BOX)
    assert reply["success"] is False and "content" not in reply
    assert reply["error"].startswith(f"2 documents are named '{CLUB_BOX}': 1518 (uploaded 2026-10-04); "
                                     "1561 (uploaded 2026-10-04).")


def test_a_call_naming_no_document_says_what_to_pass(db, shelf):
    reply = _read(db, shelf)
    assert reply["success"] is False
    assert reply["error"].startswith("Say which document to read: pass its document_id")


def test_the_action_takes_a_filename_and_the_name_auto_sent_it_under():
    from modules.tools.discovery import get_action_registry
    from modules.tools.execution.unified_executor import undeclared_params_refusal

    action = get_action_registry().get("platform_read_document")
    assert action.parameters["properties"]["filename"]["type"] == "string"
    assert action.parameters["required"] == []
    assert undeclared_params_refusal("platform_read_document", action, {"document_name": CLUB_BOX}, "t") is None

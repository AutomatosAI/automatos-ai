"""F087 (night 3) — a file uploaded again under the same name replaces the document.

Night 3 kept six copies of the Christmas sheet side by side; agents read
whichever they found and Auto answered from the old price after the new one was
uploaded. Now the same name re-ingests the earlier document under its id, with
the old source kept in its history. Agents' reports and cloud-synced files keep
their own identities.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from services import document_versions as dv


def test_a_replacement_keeps_the_old_source_and_counts_the_version():
    doc = NS(file_path="/uploads/a.csv", content_hash="h1", file_size=10, file_type="csv", status="completed",
             doc_metadata={"kept_pct": 37, "note": "kept"})
    original = doc.doc_metadata
    when = datetime(2026, 9, 23, 1, 0, tzinfo=timezone.utc)
    assert dv.record_replacement(doc, file_path="/uploads/b.csv", file_size=12, content_hash="h2",
                                 file_type="csv", replaced_by="user_2", now=when) == 2
    assert (doc.file_path, doc.content_hash, doc.file_size, doc.status) == ("/uploads/b.csv", "h2", 12, "processing")
    assert doc.doc_metadata == {"note": "kept", "versions": [
        {"file_path": "/uploads/a.csv", "content_hash": "h1", "file_size": 10, "replaced_at": when.isoformat(),
         "replaced_by": "user_2"}]}
    assert original == {"kept_pct": 37, "note": "kept"}              # rebuilt, never mutated
    assert dv.record_replacement(doc, file_path="/uploads/c.csv", file_size=13, content_hash="h3",
                                 file_type="csv", now=when) == 3


# ── which document a name replaces, on Postgres ─────────────────────────────

_DROP = "DROP TABLE IF EXISTS pg_temp.cloud_documents, pg_temp.documents"


@pytest.fixture
def db(test_engine):
    with test_engine.connect() as conn:
        session = Session(bind=conn)
        session.execute(text(_DROP))
        session.execute(text("CREATE TEMP TABLE documents (LIKE public.documents INCLUDING DEFAULTS)"))
        # the lookup reads only these two columns of cloud_documents
        session.execute(text("CREATE TEMP TABLE cloud_documents (id serial, document_id int)"))
        yield session
        session.rollback()
        session.execute(text(_DROP))
        session.commit()
        session.close()


def _doc(db, doc_id, ws, name, *, source_type=None, teams=()):
    db.execute(text("INSERT INTO documents (id, workspace_id, filename, source_type, team_access, status) "
                    "VALUES (:id, CAST(:ws AS uuid), :name, :st, CAST(:teams AS varchar[]), 'completed')"),
               {"id": doc_id, "ws": ws, "name": name, "st": source_type, "teams": list(teams)})


def test_the_latest_upload_of_that_name_in_that_scope_is_replaced(db):
    ws, other = str(uuid4()), str(uuid4())
    _doc(db, 1, ws, "christmas-box-2026.md")
    _doc(db, 2, ws, "christmas-box-2026.md")                               # the current copy
    _doc(db, 3, ws, "christmas-box-2026.md", source_type="agent_output")   # an agent's report: its own identity
    _doc(db, 4, ws, "christmas-box-2026.md", teams=("ops",))               # another team's copy
    _doc(db, 5, other, "christmas-box-2026.md")                            # another workspace
    _doc(db, 6, ws, "drive-sheet.md")
    db.execute(text("INSERT INTO cloud_documents (document_id) VALUES (6)"))
    assert dv.replaceable_document(db, ws, "christmas-box-2026.md", []).id == 2
    assert dv.replaceable_document(db, ws, "christmas-box-2026.md", ["ops"]).id == 4
    assert dv.replaceable_document(db, ws, "drive-sheet.md", []) is None          # cloud-synced
    assert dv.replaceable_document(db, ws, "new-file.md", []) is None


# ── the replacement itself ──────────────────────────────────────────────────

class _Db:
    def __init__(self):
        self.commits = 0

    def query(self, *_a):
        return self

    def filter(self, *_a):
        return self

    def first(self):
        return None                                   # no identical content already stored

    def commit(self):
        self.commits += 1

    def refresh(self, obj):
        obj.status = "completed"


class _Manager:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def clear_chunks(self, document_id):
        self.calls.append(("clear", document_id))

    async def _process_document(self, document_id, path, file_type, *a, **k):
        if self.fail:
            raise RuntimeError("no embedding provider")
        self.calls.append(("process", document_id, path, file_type))


def _earlier(**kw):
    return NS(**{"id": 716, "filename": "subscription-plans.md", "file_path": "/uploads/old.md",
                 "content_hash": "old", "file_size": 5, "file_type": "markdown", "status": "completed",
                 "doc_metadata": {}, "tags": ["pricing"], "description": "plans", **kw})


def test_the_old_text_is_cleared_before_the_new_is_ingested_under_the_same_id(monkeypatch):
    import api.documents as documents

    manager, db, doc = _Manager(), _Db(), _earlier()
    monkeypatch.setattr(documents, "get_document_manager", lambda ws: manager)
    version = asyncio.run(dv.replace_document(
        db, doc, workspace_id="ws", file_path="/uploads/new.md", file_size=9, content_hash="new",
        file_type="markdown", replaced_by="user_2", tags=["pricing", "2027"]))
    assert version == 2 and doc.status == "completed" and db.commits == 1
    assert manager.calls == [("clear", 716), ("process", 716, "/uploads/new.md", documents.processing_type("markdown"))]
    assert doc.tags == ["pricing", "2027"] and doc.description == "plans"      # left out: kept
    assert doc.doc_metadata["versions"][0]["replaced_by"] == "user_2"
    assert dv.replaced_message(doc.filename, version, doc.status).startswith("Replaced subscription-plans.md (version 2)")


def test_a_failed_replacement_says_so_and_keeps_the_history(monkeypatch):
    import api.documents as documents

    db, doc = _Db(), _earlier()
    monkeypatch.setattr(documents, "get_document_manager", lambda ws: _Manager(fail=True))
    version = asyncio.run(dv.replace_document(
        db, doc, workspace_id="ws", file_path="/uploads/new.md", file_size=9, content_hash="new",
        file_type="markdown"))
    assert doc.status == "failed" and db.commits == 2
    assert doc.doc_metadata["versions"][0]["file_path"] == "/uploads/old.md"
    assert dv.replaced_message(doc.filename, version, doc.status).startswith("Replacing subscription-plans.md failed")


# ── the upload route and Auto's own upload tool ─────────────────────────────

class _Upload:
    def __init__(self, filename, data):
        self.filename, self._data = filename, data

    async def read(self):
        return self._data

    async def seek(self, _pos):
        pass


def test_the_upload_route_replaces_a_file_of_the_same_name(monkeypatch, tmp_path):
    import api.documents as documents

    manager, doc = _Manager(), _earlier(filename="christmas-box-2026.csv", file_type="csv", tags=[])
    monkeypatch.setattr(documents, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(documents, "get_document_manager", lambda ws: manager)
    monkeypatch.setattr(dv, "replaceable_document", lambda db, ws, name, teams: doc if not teams else None)
    ctx = NS(workspace_id=uuid4(), user=NS(clerk_user_id="user_2"))
    out = asyncio.run(documents.handle_request(
        ctx=ctx, file=_Upload("christmas-box-2026.csv", "item,price\nChristmas Box,£48\n".encode()),
        description=None, tags=None, team_access=None, db=_Db()))
    assert (out.document_id, out.status) == (716, "completed")
    assert out.message.startswith("Replaced christmas-box-2026.csv (version 2)")
    assert [c[0] for c in manager.calls] == ["clear", "process"] and manager.calls[1][2].startswith(str(tmp_path))
    assert doc.doc_metadata["versions"][0]["replaced_by"] == "user_2"


def test_auto_uploading_the_same_name_again_replaces_it(monkeypatch, tmp_path):
    import api.documents as documents
    from modules.tools.discovery import handlers_documents

    manager, earlier = _Manager(), _earlier()
    monkeypatch.setattr(documents, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(documents, "get_document_manager", lambda ws: manager)
    monkeypatch.setattr(dv, "replaceable_document", lambda db, ws, name, teams: earlier)
    out = asyncio.run(handlers_documents.upload_document(_Db(), uuid4(), {
        "filename": "subscription-plans.md", "content": "Harvest Club: £21 a month."}))
    assert out["replaced"] is True and out["document_id"] == 716 and out["version"] == 2
    assert "agents now read the new copy" in out["message"]
    assert [c[:2] for c in manager.calls] == [("clear", 716), ("process", 716)]
    assert earlier.doc_metadata["versions"][0] == {**earlier.doc_metadata["versions"][0],
                                                   "file_path": "/uploads/old.md", "replaced_by": "auto"}

"""PRD-157 S5 — document pinning + honest doc widgets.

Pure layer: the widget scope-filter (drops out-of-scope [View Document] links)
and pinned-context validation/empty paths. Integration layer (marked): pin a
document and assemble its content into context across turns.
"""

from __future__ import annotations

import uuid

import pytest


# --------------------------------------------------------------------------- #
# S5b — honest doc widgets: _filter_frontend_docs_by_scope
# --------------------------------------------------------------------------- #

class _FakeSession:
    def close(self):
        pass


class TestWidgetScopeFilter:
    def _router(self):
        from modules.tools.tool_router import ToolRouter

        return ToolRouter.__new__(ToolRouter)  # bypass __init__

    def test_drops_out_of_scope_documents(self, monkeypatch):
        import core.database.database as dbmod
        import modules.rag.retrieval_filters as rf
        import modules.tools.discovery.handlers_documents as hd

        monkeypatch.setattr(dbmod, "SessionLocal", lambda: _FakeSession())
        monkeypatch.setattr(hd, "_resolve_agent_team", lambda db, aid: None)
        monkeypatch.setattr(rf, "allowed_document_ids", lambda db, ids, filters: {"1"})

        tr = self._router()
        fd = {"documents": [{"document_id": 1, "filename": "ok.pdf"}, {"document_id": 2, "filename": "other.pdf"}]}
        out = tr._filter_frontend_docs_by_scope(fd, agent_id=5, workspace_id="ws")
        kept = [d["document_id"] for d in out["documents"]]
        assert kept == [1]   # doc 2 (out of scope) suppressed

    def test_keeps_documents_without_id(self, monkeypatch):
        import core.database.database as dbmod
        import modules.rag.retrieval_filters as rf
        import modules.tools.discovery.handlers_documents as hd

        monkeypatch.setattr(dbmod, "SessionLocal", lambda: _FakeSession())
        monkeypatch.setattr(hd, "_resolve_agent_team", lambda db, aid: None)
        monkeypatch.setattr(rf, "allowed_document_ids", lambda db, ids, filters: set())

        tr = self._router()
        fd = {"documents": [{"filename": "no-id.pdf"}]}  # no document_id -> kept
        out = tr._filter_frontend_docs_by_scope(fd, agent_id=5, workspace_id="ws")
        assert len(out["documents"]) == 1

    def test_no_workspace_leaves_unchanged(self):
        tr = self._router()
        fd = {"documents": [{"document_id": 9}]}
        assert tr._filter_frontend_docs_by_scope(fd, agent_id=1, workspace_id=None) is fd


# --------------------------------------------------------------------------- #
# S5a — pinned context: pure validation / empty paths
# --------------------------------------------------------------------------- #

class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeDB:
    def __init__(self, row_batches=None):
        self._batches = list(row_batches or [])

    def execute(self, sql, params=None):
        return _Result(self._batches.pop(0) if self._batches else [])

    def commit(self):
        pass


class TestPinnedContextPure:
    def test_pin_rejects_non_int_id(self):
        from modules.rag.pinned_context import pin_document

        res = pin_document(_FakeDB(), chat_id="c", document_id="abc", workspace_id="ws")
        assert res["success"] is False

    def test_build_pinned_context_empty_when_nothing_pinned(self):
        from modules.rag.pinned_context import build_pinned_context

        text, smap = build_pinned_context(_FakeDB([[]]), chat_id="c", workspace_id="ws")
        assert text == "" and smap == []

    def test_build_pinned_system_message_none_when_empty(self):
        from modules.rag.pinned_context import build_pinned_system_message

        assert build_pinned_system_message(_FakeDB([[]]), chat_id="c", workspace_id="ws") is None


# --------------------------------------------------------------------------- #
# Integration (real Postgres)
# --------------------------------------------------------------------------- #

@pytest.mark.integration
def test_pin_inject_and_unpin(db_session, seed_workspace):
    from sqlalchemy import text
    from modules.rag.pinned_context import (
        pin_document,
        unpin_document,
        list_pinned,
        build_pinned_system_message,
    )

    ws = seed_workspace()
    # a user + chat + document with chunks
    user_id = db_session.execute(
        text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
        {"e": f"{uuid.uuid4()}@t.test", "u": f"pin-{uuid.uuid4().hex[:12]}"},
    ).scalar()
    chat_id = str(uuid.uuid4())
    db_session.execute(
        text(
            "INSERT INTO chats (id, user_id, workspace_id, title, visibility) "
            "VALUES (CAST(:id AS uuid), :uid, CAST(:ws AS uuid), 'pin-test', 'private')"
        ),
        {"id": chat_id, "uid": user_id, "ws": ws},
    )
    doc_id = db_session.execute(
        text(
            "INSERT INTO documents (filename, workspace_id, team_access, status, upload_date) "
            "VALUES ('pinned.txt', CAST(:ws AS uuid), '{}', 'processed', NOW()) RETURNING id"
        ),
        {"ws": ws},
    ).scalar()
    db_session.execute(
        text(
            "INSERT INTO document_chunks (document_id, chunk_index, content, workspace_id) "
            "VALUES (:doc, 0, 'PINNED-MARKER unique content', :ws)"
        ),
        {"doc": doc_id, "ws": ws},
    )
    db_session.flush()

    assert pin_document(db_session, chat_id=chat_id, document_id=doc_id, workspace_id=ws)["success"]
    assert any(p["document_id"] == doc_id for p in list_pinned(db_session, chat_id=chat_id, workspace_id=ws))

    msg = build_pinned_system_message(db_session, chat_id=chat_id, workspace_id=ws)
    assert msg is not None and "PINNED-MARKER" in msg   # present in context

    unpin_document(db_session, chat_id=chat_id, document_id=doc_id, workspace_id=ws)
    assert list_pinned(db_session, chat_id=chat_id, workspace_id=ws) == []

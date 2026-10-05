"""F357 (5 Oct) — an upload's duplicate check stays inside its own workspace.

``DocumentManager.upload_document()`` reuses a processed document with the same
bytes instead of storing a second copy. Before #834 the lookup was
``SELECT id, status FROM documents WHERE file_hash = %s`` with no workspace, so
an upload whose bytes matched another tenant's document got that tenant's id.

These tests pin the tenancy contract: identical bytes in workspace B get B's own
document, a repeat in the same workspace still dedupes, workspace A's id is never
returned to B, and a manager with no workspace refuses before it touches the
database. Every DocumentManager the code builds is built for a workspace.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import warnings
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ORCHESTRATOR = Path(__file__).resolve().parents[1]
SKIPPED_DIRS = {"tests", "__pycache__", "node_modules", "venv"}
CONTENT = b"# Price list\n\nWidgets are 4 euro each.\n"


def _manager(mgr, workspace_id):
    """A DocumentManager without __init__ (no S3, no embedding provider)."""
    manager = mgr.DocumentManager.__new__(mgr.DocumentManager)
    manager.db_config, manager.workspace_id, manager.s3_bucket = {}, workspace_id, "docs"
    manager.processor = NS(detect_file_type=lambda _path: mgr.DocumentType.MARKDOWN)
    manager._upload_to_s3 = lambda _path, document_id, filename: f"ws/{document_id}_{filename}"

    async def _process(document_id, file_path, file_type, s3_key=None, filename=None):
        return None

    manager._process_document = _process
    return manager


def _upload(manager, path):
    return asyncio.run(manager.upload_document(str(path), filename="prices.md", created_by="owner"))


def test_a_manager_without_a_workspace_refuses_before_the_database(monkeypatch, tmp_path):
    from modules.rag.ingestion import manager as mgr

    def _no_database(**_kw):
        raise AssertionError("a workspace-less upload must not reach the database")

    monkeypatch.setattr(mgr.psycopg2, "connect", _no_database)
    path = tmp_path / "prices.md"
    path.write_bytes(CONTENT)
    for missing in (None, ""):
        with pytest.raises(ValueError, match="workspace_id required"):
            _upload(_manager(mgr, missing), path)


def test_the_duplicate_lookup_binds_the_callers_workspace(monkeypatch, tmp_path):
    from modules.rag.ingestion import manager as mgr

    lookups = []

    class Cursor:
        def execute(self, sql, params=None):
            self._sql = " ".join(sql.split())
            if self._sql.startswith("SELECT id FROM documents"):
                lookups.append((self._sql, params))

        def fetchone(self):
            return (11,) if self._sql.startswith("INSERT INTO documents") else None

    class Conn:
        def cursor(self, *_a, **_kw):
            return Cursor()

        def commit(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(mgr.psycopg2, "connect", lambda **_kw: Conn())
    path = tmp_path / "prices.md"
    path.write_bytes(CONTENT)
    assert _upload(_manager(mgr, "ws-b"), path) == 11

    (sql, params), = lookups
    assert "content_hash = %s AND workspace_id = %s" in sql
    assert params[:2] == (hashlib.sha256(CONTENT).hexdigest(), "ws-b")


def _calls_named(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            called = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if called == name:
                yield node


def _source_files():
    for path in ORCHESTRATOR.rglob("*.py"):
        parts = set(path.relative_to(ORCHESTRATOR).parts)
        if not parts & SKIPPED_DIRS and not any(p.startswith(".") for p in parts):
            yield path


def _parse(path):
    """The module's AST; an old file's invalid escape warning is not this test's business."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_every_document_manager_in_the_code_is_built_for_a_workspace():
    unscoped = []
    for path in _source_files():
        tree = _parse(path)
        for call in _calls_named(tree, "DocumentManager"):
            if not any(kw.arg == "workspace_id" for kw in call.keywords):
                unscoped.append(f"{path.relative_to(ORCHESTRATOR)}:{call.lineno}")
        for call in _calls_named(tree, "get_document_manager"):
            if not (call.args or call.keywords):
                unscoped.append(f"{path.relative_to(ORCHESTRATOR)}:{call.lineno}")
    assert unscoped == []


class _SharedConnection:
    """The test transaction's own DBAPI connection, handed to the manager so its
    writes stay inside the transaction ``db_session`` rolls back."""

    def __init__(self, raw):
        self._raw = raw

    def cursor(self, *args, **kwargs):
        return self._raw.cursor(*args, **kwargs)

    def commit(self):
        pass

    def close(self):
        pass


def _complete(db_session, document_id):
    from sqlalchemy import text

    db_session.execute(text("UPDATE documents SET status = 'completed' WHERE id = :id"), {"id": document_id})


def _owner_of(db_session, document_id):
    from sqlalchemy import text

    return db_session.execute(text("SELECT CAST(workspace_id AS text) FROM documents WHERE id = :id"),
                              {"id": document_id}).scalar()


@pytest.mark.integration
def test_identical_bytes_in_another_workspace_get_their_own_document(
    db_session, seed_workspace, monkeypatch, tmp_path
):
    from modules.rag.ingestion import manager as mgr

    ws_a, ws_b = seed_workspace(), seed_workspace()
    shared = _SharedConnection(db_session.connection().connection.dbapi_connection)
    monkeypatch.setattr(mgr.psycopg2, "connect", lambda **_kw: shared)
    path = tmp_path / "prices.md"
    path.write_bytes(CONTENT)
    in_a, in_b = _manager(mgr, ws_a), _manager(mgr, ws_b)

    a_doc = _upload(in_a, path)
    _complete(db_session, a_doc)
    b_doc = _upload(in_b, path)
    assert b_doc != a_doc
    assert (_owner_of(db_session, a_doc), _owner_of(db_session, b_doc)) == (ws_a, ws_b)

    _complete(db_session, b_doc)
    b_repeats = [_upload(in_b, path) for _ in range(2)]
    assert b_repeats == [b_doc, b_doc]
    assert _upload(in_a, path) == a_doc
    assert a_doc not in {b_doc, *b_repeats}


@pytest.mark.integration
def test_a_workspace_with_no_processed_copy_never_gets_another_tenants_id(
    db_session, seed_workspace, monkeypatch, tmp_path
):
    from modules.rag.ingestion import manager as mgr

    ws_a, ws_b = seed_workspace(), seed_workspace()
    shared = _SharedConnection(db_session.connection().connection.dbapi_connection)
    monkeypatch.setattr(mgr.psycopg2, "connect", lambda **_kw: shared)
    path = tmp_path / "prices.md"
    path.write_bytes(CONTENT)

    a_doc = _upload(_manager(mgr, ws_a), path)
    _complete(db_session, a_doc)
    in_b = _manager(mgr, ws_b)
    b_ids = [_upload(in_b, path) for _ in range(2)]   # B's first copy is still processing
    assert a_doc not in b_ids
    assert {_owner_of(db_session, doc) for doc in b_ids} == {ws_b}

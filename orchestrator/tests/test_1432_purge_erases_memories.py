"""#1432: a workspace purge erases the workspace's memories in Qdrant.

``purge_workspace_sync`` deleted SQL rows and S3 objects only, so the admin delete and
``purge_test_workspace.py`` left every durable (L3) and field memory of the workspace
in Qdrant: unreachable, still stored (100 points in the case found). The GDPR erasure
erased them itself first; the purge now does it for every path, with the same erasers,
and reports the counts. The real purge on the real database; the erasers stubbed.
"""
from __future__ import annotations

from uuid import uuid4

import pytest


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    yield eng
    eng.dispose()


@pytest.fixture
def erasers(monkeypatch):
    """Qdrant configured, both erasers recorded; set ``fail`` to make one raise."""
    from config import config
    from modules.context.adapters import vector_field
    from modules.memory import unified_memory_service as ums
    from services import workspace_purge

    calls, fail = [], set()

    async def durable(self, workspace_id, db=None):
        calls.append(("durable", workspace_id))
        if "durable" in fail:
            raise RuntimeError("qdrant unreachable")
        return 61

    async def field(self, workspace_id):
        calls.append(("field", workspace_id))
        return 4

    monkeypatch.setattr(config, "QDRANT_URL", "http://qdrant.test")
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(ums.UnifiedMemoryService, "__init__", lambda self: None)
    monkeypatch.setattr(ums.UnifiedMemoryService, "erase_workspace_memories", durable)
    monkeypatch.setattr(vector_field.VectorFieldSharedContext, "__init__", lambda self: None)
    monkeypatch.setattr(vector_field.VectorFieldSharedContext, "erase_workspace", field)
    monkeypatch.setattr(workspace_purge, "_build_s3_client", lambda: None)
    return workspace_purge.purge_workspace_sync, calls, fail


def _workspace(new_session, *, gone: bool) -> str:
    from sqlalchemy import text

    ws = str(uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name, deleted_at) VALUES (CAST(:id AS uuid), '#1432 purge', "
                   "CASE WHEN CAST(:gone AS boolean) THEN now() END)"), {"id": ws, "gone": gone})
    s.commit()
    s.close()
    return ws


def _drop(new_session, ws):
    from sqlalchemy import text

    s = new_session.sweep()
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


def test_the_purge_erases_both_memory_stores_and_counts_them(new_session, erasers):
    run, calls, _ = erasers
    ws = _workspace(new_session, gone=True)
    try:
        result = run(ws)
        assert result.skipped_reason is None and result.workspace_row_deleted
        assert sorted(calls) == [("durable", ws), ("field", ws)]
        assert (result.durable_memory_deleted, result.field_memory_deleted, result.memory_erase_errors) == (61, 4, 0)
        assert {"durable_memory_deleted", "field_memory_deleted", "memory_erase_errors"} <= set(result.to_dict())
    finally:
        _drop(new_session, ws)


def test_a_failed_erase_is_counted_and_the_rest_of_the_purge_goes_on(new_session, erasers):
    run, calls, fail = erasers
    fail.add("durable")
    ws = _workspace(new_session, gone=True)
    try:
        result = run(ws)
        assert result.memory_erase_errors == 1 and result.durable_memory_deleted == 0
        assert result.field_memory_deleted == 4 and result.workspace_row_deleted
    finally:
        _drop(new_session, ws)


def test_with_no_qdrant_there_is_nothing_to_erase(new_session, erasers, monkeypatch):
    from config import config

    run, calls, _ = erasers
    monkeypatch.setattr(config, "QDRANT_URL", "")
    ws = _workspace(new_session, gone=True)
    try:
        result = run(ws)
        assert calls == [] and result.memory_erase_errors == 0 and result.workspace_row_deleted
    finally:
        _drop(new_session, ws)


def test_a_live_workspace_keeps_its_memories(new_session, erasers):
    """The purge refuses a workspace that was not soft-deleted first, memories included."""
    run, calls, _ = erasers
    ws = _workspace(new_session, gone=False)
    try:
        assert run(ws).skipped_reason == "not_soft_deleted" and calls == []
    finally:
        _drop(new_session, ws)

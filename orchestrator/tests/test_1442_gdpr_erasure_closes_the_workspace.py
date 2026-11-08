"""#1442: a GDPR workspace erasure erases every store, or says it did not.

``gdpr_service.erase_workspace`` erased the memories, then called the purge for the SQL
and S3 legs. The purge refuses a workspace that was not soft-deleted first, and neither
erase route soft-deletes, so every row and file stayed while the response said
``"complete": true``. A step that raised was swallowed the same way, and the durable
eraser turned its own failure into 0.

Now the erasure closes the workspace first, and ``complete`` is true only when every
step ran. The real close and purge on the real database; the memory stores stubbed.
"""
from __future__ import annotations

import asyncio
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
def gdpr(monkeypatch):
    """The GDPR service with the memory stores stubbed, S3 off and no audit row."""
    from config import config
    from services import gdpr_service, workspace_purge

    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(gdpr_service, "_erase_field_memory", lambda ws, subject_id=None: 3)
    monkeypatch.setattr(gdpr_service, "_erase_durable_memory", lambda ws, subject_id=None: 40)
    monkeypatch.setattr(gdpr_service, "_audit_gdpr", lambda db, ws, action, **kw: None)
    monkeypatch.setattr(workspace_purge, "_build_s3_client", lambda: None)
    return gdpr_service


def _live_workspace(new_session) -> str:
    from sqlalchemy import text

    ws = str(uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), '#1442 erasure')"), {"id": ws})
    s.commit()
    s.close()
    return ws


def _still_there(new_session, ws: str) -> bool:
    from sqlalchemy import text

    s = new_session()
    try:
        return s.execute(text("SELECT 1 FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws}).first() is not None
    finally:
        s.close()


def _drop(new_session, ws: str) -> None:
    from sqlalchemy import text

    s = new_session.sweep()
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


def test_erasing_a_live_workspace_erases_its_rows_too(new_session, gdpr):
    ws = _live_workspace(new_session)
    try:
        result = gdpr.erase_workspace(None, ws, requested_by="user:9")
        assert result["sql"]["skipped_reason"] is None
        assert result["sql"]["workspace_row_deleted"] is True
        assert not _still_there(new_session, ws)
        assert result["complete"] is True and result["failed"] == []
    finally:
        _drop(new_session, ws)


def test_a_step_that_raises_makes_the_erasure_incomplete(new_session, gdpr, monkeypatch):
    def _qdrant_down(ws, subject_id=None):
        raise RuntimeError("qdrant unreachable")

    monkeypatch.setattr(gdpr, "_erase_durable_memory", _qdrant_down)
    ws = _live_workspace(new_session)
    try:
        result = gdpr.erase_workspace(None, ws, requested_by="user:9")
        assert result["complete"] is False and result["failed"] == ["durable_memory"]
        assert not _still_there(new_session, ws)  # the other stores were still erased
    finally:
        _drop(new_session, ws)


@pytest.mark.parametrize(
    ("sql", "failed"),
    [
        ({"workspace_row_deleted": False, "skipped_reason": "not_soft_deleted"}, ["sql"]),
        ({"workspace_row_deleted": True, "s3_errors": 2}, ["s3"]),
    ],
)
def test_a_purge_that_left_something_makes_the_erasure_incomplete(gdpr, monkeypatch, sql, failed):
    monkeypatch.setattr(gdpr, "_close_workspace", lambda ws: True)
    monkeypatch.setattr(gdpr, "_purge_sql", lambda ws: sql)
    result = gdpr.erase_workspace(None, uuid4(), requested_by="user:9")
    assert result["complete"] is False and result["failed"] == failed


def test_with_no_qdrant_the_memory_steps_have_nothing_to_erase(monkeypatch):
    """Field memory raised without ``QDRANT_URL``; with an honest ``complete`` that would
    fail every erasure on an install without Qdrant."""
    from config import config
    from services import gdpr_service

    monkeypatch.setattr(config, "QDRANT_URL", "")
    monkeypatch.setattr(gdpr_service, "_close_workspace", lambda ws: True)
    monkeypatch.setattr(gdpr_service, "_purge_sql", lambda ws: {"workspace_row_deleted": True})
    monkeypatch.setattr(gdpr_service, "_audit_gdpr", lambda db, ws, action, **kw: None)
    result = gdpr_service.erase_workspace(None, uuid4(), requested_by="user:9")
    assert result["derived"] == {"field_memory_deleted": 0, "durable_memory_deleted": 0}
    assert result["complete"] is True and result["failed"] == []


class _FailingDurable:
    async def erase_workspace(self, workspace_id):
        raise RuntimeError("qdrant unreachable")

    async def erase_subject(self, workspace_id, subject_id):
        raise RuntimeError("qdrant unreachable")


def _service_over(durable):
    from modules.memory.unified_memory_service import UnifiedMemoryService

    svc = object.__new__(UnifiedMemoryService)
    svc._durable = durable
    return svc


def test_the_durable_erasers_raise_instead_of_reporting_nothing_erased(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "QDRANT_URL", "http://qdrant.test")
    svc = _service_over(_FailingDurable())
    with pytest.raises(RuntimeError):
        asyncio.run(svc.erase_workspace_memories("ws-1"))
    with pytest.raises(RuntimeError):
        asyncio.run(svc.erase_subject_memories("ws-1", "subject-1"))


def test_the_durable_erasers_have_nothing_to_erase_with_no_qdrant(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "QDRANT_URL", "")
    svc = _service_over(_FailingDurable())
    assert asyncio.run(svc.erase_workspace_memories("ws-1")) == 0
    assert asyncio.run(svc.erase_subject_memories("ws-1", "subject-1")) == 0

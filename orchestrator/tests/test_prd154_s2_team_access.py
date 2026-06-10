"""PRD-154 S2 — RAG team filter fails CLOSED + team-access PATCH/bulk SQL hits
the real schema.

Two verified breakages (reports/PLATFORM_DEEP_REVIEW_2026-06.md §2; BINDING Q1):

1. ``RAGService._filter_by_team`` (service.py) returned ALL candidates unfiltered
   when its PostgreSQL access-check raised — a fail-OPEN security hole that leaks
   team-restricted chunks on any DB hiccup. It must fail CLOSED: surface only the
   public (empty ``team_access``) candidates and log at error level.

2. The team-access ``UPDATE`` SQL in ``api/documents.py`` referenced
   ``documents.title`` and ``documents.updated_at`` — neither exists on the
   ``Document`` model (real columns: ``filename``; no update timestamp). Every
   PATCH/bulk call 500'd on ``UndefinedColumn``.

Fail-closed + schema-conformance are pinned deterministically (no DB). A real-DB
integration test executes the actual endpoint SQL against the CI Postgres schema
and skips when no database is reachable.
"""
from __future__ import annotations

import os
import sys
import types

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")
sys.modules.setdefault("camelot", types.ModuleType("camelot"))

import tests.conftest as _conftest  # noqa: E402
_conftest._restore_real_app_modules()

import asyncio  # noqa: E402
import inspect  # noqa: E402
import logging  # noqa: E402
import uuid  # noqa: E402
from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402

from modules.rag.service import RAGService  # noqa: E402


# ===================================================== fail-closed security ===


def test_filter_by_team_fails_closed_on_db_error():
    """A DB error in the access-check must yield ONLY public docs — never a
    team-restricted one — and must log at error level (not warning)."""
    svc = RAGService.__new__(RAGService)
    candidates = [
        {"id": "a", "metadata": {"document_id": "1", "team_access": []}},          # public
        {"id": "b", "metadata": {"document_id": "2", "team_access": ["finance"]}},  # restricted
        {"id": "c", "metadata": {"document_id": "3"}},                              # no scope -> public
    ]

    with patch("core.database.database.SessionLocal", side_effect=RuntimeError("db down")):
        with patch.object(logging.getLogger("modules.rag.service"), "error") as mock_err:
            out = asyncio.run(svc._filter_by_team(candidates, team="legal", workspace_id=str(uuid.uuid4())))

    returned_ids = {c["id"] for c in out}
    assert returned_ids == {"a", "c"}            # only the public docs survive
    assert "b" not in returned_ids               # the restricted doc is NEVER leaked
    assert mock_err.called                        # failure logged at error level


def test_filter_by_team_passthrough_when_no_doc_ids():
    """No regression: candidates without any document id still pass through (the
    early return predates the error path)."""
    svc = RAGService.__new__(RAGService)
    candidates = [{"id": "x", "metadata": {}}]
    out = asyncio.run(svc._filter_by_team(candidates, team="legal", workspace_id=str(uuid.uuid4())))
    assert out == candidates


# ================================================== schema conformance (SQL) ===


def test_team_access_sql_targets_real_columns():
    """The PATCH/bulk endpoints must not reference phantom columns."""
    from api import documents as docs_mod
    from core.models.core import Document

    cols = set(Document.__table__.columns.keys())
    assert "title" not in cols and "updated_at" not in cols, "phantom columns must not exist on the model"
    assert "filename" in cols

    src = (
        inspect.getsource(docs_mod.update_document_team_access)
        + inspect.getsource(docs_mod.bulk_update_team_access)
    )
    assert "updated_at" not in src           # phantom timestamp removed
    assert "RETURNING id, title" not in src  # phantom column removed
    assert "filename" in src                 # real column used


# ===================================================== real-schema round-trip ===


@pytest.fixture
def live_db():
    """Skip unless the real test Postgres is reachable (CI service / local)."""
    from core.database.database import engine
    from sqlalchemy import text as _t
    try:
        with engine.connect() as conn:
            conn.execute(_t("SELECT 1"))
    except Exception as exc:  # OperationalError and friends
        pytest.skip(f"no reachable Postgres for integration test: {exc}")
    return engine


@pytest.mark.integration
def test_team_access_endpoints_execute_against_real_schema(live_db):
    """The endpoints' UPDATE...RETURNING must PLAN/EXECUTE against the live
    schema. Bogus ids match nothing — but Postgres still validates every column
    at plan time, so a phantom column would raise instead of 404 / updated:0."""
    from fastapi import HTTPException

    from api.documents import (
        _BulkTeamAccessUpdate,
        _TeamAccessUpdate,
        bulk_update_team_access,
        update_document_team_access,
    )
    from core.database.database import SessionLocal

    ctx = types.SimpleNamespace(workspace_id=uuid.uuid4())
    missing_id = 2_000_000_111

    db = SessionLocal()
    try:
        with pytest.raises(HTTPException) as ei:
            asyncio.run(update_document_team_access(
                missing_id, _TeamAccessUpdate(team_access=["finance"]), ctx=ctx, db=db,
            ))
        assert ei.value.status_code == 404  # SQL executed; just no matching row

        res = asyncio.run(bulk_update_team_access(
            _BulkTeamAccessUpdate(document_ids=[missing_id], team_access=["finance"]), ctx=ctx, db=db,
        ))
        assert res["updated"] == 0
    finally:
        db.close()

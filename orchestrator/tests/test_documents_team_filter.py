"""PRD-158 S3 — server-side team filtering on GET /api/documents + per-team counts.

Integration: a >100-doc fixture proves the filter and counts are computed in SQL
(server-side), not the old client-side ≤100 hack.
"""

from __future__ import annotations

import os
import sys
import types
import uuid

for _k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
    os.environ.setdefault(_k, "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
sys.modules.setdefault("camelot", types.ModuleType("camelot"))

import pytest  # noqa: E402


def _seed(db, ws, n, team_access_sql):
    from sqlalchemy import text

    db.execute(
        text(
            "INSERT INTO documents (filename, workspace_id, team_access, status, upload_date) "
            f"SELECT 'doc'||g, CAST(:ws AS uuid), {team_access_sql}, 'processed', NOW() "
            "FROM generate_series(1, :n) g"
        ),
        {"ws": ws, "n": n},
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_team_filter_and_counts_are_server_side(db_session, seed_workspace):
    from api.documents import list_documents, document_team_counts

    ws = seed_workspace()  # FK parent for documents.workspace_id
    # 120 docs total: 60 support, 40 sales, 20 public — exceeds the 100 page.
    _seed(db_session, ws, 60, "ARRAY['support']")
    _seed(db_session, ws, 40, "ARRAY['sales']")
    _seed(db_session, ws, 20, "'{}'")
    db_session.flush()

    ctx = types.SimpleNamespace(workspace_id=uuid.UUID(ws))

    # Counts are aggregated over ALL 120 docs, not a ≤100 slice.
    counts = await document_team_counts(ctx=ctx, db=db_session)
    assert counts["counts"]["support"] == 60
    assert counts["counts"]["sales"] == 40
    assert counts["untagged"] == 20
    assert counts["total"] == 120

    # team=sales filters server-side → 40 sales + 20 public (public always visible).
    sales = await list_documents(
        ctx=ctx, skip=0, limit=1000, status=None, file_type=None, search=None, team="sales", db=db_session
    )
    assert len(sales) == 60
    for d in sales:
        assert (not d.team_access) or ("sales" in [t.lower() for t in d.team_access])

    # Normalization: 'Sales' resolves to the same set as 'sales'.
    sales_caps = await list_documents(
        ctx=ctx, skip=0, limit=1000, status=None, file_type=None, search=None, team="Sales", db=db_session
    )
    assert len(sales_caps) == 60


@pytest.mark.integration
@pytest.mark.asyncio
async def test_no_team_param_returns_all(db_session, seed_workspace):
    from api.documents import list_documents

    ws = seed_workspace()  # FK parent for documents.workspace_id
    _seed(db_session, ws, 10, "ARRAY['support']")
    _seed(db_session, ws, 5, "ARRAY['sales']")
    db_session.flush()

    ctx = types.SimpleNamespace(workspace_id=uuid.UUID(ws))
    everything = await list_documents(
        ctx=ctx, skip=0, limit=1000, status=None, file_type=None, search=None, team=None, db=db_session
    )
    assert len(everything) == 15   # no team restriction → workspace-wide

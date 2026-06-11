"""PRD-156 S5 — closures completed here: widget-memory delete ownership,
GET /api/documents/content auth, and document_usage workspace attribution.

Structural pins (the behavioral DB/service paths run in CI). NOTE — two S5 items
are intentionally surfaced for a human decision rather than silently deferred:
  * RAG-analytics READ scoping over document_usage — the write side now attributes
    each row to a workspace (below); the analytics READ endpoints still need to
    filter on metadata->>'workspace_id';
  * deleting the mock /api/v1/memory router + AdvancedMemoryManager — its
    acceptance is the PRD-155 route-contract test (not built), and removal cascades
    into live frontend callers (memory explorer / monitoring tab), so it needs the
    PRD-155 dependency or an explicit go to remove those features.
"""
from __future__ import annotations

import pathlib

ORCH = pathlib.Path(__file__).resolve().parents[2]


def test_content_path_read_requires_auth():
    txt = (ORCH / "api/documents.py").read_text()
    block = txt.split('@router.get("/content")')[1][:300]
    assert "ctx: RequestContext = Depends(get_request_context_hybrid)" in block, (
        "/api/documents/content is still an unauthenticated path-read"
    )


def test_widget_memory_delete_scopes_by_workspace():
    txt = (ORCH / "api/widget_memory.py").read_text()
    assert "delete_memory(memory_id=memory_id, workspace_id=ws)" in txt, (
        "widget memory delete is not workspace-scoped (cross-tenant delete)"
    )


def test_document_usage_writes_attributed_to_workspace():
    txt = (ORCH / "api/documents.py").read_text()
    inserts = txt.count("INSERT INTO document_usage")
    attributed = txt.count('"workspace_id": str(ctx.workspace_id)')
    assert inserts >= 2 and attributed >= 2, (
        f"unattributed document_usage write (inserts={inserts}, attributed={attributed})"
    )

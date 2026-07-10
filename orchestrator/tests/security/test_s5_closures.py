"""PRD-156 S5 — closures completed here: widget-memory delete ownership,
GET /api/documents/content auth, and document_usage workspace attribution.

Structural pins (the behavioral DB/service paths run in CI). document_usage is now
scoped on BOTH sides — writes attribute each row to its workspace (metadata JSONB,
no migration), reads filter on metadata->>'workspace_id'.

(The once-deferred S5 item — deleting the mock /api/v1/memory router +
AdvancedMemoryManager — was completed by PRD-187 S5; its frontend callers were
removed with it.)
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


def test_document_usage_reads_scoped_by_workspace():
    """Every workspace-attributable analytics read over document_usage filters on
    metadata->>'workspace_id' (the references-count read is document-owned, so it's
    already scoped by the document's workspace check)."""
    total = 0
    for rel in ("api/documents.py", "api/context.py", "modules/rag/service.py"):
        total += (ORCH / rel).read_text().count("metadata->>'workspace_id' = :workspace_id")
    assert total >= 6, f"document_usage analytics reads not all workspace-scoped (found {total})"

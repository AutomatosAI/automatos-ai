"""
Pinned-document context (PRD-157 S5).

Pin a document to a chat so its content is ALWAYS injected into that
conversation's context — in document order, up to the retrieval token budget
(D11: full-doc reads allowed within budget). Reuses the S1 scope filters (a pin
can only target a document the workspace owns) and the S3 budgeter (whole-chunk,
token-budgeted accumulation).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text as sa_text


def pin_document(
    db: Any,
    *,
    chat_id: Any,
    document_id: Any,
    workspace_id: Any,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Pin a document to a chat. Fails if the document is not in the workspace."""
    from modules.rag.retrieval_filters import build_retrieval_filters, allowed_document_ids

    try:
        document_id = int(document_id)
    except (TypeError, ValueError):
        return {"success": False, "error": "document_id must be an integer"}

    filters = build_retrieval_filters(workspace_id=str(workspace_id))
    if str(document_id) not in allowed_document_ids(db, [document_id], filters):
        return {"success": False, "error": "Document not found or not in this workspace"}

    db.execute(
        sa_text(
            """
            INSERT INTO pinned_documents (chat_id, document_id, workspace_id, created_by_user_id)
            VALUES (CAST(:chat AS uuid), :doc, CAST(:ws AS uuid), :uid)
            ON CONFLICT (chat_id, document_id) DO NOTHING
            """
        ),
        {"chat": str(chat_id), "doc": document_id, "ws": str(workspace_id), "uid": user_id},
    )
    db.commit()
    return {"success": True, "chat_id": str(chat_id), "document_id": document_id}


def unpin_document(db: Any, *, chat_id: Any, document_id: Any, workspace_id: Any) -> Dict[str, Any]:
    try:
        document_id = int(document_id)
    except (TypeError, ValueError):
        return {"success": False, "error": "document_id must be an integer"}
    db.execute(
        sa_text(
            """
            DELETE FROM pinned_documents
            WHERE chat_id = CAST(:chat AS uuid) AND document_id = :doc AND workspace_id = CAST(:ws AS uuid)
            """
        ),
        {"chat": str(chat_id), "doc": document_id, "ws": str(workspace_id)},
    )
    db.commit()
    return {"success": True, "chat_id": str(chat_id), "document_id": document_id}


def list_pinned(db: Any, *, chat_id: Any, workspace_id: Any) -> List[Dict[str, Any]]:
    rows = db.execute(
        sa_text(
            """
            SELECT p.document_id, d.filename, p.created_at
            FROM pinned_documents p
            JOIN documents d ON d.id = p.document_id
            WHERE p.chat_id = CAST(:chat AS uuid) AND p.workspace_id = CAST(:ws AS uuid)
            ORDER BY p.created_at
            """
        ),
        {"chat": str(chat_id), "ws": str(workspace_id)},
    ).fetchall()
    return [
        {
            "document_id": r.document_id,
            "filename": r.filename,
            "pinned_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


def build_pinned_context(
    db: Any,
    *,
    chat_id: Any,
    workspace_id: Any,
    token_budget: int = 2000,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Assemble a token-budgeted context block from the chat's pinned documents.

    Returns ``(text, source_map)``; ``("", [])`` when nothing is pinned. Chunks
    are taken in document/pin order (presorted) so the budgeter fills as much of
    the pinned documents as fits without reordering by score.
    """
    from modules.rag.budget import select_within_budget, assemble_with_citations, count_tokens

    rows = db.execute(
        sa_text(
            """
            SELECT p.document_id, d.filename
            FROM pinned_documents p
            JOIN documents d ON d.id = p.document_id
            WHERE p.chat_id = CAST(:chat AS uuid) AND p.workspace_id = CAST(:ws AS uuid)
            ORDER BY p.created_at
            """
        ),
        {"chat": str(chat_id), "ws": str(workspace_id)},
    ).fetchall()
    if not rows:
        return "", []

    candidates: List[Dict[str, Any]] = []
    for r in rows:
        chunk_rows = db.execute(
            sa_text(
                "SELECT chunk_index, content FROM document_chunks "
                "WHERE document_id = :doc ORDER BY chunk_index"
            ),
            {"doc": r.document_id},
        ).fetchall()
        for c in chunk_rows:
            if c.content:
                candidates.append(
                    {
                        "content": c.content,
                        "source_file": r.filename,
                        "document_id": r.document_id,
                        "similarity": 1.0,
                        "tokens": count_tokens(c.content),
                    }
                )
    if not candidates:
        return "", []

    selection = select_within_budget(candidates, token_budget, presorted=True)
    return assemble_with_citations(selection.chunks, query=None, include_query_header=False)


def build_pinned_system_message(
    db: Any,
    *,
    chat_id: Any,
    workspace_id: Any,
    token_budget: int = 2000,
) -> Optional[str]:
    """The ready-to-inject system message for a chat's pinned docs, or None."""
    text, _ = build_pinned_context(
        db, chat_id=chat_id, workspace_id=workspace_id, token_budget=token_budget
    )
    if not text:
        return None
    return (
        "Pinned documents (the user has pinned these to this conversation; "
        "treat them as always-available context):\n\n" + text
    )

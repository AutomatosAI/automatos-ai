"""What a product's server key reaches of its workspace's knowledge documents (PRD-274 §5.2).

``api/documents.py`` lets a server key list, get and search (``documents:read``) and upload and
delete (``documents:write``) documents, always in the key's own workspace
(``core/auth/server_key_gate.py``). Of those, a key reaches only what no audience and no person
limits, the rule its memory reads follow (``api/widget_memory.py``):

* **No team limits it:** ``team_access`` is empty or unset. A document listed for teams is read
  by those teams' agents only (PRD-124), so it is not every caller's.
* **No person keeps it:** ``is_shared`` is not false. A document whose ``is_shared`` is false is
  its uploader's alone (``services/calibration/sources.py`` reads it so); unset counts as shared.

A document a key can't reach answers as a missing one (404), so the key learns nothing of it,
and a key's upload neither answers with it (an identical file) nor replaces it (the same name).
A member reaches the workspace's documents exactly as before. What a key uploads names the key
(``sdk:<key id>``) as its uploader, as the tickets and memories it writes do.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Set

from sqlalchemy import func

from core.auth.server_key_gate import is_server_key
from core.models import Document

# Who an upload names when the member path knows no person (unchanged from before).
NO_UPLOADER = "system"


def reachable_by(query: Any, ctx: Any) -> Any:
    """``query`` over documents, narrowed for a server key to the documents it may reach; a
    member's query as it is."""
    if not is_server_key(ctx):
        return query
    return query.filter(
        func.coalesce(func.cardinality(Document.team_access), 0) == 0,
        Document.is_shared.isnot(False),
    )


def reachable_or_none(db: Any, ctx: Any, document: Optional[Any]) -> Optional[Any]:
    """``document`` when the caller reaches it, else None. A member always does; a key's is
    read again under :func:`reachable_by`, so the rule has one home."""
    if document is None or not is_server_key(ctx):
        return document
    return reachable_by(db.query(Document).filter(Document.id == document.id), ctx).first()


def reachable_ids(db: Any, ctx: Any, document_ids: Iterable[Any]) -> Set[int]:
    """Which of ``document_ids`` the caller reaches in its own workspace."""
    ids = sorted({int(document_id) for document_id in document_ids})
    if not ids:
        return set()
    query = db.query(Document.id).filter(Document.workspace_id == ctx.workspace_id, Document.id.in_(ids))
    return {row.id for row in reachable_by(query, ctx).all()}


def search_answer_for(db: Any, ctx: Any, answer: Dict[str, Any]) -> Dict[str, Any]:
    """A ``POST /api/documents/search`` answer as the caller may see it: for a server key, only
    the documents it reaches, and the count to match; a member's answer as it is."""
    if not is_server_key(ctx):
        return answer
    results = answer.get("results") or []
    allowed = reachable_ids(db, ctx, [result["document_id"] for result in results])
    kept = [result for result in results if result["document_id"] in allowed]
    return {**answer, "results": kept, "total_results": len(kept)}


def uploaded_by(ctx: Any) -> str:
    """Who an upload or a replacement names: a server key as ``sdk:<key id>``; a member by their
    sign-in id, else ``system``, as before."""
    if is_server_key(ctx):
        return str(ctx.user.id)
    return (ctx.user.clerk_user_id if ctx.user else None) or NO_UPLOADER


__all__ = ["NO_UPLOADER", "reachable_by", "reachable_ids", "reachable_or_none", "search_answer_for", "uploaded_by"]

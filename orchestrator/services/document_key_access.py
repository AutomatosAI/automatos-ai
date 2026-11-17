"""What a product's server key reaches of its workspace's knowledge documents (PRD-274 §5.2).

``api/documents.py`` lets a server key list, get and search (``documents:read``) and upload and
delete (``documents:write``) documents, always in the key's own workspace
(``core/auth/server_key_gate.py``). Of those, a key reaches only what no audience and no person
limits, the rule its memory reads follow (``api/widget_memory.py``):

* **No team limits it:** ``team_access`` is empty or unset. A document listed for teams is read
  by those teams' agents only (PRD-124), so it is not every caller's.
* **No person keeps it:** ``is_shared`` is not false. A document whose ``is_shared`` is false is
  its uploader's alone (``services/calibration/sources.py`` reads it so); unset counts as shared.

A document a key can't reach answers as a missing one (404), so the key learns nothing of it.
A key's search maps a hit only by the hit's own document id, and only to a document it reaches:
a hit with no id is dropped, never resolved by its file name, which may also be the name of a
document the key can't read. A key's upload may not be limited to teams (400): a key writes only
documents it can read back. It never answers with a document the key can't reach (an identical
file), and is refused (409) where it would write over one: a document of the same name (which
F087 would make the upload's earlier version), or the workspace copy of a spreadsheet
(``services/spreadsheet_workspace.py``). A member reaches the workspace's documents exactly as
before. What a key uploads names the key (``sdk:<key id>``) as its uploader, as the tickets and
memories it writes do.

**What the refusal tells a key.** A server key's upload refuses a filename that a document the
key can't reach already has, or whose workspace copy belongs to a document the key can't reach;
the refusal tells the key such a filename exists in its own workspace. That is accepted: the key
is the workspace's own and is confined to it, and the copy cannot be moved aside instead,
because its readers find it by rebuilding the path from the document's file name
(``spreadsheet_workspace.workspace_copies``, which ``read_document`` calls).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from sqlalchemy import and_, func, not_, or_

from core.auth.server_key_gate import is_server_key
from core.models import Document

# Who an upload names when the member path knows no person (unchanged from before).
NO_UPLOADER = "system"
UPLOAD_CLASH = ("A document this key can't read already has this name, or owns the workspace copy "
                "this spreadsheet would write. Upload it under another name.")
KEY_TEAM_ACCESS_REFUSED = ("A server key writes only documents it can read back, so its upload can't be limited "
                           "to teams. Leave team_access out.")


def _reachable() -> Any:
    """The rule as one clause: no team limits the document, and no person keeps it."""
    return and_(func.coalesce(func.cardinality(Document.team_access), 0) == 0, Document.is_shared.isnot(False))


def reachable_by(query: Any, ctx: Any) -> Any:
    """``query`` over documents, narrowed for a server key to the documents it may reach; a
    member's query as it is."""
    if not is_server_key(ctx):
        return query
    return query.filter(_reachable())


def reachable_ids(db: Any, ctx: Any, document_ids: Iterable[int]) -> Set[int]:
    """Which of ``document_ids`` the caller reaches in its own workspace."""
    ids = sorted(set(document_ids))
    if not ids:
        return set()
    query = db.query(Document.id).filter(Document.workspace_id == ctx.workspace_id, Document.id.in_(ids))
    return {row.id for row in reachable_by(query, ctx).all()}


def _hit_document_id(hit: Dict[str, Any]) -> Optional[int]:
    """The document id a retrieval hit names, or None when it names none."""
    raw = (hit.get("metadata") or {}).get("document_id")
    try:
        return None if raw is None else int(raw)
    except (TypeError, ValueError):
        return None


def reachable_hits(db: Any, ctx: Any, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The retrieval hits a caller's search may map. A member's: all of them, as before. A server
    key's: only those naming by id a document of its own workspace it reaches. A hit with no id is
    dropped: its file name may also be the name of a document the key can't read."""
    if not is_server_key(ctx):
        return hits
    named = [_hit_document_id(hit) for hit in hits]
    allowed = reachable_ids(db, ctx, [document_id for document_id in named if document_id is not None])
    return [hit for hit, document_id in zip(hits, named) if document_id in allowed]


def refuses_teams(ctx: Any, team_access: Sequence[str]) -> bool:
    """Whether the caller's upload asks for teams it may not: a server key's asks for none, since a
    document a team limits is one the key could never list, get, delete or upload again; a
    member's may name any, as before."""
    return is_server_key(ctx) and bool(team_access)


def clashes_with_unreachable(db: Any, ctx: Any, filename: str, file_type: str) -> bool:
    """Whether a server key's upload of ``filename`` would write where a document it can't read
    already is: a document of the same name, or, for a spreadsheet, a spreadsheet whose workspace
    copy could have the same path. A member's upload never clashes; it goes as before."""
    if not is_server_key(ctx):
        return False
    from services.spreadsheet_workspace import SPREADSHEET_TYPES, table_paths_may_meet

    spreadsheet = file_type in SPREADSHEET_TYPES
    near = Document.filename == filename
    if spreadsheet:
        near = or_(near, Document.file_type.in_(sorted(SPREADSHEET_TYPES)))
    hidden = db.query(Document.filename, Document.original_filename, Document.file_type).filter(
        Document.workspace_id == ctx.workspace_id, not_(_reachable()), near).all()
    return any(
        row.filename == filename
        or (spreadsheet and row.file_type in SPREADSHEET_TYPES
            and table_paths_may_meet(filename, row.original_filename or row.filename))
        for row in hidden
    )


def uploaded_by(ctx: Any) -> str:
    """Who an upload or a replacement names: a server key as ``sdk:<key id>``; a member by their
    sign-in id, else ``system``, as before."""
    if is_server_key(ctx):
        return str(ctx.user.id)
    return (ctx.user.clerk_user_id if ctx.user else None) or NO_UPLOADER


__all__ = [
    "KEY_TEAM_ACCESS_REFUSED", "NO_UPLOADER", "UPLOAD_CLASH", "clashes_with_unreachable", "reachable_by",
    "reachable_hits", "reachable_ids", "refuses_teams", "uploaded_by",
]

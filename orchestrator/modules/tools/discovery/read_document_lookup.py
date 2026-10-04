"""Which document ``platform_read_document`` reads: exactly the one asked for.

F327 (night 9b, chat ca9d92d2, 4 Oct 15:48Z): Auto asked to read
club-box-october-2026.md and was handed roast-rules.md, then
margin-sheet-sep-2026.csv. The handler looked up ``documents.id`` exactly (no
list index, no search fallback, workspace and team scoped); Auto had sent 1523
and then 1522 from a list where club-box is 1518. Its first try had sent the
filename as ``document_id`` and was refused with "document_id must be an
integer", so it went guessing ids. Nothing told it the id it sent named a
different document from the one it meant.

Now the document can be named by its filename (``filename``, or the filename
sent as ``document_id``), and when both an id and a filename are sent they must
name the same document, or nothing is read and the reply says which document
each one is. An id this workspace has no readable document under is refused in
plain words with how to find the right one. Never a fall-back to another
document.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# More same-named documents than this are not listed one by one in a refusal.
MAX_NAMED_MATCHES = 10
# Other names the filename arrives under (night 9b: Auto sent ``document_name``).
FILENAME_KEYS = ("filename", "document_name")

HOW_TO_FIND = ("Find the right one with platform_list_documents (its search takes part of the "
               "name) and pass that document_id, or pass the document's filename.")
SAY_WHICH = ("Say which document to read: pass its document_id (from platform_list_documents or a "
             "search result) or its filename.")

Handler = Callable[[Session, UUID, Dict[str, Any]], Awaitable[Dict[str, Any]]]
Named = Tuple[int, str, Any]  # (id, filename, upload_date)


def _whole_id(raw: Any) -> Optional[int]:
    """``raw`` as a document id, or None when it is not a whole number."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    text = str(raw).strip()
    return int(text) if text.isdigit() else None


def requested(params: Dict[str, Any]) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """``(document_id, filename, refusal)`` from the call's params.

    A ``document_id`` that is not a number is taken as the document's filename
    (night 9b's first call), unless a filename was also sent."""
    name = next((str(params[k]).strip() for k in FILENAME_KEYS if str(params.get(k) or "").strip()), None)
    raw = params.get("document_id")
    if raw in (None, ""):
        return None, name, None if name else SAY_WHICH
    doc_id = _whole_id(raw)
    if doc_id is not None:
        return doc_id, name, None
    if name is None:
        return None, str(raw).strip(), None
    return None, name, (f"document_id must be a document's number, and '{raw}' is not one, so nothing was "
                        f"read. {HOW_TO_FIND}")


def _visible(db: Session, workspace_id: UUID, ids: List[int], agent_id: Any) -> set:
    """The ids in ``ids`` this caller may read: workspace always, team when it has one.
    Looked up through the modules at call time, so the scope is the handler's own."""
    from modules.rag import retrieval_filters
    from modules.tools.discovery import handlers_documents

    team = handlers_documents._resolve_agent_team(db, agent_id)
    filters = retrieval_filters.build_retrieval_filters(workspace_id=str(workspace_id), team=team)
    return {int(i) for i in retrieval_filters.allowed_document_ids(db, ids, filters)}


def _named(db: Session, workspace_id: UUID, name: str, doc_id: Optional[int] = None) -> List[Named]:
    """This workspace's documents whose filename is ``name`` (any case); only
    document ``doc_id`` when one is given."""
    from core.models import Document

    wanted = name.lower()
    query = (db.query(Document)
             .filter(Document.workspace_id == workspace_id)
             .filter(or_(func.lower(Document.original_filename) == wanted, func.lower(Document.filename) == wanted)))
    if doc_id is not None:
        query = query.filter(Document.id == doc_id)
    rows = query.order_by(Document.id).limit(MAX_NAMED_MATCHES).all()
    return [(row.id, row.original_filename or row.filename, row.upload_date) for row in rows]


def _filename_of(db: Session, workspace_id: UUID, doc_id: int) -> Optional[str]:
    """The filename of document ``doc_id`` in this workspace, or None."""
    from core.models import Document

    row = (db.query(Document)
           .filter(Document.workspace_id == workspace_id, Document.id == doc_id).first())
    return (row.original_filename or row.filename) if row else None


def _many_named(name: str, matches: List[Named]) -> str:
    """The refusal when several readable documents share ``name``: each id, to pick one."""
    listed = "; ".join(f"{doc_id} (uploaded {uploaded.date().isoformat() if uploaded else 'at an unknown time'})"
                       for doc_id, _, uploaded in matches)
    return (f"{len(matches)} documents are named '{name}': {listed}. Nothing was read. "
            "Pass the document_id of the one you mean.")


def _by_name(db: Session, workspace_id: UUID, name: str, agent_id: Any) -> Tuple[Optional[int], Optional[str]]:
    """The one readable document named ``name``, or the refusal saying why not."""
    found = _named(db, workspace_id, name)
    readable = _visible(db, workspace_id, [doc_id for doc_id, _, _ in found], agent_id) if found else set()
    matches = [m for m in found if m[0] in readable]
    if not matches:
        return None, f"No document named '{name}' in this workspace, so nothing was read. {HOW_TO_FIND}"
    if len(matches) > 1:
        return None, _many_named(name, matches)
    return matches[0][0], None


def _not_the_named_one(db: Session, workspace_id: UUID, doc_id: int, name: str, agent_id: Any) -> str:
    """The refusal when ``doc_id`` and ``name`` are two different documents."""
    actual = _filename_of(db, workspace_id, doc_id)
    named_id, _ = _by_name(db, workspace_id, name, agent_id)
    where = f"'{name}' is document {named_id}." if named_id else HOW_TO_FIND
    return f"Document {doc_id} is '{actual}', not '{name}', so nothing was read. {where}"


def resolve(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Tuple[Optional[int], Optional[str]]:
    """``(document_id, refusal)``: the readable document the call names, or why none is read."""
    doc_id, name, refusal = requested(params)
    if refusal:
        return None, refusal
    agent_id = params.get("_agent_id")
    if doc_id is None:
        return _by_name(db, workspace_id, name, agent_id)
    if doc_id not in _visible(db, workspace_id, [doc_id], agent_id):
        return None, f"There is no document {doc_id} you can read in this workspace, so nothing was read. {HOW_TO_FIND}"
    if name and not _named(db, workspace_id, name, doc_id):
        return None, _not_the_named_one(db, workspace_id, doc_id, name, agent_id)
    return doc_id, None


def reads_exactly_the_named_document(handler: Handler) -> Handler:
    """Run ``handler`` on the one document the call names, by id, so it never
    reads another; refuse in plain words when the call names none it can read."""

    @functools.wraps(handler)
    async def wrapper(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            doc_id, refusal = resolve(db, workspace_id, params or {})
        except Exception:
            logger.exception("[F327] read_document could not look up the document in workspace %s", workspace_id)
            return {"success": False, "error": "The document could not be looked up just now, so nothing was read."}
        if refusal:
            return {"success": False, "error": refusal}
        return await handler(db, workspace_id, {**(params or {}), "document_id": doc_id})

    return wrapper

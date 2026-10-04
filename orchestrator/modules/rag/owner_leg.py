"""F311 (night 9): when agents' reports fill a search, the owner's documents are searched too.

By mid-night the workspace held 12 of the owner's documents and about 40 reports that
agents had filed while answering the same questions. Every search took its 15 best
chunks from the whole store, and the reports, written as answers to those very
questions, scored 0.70 to 0.90 against 0.55 to 0.70 for the owner's documents. Asked
"Who do I talk to about reordering Kirinyaga, and how long will it take to arrive?"
(chat bb91ddec, 13:09), all 15 hits came from reports and christmas-boxes-2026.md;
importers-and-green-buying.md, which names Maya Odum at Tidewater and a 3-week lead
time, was not among them, and Auto answered "Ellie" (ledger L99). The October-box
importers question (chat a75487a6, 13:01) found reports only; Northfield's 30 days were
missed (L85).

``owners_documents_searched`` wraps ``RAGService._get_candidates``: when the hits it
found include an agent's report, the owner's own documents (``source_type`` not
``agent_output``; the newest ``RAG_OWNER_LEG_MAX_DOCUMENTS``) are searched on their own
with the same query, limit and floor, and their hits join the candidates. A search
whose hits are all the owner's costs nothing more. Dial: ``RAG_OWNER_LEG_ENABLED``.
"""
from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from sqlalchemy import text

from config import config

logger = logging.getLogger(__name__)

AGENT_REPORT_SOURCE_TYPE = "agent_output"
_OWNER_DOCUMENTS_SQL = text(
    "SELECT id FROM documents WHERE workspace_id = CAST(:ws AS uuid) "
    "AND status IN ('completed', 'processed') AND source_type IS DISTINCT FROM :report "
    "ORDER BY upload_date DESC NULLS LAST, id DESC LIMIT :cap"
)


def owner_document_ids(workspace_id: str, cap: int) -> List[int]:
    """The workspace's own searchable documents (not an agent's report), newest first."""
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        rows = db.execute(_OWNER_DOCUMENTS_SQL, {"ws": str(workspace_id), "report": AGENT_REPORT_SOURCE_TYPE,
                                                 "cap": int(cap)}).fetchall()
        return [int(row[0]) for row in rows]
    finally:
        db.close()


def candidate_doc_id(candidate: Dict[str, Any]) -> Optional[str]:
    """A candidate's document id as text (S3 Vectors keeps it as ``external_file_id``)."""
    meta = candidate.get("metadata") or {}
    doc_id = candidate.get("document_id") or meta.get("document_id") or meta.get("external_file_id")
    return str(doc_id) if doc_id else None


def _as_candidate(hit: Dict[str, Any]) -> Dict[str, Any]:
    """A backend hit in ``RAGService._get_candidates``'s candidate shape."""
    meta = hit.get("metadata") or {}
    file_path = hit.get("file_path") or ""
    return {
        "id": hit.get("key", ""), "content": hit.get("content", ""),
        "source_file": hit.get("file_name", file_path or "unknown"),
        "document_id": hit.get("external_file_id") or meta.get("external_file_id") or meta.get("document_id") or 0,
        "file_type": file_path.rsplit(".", 1)[-1] if file_path else "",
        "similarity": hit.get("score", 0.0), "metadata": meta, "parent_content": None, "headers": {},
    }


def _merged(found: List[Dict[str, Any]], owners: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The hits found plus the owner's hits not already among them, best first."""
    seen = {candidate.get("id") for candidate in found}
    extra = [candidate for candidate in owners if candidate.get("id") not in seen]
    if not extra:
        return found
    return sorted([*found, *extra], key=lambda c: float(c.get("similarity") or 0.0), reverse=True)


async def _owners_hits(service: Any, query: str, limit: int, min_similarity: float,
                       workspace_id: str) -> List[Dict[str, Any]]:
    """The owner's documents searched on their own, as candidates."""
    ids = await asyncio.to_thread(owner_document_ids, workspace_id, config.RAG_OWNER_LEG_MAX_DOCUMENTS)
    if not ids:
        return []
    backend = await service._get_doc_backend(workspace_id)
    search = getattr(backend, "search_in_documents", None)
    if search is None:
        logger.warning("[F311] %s cannot search within documents; the owner's leg is skipped", type(backend).__name__)
        return []
    embedding = await service._embedding_manager.generate_embedding(query)
    vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
    hits = await asyncio.to_thread(search, vector, ids, limit=limit, min_score=min_similarity)
    return [_as_candidate(hit) for hit in hits]


async def with_owner_leg(service: Any, found: List[Dict[str, Any]], query: str, limit: int,
                         min_similarity: float, workspace_id: Optional[str]) -> List[Dict[str, Any]]:
    """``found`` plus the owner's own hits when ``found`` holds an agent's report."""
    if not config.RAG_OWNER_LEG_ENABLED or not workspace_id or not found:
        return found
    facts = await asyncio.to_thread(service._document_ranking_facts, [candidate_doc_id(c) for c in found])
    if not any(is_report for is_report, _uploaded in facts.values()):
        return found
    owners = await _owners_hits(service, query, limit, min_similarity, str(workspace_id))
    logger.info("[F311] owner's documents searched on their own: %d hit(s) beside %d found", len(owners), len(found))
    return _merged(found, owners)


def owners_documents_searched(get_candidates: Callable[..., Awaitable[List[Dict[str, Any]]]]):
    """Decorate ``RAGService._get_candidates`` with the owner's own search (F311).
    A failure of that search is logged and the hits found are returned as they were."""

    @functools.wraps(get_candidates)
    async def search(self: Any, query: str, limit: int = 20, min_similarity: float = 0.5,
                     workspace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        found = await get_candidates(self, query, limit=limit, min_similarity=min_similarity, workspace_id=workspace_id)
        try:
            return await with_owner_leg(self, found, query, limit, min_similarity,
                                        workspace_id or getattr(self, "_workspace_id", None))
        except Exception:  # noqa: BLE001 — the search found what it found; the owner's leg is extra
            logger.warning("[F311] owner's documents search failed; the hits found stand", exc_info=True)
            return found

    return search

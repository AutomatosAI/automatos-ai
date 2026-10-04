"""F311 (night 9): a passage from a short document of the owner's is handed over whole.

A search keeps one passage per document and hands over five: each further chunk of a
document scores 0.7 times less (``RAGService._optimize_with_context_optimizer``), so the
section beside the one that matched never comes. "A café wants 10 kg next week, what do
we charge for delivery?" (chat 9928b259, 12:33) was handed wholesale-terms-2026.md's
Prices and Ordering chunks; its Delivery chunk, with the carriage charges beside it,
scored 0.646 x 0.7 and was dropped for christmas-boxes-2026.md and roast-rules.md, and
Auto said the terms name no delivery charge (ledger L1). The October-box importers
question (chat 7db97485, 12:36) was handed importers-and-green-buying.md's first two
sections; Northfield's, with its 30 days, was the third (L7).

The owner's documents in a small business are often a page long (the night's twelve:
439 to 1,204 bytes). ``short_documents_whole`` wraps ``RAGService._retrieve_impl``: a
passage from an owner's document whose whole text is at most
``RAG_WHOLE_DOCUMENT_MAX_TOKENS`` is replaced by that text (its chunks in order, read
within the workspace), and a second passage of the same document is dropped. An agent's
report is never read whole. 0 turns it off.
"""
from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from sqlalchemy import text

from config import config
from core.context_guard import count_tokens
from modules.rag.owner_leg import candidate_doc_id
from modules.rag.passages import all_owners, rebuilt

logger = logging.getLogger(__name__)

_DOCUMENT_CHUNKS_SQL = text(
    "SELECT dc.document_id, dc.content FROM document_chunks dc JOIN documents d ON d.id = dc.document_id "
    "WHERE d.workspace_id = CAST(:ws AS uuid) AND dc.document_id = ANY(:ids) "
    "AND COALESCE(dc.chunk_type, 'child') = 'child' ORDER BY dc.document_id, dc.chunk_index"
)


def document_texts(workspace_id: str, document_ids: Sequence[int]) -> Dict[str, str]:
    """``{document id: its chunks' text in order}`` for these documents of the workspace."""
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        rows = db.execute(_DOCUMENT_CHUNKS_SQL, {"ws": str(workspace_id), "ids": list(document_ids)}).fetchall()
    finally:
        db.close()
    pieces: Dict[str, List[str]] = {}
    for doc_id, content in rows:
        pieces.setdefault(str(doc_id), []).append(content or "")
    return {doc_id: "\n\n".join(parts) for doc_id, parts in pieces.items()}


def _short_owners_texts(service: Any, chunks: Sequence[Dict[str, Any]], workspace_id: str) -> Dict[str, str]:
    """The whole text of each owner's document among the passages that is short enough."""
    owners = [int(doc_id) for doc_id in all_owners(service, chunks) if doc_id.isdigit()]
    if not owners:
        return {}
    texts = document_texts(workspace_id, owners)
    return {doc_id: body for doc_id, body in texts.items()
            if body.strip() and count_tokens(body) <= config.RAG_WHOLE_DOCUMENT_MAX_TOKENS}


def _read_whole(chunks: Sequence[Dict[str, Any]], whole: Dict[str, str]) -> List[Dict[str, Any]]:
    """The passages with a short document's first passage made its whole text and its others dropped."""
    read: List[Dict[str, Any]] = []
    done = set()
    for chunk in chunks:
        doc_id = candidate_doc_id(chunk)
        if doc_id not in whole:
            read.append(chunk)
        elif doc_id not in done:
            done.add(doc_id)
            read.append({**chunk, "content": whole[doc_id], "tokens": count_tokens(whole[doc_id]),
                         "whole_document": True})
    return read


async def whole_where_short(service: Any, result: Any, workspace_id: Optional[str]) -> Any:
    """``result`` with each short owner's document handed over whole."""
    if config.RAG_WHOLE_DOCUMENT_MAX_TOKENS <= 0 or not workspace_id or not result.chunks:
        return result
    whole = await asyncio.to_thread(_short_owners_texts, service, result.chunks, str(workspace_id))
    if not whole:
        return result
    logger.info("[F311] %d short document(s) of the owner's handed over whole", len(whole))
    return rebuilt(result, _read_whole(result.chunks, whole))


def short_documents_whole(retrieve_impl: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Decorate ``RAGService._retrieve_impl`` (F311). A failure is logged and the
    passages are handed over as they were selected."""

    @functools.wraps(retrieve_impl)
    async def retrieve(self: Any, query: str, max_chunks: int = 8, max_tokens: Optional[int] = None,
                       diversity: Optional[float] = None, context_type: str = "chatbot",
                       workspace_id: Optional[str] = None, team: Optional[str] = None) -> Any:
        result = await retrieve_impl(self, query, max_chunks=max_chunks, max_tokens=max_tokens, diversity=diversity,
                                     context_type=context_type, workspace_id=workspace_id, team=team)
        try:
            return await whole_where_short(self, result, workspace_id)
        except Exception:  # noqa: BLE001 — the passages stand as they were selected
            logger.warning("[F311] reading short documents whole failed; the passages stand", exc_info=True)
            return result

    return retrieve

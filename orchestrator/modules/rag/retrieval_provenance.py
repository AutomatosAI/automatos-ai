"""PRD-185 S7: per-turn retrieval provenance.

Capture the document/chunk ids a chat turn actually retrieved so that a later
thumbs vote can write a complete ``rag_feedback`` row. The PRD-179 live ranker
reads that table via ``UNNEST(document_ids)`` to down-weight flagged docs, so
without the retrieved ids on hand at vote time the feedback loop ships hollow.

Pure functions — no DB, network, or framework — so they run in CI with no
external service (per ``feedback-no-local-servers``).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple

# The chat tools that perform document retrieval. Provenance is collected ONLY
# for these — a tool like ``query_database`` can surface a user column literally
# named ``document_id`` that is not a retrieved RAG doc, and must not pollute
# ``rag_feedback``. Mirrors the retrieval branch in
# ``consumers/chatbot/tool_router.build_tool_context_message``.
RETRIEVAL_TOOL_NAMES = frozenset({
    "search_knowledge",
    "search_documents",
    "semantic_search",
})

# Keys carrying a single id / an id list, scanned within a retrieval result.
_DOC_ID_KEYS = ("document_id", "doc_id")
_DOC_IDS_KEYS = ("document_ids", "doc_ids")
_CHUNK_ID_KEYS = ("chunk_id",)
_CHUNK_IDS_KEYS = ("chunk_ids",)

# Tool results are shallow in practice; bound the scan so a pathological payload
# can never spin.
_MAX_DEPTH = 6

# rag_feedback stores TEXT NOT NULL for query — bound what we persist.
_MAX_QUERY_LEN = 2000


def is_retrieval_tool(tool_name: Optional[str]) -> bool:
    """True when a tool's results represent retrieved documents."""
    return bool(tool_name) and tool_name in RETRIEVAL_TOOL_NAMES


def _coerce_int(value: Any) -> Optional[int]:
    """Return ``value`` as an int only if it cleanly represents one.

    ``rag_feedback.document_ids`` / ``chunk_ids`` are ``INTEGER[]``; string chunk
    uuids and floats are dropped rather than corrupt the array.
    """
    if isinstance(value, bool):  # bool is an int subclass — never an id
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.lstrip("-").isdigit():
            return int(s)
    return None


def _scan(obj: Any, docs: Set[int], chunks: Set[int], depth: int) -> None:
    if depth > _MAX_DEPTH:
        return
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key in _DOC_ID_KEYS:
                cid = _coerce_int(val)
                if cid is not None:
                    docs.add(cid)
            elif key in _DOC_IDS_KEYS and isinstance(val, (list, tuple)):
                for item in val:
                    cid = _coerce_int(item)
                    if cid is not None:
                        docs.add(cid)
            elif key in _CHUNK_ID_KEYS:
                cid = _coerce_int(val)
                if cid is not None:
                    chunks.add(cid)
            elif key in _CHUNK_IDS_KEYS and isinstance(val, (list, tuple)):
                for item in val:
                    cid = _coerce_int(item)
                    if cid is not None:
                        chunks.add(cid)
            else:
                _scan(val, docs, chunks, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _scan(item, docs, chunks, depth + 1)


def collect_doc_ids_from_tool_result(
    result: Optional[Dict[str, Any]],
) -> Tuple[Set[int], Set[int]]:
    """Extract ``(document_ids, chunk_ids)`` from an ``execute_and_format`` result.

    Scans ``raw_result`` and ``frontend_data`` defensively so it works across the
    varied shapes the retrieval tools return. Caller must gate on
    :func:`is_retrieval_tool` — this does not know the tool name.
    """
    docs: Set[int] = set()
    chunks: Set[int] = set()
    if not isinstance(result, dict):
        return docs, chunks
    for key in ("raw_result", "frontend_data"):
        _scan(result.get(key), docs, chunks, 0)
    return docs, chunks


def build_retrieval_context(
    document_ids: Set[int],
    chunk_ids: Set[int],
    query: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Assemble the blob stored on ``messages.retrieval_context``.

    Returns ``None`` when nothing was retrieved so the column stays NULL rather
    than recording an empty turn. Ids are sorted for stable, diffable storage.
    """
    if not document_ids and not chunk_ids:
        return None
    ctx: Dict[str, Any] = {
        "document_ids": sorted(document_ids),
        "chunk_ids": sorted(chunk_ids),
    }
    if query:
        ctx["query"] = query[:_MAX_QUERY_LEN]
    return ctx

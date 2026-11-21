"""The full text of retrieved chunks and their neighbours, in one round-trip (PRD-154 S1).

``RAGService._expand_to_parent_context`` plans each candidate's window here, then fetches
every (document_id, chunk_index) key it needs in ONE asyncpg query. The full chunk text
lives in ``document_chunks.content``; a candidate's ``content`` is only the 500-char S3
preview.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# (candidate, document_id or None, [chunk_index, ...] or None)
Plan = Tuple[Dict, Optional[int], Optional[List[int]]]

# PRD-172 F005: documents.workspace_id is pinned, so a chunk is hydrated ONLY when its
# parent document belongs to this workspace. When workspace_id is unavailable ($3 IS NULL)
# the predicate is a no-op (only reachable for a non-multi-tenant / workspace-less caller),
# preserving prior behaviour without widening a tenant's scope.
CHUNK_TEXTS_SQL = """
    SELECT dc.document_id, dc.chunk_index, dc.content
    FROM document_chunks dc
    JOIN unnest($1::int[], $2::int[]) AS k(document_id, chunk_index)
      ON dc.document_id = k.document_id
     AND dc.chunk_index = k.chunk_index
    JOIN documents d ON d.id = dc.document_id
    WHERE ($3::uuid IS NULL OR d.workspace_id = $3::uuid)
"""


def _document_id(raw: Any) -> Optional[int]:
    """The candidate's document id as an int (S3 Vectors stores it as a string)."""
    try:
        return int(raw) if raw else None
    except (ValueError, TypeError):
        return None


def expansion_plans(candidates: Iterable[Dict], window: int) -> Tuple[List[Plan], Set[Tuple[int, int]]]:
    """Each candidate's window, planned once, and the union of keys to hydrate."""
    plans: List[Plan] = []
    needed: Set[Tuple[int, int]] = set()
    for candidate in candidates:
        chunk_metadata = candidate.get("metadata", {})
        chunk_index = chunk_metadata.get("chunk_index") if isinstance(chunk_metadata, dict) else None
        doc_id = _document_id(candidate.get("document_id"))
        if doc_id is None or chunk_index is None:
            plans.append((candidate, None, None))
            continue
        idx_window = list(range(max(0, chunk_index - window), chunk_index + window + 1))
        needed.update((doc_id, ci) for ci in idx_window)
        plans.append((candidate, doc_id, idx_window))
    return plans, needed


async def fetch_chunk_texts(needed: Iterable[Tuple[int, int]], workspace_id: Optional[str]) -> List[Any]:
    """ONE round-trip: the requested (doc_id, chunk_index) keys joined against
    document_chunks through two parallel arrays, on a connection secured like every
    other connection to the application's database (PRD-278 W3)."""
    import asyncpg
    from config import config as app_config
    from core.database.tls import database_tls

    pairs = list(needed)
    conn = await asyncpg.connect(app_config.DATABASE_URL, **database_tls().asyncpg_kwargs())
    try:
        return await conn.fetch(
            CHUNK_TEXTS_SQL,
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            str(workspace_id) if workspace_id else None,
        )
    finally:
        await conn.close()

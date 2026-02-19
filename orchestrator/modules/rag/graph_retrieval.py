"""
Multi-Hop Graph Retrieval
==========================

Traverses the entity relationship graph to find related documents beyond
direct vector similarity.  Merges graph-discovered results with standard
vector search results, deduplicating and tagging each result with its
retrieval method ('vector', 'graph', or 'both').

Depends on the document_entities and document_relationships tables
populated by entity_extractor.py (US-018).
"""

import logging
from typing import Dict, List, Optional

from sqlalchemy import text

from .entity_extractor import extract_entities

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph traversal
# ---------------------------------------------------------------------------

async def get_related_entities(
    db_session,
    entity_names: List[str],
    workspace_id: str,
    max_hops: int = 2,
) -> List[Dict]:
    """
    Given a set of entity names, traverse the relationship graph up to
    *max_hops* to discover related entities and their documents.

    Uses a recursive CTE in PostgreSQL for breadth-first graph expansion.

    Args:
        db_session: SQLAlchemy Session (sync or async-compatible)
        entity_names: Seed entity names to start traversal from
        workspace_id: Workspace UUID string
        max_hops: Maximum traversal depth (default 2)

    Returns:
        List of dicts with keys:
            document_id, entity_name, entity_type, depth
        Sorted by depth (nearest first), then document_id.
    """
    if not entity_names:
        return []

    graph_cte_sql = text("""
        WITH RECURSIVE entity_graph AS (
            -- Base case: entities matching seed names
            SELECT
                de.id,
                de.name,
                de.entity_type,
                de.document_id,
                0 AS depth
            FROM document_entities de
            WHERE de.name = ANY(:entity_names)
              AND de.workspace_id = :workspace_id

            UNION ALL

            -- Recursive case: follow relationships in both directions
            SELECT
                de2.id,
                de2.name,
                de2.entity_type,
                de2.document_id,
                eg.depth + 1
            FROM entity_graph eg
            JOIN document_relationships dr
                ON dr.source_entity_id = eg.id
                OR dr.target_entity_id = eg.id
            JOIN document_entities de2
                ON de2.id = CASE
                    WHEN dr.source_entity_id = eg.id THEN dr.target_entity_id
                    ELSE dr.source_entity_id
                END
            WHERE eg.depth < :max_hops
              AND de2.workspace_id = :workspace_id
        )
        SELECT DISTINCT
            document_id,
            name         AS entity_name,
            entity_type,
            MIN(depth)   AS min_depth
        FROM entity_graph
        GROUP BY document_id, name, entity_type
        ORDER BY min_depth, document_id
    """)

    try:
        result = db_session.execute(graph_cte_sql, {
            "entity_names": entity_names,
            "workspace_id": workspace_id,
            "max_hops": max_hops,
        })
        rows = result.fetchall()

        related = [
            {
                "document_id": row.document_id,
                "entity_name": row.entity_name,
                "entity_type": row.entity_type,
                "depth": row.min_depth,
            }
            for row in rows
        ]

        logger.debug(
            "Graph traversal from %d seed entities returned %d results (max_hops=%d)",
            len(entity_names), len(related), max_hops,
        )
        return related

    except Exception as e:
        logger.error("Graph traversal failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Fetch chunk content for graph-discovered documents
# ---------------------------------------------------------------------------

async def _fetch_document_chunks(
    db_session,
    document_ids: List[int],
    workspace_id: str,
    limit_per_doc: int = 2,
) -> List[Dict]:
    """
    Retrieve the top chunks for a set of document IDs.

    Fetches the first *limit_per_doc* chunks per document, ordered by
    chunk_index so we get the most relevant (usually intro) content.
    """
    if not document_ids:
        return []

    fetch_sql = text("""
        SELECT
            dc.id          AS chunk_id,
            dc.document_id,
            dc.content,
            dc.chunk_index,
            dc.metadata,
            d.filename
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.document_id
        WHERE dc.document_id = ANY(:document_ids)
          AND d.workspace_id = :workspace_id
        ORDER BY dc.document_id, dc.chunk_index
    """)

    try:
        result = db_session.execute(fetch_sql, {
            "document_ids": document_ids,
            "workspace_id": workspace_id,
        })
        rows = result.fetchall()

        # Limit per document
        doc_counts: Dict[int, int] = {}
        chunks: List[Dict] = []
        for row in rows:
            count = doc_counts.get(row.document_id, 0)
            if count >= limit_per_doc:
                continue
            doc_counts[row.document_id] = count + 1

            chunks.append({
                "chunk_id": row.chunk_id,
                "document_id": row.document_id,
                "content": row.content,
                "chunk_index": row.chunk_index,
                "metadata": row.metadata,
                "source_file": row.filename,
                "retrieval_method": "graph",
            })

        return chunks

    except Exception as e:
        logger.error("Failed to fetch document chunks: %s", e)
        return []


# ---------------------------------------------------------------------------
# Main entry point: graph-enhanced retrieval
# ---------------------------------------------------------------------------

async def graph_enhanced_retrieval(
    db_session,
    query: str,
    vector_results: List[Dict],
    workspace_id: str,
    max_graph_results: int = 5,
    max_hops: int = 2,
) -> List[Dict]:
    """
    Augment vector search results with graph-discovered documents.

    Pipeline:
    1. Extract entities from the query text.
    2. Also extract entities from top vector-result content.
    3. Combine seed entity names and traverse the relationship graph.
    4. Fetch chunks for graph-discovered documents not already in vector results.
    5. Merge and deduplicate, tagging each result with retrieval_method.

    Args:
        db_session: SQLAlchemy Session
        query: Original user query string
        vector_results: List of chunk dicts from vector search (must have 'document_id')
        workspace_id: Workspace UUID string
        max_graph_results: Max additional graph results to return
        max_hops: Graph traversal depth

    Returns:
        Merged list: vector_results + graph results, deduplicated by document_id,
        each tagged with retrieval_method ('vector', 'graph', or 'both').
    """
    # ------------------------------------------------------------------
    # Step 1 & 2: Collect seed entity names
    # ------------------------------------------------------------------
    query_entities = extract_entities(query)
    seed_names = [e["name"] for e in query_entities]

    # Also mine entities from top vector results for richer seed set
    top_n = min(3, len(vector_results))
    for chunk in vector_results[:top_n]:
        content = chunk.get("content", "")
        if content:
            chunk_entities = extract_entities(content)
            seed_names.extend(e["name"] for e in chunk_entities[:5])

    # Deduplicate seed names (preserve order)
    seen = set()
    unique_seeds: List[str] = []
    for name in seed_names:
        norm = name.lower()
        if norm not in seen:
            seen.add(norm)
            unique_seeds.append(name)

    if not unique_seeds:
        logger.debug("No seed entities found; returning vector results as-is.")
        # Tag vector results
        for r in vector_results:
            r.setdefault("retrieval_method", "vector")
        return vector_results

    # ------------------------------------------------------------------
    # Step 3: Graph traversal
    # ------------------------------------------------------------------
    graph_entities = await get_related_entities(
        db_session, unique_seeds, workspace_id, max_hops=max_hops,
    )

    # Collect document IDs not already present in vector results
    vector_doc_ids = {r.get("document_id") for r in vector_results if r.get("document_id")}
    graph_doc_ids = []
    for ge in graph_entities:
        did = ge["document_id"]
        if did not in vector_doc_ids and did not in graph_doc_ids:
            graph_doc_ids.append(did)
    graph_doc_ids = graph_doc_ids[:max_graph_results]

    # ------------------------------------------------------------------
    # Step 4: Fetch chunks for new graph documents
    # ------------------------------------------------------------------
    graph_chunks: List[Dict] = []
    if graph_doc_ids:
        graph_chunks = await _fetch_document_chunks(
            db_session, graph_doc_ids, workspace_id,
        )

    # ------------------------------------------------------------------
    # Step 5: Merge and deduplicate
    # ------------------------------------------------------------------
    merged = _merge_results(vector_results, graph_chunks)

    logger.info(
        "Graph-enhanced retrieval: %d vector + %d graph = %d merged results",
        len(vector_results), len(graph_chunks), len(merged),
    )
    return merged


def _merge_results(
    vector_results: List[Dict],
    graph_results: List[Dict],
) -> List[Dict]:
    """
    Merge vector and graph results, deduplicating by document_id.

    Priority:
    - If a document appears in both, mark retrieval_method = 'both'
    - Vector results come first (they have similarity scores)
    - Graph results are appended after
    """
    # Index graph results by document_id for quick lookup
    graph_by_doc: Dict[int, Dict] = {}
    for gr in graph_results:
        did = gr.get("document_id")
        if did and did not in graph_by_doc:
            graph_by_doc[did] = gr

    seen_doc_ids: set = set()
    merged: List[Dict] = []

    # First pass: vector results
    for vr in vector_results:
        vr = dict(vr)  # shallow copy to avoid mutating caller's data
        did = vr.get("document_id")
        if did in graph_by_doc:
            vr["retrieval_method"] = "both"
        else:
            vr["retrieval_method"] = "vector"
        merged.append(vr)
        if did:
            seen_doc_ids.add(did)

    # Second pass: graph-only results
    for gr in graph_results:
        did = gr.get("document_id")
        if did and did not in seen_doc_ids:
            gr = dict(gr)
            gr["retrieval_method"] = "graph"
            merged.append(gr)
            seen_doc_ids.add(did)

    return merged

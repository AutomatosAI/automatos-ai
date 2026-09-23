"""
pgvector-local document backend (PRD-197 S5)
============================================

The open-core/local edition's document-vector READ leg. With
``S3_VECTORS_ENABLED=false`` (the OSS default — S3 Vectors is AWS-only),
document ingestion already writes each chunk's embedding inline into
``document_chunks.embedding`` ("legacy pgvector mode" in
``modules/rag/ingestion/manager._persist_chunks``); until this backend
existed nothing could read those vectors back, so a fresh clone's
``RAGService.retrieve`` constructed no backend and returned empty.

The result-dict contract mirrors ``S3VectorsBackend.search()`` (key / score /
metadata / content / file_name / …) so ``RAGService._get_candidates`` consumes
either backend interchangeably. Cosine uses the pgvector ``<=>`` operator
(distance; similarity = 1 − distance), which the shipped
``idx_document_chunks_embedding`` HNSW index (``vector_cosine_ops``) serves —
NOT the ``<->`` L2 operator the deleted F079 zombie store mislabeled as
cosine (PRD-197 S1).

S3 Vectors remains the SaaS path; selection happens in
``RAGService._get_doc_backend`` and ``get_vector_store`` off the committed
config, never here.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

logger = logging.getLogger(__name__)

# ``document_chunks`` is not the same shape in every deployment. On this local
# stack (verified 2026-09-19) ``embedding`` is TEXT holding a pgvector literal
# and ``workspace_id`` is TEXT, while ``documents``/``board_tasks`` use uuid and
# a migrated deployment has ``embedding vector(N)``. Querying either shape with
# the other's SQL fails outright:
#   "operator does not exist: text <=> vector"
#   "operator does not exist: text = uuid"
# So the column types are read ONCE per process and the SQL is built to match —
# a real vector column is compared directly (the HNSW index is usable), a text
# one is cast (no index exists on it to lose).
_COLUMN_TYPES: Optional[Tuple[str, str]] = None


def _column_types(db) -> Tuple[str, str]:
    """``(embedding_type, workspace_id_type)`` for ``document_chunks``, memoized."""
    global _COLUMN_TYPES
    if _COLUMN_TYPES is not None:
        return _COLUMN_TYPES
    embedding_type, workspace_type = "USER-DEFINED", "uuid"
    try:
        rows = db.execute(
            text(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = 'document_chunks' "
                "  AND column_name IN ('embedding', 'workspace_id')"
            )
        ).fetchall()
        found = {r.column_name: r.data_type for r in rows}
        embedding_type = found.get("embedding", embedding_type)
        workspace_type = found.get("workspace_id", workspace_type)
    except Exception:  # noqa: BLE001 — fall back to the migrated shape
        logger.warning("pgvector-local: could not read document_chunks column types", exc_info=True)
        return embedding_type, workspace_type
    _COLUMN_TYPES = (embedding_type, workspace_type)
    logger.info(
        "pgvector-local: document_chunks.embedding=%s workspace_id=%s",
        embedding_type, workspace_type,
    )
    return _COLUMN_TYPES


def embedding_expr(embedding_type: str, column: str = "dc.embedding") -> str:
    """The column as a ``vector``, whatever shape it is stored in.

    A real ``vector`` column is used directly (the HNSW index stays usable). A
    TEXT column is cast — and on this stack the text is a Postgres ARRAY literal
    ``{0.1,0.2}``, which ``vector`` refuses ("invalid input syntax for type
    vector"), so the braces are translated to the brackets ``vector`` expects.
    ``replace`` is a no-op on text already stored as ``[0.1,0.2]``.
    """
    if embedding_type not in ("text", "character varying"):
        return column
    return f"CAST(replace(replace({column}, '{{', '['), '}}', ']') AS vector)"


def workspace_predicate(workspace_type: str, column: str = "dc.workspace_id") -> str:
    """``<column> = <bind>``, with whichever cast makes the types agree."""
    if workspace_type in ("text", "character varying"):
        return f"{column} = :ws"
    return f"{column} = CAST(:ws AS uuid)"


class PgVectorLocalBackend:
    """Workspace-scoped document-vector search over ``document_chunks``."""

    def __init__(self, workspace_id: str):
        if not workspace_id:
            raise ValueError("workspace_id is required for the pgvector-local backend")
        self.workspace_id = str(workspace_id)

    async def initialize(self) -> None:
        """No-op — the table and HNSW index are migration-managed."""

    def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        min_score: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Cosine search over this workspace's embedded chunks.

        Same fail-closed workspace contract as the S3 backend: the backend is
        bound to one workspace at construction; an explicit
        ``filters['workspace_id']`` that disagrees returns ``[]`` rather than
        silently widening scope.
        """
        required_ws = self.workspace_id
        if filters:
            filter_ws = filters.get("workspace_id")
            if filter_ws is not None and str(filter_ws) != required_ws:
                logger.warning(
                    "pgvector-local search: filter workspace_id=%s != backend "
                    "workspace_id=%s — returning no results",
                    filter_ws, required_ws,
                )
                return []

        # pgvector text literal; CAST(:p AS vector) — ':p::vector' does not
        # bind in SQLAlchemy 2.0 text().
        embedding_literal = "[" + ",".join(f"{float(x):.8f}" for x in query_embedding) + "]"

        from core.database.database import SessionLocal

        db = SessionLocal()
        try:
            embedding_type, workspace_type = _column_types(db)
            emb_col = embedding_expr(embedding_type)
            ws_where = workspace_predicate(workspace_type)
            rows = db.execute(
                text(
                    f"""
                    SELECT dc.document_id,
                           dc.chunk_index,
                           dc.content,
                           d.filename AS file_name,
                           d.file_path AS file_path,
                           1 - ({emb_col} <=> CAST(:emb AS vector)) AS similarity
                    FROM document_chunks dc
                    JOIN documents d ON d.id = dc.document_id
                    WHERE {ws_where}
                      AND dc.embedding IS NOT NULL
                      AND 1 - ({emb_col} <=> CAST(:emb AS vector)) >= :min_score
                    ORDER BY {emb_col} <=> CAST(:emb AS vector)
                    LIMIT :limit
                    """
                ),
                {
                    "emb": embedding_literal,
                    "ws": required_ws,
                    "min_score": float(min_score),
                    "limit": int(limit),
                },
            ).fetchall()
        except Exception as e:
            logger.error(f"pgvector-local search failed: {e}", exc_info=True)
            return []
        finally:
            db.close()

        results: List[Dict[str, Any]] = []
        for row in rows:
            document_id = row.document_id
            chunk_index = row.chunk_index
            metadata = {
                "external_file_id": str(document_id),
                "document_id": str(document_id),
                "chunk_index": chunk_index,
                "workspace_id": required_ws,
                "file_name": row.file_name or "",
                "file_path": row.file_path or "",
            }
            results.append({
                "key": f"doc_{document_id}_chunk_{chunk_index}",
                "score": float(row.similarity),
                "metadata": metadata,
                "content": row.content or "",
                "source": "pgvector_local",
                "file_name": row.file_name or "",
                "file_path": row.file_path or "",
                "external_file_id": str(document_id),
                "chunk_index": chunk_index,
            })
        return results

    async def close(self) -> None:
        """No-op — sessions are opened and closed per call."""

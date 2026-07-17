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
from typing import Any, Dict, List, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)


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
            rows = db.execute(
                text(
                    """
                    SELECT dc.document_id,
                           dc.chunk_index,
                           dc.content,
                           d.filename AS file_name,
                           d.file_path AS file_path,
                           1 - (dc.embedding <=> CAST(:emb AS vector)) AS similarity
                    FROM document_chunks dc
                    JOIN documents d ON d.id = dc.document_id
                    WHERE dc.workspace_id = CAST(:ws AS uuid)
                      AND dc.embedding IS NOT NULL
                      AND 1 - (dc.embedding <=> CAST(:emb AS vector)) >= :min_score
                    ORDER BY dc.embedding <=> CAST(:emb AS vector)
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

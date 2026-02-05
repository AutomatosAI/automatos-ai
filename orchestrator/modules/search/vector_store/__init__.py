"""
Vector Store Module
===================

pgvector-based vector storage and retrieval, with pluggable backend support.
"""

from typing import Literal, Optional

from .store import (
    EnhancedVectorStore,
    VectorDocument,
    SearchFilter,
    SearchResult,
    SearchMode,
    RankingStrategy,
)

VectorBackendType = Literal["pgvector", "s3_vectors", "s3_vectors_mock"]


def get_vector_store(
    backend: VectorBackendType = "pgvector",
    workspace_id: Optional[str] = None,
    **kwargs,
):
    """
    Factory for vector stores with pluggable backends.

    Args:
        backend: "pgvector" (existing PostgreSQL), "s3_vectors" (AWS S3 Vectors),
                 or "s3_vectors_mock" (in-memory mock for local testing).
        workspace_id: Required for s3_vectors and s3_vectors_mock (used for bucket scoping).
        **kwargs: Passed through to the backend constructor.

    Returns:
        EnhancedVectorStore for pgvector, S3VectorsBackend for s3_vectors,
        MockS3VectorsBackend for s3_vectors_mock.
    """
    if backend == "pgvector":
        return EnhancedVectorStore(**kwargs)

    if backend == "s3_vectors_mock":
        if not workspace_id:
            raise ValueError("workspace_id is required for the s3_vectors_mock backend")
        from .backends.s3_vectors_mock import MockS3VectorsBackend
        return MockS3VectorsBackend(workspace_id=workspace_id, **kwargs)

    if backend == "s3_vectors":
        if not workspace_id:
            raise ValueError("workspace_id is required for the s3_vectors backend")
        from .backends.s3_vectors_backend import S3VectorsBackend
        return S3VectorsBackend(workspace_id=workspace_id, **kwargs)

    raise ValueError(f"Unknown vector backend: {backend}")


__all__ = [
    "EnhancedVectorStore",
    "VectorDocument",
    "SearchFilter",
    "SearchResult",
    "SearchMode",
    "RankingStrategy",
    "get_vector_store",
    "VectorBackendType",
]

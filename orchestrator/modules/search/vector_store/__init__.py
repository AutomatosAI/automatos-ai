"""
Vector Store Module
===================

Pluggable document-vector backends behind one factory.

PRD-197 S1/S5: the F079 ``EnhancedVectorStore`` (wrong-math cosine via the L2
operator, namesake table dropped in PRD-135) is deleted; the ``pgvector``
backend name now returns the correct-math, workspace-scoped
``PgVectorLocalBackend`` — the open-core/local edition's document read leg
over ``document_chunks.embedding``. S3 Vectors stays the SaaS document plane.
"""

from typing import Literal, Optional

VectorBackendType = Literal["pgvector", "s3_vectors", "s3_vectors_mock"]


def get_vector_store(
    backend: VectorBackendType = "pgvector",
    workspace_id: Optional[str] = None,
    **kwargs,
):
    """
    Factory for vector stores with pluggable backends.

    Args:
        backend: "pgvector" (local edition — document_chunks read leg),
                 "s3_vectors" (AWS S3 Vectors, the SaaS document plane),
                 or "s3_vectors_mock" (in-memory mock for local testing).
        workspace_id: Required for every backend — all three are
                 workspace-scoped (fail-closed isolation).
        **kwargs: Passed through to the backend constructor.

    Returns:
        PgVectorLocalBackend, S3VectorsBackend, or MockS3VectorsBackend.
    """
    if backend == "pgvector":
        if not workspace_id:
            raise ValueError("workspace_id is required for the pgvector backend")
        from .backends.pgvector_local_backend import PgVectorLocalBackend
        return PgVectorLocalBackend(workspace_id=workspace_id, **kwargs)

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
    "get_vector_store",
    "VectorBackendType",
]

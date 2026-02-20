"""
Mock S3 Vectors Backend (PRD-42 - Local Testing)
=================================================

In-memory mock implementation of S3 Vectors for local development and testing.
Implements the same interface as S3VectorsBackend but stores vectors in memory.

When ready to switch to real S3 Vectors:
1. Set AWS credentials in .env
2. Change backend="s3_vectors" in vector_store/__init__.py
3. Remove this mock
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class MockS3VectorsBackend:
    """
    Mock S3 Vectors backend for local testing.

    Stores vectors in memory instead of S3. Implements same interface
    as S3VectorsBackend for easy drop-in replacement.
    """

    INDEX_NAME = "documents-index"
    INDEX_DIMENSION = 4096
    DISTANCE_METRIC = "cosine"

    def __init__(self, workspace_id: str, region: str = None):
        self.workspace_id = str(workspace_id)
        self.bucket_name = f"automatos-vectors-{self.workspace_id}"
        self.region = region or "us-east-1"

        # In-memory storage: {key: {embedding: [...], metadata: {...}}}
        self._vectors = {}
        self._setup_complete = False

        logger.info(f"🔧 MockS3VectorsBackend initialized for workspace {workspace_id}")

    async def initialize(self) -> None:
        """Ensure bucket and index exist. Safe to call multiple times."""
        if self._setup_complete:
            return
        self._ensure_setup()

    def _ensure_setup(self) -> None:
        """Mock setup - just sets flag."""
        if not self._setup_complete:
            logger.info(f"✅ Mock S3 bucket/index ready: {self.bucket_name}/{self.INDEX_NAME}")
            self._setup_complete = True

    def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        min_score: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search vectors by cosine similarity (in-memory).

        Returns list of dicts with keys: key, score, metadata, content, etc.
        """
        self._ensure_setup()

        if not self._vectors:
            logger.debug("No vectors in mock storage yet")
            return []

        # Calculate cosine similarities
        query_vec = np.array(query_embedding, dtype=np.float32)
        results = []

        for key, data in self._vectors.items():
            vec = np.array(data["embedding"], dtype=np.float32)

            # Cosine similarity: dot(a,b) / (norm(a) * norm(b))
            similarity = np.dot(query_vec, vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(vec)
            )

            if similarity < min_score:
                continue

            metadata = data.get("metadata", {})
            results.append({
                "key": key,
                "score": float(similarity),
                "metadata": metadata,
                "content": metadata.get("chunk_text", ""),
                "source": metadata.get("app_name", ""),
                "file_name": metadata.get("file_name", ""),
                "file_path": metadata.get("file_path", ""),
                "external_file_id": metadata.get("external_file_id", ""),
                "chunk_index": metadata.get("chunk_index", 0),
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"🔍 Mock search found {len(results[:limit])} results (of {len(results)} total)")
        return results[:limit]

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> List[str]:
        """
        Add documents with embeddings to mock storage.

        Each document dict should contain:
            - external_file_id: str
            - chunk_index: int
            - chunk_text: str (first 500 chars for preview)
            - app_name: str
            - file_name: str
            - file_path: str

        Returns list of vector keys that were stored.
        """
        self._ensure_setup()

        keys = []
        for doc, embedding in zip(documents, embeddings):
            external_id = doc.get("external_file_id", "unknown")
            chunk_idx = doc.get("chunk_index", 0)
            key = f"doc_{external_id}_chunk_{chunk_idx}"

            metadata = {
                "external_file_id": str(external_id),
                "chunk_index": chunk_idx,
                "chunk_text": doc.get("chunk_text", "")[:500],
                "app_name": doc.get("app_name", ""),
                "workspace_id": self.workspace_id,
                "file_name": doc.get("file_name", ""),
                "file_path": doc.get("file_path", ""),
            }

            self._vectors[key] = {
                "embedding": embedding,
                "metadata": metadata,
            }
            keys.append(key)

        logger.info(f"📝 Stored {len(keys)} vectors in mock storage (total: {len(self._vectors)})")
        return keys

    def delete_documents(self, external_file_id: str) -> int:
        """
        Delete all vectors for a given cloud document.

        Uses the key prefix pattern: doc_{external_file_id}_chunk_*

        Returns number of vectors deleted.
        """
        self._ensure_setup()

        prefix = f"doc_{external_file_id}_chunk_"
        keys_to_delete = [k for k in self._vectors.keys() if k.startswith(prefix)]

        for key in keys_to_delete:
            del self._vectors[key]

        deleted = len(keys_to_delete)
        if deleted > 0:
            logger.info(f"🗑️  Deleted {deleted} vectors for file {external_file_id}")

        return deleted

    def delete_all_for_connection(self, connection_id: int) -> int:
        """
        Delete ALL vectors associated with a Composio connection.

        Used when disconnecting a cloud storage provider with delete_vectors=true.
        Note: In mock, we delete ALL vectors (can't filter by connection_id in metadata).
        """
        self._ensure_setup()

        # In a real implementation, you'd filter by connection metadata
        # For mock, we'll just clear everything as a simplification
        deleted = len(self._vectors)
        self._vectors.clear()

        logger.info(f"🗑️  Deleted all {deleted} vectors (mock connection cleanup)")
        return deleted

    async def close(self) -> None:
        """Cleanup - clear in-memory storage."""
        self._vectors.clear()
        logger.info("🔒 Mock S3 Vectors backend closed")

    # Debug/testing helpers
    def get_vector_count(self) -> int:
        """Get total number of vectors in mock storage."""
        return len(self._vectors)

    def clear_all(self) -> None:
        """Clear all vectors (testing only)."""
        count = len(self._vectors)
        self._vectors.clear()
        logger.warning(f"⚠️  Cleared {count} vectors from mock storage")

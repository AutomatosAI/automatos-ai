"""
S3 Vectors Backend (PRD-42)
============================

AWS S3 Vectors backend for vector storage and retrieval.
Creates workspace-scoped buckets with cosine similarity search.

Each workspace gets its own S3 vector bucket: automatos-vectors-{workspace_id}
with a single index: documents-index (dimension from S3_VECTORS_DIMENSION, metric=COSINE).
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

from config import config

logger = logging.getLogger(__name__)


class S3VectorsBackend:
    """
    AWS S3 Vectors backend for EnhancedVectorStore.

    Implements search, add_documents, and delete_documents using the
    S3 Vectors API (boto3 s3vectors client).

    Configuration loaded from config.py (which reads from .env):
    - S3_VECTORS_BUCKET: Bucket name (supports {workspace_id} template)
    - S3_VECTORS_INDEX_NAME: Index name within bucket
    - S3_VECTORS_DIMENSION: Embedding dimension (must match embedding model)
    - S3_VECTORS_METRIC: Distance metric (cosine, euclidean, dot_product)
    - AWS_REGION: AWS region where S3 Vectors is deployed
    """

    def __init__(self, workspace_id: str, region: str = None):
        self.workspace_id = str(workspace_id)

        # Get bucket name from config (supports {workspace_id} template for multi-tenant)
        bucket_template = config.S3_VECTORS_BUCKET
        if not bucket_template:
            raise ValueError("S3_VECTORS_BUCKET not configured in .env")
        self.bucket_name = bucket_template.replace("{workspace_id}", self.workspace_id)

        # Get index configuration from config
        self.index_name = config.S3_VECTORS_INDEX_NAME
        self.index_dimension = config.S3_VECTORS_DIMENSION
        self.distance_metric = config.S3_VECTORS_METRIC

        # Get region from config
        self.region = region or config.AWS_REGION

        self.client = boto3.client(
            "s3vectors",
            region_name=self.region,
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
        )

        self._setup_complete = False

    async def initialize(self) -> None:
        """Ensure bucket and index exist. Safe to call multiple times."""
        if self._setup_complete:
            return
        self._ensure_setup()

    def _ensure_setup(self) -> None:
        """Create bucket and index if they don't already exist.

        If an existing index has a different dimension than configured,
        it is deleted and recreated so embeddings can be stored correctly.
        """
        # Create bucket
        try:
            self.client.create_vector_bucket(vectorBucketName=self.bucket_name)
            logger.info(f"Created S3 vector bucket: {self.bucket_name}")
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("BucketAlreadyExists", "BucketAlreadyOwnedByYou", "ConflictException"):
                logger.debug(f"S3 vector bucket already exists: {self.bucket_name}")
            else:
                raise

        # Create index (with dimension mismatch detection)
        try:
            self.client.create_index(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
                dimension=self.index_dimension,
                dataType="float32",
                distanceMetric=self.distance_metric,
            )
            logger.info(f"Created S3 vector index: {self.index_name} (dimension={self.index_dimension})")
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("ConflictException", "ResourceAlreadyExistsException"):
                # Index exists — verify the dimension matches
                self._verify_or_recreate_index()
            else:
                raise

        self._setup_complete = True

    def _verify_or_recreate_index(self) -> None:
        """Check existing index dimension and recreate if it doesn't match config."""
        try:
            response = self.client.get_index(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
            )
            existing_dim = response.get("dimension", 0)

            if existing_dim == self.index_dimension:
                logger.debug(
                    f"S3 vector index already exists with correct dimension {existing_dim}"
                )
                return

            logger.warning(
                f"S3 vector index dimension mismatch: index has {existing_dim}, "
                f"config expects {self.index_dimension}. Recreating index..."
            )

            # Delete the old index
            self.client.delete_index(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
            )
            logger.info(f"Deleted stale S3 vector index (dimension={existing_dim})")

            # Create with the correct dimension
            self.client.create_index(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
                dimension=self.index_dimension,
                dataType="float32",
                distanceMetric=self.distance_metric,
            )
            logger.info(
                f"Recreated S3 vector index: {self.index_name} "
                f"(dimension={self.index_dimension})"
            )

        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("NotFoundException", "ResourceNotFoundException"):
                # Index was deleted between check and recreate — just create it
                self.client.create_index(
                    vectorBucketName=self.bucket_name,
                    indexName=self.index_name,
                    dimension=self.index_dimension,
                    dataType="float32",
                    distanceMetric=self.distance_metric,
                )
                logger.info(f"Created S3 vector index: {self.index_name}")
            else:
                logger.error(f"Failed to verify/recreate S3 vector index: {e}")
                raise

    def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        min_score: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search vectors by cosine similarity.

        Returns list of dicts with keys: key, score, metadata.
        """
        self._ensure_setup()

        try:
            response = self.client.query_vectors(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
                queryVector=query_embedding,
                queryVectorDataType="float32",
                topK=limit,
            )

            results = []
            for match in response.get("vectors", []):
                score = match.get("distance", 0.0)
                # S3 Vectors cosine returns distance; convert to similarity
                similarity = 1.0 - score if score <= 1.0 else 0.0
                if similarity < min_score:
                    continue

                metadata = match.get("metadata", {})
                results.append({
                    "key": match.get("key", ""),
                    "score": similarity,
                    "metadata": metadata,
                    "content": metadata.get("chunk_text", ""),
                    "source": metadata.get("app_name", ""),
                    "file_name": metadata.get("file_name", ""),
                    "file_path": metadata.get("file_path", ""),
                    "external_file_id": metadata.get("external_file_id", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                })

            return results

        except ClientError as e:
            logger.error(f"S3 Vectors search failed: {e}")
            return []

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> List[str]:
        """
        Add documents with embeddings to S3 Vectors.

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

        vector_objects = []
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

            vector_objects.append({
                "key": key,
                "data": {"float32": embedding},
                "metadata": metadata,
            })
            keys.append(key)

        if not vector_objects:
            return []

        # Batch puts to stay under S3 Vectors max request size (~10MB)
        # With 4096-dim float32 vectors (~16KB each) + metadata, ~50 vectors per batch is safe
        BATCH_SIZE = 50
        try:
            for i in range(0, len(vector_objects), BATCH_SIZE):
                batch = vector_objects[i:i + BATCH_SIZE]
                self.client.put_vectors(
                    vectorBucketName=self.bucket_name,
                    indexName=self.index_name,
                    vectors=batch,
                )
                if len(vector_objects) > BATCH_SIZE:
                    logger.info(
                        f"Stored batch {i // BATCH_SIZE + 1}/{(len(vector_objects) + BATCH_SIZE - 1) // BATCH_SIZE} "
                        f"({len(batch)} vectors) in {self.bucket_name}/{self.index_name}"
                    )
            logger.info(
                f"Stored {len(vector_objects)} vectors in {self.bucket_name}/{self.index_name}"
            )
            return keys

        except ClientError as e:
            logger.error(f"S3 Vectors put failed: {e}")
            raise  # Re-raise to propagate error to caller

    def delete_documents(self, external_file_id: str) -> int:
        """
        Delete all vectors for a given cloud document.

        Uses the key prefix pattern: doc_{external_file_id}_chunk_*

        Returns number of vectors deleted.
        """
        self._ensure_setup()

        deleted = 0
        try:
            # List vectors with prefix to find all chunks
            response = self.client.list_vectors(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
                keyPrefix=f"doc_{external_file_id}_chunk_",
            )

            keys_to_delete = [v["key"] for v in response.get("vectors", [])]

            if keys_to_delete:
                self.client.delete_vectors(
                    vectorBucketName=self.bucket_name,
                    indexName=self.index_name,
                    keys=keys_to_delete,
                )
                deleted = len(keys_to_delete)
                logger.info(
                    f"Deleted {deleted} vectors for file {external_file_id}"
                )

        except ClientError as e:
            logger.error(f"S3 Vectors delete failed for {external_file_id}: {e}")

        return deleted

    def delete_all_for_connection(self, connection_id: int) -> int:
        """
        Delete ALL vectors associated with a Composio connection.

        Used when disconnecting a cloud storage provider with delete_vectors=true.
        """
        # This requires listing all vectors and filtering by metadata.
        # S3 Vectors list_vectors returns keys; we delete in batches.
        self._ensure_setup()
        deleted = 0

        try:
            # List all vectors in the index
            paginator_token = None
            while True:
                kwargs = {
                    "vectorBucketName": self.bucket_name,
                    "indexName": self.index_name,
                }
                if paginator_token:
                    kwargs["nextToken"] = paginator_token

                response = self.client.list_vectors(**kwargs)
                keys_to_delete = [v["key"] for v in response.get("vectors", [])]

                if keys_to_delete:
                    self.client.delete_vectors(
                        vectorBucketName=self.bucket_name,
                        indexName=self.index_name,
                        keys=keys_to_delete,
                    )
                    deleted += len(keys_to_delete)

                paginator_token = response.get("nextToken")
                if not paginator_token:
                    break

            logger.info(f"Deleted {deleted} vectors for connection cleanup")

        except ClientError as e:
            logger.error(f"S3 Vectors bulk delete failed: {e}")

        return deleted

    async def close(self) -> None:
        """No-op — boto3 clients don't need explicit cleanup."""
        pass

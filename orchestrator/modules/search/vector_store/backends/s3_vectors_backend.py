"""
S3 Vectors Backend (PRD-42)
============================

AWS S3 Vectors backend for vector storage and retrieval.

The bucket may be shared across workspaces or templated per workspace
(``{workspace_id}`` placeholder) — both layouts are supported. Tenant
isolation does not depend on the layout: ``search()`` is fail-closed on
``workspace_id`` metadata (PRD-186 S1), and deletes are scoped to the
caller's own file keys. One index per bucket: dimension from
S3_VECTORS_DIMENSION, metric from S3_VECTORS_METRIC.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

from config import config

logger = logging.getLogger(__name__)


class IndexDimensionMismatchError(RuntimeError):
    """A populated S3 Vectors index reports a dimension different from config.

    Serving across the mismatch is silently wrong (vectors written or scored
    against a foreign geometry), so setup aborts instead of proceeding — and
    never deletes the index, which would destroy stored vectors (PRD-186 S2).
    """


class S3VectorsBackend:
    """
    AWS S3 Vectors document backend (the SaaS document plane).

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
        # A bucket name may be a single shared bucket, or carry a {workspace_id}
        # placeholder for physically per-workspace buckets — both are supported.
        # Tenant isolation does NOT depend on the bucket layout: search() is
        # fail-closed on workspace_id (it drops any hit whose metadata
        # workspace_id != this backend's, and returns [] on a mismatched filter),
        # so a shared bucket is still isolated at query time. PRD-172 F005 also
        # *required* per-workspace buckets as belt-and-suspenders; that hard
        # requirement broke a working shared-bucket deployment (2026-07-02) and is
        # dropped — the query-level filter is the enforced isolation guarantee.
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

        If an existing index reports a dimension different from the configured
        one, setup raises ``IndexDimensionMismatchError`` (PRD-186 S2) — a
        populated index is never deleted or recreated from here.
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
                # Index exists — its dimension must match config, loudly
                self._assert_index_dimension()
            else:
                raise

        self._setup_complete = True

    def _assert_index_dimension(self) -> None:
        """Raise on a confirmed index-vs-config dimension mismatch (PRD-186 S2).

        Never deletes or recreates — that destroys stored vectors. A mismatch
        that can't be confirmed (index unreadable) only warns: reachability
        problems surface elsewhere; this guard is for confirmed wrong geometry,
        which previously was logged and served through.
        """
        try:
            response = self.client.get_index(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
            )
        except ClientError as e:
            logger.warning(f"Could not verify S3 vector index: {e}")
            return

        existing_dim = (
            response.get("dimension")
            or (response.get("index", {}) or {}).get("dimension")
            or 0
        )
        if existing_dim and existing_dim != self.index_dimension:
            raise IndexDimensionMismatchError(
                f"S3 Vectors index '{self.index_name}' in bucket "
                f"'{self.bucket_name}' reports dimension {existing_dim} but "
                f"config.S3_VECTORS_DIMENSION is {self.index_dimension}. "
                "Refusing to write/query mismatched geometry. Fix the "
                "configured dimension, or re-embed into a correctly-"
                "dimensioned index (scripts/migrate_to_s3_vectors.py) — this "
                "index is never auto-deleted."
            )
        logger.info(
            f"S3 vector index exists: {self.index_name} "
            f"(dimension={existing_dim or 'unreported'}, config={self.index_dimension})"
        )

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

        PRD-172 F005: the ``filters`` param is now enforced. Isolation on S3
        Vectors previously rested entirely on a per-workspace bucket name; if a
        deploy ever pointed multiple workspaces at one bucket (no
        ``{workspace_id}`` placeholder), search leaked cross-workspace chunk
        text into LLM context. We ALWAYS post-filter on the backend's own
        ``workspace_id`` and, when a ``workspace_id`` filter is supplied, on that
        too — dropping any hit whose metadata ``workspace_id`` does not match.
        """
        self._ensure_setup()

        # Fail-closed workspace scope: the backend is bound to exactly one
        # workspace at construction; every hit must carry that workspace_id.
        # An explicit filters['workspace_id'] must agree with it.
        required_ws = str(self.workspace_id)
        if filters:
            filter_ws = filters.get("workspace_id")
            if filter_ws is not None and str(filter_ws) != required_ws:
                # A caller asked to search a different workspace through a
                # workspace-bound backend — refuse rather than silently widen.
                logger.warning(
                    "S3 Vectors search: filter workspace_id=%s != backend "
                    "workspace_id=%s — returning no results",
                    filter_ws, required_ws,
                )
                return []

        try:
            response = self.client.query_vectors(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
                queryVector={"float32": query_embedding},
                topK=limit,
                returnMetadata=True,
                returnDistance=True,
            )

            results = []
            for match in response.get("vectors", []):
                score = match.get("distance", 0.0)
                # S3 Vectors cosine returns distance; convert to similarity
                similarity = 1.0 - score if score <= 1.0 else 0.0
                if similarity < min_score:
                    continue

                metadata = match.get("metadata", {})
                # PRD-172 F005 + PRD-186 S1: drop any hit not PROVEN scoped to
                # this workspace — mismatched OR unlabeled. On a shared bucket
                # an unstamped chunk must never reach another tenant's context,
                # so isolation cannot depend on every writer remembering to
                # stamp; legitimate chunks are stamped by add_documents.
                hit_ws = metadata.get("workspace_id")
                if hit_ws is None or str(hit_ws) != required_ws:
                    continue

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
            logger.error(f"S3 Vectors search failed: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"S3 Vectors search unexpected error: {e}", exc_info=True)
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

        # Batch puts to stay under S3 Vectors max request size (~10MB).
        # At the configured dimension (config.S3_VECTORS_DIMENSION, live 2048
        # → ~8KB float32 each) + metadata, 50 vectors per batch is safe.
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

    def delete_for_files(self, external_file_ids: List[str]) -> int:
        """
        Delete vectors for the given cloud documents, file by file.

        Used when disconnecting a cloud storage provider with
        delete_vectors=true. Replaces the index-wide sweep its predecessor
        (``delete_all_for_connection``) performed: on a shared bucket, listing
        the whole index reaches EVERY tenant's vectors, so deletes must stay
        scoped to the caller's own file ids — each one a workspace-checked
        CloudDocument, deleted via the key-prefix pattern (PRD-186 S1).
        """
        deleted = 0
        for external_file_id in external_file_ids:
            deleted += self.delete_documents(str(external_file_id))
        logger.info(
            f"Deleted {deleted} vectors across {len(external_file_ids)} files "
            "for connection cleanup"
        )
        return deleted

    async def close(self) -> None:
        """No-op — boto3 clients don't need explicit cleanup."""
        pass

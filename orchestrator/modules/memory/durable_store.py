"""
Durable Memory Store — PRD-187 S1 (P2-06, the un-split)
========================================================

In-process L3 durable memory on the platform's own Qdrant — the same
running instance field memory uses (``vector_field.py``), second collection.
Replaces the mem0-fork-as-a-service: no HTTP hop, no circuit breaker, no
remote deployment to die silently. When this store is unreachable the
failure is LOUD (error-level with traceback) — a health signal, never a
silent skip.

Interface mirrors the retired fork client (add/search/get_all/delete)
so ``UnifiedMemoryService``'s L3 seam rewires mechanically, plus the
filter-delete surfaces (workspace GDPR erase/export, exact-namespace
erase) the old per-namespace HTTP API could never offer.

Points carry both the ``namespace`` (the MemoryNamespace user_id string —
``mem:{ws}``, ``mem:{ws}:agent:{id}``, …) and a top-level ``workspace_id``
payload so tenancy is fail-closed and GDPR erasure is one filter delete.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    HnswConfigDiff,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from config import config
from core.llm.embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)


def filter_by_relevance_floor(results: List[Dict], floor: float) -> List[Dict]:
    """Drop scored results below the similarity floor (PRD-159 S3).

    Results without a score are kept (cannot judge); scored-but-below-floor are
    dropped so low-relevance junk is never injected into context. A floor <= 0
    disables filtering. (Rehoused from the retired mem0 client — the rule is
    store-agnostic.)
    """
    if not floor or floor <= 0:
        return results
    return [r for r in results if r.get("score") is None or (r.get("score") or 0) >= floor]


def workspace_from_namespace(user_id: str) -> Optional[str]:
    """Extract the workspace id from a MemoryNamespace user_id string.

    All L3 namespaces are ``mem:{workspace_id}[:scope...]`` (PRD-79); scoped
    callers that don't pass an explicit workspace_id still get fail-closed
    tenancy on the payload via this parse.
    """
    if not user_id or not user_id.startswith("mem:"):
        return None
    parts = user_id.split(":")
    return parts[1] if len(parts) > 1 and parts[1] else None


class DurableMemoryStore:
    """Qdrant-backed L3 durable memory (collection ``durable_memory``).

    Same client/ensure-collection/payload-index pattern as
    ``VectorFieldSharedContext`` — deliberately, so Wave-3 P2-16 can
    consolidate the two collections under one client by renaming, not
    migrating.
    """

    def __init__(self) -> None:
        self._client = AsyncQdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY or None,
            timeout=30,
        )
        self._embedder = EmbeddingManager()
        self._collection = config.DURABLE_MEMORY_COLLECTION
        self._dimension = config.FIELD_EMBEDDING_DIM
        self._bootstrap_done = False

    # ── Bootstrap ───────────────────────────────────────────────

    async def ensure_collection(self) -> None:
        """Idempotent — create the durable collection and its payload indexes
        if missing. Safe to call on every operation."""
        if self._bootstrap_done:
            return

        try:
            exists = await self._client.collection_exists(self._collection)
        except Exception:
            logger.error("[Durable] collection_exists check failed", exc_info=True)
            raise

        if not exists:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._dimension,
                    distance=Distance.COSINE,
                    on_disk=True,
                ),
                hnsw_config=HnswConfigDiff(on_disk=True),
                on_disk_payload=True,
            )
            logger.info("[Durable] Created collection %s", self._collection)

        for field_name, schema in [
            ("namespace", PayloadSchemaType.KEYWORD),
            ("workspace_id", PayloadSchemaType.KEYWORD),
            # PRD-196 S6 (GDPR): the data-subject tag. A keyword payload index so
            # subject-level erasure is one filter-delete (workspace_id AND
            # subject_id), the same shape as the workspace erase.
            ("subject_id", PayloadSchemaType.KEYWORD),
            ("content_hash", PayloadSchemaType.KEYWORD),
            ("created_at", PayloadSchemaType.KEYWORD),
        ]:
            try:
                await self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field_name,
                    field_schema=schema,
                    wait=True,
                )
            except Exception:
                logger.debug("[Durable] payload index %s present", field_name)

        self._bootstrap_done = True

    # ── Filter helpers ──────────────────────────────────────────

    @staticmethod
    def _namespace_filter(user_id: str) -> Filter:
        return Filter(must=[
            FieldCondition(key="namespace", match=MatchValue(value=user_id)),
        ])

    @staticmethod
    def _workspace_filter(workspace_id: str) -> Filter:
        return Filter(must=[
            FieldCondition(key="workspace_id", match=MatchValue(value=str(workspace_id))),
        ])

    @staticmethod
    def _subject_filter(workspace_id: str, subject_id: str) -> Filter:
        """PRD-196 S6: workspace_id AND subject_id — fail-closed tenancy, so a
        subject erase can never reach beyond the requesting workspace."""
        return Filter(must=[
            FieldCondition(key="workspace_id", match=MatchValue(value=str(workspace_id))),
            FieldCondition(key="subject_id", match=MatchValue(value=str(subject_id))),
        ])

    @staticmethod
    def _item_from_payload(point_id: Any, payload: Dict, score: Optional[float] = None) -> Dict:
        """Map a point to the memory-item shape L3 consumers already read
        (same keys the mem0 search/get_all results carried)."""
        payload = payload or {}
        return {
            "id": str(point_id),
            "memory": payload.get("content"),
            "score": score,
            "metadata": payload.get("metadata"),
            "created_at": payload.get("created_at"),
            "namespace": payload.get("namespace"),
        }

    # ── Write ───────────────────────────────────────────────────

    async def add(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        metadata: Optional[Dict] = None,
        workspace_id: Optional[str] = None,
        subject_id: Optional[str] = None,
    ) -> Dict:
        """Store one durable memory under a namespace.

        Messages are joined to a single text (user content verbatim, other
        roles prefixed) — the same normalisation the retired client applied.
        Same-content writes in the same namespace dedup to the existing point.

        PRD-196 S6: ``subject_id`` is an optional namespaced data-subject tag
        (``user:{users.id}`` — the INTERNAL id, never a Clerk string) written to
        the payload so a single human's memories can be filter-deleted for GDPR.
        Untagged writes (heartbeat / playbook / agent-internal) store a null tag
        and are reported as untagged history on a subject erase, never as erased.
        """
        await self.ensure_collection()

        text_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                text_parts.append(f"{role}: {content}" if role != "user" else content)
        text = "\n".join(text_parts)
        if not text:
            return {"success": False, "error": "empty content"}

        ws = str(workspace_id) if workspace_id else workspace_from_namespace(user_id)
        content_hash = hashlib.sha256(text.encode()).hexdigest()

        existing = await self._find_by_hash(user_id, content_hash)
        if existing is not None:
            logger.debug("[Durable] Deduplicated add in namespace=%s", user_id)
            return {"success": True, "id": str(existing.id), "deduped": True}

        embedding = await self._embedder.generate_embedding(text)
        point_id = str(uuid.uuid4())
        await self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "namespace": user_id,
                    "workspace_id": ws,
                    "subject_id": subject_id,  # PRD-196 S6 (GDPR data-subject tag)
                    "content": text,
                    "metadata": metadata or {},
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "content_hash": content_hash,
                },
            )],
        )
        logger.info("[Durable] Stored memory namespace=%s len=%d", user_id, len(text))
        return {"success": True, "id": point_id}

    # ── Read ────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        workspace_id: Optional[str] = None,
    ) -> List[Dict]:
        """Semantic search within a namespace, floor-filtered (PRD-159 S3)."""
        await self.ensure_collection()

        # PRD-197 S4: memory-seam substrate telemetry — candidates/latency/
        # errors feed the Command Center substrate tile. Fire-and-forget;
        # a raise still propagates to the caller unchanged.
        seam_t0 = time.perf_counter()
        try:
            query_embedding = await self._embedder.generate_embedding(query)
            response = await self._client.query_points(
                collection_name=self._collection,
                query=query_embedding,
                query_filter=self._namespace_filter(user_id),
                limit=max(limit, 1),
            )
            results = [
                self._item_from_payload(hit.id, hit.payload, score=hit.score)
                for hit in response.points
            ]
            results = filter_by_relevance_floor(results, config.MEMORY_RELEVANCE_FLOOR)
        except Exception:
            from core.observability.substrate_metrics import (
                SEAM_MEMORY,
                STATUS_ERROR,
                record_substrate_search_nowait,
            )

            record_substrate_search_nowait(
                seam=SEAM_MEMORY,
                workspace_id=workspace_id,
                candidates=0,
                latency_ms=(time.perf_counter() - seam_t0) * 1000.0,
                status=STATUS_ERROR,
                query=query,
            )
            raise
        logger.debug(
            "[Durable] search namespace=%s query=%r → %d results",
            user_id, query[:60], len(results),
        )
        returned = results[:limit]

        from core.observability.substrate_metrics import (
            SEAM_MEMORY,
            STATUS_EMPTY,
            STATUS_HIT,
            record_substrate_search_nowait,
        )

        record_substrate_search_nowait(
            seam=SEAM_MEMORY,
            workspace_id=workspace_id,
            candidates=len(returned),
            latency_ms=(time.perf_counter() - seam_t0) * 1000.0,
            status=STATUS_HIT if returned else STATUS_EMPTY,
            query=query,
            top_score=float(returned[0].get("score") or 0.0) if returned else 0.0,
        )
        return returned

    async def get_all(
        self,
        user_id: str,
        limit: int = 100,
        workspace_id: Optional[str] = None,
    ) -> List[Dict]:
        """Every memory in a namespace (unscored, scroll order)."""
        await self.ensure_collection()

        out: List[Dict] = []
        next_offset: Any = None
        while len(out) < limit:
            points, next_offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=self._namespace_filter(user_id),
                limit=min(256, limit - len(out)),
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            out.extend(self._item_from_payload(p.id, p.payload) for p in points)
            if next_offset is None:
                break
        return out[:limit]

    # ── Delete ──────────────────────────────────────────────────

    async def delete(
        self,
        memory_ids: List[str],
        user_id: str,
        workspace_id: Optional[str] = None,
    ) -> bool:
        """Delete memories by id, only where the point belongs to ``user_id``
        (preserves the namespace-ownership check the old bulk API enforced)."""
        await self.ensure_collection()

        points = await self._client.retrieve(self._collection, ids=list(memory_ids))
        owned = [p.id for p in points if (p.payload or {}).get("namespace") == user_id]
        if not owned:
            return False
        await self._client.delete(
            collection_name=self._collection,
            points_selector=owned,
        )
        logger.info("[Durable] Deleted %d memor(ies) namespace=%s", len(owned), user_id)
        return True

    # ── GDPR (PRD-181 S3/S4 surface) ────────────────────────────

    async def erase_workspace(self, workspace_id: str) -> int:
        """GDPR erasure — one filter delete over the ``workspace_id`` payload
        index (no namespace enumeration; the whole point of the un-split)."""
        await self.ensure_collection()
        flt = self._workspace_filter(workspace_id)
        count = await self._count_by_filter(flt)
        await self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(filter=flt),
        )
        logger.warning(
            "[Durable] GDPR erased %d durable memor(ies) for workspace %s",
            count, workspace_id,
        )
        return count

    async def erase_subject(self, workspace_id: str, subject_id: str) -> int:
        """GDPR subject-level erasure (PRD-196 S6) — one filter-delete over the
        ``workspace_id`` AND ``subject_id`` payload indexes. Fail-closed: never
        workspace-wide. Returns the number of memories erased (0 if the subject
        has no tagged rows — e.g. only untagged pre-tag history exists)."""
        await self.ensure_collection()
        flt = self._subject_filter(workspace_id, subject_id)
        count = await self._count_by_filter(flt)
        await self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(filter=flt),
        )
        logger.warning(
            "[Durable] GDPR subject-erased %d durable memor(ies) for subject=%s ws=%s",
            count, subject_id, workspace_id,
        )
        return count

    async def erase_namespace(self, user_id: str) -> int:
        """Delete every memory in one exact namespace (e.g. a deleted
        playbook's ``mem:{ws}:recipe:{id}`` bucket). Same filter-delete shape
        as ``erase_workspace``."""
        await self.ensure_collection()
        flt = self._namespace_filter(user_id)
        count = await self._count_by_filter(flt)
        await self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(filter=flt),
        )
        logger.info("[Durable] Erased %d memor(ies) in namespace=%s", count, user_id)
        return count

    async def export_workspace(self, workspace_id: str, limit: int = 10000) -> List[Dict]:
        """GDPR export — every durable memory for a workspace, portable dicts."""
        await self.ensure_collection()
        out: List[Dict] = []
        next_offset: Any = None
        while len(out) < limit:
            points, next_offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=self._workspace_filter(workspace_id),
                limit=min(256, limit - len(out)),
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            out.extend(self._item_from_payload(p.id, p.payload) for p in points)
            if next_offset is None:
                break
        return out

    async def _count_by_filter(self, flt: Filter) -> int:
        try:
            res = await self._client.count(
                collection_name=self._collection, count_filter=flt, exact=True,
            )
            return int(getattr(res, "count", 0) or 0)
        except Exception:
            logger.debug("[Durable] count_by_filter failed", exc_info=True)
            return 0

    # ── Health ──────────────────────────────────────────────────

    async def health(self) -> Dict[str, Any]:
        """Real backend health — pings Qdrant, mirrors the field adapter's
        contract so monitoring can tell live-but-empty from down."""
        try:
            await self._client.get_collections()
            return {
                "healthy": True,
                "backend": "durable_memory",
                "collection": self._collection,
            }
        except Exception as exc:
            return {
                "healthy": False,
                "backend": "durable_memory",
                "error": f"{type(exc).__name__}: {exc}",
            }

    # ── Internals ───────────────────────────────────────────────

    async def _find_by_hash(self, user_id: str, content_hash: str):
        results, _ = await self._client.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="namespace", match=MatchValue(value=user_id)),
                FieldCondition(key="content_hash", match=MatchValue(value=content_hash)),
            ]),
            limit=1,
        )
        return results[0] if results else None

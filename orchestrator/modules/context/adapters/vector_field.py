"""
Vector Field Shared Context — PRD-108
======================================

Shared semantic field where agent knowledge resonates, decays, and
forms attractors. Implements SharedContextPort so it's swappable
with RedisSharedContext (PRD-107) for A/B comparison.

Each mission field is a Qdrant collection. Each pattern is a point
with a 2048-dim embedding and metadata payload. Resonance is computed
as cosine_similarity² × decayed_strength at query time.

This is how agents share a brain instead of playing telephone.
"""

from __future__ import annotations

import hashlib
import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from config import config
from core.llm.embedding_manager import EmbeddingManager
from core.ports.context import SharedContextPort

logger = logging.getLogger(__name__)


class VectorFieldSharedContext(SharedContextPort):
    """PRD-108: vector field backend for SharedContextPort.

    Patterns injected by one agent become queryable by all agents
    in the field. Relevance is surfaced by resonance (cosine² × strength),
    not by recency or insertion order. Old patterns fade. Frequently
    accessed patterns resist decay. No telephone game.
    """

    def __init__(self) -> None:
        self._client = AsyncQdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY or None,
            timeout=30,  # Default 5s too short for index creation with wait=True
        )
        self._embedder = EmbeddingManager()
        self._decay_rate = config.FIELD_DECAY_RATE
        self._reinforce_bonus = config.FIELD_REINFORCE_BONUS
        self._reinforce_cap = config.FIELD_REINFORCE_CAP
        self._archival_threshold = config.FIELD_ARCHIVAL_THRESHOLD
        self._boundary_permeability = config.FIELD_BOUNDARY_PERMEABILITY
        self._dimension = config.FIELD_EMBEDDING_DIM

    # ── Create / Destroy ────────────────────────────────────────

    async def create_context(
        self,
        team_agent_ids: list[int],
        initial_data: Optional[dict[str, Any]] = None,
    ) -> str:
        field_id = str(uuid.uuid4())
        collection = f"field_{field_id}"

        await self._client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=self._dimension,
                distance=Distance.COSINE,
                on_disk=True,
            ),
            hnsw_config=HnswConfigDiff(on_disk=True),
            on_disk_payload=True,
        )

        # Payload indexes for filtered queries — wait=True to avoid race
        # between collection creation and index creation on Qdrant.
        for field_name, schema in [
            ("content_hash", PayloadSchemaType.KEYWORD),
            ("agent_id", PayloadSchemaType.INTEGER),
            ("created_at", PayloadSchemaType.KEYWORD),
        ]:
            await self._client.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )

        # Seed with initial data (e.g. mission brief)
        if initial_data:
            for key, value in initial_data.items():
                await self.inject(field_id, key, str(value), agent_id=0, strength=1.0)

        logger.info(
            "[Field] Created field %s (collection=%s, team=%s)",
            field_id, collection, team_agent_ids,
        )
        return field_id

    async def destroy_context(self, context_id: str) -> None:
        collection = f"field_{context_id}"
        try:
            await self._client.delete_collection(collection)
            logger.info("[Field] Destroyed field %s", context_id)
        except Exception:
            logger.warning("[Field] Failed to delete collection %s", collection, exc_info=True)

    async def context_exists(self, context_id: str) -> bool:
        """Check whether the underlying Qdrant collection still exists.

        Used by the coordinator to detect stale field_ids inherited from
        a parent/template mission whose collection has already been
        destroyed by _cleanup_terminal_fields.
        """
        try:
            return await self._client.collection_exists(f"field_{context_id}")
        except Exception:
            logger.debug("[Field] collection_exists check failed for %s", context_id, exc_info=True)
            return False

    # ── Inject ──────────────────────────────────────────────────

    async def inject(
        self,
        context_id: str,
        key: str,
        value: str,
        agent_id: int,
        strength: float = 1.0,
    ) -> None:
        content = f"{key}: {value}"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Dedup: if same content exists, reinforce instead of duplicating
        existing = await self._find_by_hash(context_id, content_hash)
        if existing:
            await self._reinforce_single(context_id, existing.id)
            logger.debug("[Field] Deduplicated inject — reinforced existing pattern")
            return

        embedding = await self._embedder.generate_embedding(content)
        effective_strength = strength * self._boundary_permeability
        now = datetime.now(timezone.utc).isoformat()
        point_id = str(uuid.uuid4())

        await self._client.upsert(
            collection_name=f"field_{context_id}",
            points=[PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "agent_id": agent_id,
                    "key": key,
                    "value": value,
                    "strength": effective_strength,
                    "created_at": now,
                    "last_accessed": now,
                    "access_count": 0,
                    "content_hash": content_hash,
                },
            )],
        )
        logger.debug("[Field] Injected pattern key=%s agent=%s strength=%.2f", key, agent_id, effective_strength)

    # ── Query (with resonance scoring) ──────────────────────────

    async def query(
        self,
        context_id: str,
        query: str,
        agent_id: int,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        query_embedding = await self._embedder.generate_embedding(query)

        # Over-fetch — decay filtering will reduce the set
        response = await self._client.query_points(
            collection_name=f"field_{context_id}",
            query=query_embedding,
            limit=top_k * 3,
        )
        raw_results = response.points

        now = datetime.now(timezone.utc)
        scored: list[dict[str, Any]] = []

        for hit in raw_results:
            payload = hit.payload
            last_accessed = datetime.fromisoformat(payload["last_accessed"])
            age_hours = (now - last_accessed).total_seconds() / 3600

            decayed_strength = self._compute_decayed_strength(
                initial_strength=payload["strength"],
                age_hours=age_hours,
                access_count=payload["access_count"],
            )

            if decayed_strength < self._archival_threshold:
                continue

            # Resonance = cosine² × decayed_strength
            resonance = (hit.score ** 2) * decayed_strength

            scored.append({
                "id": hit.id,
                "key": payload["key"],
                "value": payload["value"],
                "score": resonance,
                "agent_id": payload["agent_id"],
                "decayed_strength": decayed_strength,
                "cosine_similarity": hit.score,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        top_results = scored[:top_k]

        # Hebbian reinforcement — accessed patterns resist future decay
        accessed_ids = [r["id"] for r in top_results]
        if accessed_ids:
            await self._reinforce_batch(context_id, accessed_ids)

        logger.info(
            "[Field] Query field=%s agent=%s results=%d top_score=%.4f query=%s",
            context_id, agent_id, len(top_results),
            top_results[0]["score"] if top_results else 0.0,
            query[:60],
        )
        return top_results

    # ── Stability measurement ───────────────────────────────────

    async def measure_stability(self, context_id: str) -> dict[str, Any]:
        """How converged is the field? Used for telemetry and experiment analysis.

        stability = (avg_strength × 0.6) + (organization × 0.4)
        organization = 1 - (stddev / mean) if mean > 0
        """
        collection = f"field_{context_id}"
        try:
            points, _ = await self._client.scroll(collection, limit=10000)
        except Exception:
            # Missing collection → return zero stats rather than crashing
            # the field-viewer endpoint. PR #312 covers automatic recovery.
            logger.debug("[Field] measure_stability: collection %s missing", collection, exc_info=True)
            return {"stability": 0.0, "pattern_count": 0, "avg_strength": 0.0, "missing": True}

        if not points:
            return {"stability": 0.0, "pattern_count": 0, "avg_strength": 0.0}

        now = datetime.now(timezone.utc)
        strengths = []
        for p in points:
            age_hours = (now - datetime.fromisoformat(p.payload["last_accessed"])).total_seconds() / 3600
            ds = self._compute_decayed_strength(
                p.payload["strength"], age_hours, p.payload["access_count"],
            )
            strengths.append(ds)

        avg = sum(strengths) / len(strengths)
        if avg > 0:
            stddev = (sum((s - avg) ** 2 for s in strengths) / len(strengths)) ** 0.5
            organization = max(0.0, 1.0 - (stddev / avg))
        else:
            organization = 0.0

        stability = (avg * 0.6) + (organization * 0.4)

        return {
            "stability": round(stability, 4),
            "pattern_count": len(points),
            "avg_strength": round(avg, 4),
            "organization": round(organization, 4),
            "active_patterns": sum(1 for s in strengths if s >= self._archival_threshold),
            "decayed_patterns": sum(1 for s in strengths if s < self._archival_threshold),
        }

    # ── Pattern listing (for field visualizer) ─────────────────

    async def get_patterns(self, context_id: str) -> list[dict[str, Any]]:
        """Return all patterns in the field with computed decayed strength.

        Used by the field visualizer API to show the live state of the field.
        Does NOT trigger Hebbian reinforcement (read-only).
        """
        collection = f"field_{context_id}"
        try:
            points, _ = await self._client.scroll(collection, limit=10000)
        except Exception:
            logger.debug("[Field] get_patterns: collection %s missing or unreachable", collection, exc_info=True)
            return []

        now = datetime.now(timezone.utc)
        patterns = []
        for p in points:
            payload = p.payload
            age_hours = (now - datetime.fromisoformat(payload["last_accessed"])).total_seconds() / 3600
            decayed = self._compute_decayed_strength(
                payload["strength"], age_hours, payload["access_count"],
            )
            patterns.append({
                "id": str(p.id),
                "key": payload["key"],
                "value": payload["value"][:500],  # truncate for UI
                "agent_id": payload["agent_id"],
                "strength": payload["strength"],
                "decayed_strength": round(decayed, 4),
                "access_count": payload["access_count"],
                "created_at": payload["created_at"],
                "last_accessed": payload["last_accessed"],
                "is_archived": decayed < self._archival_threshold,
            })

        # Sort by decayed strength descending (strongest patterns first)
        patterns.sort(key=lambda x: x["decayed_strength"], reverse=True)
        return patterns

    # ── Internals ───────────────────────────────────────────────

    def _compute_decayed_strength(
        self,
        initial_strength: float,
        age_hours: float,
        access_count: int,
    ) -> float:
        """S(t) = S₀ × e^(-λt) × access_boost

        λ = 0.1 → half-life ≈ 6.93 hours
        access_boost = 1 + (access_count × 0.05), capped at 2.0
        """
        decay = math.exp(-self._decay_rate * age_hours)
        access_boost = min(
            1.0 + (access_count * self._reinforce_bonus),
            self._reinforce_cap,
        )
        return initial_strength * decay * access_boost

    async def _find_by_hash(self, context_id: str, content_hash: str):
        collection = f"field_{context_id}"
        results, _ = await self._client.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="content_hash", match=MatchValue(value=content_hash))]
            ),
            limit=1,
        )
        return results[0] if results else None

    async def _reinforce_single(self, context_id: str, point_id: str) -> None:
        """Reinforce a single pattern (used during dedup inject)."""
        collection = f"field_{context_id}"
        points = await self._client.retrieve(collection, ids=[point_id])
        if not points:
            return

        point = points[0]
        now = datetime.now(timezone.utc).isoformat()
        new_count = point.payload["access_count"] + 1

        await self._client.set_payload(
            collection_name=collection,
            payload={
                "access_count": new_count,
                "last_accessed": now,
            },
            points=[point_id],
        )

    async def _reinforce_batch(self, context_id: str, point_ids: list[str]) -> None:
        """Hebbian reinforcement: accessed patterns resist decay.

        Co-access bonus: when multiple patterns are retrieved together,
        each gets +2% per co-accessed pattern (neurons that fire together
        wire together). Capped at reinforce_cap × initial strength.
        """
        collection = f"field_{context_id}"
        now = datetime.now(timezone.utc).isoformat()

        all_points = await self._client.retrieve(collection, ids=point_ids)
        if not all_points:
            return

        point_map = {str(p.id): p for p in all_points}

        for pid in point_ids:
            point = point_map.get(pid)
            if not point:
                continue

            new_count = point.payload["access_count"] + 1
            initial_strength = point.payload["strength"]

            # Co-access bonus
            if len(point_ids) > 1:
                boosted = min(
                    initial_strength * (1.0 + 0.02 * (len(point_ids) - 1)),
                    initial_strength * self._reinforce_cap,
                )
            else:
                boosted = initial_strength

            await self._client.set_payload(
                collection_name=collection,
                payload={
                    "access_count": new_count,
                    "last_accessed": now,
                    "strength": boosted,
                },
                points=[pid],
            )

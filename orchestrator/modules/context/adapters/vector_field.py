"""
Vector Field Shared Context — PRD-108
======================================

Shared semantic field where agent knowledge resonates, decays, and
forms attractors. Implements SharedContextPort so it's swappable
with RedisSharedContext (PRD-107) for A/B comparison.

All fields live in a SINGLE Qdrant collection. Each pattern is a
point with a 2048-dim embedding and a payload that includes the
owning ``field_id`` (one UUID per mission). Queries filter by
``field_id``; destroy = delete-by-filter. One HNSW for the whole
platform, scales by point count rather than collection count.

Resonance = cosine_similarity² × decayed_strength at query time.
This is how agents share a brain instead of playing telephone.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

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
from core.ports.context import SharedContextPort
from modules.context import field_scoring

logger = logging.getLogger(__name__)

# Single shared collection for all field memory. Per-mission isolation
# is enforced via the ``field_id`` payload filter.
SHARED_COLLECTION = "field_memory"


@dataclass(frozen=True)
class CompactionResult:
    """PRD-178 S3 (F063): outcome of one bounded compaction pass.

    ``next_offset`` is the opaque Qdrant scroll cursor to resume from on the
    next run; ``None`` means the sweep reached the end of the (scoped)
    collection, so the next run starts a fresh full pass. The caller persists
    ``next_offset`` (see ``modules.context.compaction_cursor``) — the adapter
    stays DB-free."""

    pruned: int
    next_offset: Optional[Any]
    scanned: int


class VectorFieldSharedContext(SharedContextPort):
    """PRD-108: vector field backend for SharedContextPort.

    Patterns injected by one agent become queryable by all agents
    in the same field. Relevance is surfaced by resonance
    (cosine² × strength), not by recency or insertion order.

    All fields share one Qdrant collection (``field_memory``).
    Per-mission isolation is enforced via the ``field_id`` payload
    on every point and a payload index on that key.
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
        self._half_life_access_scale = config.FIELD_HALF_LIFE_ACCESS_SCALE
        self._bootstrap_done = False

    def _scoring_params(self) -> field_scoring.ScoringParams:
        """PRD-166 S2: the config-sourced curve shared by query, viz, archival,
        and compaction (one honest definition, D11 config-driven)."""
        return field_scoring.ScoringParams(
            decay_rate=self._decay_rate,
            reinforce_bonus=self._reinforce_bonus,
            reinforce_cap=self._reinforce_cap,
            archival_threshold=self._archival_threshold,
            half_life_access_scale=self._half_life_access_scale,
        )

    # ── Bootstrap ───────────────────────────────────────────────

    async def ensure_shared_collection(self) -> None:
        """Idempotent — create the shared collection and its payload
        indexes if missing. Safe to call on every startup.
        """
        if self._bootstrap_done:
            return

        try:
            exists = await self._client.collection_exists(SHARED_COLLECTION)
        except Exception:
            logger.warning("[Field] collection_exists check failed", exc_info=True)
            exists = False

        if not exists:
            await self._client.create_collection(
                collection_name=SHARED_COLLECTION,
                vectors_config=VectorParams(
                    size=self._dimension,
                    distance=Distance.COSINE,
                    on_disk=True,
                ),
                hnsw_config=HnswConfigDiff(on_disk=True),
                on_disk_payload=True,
            )
            logger.info("[Field] Created shared collection %s", SHARED_COLLECTION)

        # Payload indexes — idempotent in Qdrant, safe to call repeatedly.
        for field_name, schema in [
            ("field_id", PayloadSchemaType.KEYWORD),
            ("workspace_id", PayloadSchemaType.KEYWORD),  # PRD-166 S1: workspace-scoped recall
            ("content_hash", PayloadSchemaType.KEYWORD),
            ("agent_id", PayloadSchemaType.INTEGER),
            ("created_at", PayloadSchemaType.KEYWORD),
        ]:
            try:
                await self._client.create_payload_index(
                    collection_name=SHARED_COLLECTION,
                    field_name=field_name,
                    field_schema=schema,
                    wait=True,
                )
            except Exception:
                # "already exists" is the expected case after first boot
                logger.debug("[Field] payload index %s present", field_name)

        self._bootstrap_done = True

    # ── Filter helpers ──────────────────────────────────────────

    @staticmethod
    def _field_filter(field_id: str, extra: Optional[list[FieldCondition]] = None) -> Filter:
        must: list[FieldCondition] = [
            FieldCondition(key="field_id", match=MatchValue(value=field_id)),
        ]
        if extra:
            must.extend(extra)
        return Filter(must=must)

    @staticmethod
    def _workspace_filter(workspace_id: str) -> Filter:
        """PRD-166 S1: match every pattern accumulated in a workspace, across all
        its missions' fields."""
        return Filter(must=[
            FieldCondition(key="workspace_id", match=MatchValue(value=workspace_id)),
        ])

    # ── Create / Destroy ────────────────────────────────────────

    async def create_context(
        self,
        team_agent_ids: list[int],
        initial_data: Optional[dict[str, Any]] = None,
        provenance: Optional[dict[str, Any]] = None,
    ) -> str:
        await self.ensure_shared_collection()
        field_id = str(uuid.uuid4())

        if initial_data:
            for key, value in initial_data.items():
                await self.inject(
                    field_id, key, str(value), agent_id=0, strength=1.0,
                    provenance=provenance,
                )

        logger.info(
            "[Field] Created field %s (team=%s, seeded=%d)",
            field_id, team_agent_ids, len(initial_data or {}),
        )
        return field_id

    async def destroy_context(self, context_id: str) -> None:
        """Delete every point belonging to this field. Atomic."""
        try:
            await self._client.delete(
                collection_name=SHARED_COLLECTION,
                points_selector=FilterSelector(
                    filter=self._field_filter(context_id),
                ),
            )
            logger.info("[Field] Destroyed field %s", context_id)
        except Exception:
            logger.warning("[Field] Failed to destroy field %s", context_id, exc_info=True)

    async def context_exists(self, context_id: str) -> bool:
        """A field 'exists' if at least one point references it.

        Used by the coordinator to detect stale field_ids inherited
        from a parent/template mission whose data has already been
        destroyed.
        """
        try:
            points, _ = await self._client.scroll(
                collection_name=SHARED_COLLECTION,
                scroll_filter=self._field_filter(context_id),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            return bool(points)
        except Exception:
            logger.debug("[Field] context_exists check failed for %s", context_id, exc_info=True)
            return False

    # ── Inject ──────────────────────────────────────────────────

    async def inject(
        self,
        context_id: str,
        key: str,
        value: str,
        agent_id: int,
        strength: float = 1.0,
        provenance: Optional[dict[str, Any]] = None,
    ) -> None:
        await self.ensure_shared_collection()

        content = f"{key}: {value}"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Dedup within this field only
        existing = await self._find_by_hash(context_id, content_hash)
        if existing:
            await self._reinforce_single(existing.id)
            logger.debug("[Field] Deduplicated inject — reinforced existing pattern")
            return

        embedding = await self._embedder.generate_embedding(content)
        effective_strength = strength * self._boundary_permeability
        now = datetime.now(timezone.utc).isoformat()
        point_id = str(uuid.uuid4())
        prov = provenance or {}

        await self._client.upsert(
            collection_name=SHARED_COLLECTION,
            points=[PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "field_id": context_id,
                    # PRD-166 S1: provenance — survives the mission so the
                    # workspace field keeps cross-mission lineage.
                    "workspace_id": prov.get("workspace_id"),
                    "mission_id": prov.get("mission_id"),
                    "task_id": prov.get("task_id"),
                    "agent_id": agent_id,
                    "key": key,
                    "value": value,
                    "strength": effective_strength,
                    "created_at": now,
                    "last_accessed": now,
                    "access_count": 0,
                    "expired_at": None,  # set when the mission field is archived (S1)
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
        top_k: int = 0,
        record_access: bool = True,
    ) -> list[dict[str, Any]]:
        """Mission-scoped query: rank this field's patterns by three-factor
        resonance (similarity × stability × recency).

        PRD-178 S2 (F062): ``record_access=False`` skips Hebbian reinforcement so
        the retrieval-trace inspector can observe the field WITHOUT mutating the
        access_count/last_accessed/strength it reports on."""
        return await self._scored_search(
            self._field_filter(context_id), query, top_k, label=f"field={context_id}",
            record_access=record_access,
        )

    async def query_workspace(
        self,
        workspace_id: str,
        query: str,
        agent_id: int = 0,
        top_k: int = 0,
        query_vector: Optional[list[float]] = None,
    ) -> list[dict[str, Any]]:
        """PRD-166 S1: workspace-persistent recall — rank patterns accumulated
        across every mission in the workspace (filter on ``workspace_id``, not a
        single ``field_id``). Powers cross-mission learning.

        PRD-164 S2 (Q21): pass ``query_vector`` to reuse an embedding the caller
        already computed (keeps agent dispatch at one embedding call)."""
        return await self._scored_search(
            self._workspace_filter(workspace_id), query, top_k,
            label=f"ws={workspace_id}", query_vector=query_vector,
        )

    async def _scored_search(
        self,
        scroll_filter: Filter,
        query: str,
        top_k: int,
        label: str,
        query_vector: Optional[list[float]] = None,
        record_access: bool = True,
    ) -> list[dict[str, Any]]:
        await self.ensure_shared_collection()
        top_k = top_k or config.FIELD_QUERY_TOP_K
        params = self._scoring_params()

        query_embedding = (
            query_vector if query_vector is not None
            else await self._embedder.generate_embedding(query)
        )

        # Over-fetch — decay/archival filtering will reduce the set.
        response = await self._client.query_points(
            collection_name=SHARED_COLLECTION,
            query=query_embedding,
            query_filter=scroll_filter,
            limit=top_k * config.FIELD_QUERY_OVER_FETCH,
        )

        now = datetime.now(timezone.utc)
        scored: list[dict[str, Any]] = []
        for hit in response.points:
            payload = hit.payload
            age_hours = (now - datetime.fromisoformat(payload["last_accessed"])).total_seconds() / 3600
            ds = self._compute_decayed_strength(
                payload["strength"], age_hours, payload["access_count"],
            )
            if ds < self._archival_threshold:
                continue
            scored.append({
                "id": hit.id,
                "key": payload["key"],
                "value": payload["value"],
                "score": field_scoring.resonance(
                    hit.score, payload["strength"], age_hours, payload["access_count"], params,
                ),
                "agent_id": payload.get("agent_id", 0),
                "mission_id": payload.get("mission_id"),
                "decayed_strength": ds,
                "cosine_similarity": hit.score,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        top_results = scored[:top_k]

        # Hebbian reinforcement — accessed patterns resist future decay.
        # PRD-178 S2 (F062): the trace inspector passes record_access=False so
        # observing the field never writes access patterns back into it.
        if record_access:
            accessed_ids = [r["id"] for r in top_results]
            if accessed_ids:
                await self._reinforce_batch(accessed_ids)

        logger.info(
            "[Field] Query %s results=%d top_score=%.4f query=%s",
            label, len(top_results),
            top_results[0]["score"] if top_results else 0.0,
            query[:60],
        )
        return top_results

    # ── Workspace lifecycle (PRD-166 S1) ────────────────────────

    async def archive_into_workspace(self, field_id: str, workspace_id: str) -> None:
        """Merge a terminal mission's field into the workspace-persistent field:
        stamp ``workspace_id`` (so the patterns join cross-mission recall) and
        ``expired_at`` (so mission-scoped views can soft-archive) on every point.
        Cheap and in-place — the compaction job consolidates over time."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            await self._client.set_payload(
                collection_name=SHARED_COLLECTION,
                payload={"workspace_id": str(workspace_id), "expired_at": now},
                points=self._field_filter(field_id),
            )
            logger.info("[Field] Archived field %s into workspace %s", field_id, workspace_id)
        except Exception:
            logger.warning(
                "[Field] Failed to archive field %s into workspace %s",
                field_id, workspace_id, exc_info=True,
            )

    async def compact(
        self,
        workspace_id: Optional[str] = None,
        prune_threshold: Optional[float] = None,
        resume_offset: Optional[Any] = None,
        max_scan: Optional[int] = None,
    ) -> CompactionResult:
        """Bound Qdrant: delete points whose decayed strength has fallen below the
        hard prune floor (``FIELD_PRUNE_THRESHOLD`` — stricter than archival, so
        archived-but-live patterns survive). The prune decision is the pure
        ``field_scoring.is_prunable`` (unit-tested).

        PRD-178 S3 (F063):
          * **Workspace scope** — when ``workspace_id`` is given the scroll is
            filtered to that workspace, so another workspace's points are never
            scanned. (Unscoped only for an explicit whole-collection sweep.)
          * **Resume cursor** — start scanning from ``resume_offset`` (the
            opaque Qdrant scroll cursor persisted by the previous run) instead
            of restarting at the top every time. Returns ``CompactionResult``
            whose ``next_offset`` the caller persists; ``None`` means a full
            pass completed and the next run starts fresh.
        """
        threshold = config.FIELD_PRUNE_THRESHOLD if prune_threshold is None else prune_threshold
        scan_budget = config.FIELD_COMPACTION_MAX_SCAN if max_scan is None else max_scan
        params = self._scoring_params()
        scroll_filter = self._workspace_filter(str(workspace_id)) if workspace_id else None
        now = datetime.now(timezone.utc)

        to_delete: list[Any] = []
        offset = resume_offset
        scanned = 0
        while scanned < scan_budget:
            points, offset = await self._client.scroll(
                collection_name=SHARED_COLLECTION,
                scroll_filter=scroll_filter,
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                offset = None  # end of collection — next run starts fresh
                break
            for p in points:
                scanned += 1
                payload = p.payload or {}
                try:
                    age_hours = (
                        now - datetime.fromisoformat(payload["last_accessed"])
                    ).total_seconds() / 3600
                except (KeyError, ValueError, TypeError):
                    continue
                if field_scoring.is_prunable(
                    payload.get("strength", 0.0), age_hours,
                    payload.get("access_count", 0), params, threshold,
                ):
                    to_delete.append(p.id)
            if offset is None:
                break

        if to_delete:
            await self._client.delete(
                collection_name=SHARED_COLLECTION,
                points_selector=to_delete,
            )
        logger.info(
            "[Field] Compaction pruned %d/%d scanned point(s) (ws=%s, resume=%s)",
            len(to_delete), scanned, workspace_id, offset is not None,
        )
        return CompactionResult(
            pruned=len(to_delete), next_offset=offset, scanned=scanned,
        )

    async def health(self) -> dict[str, Any]:
        """PRD-166 S2: real backend health — pings Qdrant instead of reporting a
        hardcoded ``'healthy'``. Reflects an actual outage so callers can tell a
        live-but-empty field from a down backend."""
        try:
            await self._client.get_collections()
            return {
                "healthy": True,
                "backend": "vector_field",
                "collection": SHARED_COLLECTION,
            }
        except Exception as exc:
            return {
                "healthy": False,
                "backend": "vector_field",
                "error": f"{type(exc).__name__}: {exc}",
            }

    # ── Stability measurement ───────────────────────────────────

    async def measure_stability(self, context_id: str) -> dict[str, Any]:
        """How converged is the field? Used for telemetry and experiment analysis.

        stability = (avg_strength × 0.6) + (organization × 0.4)
        organization = 1 - (stddev / mean) if mean > 0
        """
        try:
            points, _ = await self._client.scroll(
                collection_name=SHARED_COLLECTION,
                scroll_filter=self._field_filter(context_id),
                limit=10000,
            )
        except Exception:
            logger.debug("[Field] measure_stability scroll failed for %s", context_id, exc_info=True)
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
        """Return all patterns in ONE mission field with decayed strength.

        Used by the field visualizer API to show the live state of the field.
        Does NOT trigger Hebbian reinforcement (read-only).
        """
        return await self._list_patterns(self._field_filter(context_id), f"field={context_id}")

    async def get_workspace_patterns(self, workspace_id: str) -> list[dict[str, Any]]:
        """PRD-166 S1/S4: every pattern accumulated across a workspace's missions,
        for the workspace-scoped Field view. Read-only."""
        return await self._list_patterns(self._workspace_filter(workspace_id), f"ws={workspace_id}")

    async def _list_patterns(self, scroll_filter: Filter, label: str) -> list[dict[str, Any]]:
        try:
            points, _ = await self._client.scroll(
                collection_name=SHARED_COLLECTION,
                scroll_filter=scroll_filter,
                limit=config.FIELD_COMPACTION_MAX_SCAN,
            )
        except Exception:
            logger.debug("[Field] get_patterns scroll failed for %s", label, exc_info=True)
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
                # PRD-166 S1: provenance + soft-archive for the viz / inspector.
                "mission_id": payload.get("mission_id"),
                "expired_at": payload.get("expired_at"),
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
        """PRD-166 S2: stability × recency (adaptive half-life), delegated to the
        pure ``field_scoring`` module so query/viz/archival/compaction agree."""
        return field_scoring.decayed_strength(
            initial_strength, age_hours, access_count, self._scoring_params(),
        )

    async def _find_by_hash(self, context_id: str, content_hash: str):
        """Lookup an existing point in this field with the same content hash.

        Filter is ``field_id AND content_hash``; both are payload-indexed.
        """
        results, _ = await self._client.scroll(
            collection_name=SHARED_COLLECTION,
            scroll_filter=self._field_filter(
                context_id,
                extra=[FieldCondition(key="content_hash", match=MatchValue(value=content_hash))],
            ),
            limit=1,
        )
        return results[0] if results else None

    async def _reinforce_single(self, point_id: str) -> None:
        """Reinforce a single pattern (used during dedup inject).

        Point IDs are globally unique UUIDs across the shared
        collection, so no field_id filter is needed for the lookup.
        """
        points = await self._client.retrieve(SHARED_COLLECTION, ids=[point_id])
        if not points:
            return

        point = points[0]
        now = datetime.now(timezone.utc).isoformat()
        new_count = point.payload["access_count"] + 1

        await self._client.set_payload(
            collection_name=SHARED_COLLECTION,
            payload={
                "access_count": new_count,
                "last_accessed": now,
            },
            points=[point_id],
        )

    async def _reinforce_batch(self, point_ids: list[str]) -> None:
        """Hebbian reinforcement: accessed patterns resist decay.

        Co-access bonus: when multiple patterns are retrieved together,
        each gets +2% per co-accessed pattern (neurons that fire together
        wire together). Capped at reinforce_cap × initial strength.
        """
        now = datetime.now(timezone.utc).isoformat()

        all_points = await self._client.retrieve(SHARED_COLLECTION, ids=point_ids)
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
                collection_name=SHARED_COLLECTION,
                payload={
                    "access_count": new_count,
                    "last_accessed": now,
                    "strength": boosted,
                },
                points=[pid],
            )

"""Field → durable memory promotion — PRD-178 S4 (the moat arm).

Strong, frequently-recalled field-memory patterns decay and are hard-deleted by
compaction before they can become durable (so the field never compounds into
long-term memory). This job distills those patterns into durable mem0 (L3)
memory BEFORE the delete, so accumulated field knowledge survives as durable,
searchable memory.

Ordering is load-bearing:
  1. **Taint gate FIRST** (top-risk #4 — promotion is the memory-poisoning
     surface): a pattern whose provenance names untrusted external content
     (inbound email/web/webhook) is NEVER promoted. Its data-subject/source tags
     travel with the entry on the Qdrant payload.
  2. Survivors are distilled into a TYPED durable memory with provenance
     preserved, via the existing ``UnifiedMemoryService.store_long_term`` path
     (PRD-159) — no parallel durable writer is built.
  3. Only a SUCCESSFUL durable write earns a delete from the field. A failed
     write leaves the pattern in place (never lose it to a failed promotion).

Registered on the shared APScheduler by ``FieldPromotionJobScheduler`` (mirrors
``services.memory_jobs.MemoryJobScheduler`` — the L2→L3 promotion sibling).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import config
from modules.context import field_scoring

logger = logging.getLogger(__name__)

# Typed category for promoted field patterns (the "typed durable memory").
PROMOTED_CATEGORY = "field_pattern"


def _untrusted_sources() -> set:
    raw = config.FIELD_PROMOTION_UNTRUSTED_SOURCES or ""
    return {s.strip().lower() for s in raw.split(",") if s.strip()}


class FieldMemoryPromoter:
    """Promotes strong, clean field patterns to durable mem0 memory.

    Constructed with the inner vector-field adapter (for its Qdrant client +
    workspace filter + scoring params) and a durable memory service exposing
    ``store_long_term`` (the PRD-159 L3 writer). Both are injected so the job is
    unit-testable with Qdrant/mem0 mocked at the boundary.
    """

    def __init__(self, field_inner: Any, memory_service: Any) -> None:
        self._inner = field_inner
        self._memory = memory_service

    async def promote_workspace(self, workspace_id: str) -> Dict[str, int]:
        """Scan a workspace's field, promote every strong+clean pattern to
        durable memory (then delete it from the field), and skip tainted ones.
        Returns counts: ``promoted``, ``skipped_tainted``, ``failed``, ``scanned``."""
        counts = {"promoted": 0, "skipped_tainted": 0, "failed": 0, "scanned": 0}

        if not config.FIELD_PROMOTION_ENABLED:
            return counts

        client = getattr(self._inner, "_client", None)
        if client is None:
            return counts

        try:
            from modules.context.adapters.vector_field import SHARED_COLLECTION
        except Exception:
            logger.warning("[FieldPromotion] adapter import failed", exc_info=True)
            return counts

        params = self._inner._scoring_params()
        untrusted = _untrusted_sources()
        now = datetime.now(timezone.utc)
        scroll_filter = self._inner._workspace_filter(str(workspace_id))

        offset = None
        promoted_ids: List[Any] = []
        while counts["scanned"] < config.FIELD_PROMOTION_MAX_SCAN:
            try:
                points, offset = await client.scroll(
                    collection_name=SHARED_COLLECTION,
                    scroll_filter=scroll_filter,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception:
                logger.warning(
                    "[FieldPromotion] scroll failed for ws=%s", workspace_id, exc_info=True,
                )
                break
            if not points:
                break
            for p in points:
                counts["scanned"] += 1
                payload = p.payload or {}
                provenance = self._provenance_of(payload)

                # Taint gate FIRST — never even score a tainted trajectory for
                # promotion; leave it in the field (do not delete).
                if field_scoring.is_tainted(provenance, untrusted):
                    counts["skipped_tainted"] += 1
                    logger.info(
                        "[FieldPromotion] taint-blocked pattern id=%s ws=%s source=%s",
                        p.id, workspace_id, provenance.get("source"),
                    )
                    continue

                ds = self._decayed_strength(payload, params, now)
                if not field_scoring.is_promotable(
                    ds, payload.get("access_count", 0), provenance,
                    min_strength=config.FIELD_PROMOTION_MIN_STRENGTH,
                    min_access_count=config.FIELD_PROMOTION_MIN_ACCESS_COUNT,
                    untrusted_sources=untrusted,
                ):
                    continue

                if await self._promote_one(workspace_id, payload, provenance):
                    counts["promoted"] += 1
                    promoted_ids.append(p.id)
                else:
                    counts["failed"] += 1
            if offset is None:
                break

        # Delete ONLY successfully-promoted points — promotion happens before
        # the delete, and a delete is the reward for a durable write.
        if promoted_ids:
            try:
                await client.delete(
                    collection_name=SHARED_COLLECTION,
                    points_selector=promoted_ids,
                )
            except Exception:
                logger.warning(
                    "[FieldPromotion] post-promotion delete failed for ws=%s",
                    workspace_id, exc_info=True,
                )

        logger.info(
            "[FieldPromotion] ws=%s promoted=%d tainted=%d failed=%d scanned=%d",
            workspace_id, counts["promoted"], counts["skipped_tainted"],
            counts["failed"], counts["scanned"],
        )
        return counts

    async def _promote_one(
        self, workspace_id: str, payload: Dict[str, Any], provenance: Dict[str, Any]
    ) -> bool:
        """Distill one field pattern into a typed durable memory, provenance
        preserved. Reuses the PRD-159 durable writer. Returns True on success."""
        key = payload.get("key", "")
        value = payload.get("value", "")
        content = f"{key}: {value}".strip(": ").strip() if key else str(value)
        agent_id = payload.get("agent_id")

        metadata = {
            "promoted_from_field": True,
            "field_id": payload.get("field_id"),
            "mission_id": payload.get("mission_id"),
            "task_id": payload.get("task_id"),
            "source": provenance.get("source"),
            "field_strength": payload.get("strength"),
            "field_access_count": payload.get("access_count"),
        }
        # Drop null keys so the durable metadata stays clean.
        metadata = {k: v for k, v in metadata.items() if v is not None}

        try:
            result = await self._memory.store_long_term(
                workspace_id=str(workspace_id),
                content=content,
                agent_id=agent_id,
                category=PROMOTED_CATEGORY,
                metadata=metadata,
            )
        except Exception:
            logger.warning(
                "[FieldPromotion] durable write raised for ws=%s", workspace_id,
                exc_info=True,
            )
            return False

        if isinstance(result, dict) and result.get("success") is False:
            return False
        return True

    @staticmethod
    def _provenance_of(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the provenance/taint tags that travel with a field point.

        A pattern may carry a nested ``provenance`` dict or flat tags on the
        payload (``source``/``source_type``/``untrusted``/``tainted``/lineage
        ids). Merge both into one dict for the taint gate + metadata."""
        prov: Dict[str, Any] = {}
        nested = payload.get("provenance")
        if isinstance(nested, dict):
            prov.update(nested)
        for tag in ("source", "source_type", "untrusted", "tainted",
                    "field_id", "mission_id", "task_id"):
            if tag in payload and payload[tag] is not None and tag not in prov:
                prov[tag] = payload[tag]
        return prov

    def _decayed_strength(
        self, payload: Dict[str, Any], params: Any, now: datetime
    ) -> float:
        try:
            age_hours = (
                now - datetime.fromisoformat(payload["last_accessed"])
            ).total_seconds() / 3600
        except (KeyError, ValueError, TypeError):
            return 0.0
        return field_scoring.decayed_strength(
            payload.get("strength", 0.0), age_hours,
            payload.get("access_count", 0), params,
        )


# ---------------------------------------------------------------------------
# Run-all + scheduler registration (mirrors services.memory_jobs)
# ---------------------------------------------------------------------------

async def run_field_promotion_all() -> Dict[str, Any]:
    """Promote across every workspace that has accumulated field data. One
    workspace failure never stops the others."""
    if not config.FIELD_PROMOTION_ENABLED:
        return {"workspaces_processed": 0, "total_promoted": 0,
                "total_tainted": 0, "total_failed": 0, "errors": 0}

    from modules.context.factory import get_shared_context

    field = get_shared_context()
    inner = getattr(field, "_inner", field) if field else None
    if inner is None or getattr(inner, "_client", None) is None:
        return {"workspaces_processed": 0, "total_promoted": 0,
                "total_tainted": 0, "total_failed": 0, "errors": 0}

    from modules.memory.unified_memory_service import get_unified_memory_service

    promoter = FieldMemoryPromoter(
        field_inner=inner, memory_service=get_unified_memory_service(),
    )

    workspace_ids = _workspaces_with_field_data()
    totals = {"workspaces_processed": 0, "total_promoted": 0,
              "total_tainted": 0, "total_failed": 0, "errors": 0}
    for ws_id in workspace_ids:
        try:
            r = await promoter.promote_workspace(str(ws_id))
            totals["workspaces_processed"] += 1
            totals["total_promoted"] += r["promoted"]
            totals["total_tainted"] += r["skipped_tainted"]
            totals["total_failed"] += r["failed"]
        except Exception:
            totals["errors"] += 1
            logger.error(
                "[FieldPromotion] run_all failed for ws=%s", ws_id, exc_info=True,
            )
    return totals


def _workspaces_with_field_data() -> List[str]:
    """Distinct workspace ids that have accumulated field memory (a run with a
    ``field_id``)."""
    from sqlalchemy import text

    from core.database.database import get_db_session

    try:
        with get_db_session() as db:
            rows = db.execute(
                text(
                    "SELECT DISTINCT workspace_id FROM orchestration_runs "
                    "WHERE config->>'field_id' IS NOT NULL"
                )
            ).fetchall()
        return [str(r[0]) for r in rows if r[0] is not None]
    except Exception:
        logger.warning("[FieldPromotion] workspace scan failed", exc_info=True)
        return []


class FieldPromotionJobScheduler:
    """Registers the daily field→durable promotion job on the shared scheduler
    (mirrors ``services.memory_jobs.MemoryJobScheduler``)."""

    JOB_ID = "field_memory_promotion"

    def __init__(self) -> None:
        self._scheduler = None

    async def start(self, scheduler) -> None:
        if not config.FIELD_PROMOTION_ENABLED:
            logger.info("[FieldPromotion] disabled (FIELD_PROMOTION_ENABLED=false)")
            return
        self._scheduler = scheduler
        self._scheduler.add_job(
            self._run,
            "cron",
            hour=config.FIELD_PROMOTION_HOUR_UTC,
            minute=30,
            id=self.JOB_ID,
            replace_existing=True,
            max_instances=1,
        )
        logger.info(
            "[FieldPromotion] Started — daily at %02d:30 UTC",
            config.FIELD_PROMOTION_HOUR_UTC,
        )

    async def stop(self) -> None:
        if self._scheduler and self._scheduler.get_job(self.JOB_ID):
            self._scheduler.remove_job(self.JOB_ID)
            logger.info("[FieldPromotion] Stopped")

    async def _run(self) -> None:
        try:
            result = await run_field_promotion_all()
            logger.info(
                "[FieldPromotion] complete: workspaces=%d promoted=%d tainted=%d "
                "failed=%d errors=%d",
                result.get("workspaces_processed", 0),
                result.get("total_promoted", 0),
                result.get("total_tainted", 0),
                result.get("total_failed", 0),
                result.get("errors", 0),
            )
        except Exception:
            logger.error("[FieldPromotion] job run failed", exc_info=True)


_field_promotion_scheduler: Optional[FieldPromotionJobScheduler] = None


def get_field_promotion_scheduler() -> FieldPromotionJobScheduler:
    global _field_promotion_scheduler
    if _field_promotion_scheduler is None:
        _field_promotion_scheduler = FieldPromotionJobScheduler()
    return _field_promotion_scheduler

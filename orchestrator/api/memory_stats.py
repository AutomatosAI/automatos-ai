"""
Memory Stats API — Mem0-first with local DB fallback
=====================================================
Queries the Mem0 service (via UnifiedMemoryService) for memory data.
Falls back to the local memory_items table if Mem0 is unavailable.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, text
from datetime import datetime

from core.database.database import get_db
from modules.memory.storage.knowledge_system import MemoryItem
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/memory", tags=["Real Memory Stats"])


# ---------------------------------------------------------------------------
# UnifiedMemoryService helper (lazy, optional)
# ---------------------------------------------------------------------------

_memory_service: Optional[Any] = None
_memory_service_checked: bool = False


def _get_memory_service():
    """Return the UnifiedMemoryService singleton or None if unavailable."""
    global _memory_service, _memory_service_checked
    if _memory_service_checked:
        return _memory_service
    _memory_service_checked = True
    try:
        from modules.memory.unified_memory_service import get_unified_memory_service
        svc = get_unified_memory_service()
        if svc.is_mem0_configured:
            _memory_service = svc
            logger.info("[memory_stats] Using UnifiedMemoryService")
        else:
            logger.warning("[memory_stats] Mem0 not configured")
    except Exception as exc:
        logger.warning("[memory_stats] UnifiedMemoryService unavailable: %s", exc)
    return _memory_service


def _get_agent_ids(workspace_id, db: Session) -> List[int]:
    """Get all agent IDs for a workspace."""
    try:
        from core.models.core import Agent
        rows = db.query(Agent.id).filter(Agent.workspace_id == workspace_id).all()
        return [aid for (aid,) in rows]
    except Exception as e:
        logger.warning("Failed to fetch agent IDs for memory stats: %s", e)
        return []


async def _fetch_all_scoped_memories(
    service,
    workspace_id: str,
    agent_ids: List[int],
    limit: int = 500,
    query: Optional[str] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Fetch memories from all scopes (workspace, agent, daily) using
    UnifiedMemoryService. Returns list of (tier_label, memory_dict) tuples,
    deduplicated by ID.
    """
    from modules.memory.unified_memory_service import MemoryNamespace

    ws = str(workspace_id)
    ns = MemoryNamespace(workspace_id=ws)

    tasks: List = []
    scope_labels: List[str] = []

    if query:
        tasks.append(service.search_long_term(ws, query, limit=limit))
        scope_labels.append("global")
        for aid in agent_ids:
            tasks.append(service.search_long_term(ws, query, agent_id=aid, limit=limit))
            scope_labels.append("agent")
        tasks.append(service.search_long_term_scoped(ns.daily(), query, limit=limit))
        scope_labels.append("daily")
    else:
        tasks.append(service.get_all_memories(ws, limit=limit))
        scope_labels.append("global")
        for aid in agent_ids:
            tasks.append(service.get_all_memories(ws, agent_id=aid, limit=limit))
            scope_labels.append("agent")
        tasks.append(service.get_all_daily_logs(ws, limit=limit))
        scope_labels.append("daily")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    seen_ids: set = set()
    all_items: List[Tuple[str, Dict[str, Any]]] = []
    for label, result in zip(scope_labels, results):
        if isinstance(result, Exception):
            logger.warning("[memory_stats] Fetch for scope %s failed: %s", label, result)
            continue
        items = result if isinstance(result, list) else []
        for m in items:
            mid = str(m.get("id", ""))
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                all_items.append((label, m))

    return all_items


@router.get("/stats/real")
async def get_real_memory_stats(ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get memory statistics — Mem0 first, local DB fallback.
    """
    service = _get_memory_service()
    mem0_total = 0
    mem0_available = False

    # Try Mem0 for total count — query ALL tiers via UnifiedMemoryService
    if service:
        try:
            agent_ids = _get_agent_ids(ctx.workspace_id, db)
            all_items = await _fetch_all_scoped_memories(
                service, str(ctx.workspace_id), agent_ids, limit=1000,
            )
            mem0_total = len(all_items)
            mem0_available = True
        except Exception as e:
            logger.warning("Memory stats query failed, falling back to local DB: %s", e, exc_info=True)

    # Local DB stats (always available as secondary source)
    ws_filter = MemoryItem.workspace_id == ctx.workspace_id
    local_total = db.query(func.count(MemoryItem.id)).filter(ws_filter).scalar() or 0

    total_memories = mem0_total if mem0_available else local_total

    # Additional local DB stats
    memory_by_type = db.query(
        MemoryItem.memory_type,
        func.count(MemoryItem.id).label('count')
    ).filter(ws_filter).group_by(MemoryItem.memory_type).all()

    memory_by_level = db.query(
        MemoryItem.memory_level,
        func.count(MemoryItem.id).label('count')
    ).filter(ws_filter).group_by(MemoryItem.memory_level).all()

    agents_with_memories = db.query(
        func.count(func.distinct(MemoryItem.agent_id))
    ).filter(ws_filter).scalar() or 0

    total_accesses = db.query(
        func.sum(MemoryItem.access_count)
    ).filter(ws_filter).scalar() or 0

    # Hit rate: calculated from memory_access_log (real search-based metric)
    hit_rate = 0
    total_searches = 0
    try:
        access_stats = db.execute(
            text("""
                SELECT
                    COUNT(*) as total_searches,
                    SUM(CASE WHEN had_results THEN 1 ELSE 0 END) as hits
                FROM memory_access_log
                WHERE workspace_id = :ws_id
            """),
            {"ws_id": str(ctx.workspace_id)},
        ).fetchone()
        total_searches = access_stats.total_searches or 0
        hits = access_stats.hits or 0
        hit_rate = round(hits / max(total_searches, 1), 2) if total_searches > 0 else 0
    except Exception as e:
        logger.debug(f"memory_access_log query failed (table may not exist yet): {e}")
        # Fallback to old calculation
        if total_memories > 0 and total_accesses > 0:
            hit_rate = round(min(total_accesses / max(total_memories, 1), 1.0), 2)

    return {
        "system_stats": {
            "total_memories": total_memories,
            "memory_levels": {level: count for level, count in memory_by_level},
            "memory_types": {mtype: count for mtype, count in memory_by_type},
            "agents_with_memories": agents_with_memories,
            "source": "mem0" if mem0_available else "local_db",
        },
        "access_metrics": {
            "total_accesses": total_searches,
            "hit_rate": hit_rate,
            "cache_utilization": min(100, (total_memories / 1000) * 100) if total_memories else 0,
        },
        "is_real_data": True,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/stats/agents")
async def get_agent_memory_stats(ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """Get memory stats per agent with type/level breakdown."""
    try:
        ws_filter = MemoryItem.workspace_id == ctx.workspace_id

        agent_stats = db.query(
            MemoryItem.agent_id,
            func.count(MemoryItem.id).label('memory_count'),
            func.avg(MemoryItem.importance).label('avg_importance'),
            func.sum(MemoryItem.access_count).label('total_accesses'),
            func.max(MemoryItem.created_at).label('last_memory_at'),
        ).filter(ws_filter).group_by(MemoryItem.agent_id).all()

        # Memory type distribution per agent
        type_rows = db.query(
            MemoryItem.agent_id, MemoryItem.memory_type, func.count(MemoryItem.id)
        ).filter(ws_filter).group_by(MemoryItem.agent_id, MemoryItem.memory_type).all()

        type_map: Dict[Any, Dict[str, int]] = {}
        for agent_id, mtype, cnt in type_rows:
            type_map.setdefault(agent_id, {})[mtype or "UNKNOWN"] = cnt

        # Memory level distribution per agent
        level_rows = db.query(
            MemoryItem.agent_id, MemoryItem.memory_level, func.count(MemoryItem.id)
        ).filter(ws_filter).group_by(MemoryItem.agent_id, MemoryItem.memory_level).all()

        level_map: Dict[Any, Dict[str, int]] = {}
        for agent_id, mlevel, cnt in level_rows:
            level_map.setdefault(agent_id, {})[mlevel or "UNKNOWN"] = cnt

        return [
            {
                "agent_id": stat.agent_id,
                "memory_count": stat.memory_count,
                "avg_importance": round(float(stat.avg_importance or 0), 2),
                "total_accesses": stat.total_accesses or 0,
                "memory_types": type_map.get(stat.agent_id, {}),
                "memory_levels": level_map.get(stat.agent_id, {}),
                "last_memory_at": stat.last_memory_at.isoformat() if stat.last_memory_at else None,
            }
            for stat in agent_stats
        ]
    except Exception as e:
        logger.error(f"Failed to get agent memory stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stats/recent")
async def get_recent_memories(
    limit: int = 10,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """
    Get most recent memories — Mem0 first, local DB fallback.
    """
    # Try Mem0 — query ALL tiers via UnifiedMemoryService
    service = _get_memory_service()
    if service:
        try:
            agent_ids = _get_agent_ids(ctx.workspace_id, db)
            scoped_items = await _fetch_all_scoped_memories(
                service, str(ctx.workspace_id), agent_ids, limit=limit,
            )

            all_items: List[Dict[str, Any]] = [
                {
                    "id": str(m.get("id", "")),
                    "agent_id": None,
                    "memory_type": tier,
                    "memory_level": "long_term",
                    "content": _truncate(m.get("content") or m.get("memory") or "", 120),
                    "importance": m.get("score"),
                    "access_count": None,
                    "created_at": m.get("created_at"),
                    "source": "mem0",
                }
                for tier, m in scoped_items
            ]

            if all_items:
                # Sort by created_at descending
                all_items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
                return all_items[:limit]
        except Exception as e:
            logger.warning("Memory recent fetch failed, falling back to local: %s", e, exc_info=True)

    # Fallback: local DB
    recent = db.query(MemoryItem).filter(
        MemoryItem.workspace_id == ctx.workspace_id
    ).order_by(desc(MemoryItem.created_at)).limit(limit).all()

    return [
        {
            "id": str(mem.id),
            "agent_id": mem.agent_id,
            "memory_type": mem.memory_type,
            "memory_level": mem.memory_level,
            "content": _truncate(mem.content, 120),
            "importance": mem.importance,
            "access_count": mem.access_count,
            "created_at": mem.created_at.isoformat() if mem.created_at else None,
            "source": "local_db",
        }
        for mem in recent
    ]


@router.get("/browse")
async def browse_memories(
    query: Optional[str] = None,
    limit: int = 20,
    content_type: Optional[str] = None,
    tier: Optional[str] = None,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Browse/search all memories — PRD-77 Memory Explorer, extended PRD-131d.

    Queries BOTH memory layers for the workspace:
      - L3 (Mem0): distilled facts across global/agent/daily tiers
      - L2 (Postgres memory_short_term): raw transcripts, mission summaries,
        task failures, retry recoveries — preserved verbatim

    Args:
        query: Optional substring/semantic search.
        limit: Max total rows returned.
        content_type: Optional CSV filter restricted to L2 (e.g.
            "transcript,mission_summary,task_failure,retry_recovery").
        tier: Optional scope filter: "l2" (short-term only), "l3" (Mem0 only),
            or None/"all" for both.
    """
    service = _get_memory_service()
    if not service:
        return {"success": False, "error": "Memory service unavailable", "memories": []}

    tier_scope = (tier or "all").lower()
    content_type_list: Optional[List[str]] = None
    if content_type:
        content_type_list = [
            ct.strip() for ct in content_type.split(",") if ct.strip()
        ]

    all_results: List[Dict[str, Any]] = []

    try:
        # --- L3 (Mem0) ---
        if tier_scope in ("all", "l3") and not content_type_list:
            # Mem0 has no content_type concept; if the caller explicitly asked
            # for specific content types, skip Mem0 and query L2 only.
            agent_ids = _get_agent_ids(ctx.workspace_id, db)
            logger.info("[browse] L3 scan workspace=%s", ctx.workspace_id)
            scoped_items = await _fetch_all_scoped_memories(
                service, str(ctx.workspace_id), agent_ids,
                limit=limit, query=query,
            )
            all_results.extend(
                {
                    "id": str(m.get("id", "")),
                    "content": m.get("memory") or m.get("content", ""),
                    "score": m.get("score"),
                    "metadata": m.get("metadata") or m.get("metadata_"),
                    "created_at": m.get("created_at"),
                    "updated_at": m.get("updated_at"),
                    "tier": scope_tier,
                    "layer": "l3",
                    "content_type": None,
                }
                for scope_tier, m in scoped_items
            )

        # --- L2 (Postgres short-term) — PRD-131d ---
        if tier_scope in ("all", "l2"):
            logger.info(
                "[browse] L2 scan workspace=%s content_types=%s",
                ctx.workspace_id, content_type_list,
            )
            l2_rows = await service.list_short_term_by_type(
                workspace_id=str(ctx.workspace_id),
                content_types=content_type_list,
                limit=limit,
                query=query,
            )
            all_results.extend(
                {
                    "id": row.get("id", ""),
                    "content": row.get("content", ""),
                    "score": None,
                    "metadata": row.get("metadata") or {},
                    "created_at": row.get("created_at"),
                    "updated_at": row.get("last_accessed_at"),
                    "tier": "short_term",
                    "layer": "l2",
                    "content_type": row.get("content_type"),
                    "importance": row.get("importance"),
                }
                for row in l2_rows
            )

        logger.info("[browse] Collected %d memories (L2+L3)", len(all_results))

        # Sort by created_at DESC and truncate to caller's limit
        all_results.sort(
            key=lambda x: x.get("created_at") or "",
            reverse=True,
        )
        memories = all_results[:limit]

        if memories:
            logger.info(
                "[browse] Returning %d memories, first: id=%s layer=%s content=%s",
                len(memories), memories[0].get("id"), memories[0].get("layer"),
                str(memories[0].get("content", ""))[:60],
            )
        else:
            logger.warning("[browse] Returning 0 memories")

        return {
            "success": True,
            "memories": memories,
            "total": len(all_results),
            "source": "unified_l2_l3",
            "search_query": query,
        }
    except Exception as e:
        logger.error("Memory browse failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)[:200], "memories": []}


@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Delete a specific memory by ID — PRD-77."""
    service = _get_memory_service()
    if not service:
        return {"success": False, "error": "Memory service unavailable"}

    try:
        # Ownership check: verify memory belongs to this workspace (any tier)
        agent_ids = _get_agent_ids(ctx.workspace_id, db)
        scoped_items = await _fetch_all_scoped_memories(
            service, str(ctx.workspace_id), agent_ids, limit=500,
        )
        owned_ids = {str(m.get("id", "")) for _, m in scoped_items}

        if memory_id not in owned_ids:
            return {"success": False, "error": "Memory not found or not owned by this workspace"}

        deleted = await service.delete_memory(memory_id)
        if deleted:
            logger.info("Memory %s deleted by workspace %s", memory_id, ctx.workspace_id)
            return {"success": True, "message": f"Memory {memory_id} deleted"}
        return {"success": False, "error": f"Failed to delete memory {memory_id}"}
    except Exception as e:
        logger.error("Memory delete failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)[:200]}


@router.get("/health")
async def get_memory_health(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Memory health report — PRD-77.
    Shows total count, staleness, and basic health indicators.
    Queries ALL memory tiers (global, per-agent, daily).
    """
    service = _get_memory_service()

    total = 0
    oldest_memory = None
    newest_memory = None
    mem0_available = False

    if service:
        try:
            agent_ids = _get_agent_ids(ctx.workspace_id, db)
            scoped_items = await _fetch_all_scoped_memories(
                service, str(ctx.workspace_id), agent_ids, limit=500,
            )

            total = len(scoped_items)
            mem0_available = True

            all_dates = [
                m.get("created_at")
                for _, m in scoped_items
                if m.get("created_at")
            ]
            if all_dates:
                oldest_memory = min(all_dates)
                newest_memory = max(all_dates)
        except Exception as e:
            logger.warning("Memory health check failed: %s", e, exc_info=True)

    # Search effectiveness from access log
    search_stats = {"total_searches": 0, "hits": 0, "hit_rate": 0}
    try:
        row = db.execute(
            text("""
                SELECT
                    COUNT(*) as total_searches,
                    SUM(CASE WHEN had_results THEN 1 ELSE 0 END) as hits
                FROM memory_access_log
                WHERE workspace_id = :ws_id
            """),
            {"ws_id": str(ctx.workspace_id)},
        ).fetchone()
        if row and row.total_searches:
            search_stats = {
                "total_searches": row.total_searches,
                "hits": row.hits or 0,
                "hit_rate": round((row.hits or 0) / max(row.total_searches, 1), 2),
            }
    except Exception as e:
        logger.debug("memory_access_log query failed: %s", e)

    return {
        "success": True,
        "mem0_available": mem0_available,
        "total_memories": total,
        "oldest_memory": oldest_memory,
        "newest_memory": newest_memory,
        "search_effectiveness": search_stats,
        "health_status": "healthy" if mem0_available and total > 0 else "degraded" if mem0_available else "unavailable",
    }


@router.get("/layers")
async def get_memory_layers(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Memory layer health endpoint — PRD-79 US-026.

    Returns per-layer stats across the 5-layer memory stack:
      L1 (Redis sessions), L2 (Postgres short-term), L3 (Mem0 long-term),
      L4 (org knowledge: databases + documents).
    Includes overall health status and per-layer latency.
    Result is cached for 60s (configurable via MEMORY_LAYERS_CACHE_TTL_SECONDS).
    """
    import time as _time

    from config import config as app_config

    service = _get_memory_service()
    ws_id = str(ctx.workspace_id)

    # ------------------------------------------------------------------
    # Check Redis cache first
    # ------------------------------------------------------------------
    cache_key = f"mem:layers:{ws_id}"
    cache_ttl = getattr(app_config, "MEMORY_LAYERS_CACHE_TTL_SECONDS", 60)

    if service:
        try:
            redis_client = service._get_redis()
            if redis_client:
                import json as _json

                conn = redis_client.get_redis()
                cached = conn.get(cache_key)
                if cached:
                    return _json.loads(cached)
        except Exception:
            pass  # cache miss — compute fresh

    layers: Dict[str, Any] = {}
    health_issues: List[str] = []

    # ------------------------------------------------------------------
    # L1: Redis Session Stats
    # ------------------------------------------------------------------
    l1_start = _time.monotonic()
    l1: Dict[str, Any] = {"active_sessions": 0, "responding": False, "latency_ms": 0}
    try:
        if service:
            redis_client = service._get_redis()
            if redis_client:
                conn = redis_client.get_redis()
                session_pattern = f"mem:session:{ws_id}:*"
                count = 0
                for _ in conn.scan_iter(match=session_pattern, count=100):
                    count += 1
                l1["active_sessions"] = count
                l1["responding"] = True
    except Exception as e:
        logger.warning("[layers] L1 Redis check failed: %s", e, exc_info=True)
        health_issues.append("L1 Redis unavailable")
    l1["latency_ms"] = round((_time.monotonic() - l1_start) * 1000, 1)
    layers["L1_session"] = l1

    # ------------------------------------------------------------------
    # L2: Postgres Short-term Stats
    # ------------------------------------------------------------------
    l2_start = _time.monotonic()
    l2: Dict[str, Any] = {
        "total_rows": 0,
        "avg_decay_score": 0.0,
        "pending_promotion": 0,
        "responding": False,
        "latency_ms": 0,
    }
    try:
        from modules.memory.models import MemoryShortTerm

        row_count = (
            db.query(func.count(MemoryShortTerm.id))
            .filter(
                MemoryShortTerm.workspace_id == ctx.workspace_id,
                MemoryShortTerm.archived_at.is_(None),
            )
            .scalar()
            or 0
        )
        avg_decay = (
            db.query(func.avg(MemoryShortTerm.decay_score))
            .filter(
                MemoryShortTerm.workspace_id == ctx.workspace_id,
                MemoryShortTerm.archived_at.is_(None),
            )
            .scalar()
        )
        pending_promo = (
            db.query(func.count(MemoryShortTerm.id))
            .filter(
                MemoryShortTerm.workspace_id == ctx.workspace_id,
                MemoryShortTerm.archived_at.is_(None),
                MemoryShortTerm.promoted_to_l3 == False,  # noqa: E712
                MemoryShortTerm.importance > getattr(app_config, "MEMORY_PROMOTION_MIN_IMPORTANCE", 0.7),
                MemoryShortTerm.access_count > getattr(app_config, "MEMORY_PROMOTION_MIN_ACCESS_COUNT", 3),
            )
            .scalar()
            or 0
        )
        l2["total_rows"] = row_count
        l2["avg_decay_score"] = round(float(avg_decay or 0), 3)
        l2["pending_promotion"] = pending_promo
        l2["responding"] = True
    except Exception as e:
        logger.warning("[layers] L2 Postgres check failed: %s", e, exc_info=True)
        health_issues.append("L2 Postgres unavailable")
    l2["latency_ms"] = round((_time.monotonic() - l2_start) * 1000, 1)
    layers["L2_short_term"] = l2

    # ------------------------------------------------------------------
    # L3: Mem0 Long-term Stats
    # ------------------------------------------------------------------
    l3_start = _time.monotonic()
    l3: Dict[str, Any] = {
        "total_memories": 0,
        "responding": False,
        "latency_ms": 0,
    }
    if service and service.is_mem0_configured:
        try:
            mems = await service.get_all_memories(ws_id, limit=1)
            # get_all_memories returns a list; we just need to confirm it works
            # For total count, do a broader fetch (capped at 500 to stay fast)
            all_mems = await service.get_all_memories(ws_id, limit=500)
            l3["total_memories"] = len(all_mems)
            l3["responding"] = True
        except Exception as e:
            logger.warning("[layers] L3 Mem0 check failed: %s", e, exc_info=True)
            health_issues.append("L3 Mem0 unavailable")
    else:
        health_issues.append("L3 Mem0 not configured")
    l3["latency_ms"] = round((_time.monotonic() - l3_start) * 1000, 1)
    layers["L3_long_term"] = l3

    # ------------------------------------------------------------------
    # L4: Organizational Knowledge Stats
    # ------------------------------------------------------------------
    l4_start = _time.monotonic()
    l4: Dict[str, Any] = {
        "connected_databases": 0,
        "documents": 0,
        "responding": False,
        "latency_ms": 0,
    }
    try:
        from core.models.database_knowledge import DatabaseKnowledgeSource
        from core.models.core import Document

        db_count = (
            db.query(func.count(DatabaseKnowledgeSource.id))
            .filter(
                DatabaseKnowledgeSource.workspace_id == ctx.workspace_id,
                DatabaseKnowledgeSource.is_active == True,  # noqa: E712
            )
            .scalar()
            or 0
        )
        doc_count = (
            db.query(func.count(Document.id))
            .filter(Document.workspace_id == ctx.workspace_id)
            .scalar()
            or 0
        )
        l4["connected_databases"] = db_count
        l4["documents"] = doc_count
        l4["responding"] = True
    except Exception as e:
        logger.warning("[layers] L4 knowledge check failed: %s", e, exc_info=True)
        health_issues.append("L4 knowledge query failed")
    l4["latency_ms"] = round((_time.monotonic() - l4_start) * 1000, 1)
    layers["L4_knowledge"] = l4

    # ------------------------------------------------------------------
    # Background Jobs Status
    # ------------------------------------------------------------------
    jobs_status: Dict[str, Any] = {"active": False, "jobs": {}}
    try:
        from services.memory_jobs import get_memory_job_scheduler

        jobs_status = get_memory_job_scheduler().get_status()
    except Exception as e:
        logger.debug("[layers] Job scheduler status unavailable: %s", e)

    # ------------------------------------------------------------------
    # Overall Health
    # ------------------------------------------------------------------
    responding_layers = sum(
        1
        for layer in layers.values()
        if layer.get("responding")
    )
    total_layers = len(layers)

    if responding_layers == total_layers:
        health_status = "healthy"
    elif responding_layers == 0:
        health_status = "critical"
    else:
        health_status = "degraded"

    result = {
        "health_status": health_status,
        "responding_layers": f"{responding_layers}/{total_layers}",
        "layers": layers,
        "jobs": jobs_status,
        "issues": health_issues if health_issues else None,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # ------------------------------------------------------------------
    # Cache the result
    # ------------------------------------------------------------------
    if service:
        try:
            redis_client = service._get_redis()
            if redis_client:
                import json as _json

                conn = redis_client.get_redis()
                conn.setex(cache_key, cache_ttl, _json.dumps(result, default=str))
        except Exception:
            pass  # caching is non-critical

    return result


class ConsolidateRequest(BaseModel):
    memory_ids: List[str]
    strategy: str = "merge"  # "merge" | "summarise"


@router.post("/consolidate")
async def consolidate_memories(
    body: ConsolidateRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Consolidate multiple memories into one — PRD-77 Phase 3.
    'merge' concatenates content; 'summarise' uses LLM to produce a summary.
    Deletes originals after creating the merged entry.
    """
    if len(body.memory_ids) < 2:
        return {"success": False, "error": "Need at least 2 memories to consolidate"}

    if body.strategy not in ("merge", "summarise"):
        return {"success": False, "error": "strategy must be 'merge' or 'summarise'"}

    service = _get_memory_service()
    if not service:
        return {"success": False, "error": "Memory service unavailable"}

    # Fetch target memories across all tiers
    try:
        agent_ids = _get_agent_ids(ctx.workspace_id, db)
        scoped_items = await _fetch_all_scoped_memories(
            service, str(ctx.workspace_id), agent_ids, limit=500,
        )

        id_set = set(body.memory_ids)
        targets: List[Dict] = [
            m for _, m in scoped_items
            if str(m.get("id", "")) in id_set
        ]

        if len(targets) < 2:
            return {"success": False, "error": f"Found only {len(targets)} of {len(body.memory_ids)} memories"}
    except Exception as e:
        logger.error("Failed to fetch memories for consolidation: %s", e, exc_info=True)
        return {"success": False, "error": str(e)[:200]}

    # Build consolidated content
    contents = [
        m.get("memory") or m.get("content", "")
        for m in targets
    ]

    if body.strategy == "merge":
        merged_content = "\n\n".join(c for c in contents if c)
    else:
        # Summarise using LLM
        try:
            from config import config
            import openai

            client = openai.OpenAI(
                api_key=config.OPENROUTER_API_KEY,
                base_url=config.OPENROUTER_BASE_URL,
            )
            summary_resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Summarise the following memory entries into a single, concise memory. Preserve all key facts and preferences. Output only the consolidated memory text, no preamble."},
                    {"role": "user", "content": "\n---\n".join(contents)},
                ],
                max_tokens=500,
            )
            merged_content = summary_resp.choices[0].message.content or "\n\n".join(contents)
        except Exception as e:
            logger.warning("LLM summarisation failed, falling back to merge: %s", e)
            merged_content = "\n\n".join(c for c in contents if c)

    # Store the consolidated memory under global tier via UnifiedMemoryService
    try:
        await service.store_long_term(
            workspace_id=str(ctx.workspace_id),
            content=merged_content,
            metadata={"source": "consolidation", "merged_from": len(targets)},
        )
    except Exception as e:
        logger.error("Failed to store consolidated memory: %s", e, exc_info=True)
        return {"success": False, "error": f"Failed to store: {str(e)[:200]}"}

    # Delete originals
    deleted = 0
    for m in targets:
        mid = str(m.get("id", ""))
        if mid and await service.delete_memory(mid):
            deleted += 1

    logger.info(
        "Consolidated %d memories (strategy=%s, deleted=%d) for workspace %s",
        len(targets), body.strategy, deleted, ctx.workspace_id,
    )

    return {
        "success": True,
        "deleted_count": deleted,
        "strategy": body.strategy,
        "message": f"Consolidated {len(targets)} memories into 1 ({body.strategy})",
    }


def _truncate(text: str, max_len: int = 120) -> str:
    if not text:
        return ""
    return (text[:max_len] + "...") if len(text) > max_len else text

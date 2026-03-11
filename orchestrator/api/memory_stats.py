"""
Memory Stats API — Mem0-first with local DB fallback
=====================================================
Queries the Mem0 service (OpenMemory) for memory data.
Falls back to the local memory_items table if Mem0 is unavailable.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, text
from datetime import datetime, timedelta

from core.database.database import get_db
from modules.memory.storage.knowledge_system import MemoryItem
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/memory", tags=["Real Memory Stats"])


def _get_mem0_client():
    """Lazy-load a Mem0Client; returns None if unavailable."""
    try:
        from modules.memory.integrations.mem0_client import Mem0Client
        return Mem0Client()
    except Exception as e:
        logger.warning("Mem0 client not available: %s", e)
        return None


def _mem0_user_id(workspace_id) -> str:
    """Build the GLOBAL scoped user_id that SmartMemoryManager uses."""
    return f"ws_{workspace_id}"


def _all_mem0_user_ids(workspace_id, db: Session) -> List[str]:
    """
    Build ALL user_id variants that SmartMemoryManager may have stored under.

    SmartMemoryManager uses three tiers:
      - Global:  ws_{workspace_id}
      - Agent:   ws_{workspace_id}_agent_{agent_id}
      - Daily:   ws_{workspace_id}_daily
    """
    ws = str(workspace_id)
    user_ids = [f"ws_{ws}", f"ws_{ws}_daily"]

    # Get all agent IDs in this workspace
    try:
        from core.models.core import Agent
        agent_ids = db.query(Agent.id).filter(Agent.workspace_id == workspace_id).all()
        for (aid,) in agent_ids:
            user_ids.append(f"ws_{ws}_agent_{aid}")
    except Exception as e:
        logger.warning("Failed to fetch agent IDs for memory browse: %s", e)

    return user_ids


@router.get("/stats/real")
async def get_real_memory_stats(ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get memory statistics — Mem0 first, local DB fallback.
    """
    mem0 = _get_mem0_client()
    mem0_total = 0
    mem0_available = False

    # Try Mem0 for total count — query ALL tiers
    if mem0:
        try:
            user_ids = _all_mem0_user_ids(ctx.workspace_id, db)
            seen_ids: set = set()
            for uid in user_ids:
                items_raw = mem0.get_all(user_id=uid, limit=1000)
                if isinstance(items_raw, list):
                    for m in items_raw:
                        seen_ids.add(str(m.get("id", "")))
                elif isinstance(items_raw, dict):
                    for m in items_raw.get("items", []):
                        seen_ids.add(str(m.get("id", "")))
            mem0_total = len(seen_ids)
            mem0_available = True
        except Exception as e:
            logger.warning(f"Mem0 stats query failed, falling back to local DB: {e}")

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
    # Try Mem0 — query ALL tiers
    mem0 = _get_mem0_client()
    if mem0:
        try:
            user_ids = _all_mem0_user_ids(ctx.workspace_id, db)
            all_items: List[Dict[str, Any]] = []
            seen_ids: set = set()

            for uid in user_ids:
                results = mem0.get_all(user_id=uid, limit=limit)
                items = results if isinstance(results, list) else results.get("items", []) if isinstance(results, dict) else []
                for m in items:
                    mid = str(m.get("id", ""))
                    if mid and mid not in seen_ids:
                        seen_ids.add(mid)
                        tier = "global"
                        if "_agent_" in uid:
                            tier = "agent"
                        elif uid.endswith("_daily"):
                            tier = "daily"
                        all_items.append({
                            "id": mid,
                            "agent_id": None,
                            "memory_type": tier,
                            "memory_level": "long_term",
                            "content": _truncate(m.get("content") or m.get("memory") or "", 120),
                            "importance": m.get("score"),
                            "access_count": None,
                            "created_at": m.get("created_at"),
                            "source": "mem0",
                        })

            if all_items:
                # Sort by created_at descending
                all_items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
                return all_items[:limit]
        except Exception as e:
            logger.warning(f"Mem0 recent fetch failed, falling back to local: {e}")

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
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Browse/search all memories — PRD-77 Memory Explorer.
    Queries ALL memory tiers (global, per-agent, daily) for the workspace.
    If query is provided, performs vector similarity search.
    Otherwise returns all memories sorted by recency.
    """
    mem0 = _get_mem0_client()
    if not mem0:
        return {"success": False, "error": "Memory service unavailable", "memories": []}

    user_ids = _all_mem0_user_ids(ctx.workspace_id, db)

    try:
        all_results: List[Dict] = []
        seen_ids: set = set()

        logger.info("[browse] Querying %d user_ids for workspace %s", len(user_ids), ctx.workspace_id)

        for uid in user_ids:
            if query:
                results = mem0.search(query=query, user_id=uid, limit=limit)
            else:
                results = mem0.get_all(user_id=uid, limit=limit)

            logger.info("[browse] uid=%s type=%s len=%s", uid, type(results).__name__, len(results) if isinstance(results, (list, dict)) else "?")

            items = results if isinstance(results, list) else []
            for m in items:
                mid = str(m.get("id", ""))
                if mid and mid not in seen_ids:
                    seen_ids.add(mid)
                    # Tag which tier this came from
                    tier = "global"
                    if "_agent_" in uid:
                        tier = "agent"
                    elif uid.endswith("_daily"):
                        tier = "daily"
                    all_results.append({
                        "id": mid,
                        "content": m.get("memory") or m.get("content", ""),
                        "score": m.get("score"),
                        "metadata": m.get("metadata") or m.get("metadata_"),
                        "created_at": m.get("created_at"),
                        "updated_at": m.get("updated_at"),
                        "tier": tier,
                    })

        logger.info("[browse] Collected %d unique memories from %d user_ids", len(all_results), len(user_ids))

        # Sort by created_at descending (newest first), then truncate
        all_results.sort(
            key=lambda x: x.get("created_at") or "",
            reverse=True,
        )
        memories = all_results[:limit]

        if memories:
            logger.info("[browse] Returning %d memories, first: id=%s content=%s",
                        len(memories), memories[0].get("id"), str(memories[0].get("content", ""))[:60])
        else:
            logger.warning("[browse] Returning 0 memories despite querying %d user_ids", len(user_ids))

        return {
            "success": True,
            "memories": memories,
            "total": len(all_results),
            "source": "mem0",
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
    mem0 = _get_mem0_client()
    if not mem0:
        return {"success": False, "error": "Memory service unavailable"}

    user_ids = _all_mem0_user_ids(ctx.workspace_id, db)

    try:
        # Ownership check: verify memory belongs to this workspace (any tier)
        owned_ids: set = set()
        for uid in user_ids:
            items_raw = mem0.get_all(user_id=uid, limit=500)
            items = items_raw if isinstance(items_raw, list) else []
            for m in items:
                owned_ids.add(str(m.get("id", "")))

        if memory_id not in owned_ids:
            return {"success": False, "error": "Memory not found or not owned by this workspace"}

        deleted = mem0.delete(memory_id)
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
    mem0 = _get_mem0_client()
    user_ids = _all_mem0_user_ids(ctx.workspace_id, db)

    total = 0
    oldest_memory = None
    newest_memory = None
    mem0_available = False

    if mem0:
        try:
            all_dates: List[str] = []
            seen_ids: set = set()

            for uid in user_ids:
                items_raw = mem0.get_all(user_id=uid, limit=500)
                items = items_raw if isinstance(items_raw, list) else []
                for m in items:
                    mid = str(m.get("id", ""))
                    if mid and mid not in seen_ids:
                        seen_ids.add(mid)
                        dt = m.get("created_at")
                        if dt:
                            all_dates.append(dt)

            total = len(seen_ids)
            mem0_available = True

            if all_dates:
                oldest_memory = min(all_dates)
                newest_memory = max(all_dates)
        except Exception as e:
            logger.warning("Memory health check failed: %s", e)

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

    mem0 = _get_mem0_client()
    if not mem0:
        return {"success": False, "error": "Memory service unavailable"}

    # Fetch target memories across all tiers
    try:
        user_ids = _all_mem0_user_ids(ctx.workspace_id, db)

        id_set = set(body.memory_ids)
        targets: List[Dict] = []
        seen: set = set()

        for uid in user_ids:
            items_raw = mem0.get_all(user_id=uid, limit=500)
            items = items_raw if isinstance(items_raw, list) else []
            for m in items:
                mid = str(m.get("id", ""))
                if mid in id_set and mid not in seen:
                    seen.add(mid)
                    targets.append(m)

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

    # Store the consolidated memory under global tier
    try:
        global_user_id = _mem0_user_id(ctx.workspace_id)
        mem0.add(
            messages=[{"role": "system", "content": merged_content}],
            user_id=global_user_id,
            metadata={"source": "consolidation", "merged_from": len(targets)},
        )
    except Exception as e:
        logger.error("Failed to store consolidated memory: %s", e, exc_info=True)
        return {"success": False, "error": f"Failed to store: {str(e)[:200]}"}

    # Delete originals
    deleted = 0
    for m in targets:
        mid = str(m.get("id", ""))
        if mid and mem0.delete(mid):
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

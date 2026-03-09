"""
Memory Stats API — Mem0-first with local DB fallback
=====================================================
Queries the Mem0 service (OpenMemory) for memory data.
Falls back to the local memory_items table if Mem0 is unavailable.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
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
        logger.debug(f"Mem0 client not available: {e}")
        return None


def _mem0_user_id(workspace_id) -> str:
    """Build the scoped user_id that SmartMemoryManager uses."""
    return f"ws_{workspace_id}_agent_global"


@router.get("/stats/real")
async def get_real_memory_stats(ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get memory statistics — Mem0 first, local DB fallback.
    """
    mem0 = _get_mem0_client()
    mem0_total = 0
    mem0_available = False

    # Try Mem0 for total count
    if mem0:
        try:
            user_id = _mem0_user_id(ctx.workspace_id)
            all_memories = mem0.get_all(user_id=user_id, limit=1000)
            if isinstance(all_memories, list):
                mem0_total = len(all_memories)
                mem0_available = True
            elif isinstance(all_memories, dict):
                mem0_total = all_memories.get("total", len(all_memories.get("items", [])))
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
    # Try Mem0
    mem0 = _get_mem0_client()
    if mem0:
        try:
            user_id = _mem0_user_id(ctx.workspace_id)
            results = mem0.get_all(user_id=user_id, limit=limit)

            items = results if isinstance(results, list) else results.get("items", [])
            if items:
                return [
                    {
                        "id": str(m.get("id", "")),
                        "agent_id": None,
                        "memory_type": "mem0",
                        "memory_level": "long_term",
                        "content": _truncate(m.get("content") or m.get("memory") or "", 120),
                        "importance": m.get("score"),
                        "access_count": None,
                        "created_at": m.get("created_at"),
                        "source": "mem0",
                    }
                    for m in items[:limit]
                ]
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


def _truncate(text: str, max_len: int = 120) -> str:
    if not text:
        return ""
    return (text[:max_len] + "...") if len(text) > max_len else text

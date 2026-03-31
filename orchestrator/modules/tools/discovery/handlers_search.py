"""Search handlers for PlatformActionExecutor — chat history, memory search, browse/delete memories."""

import asyncio
import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def search_chat_history(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Search across all chat messages by keyword."""
    from sqlalchemy import text

    query = params.get("query", "").strip()
    if not query:
        return {"success": False, "error": "query parameter is required"}

    try:
        days = min(int(params.get("days", 30)), 365)
    except (TypeError, ValueError):
        days = 30
    try:
        limit = min(int(params.get("limit", 20)), 100)
    except (TypeError, ValueError):
        limit = 20
    search_term = f"%{query}%"

    try:
        rows = db.execute(
            text("""
                SELECT m.id, m.chat_id, m.role, m.parts, m.created_at,
                       c.title AS chat_title
                FROM messages m
                JOIN chats c ON c.id = m.chat_id
                JOIN workspace_members wm ON wm.user_id = c.user_id
                WHERE wm.workspace_id = :workspace_id
                  AND wm.is_active = true
                  AND m.created_at >= NOW() - make_interval(days => :days)
                  AND EXISTS (
                      SELECT 1 FROM jsonb_array_elements(m.parts) AS p
                      WHERE p->>'text' ILIKE :search
                  )
                ORDER BY m.created_at DESC
                LIMIT :lim
            """),
            {"workspace_id": str(workspace_id), "days": days, "search": search_term, "lim": limit},
        ).fetchall()

        results = []
        for r in rows:
            parts = r.parts if isinstance(r.parts, list) else []
            text_content = " ".join(
                p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")
            )
            results.append({
                "chat_title": r.chat_title,
                "role": r.role,
                "content": text_content[:300],
                "date": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else None,
                "chat_id": str(r.chat_id),
            })

        # Format for LLM
        lines = [f"Found {len(results)} message(s) matching '{query}':\n"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. [{r['date']}] ({r['role']}) in \"{r['chat_title']}\":\n"
                f"   {r['content']}\n"
            )

        return {
            "success": True,
            "query": query,
            "total": len(results),
            "results": results,
            "formatted": "\n".join(lines),
        }
    except Exception as exc:
        logger.error("[PlatformExecutor] Chat search failed: %s", exc, exc_info=True)
        return {"success": False, "error": f"Chat search failed: {exc}"}


async def search_memory(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Search Mem0 memories by query."""
    from modules.memory.unified_memory_service import get_unified_memory_service

    query = params.get("query", "").strip()
    if not query:
        return {"success": False, "error": "query parameter is required"}

    agent_id = params.get("agent_id")
    limit = min(params.get("limit", 10), 50)
    result_char_limit = 150

    try:
        service = get_unified_memory_service()
        if not service.is_mem0_configured:
            return {"success": False, "error": "Memory service not configured (MEM0_API_URL empty)"}

        ws_id = str(workspace_id)

        # Search global memories
        global_results = await service.search_long_term(
            workspace_id=ws_id, query=query, limit=limit,
        )

        # Search agent-specific if agent_id given, otherwise search all agents
        agent_results = []
        partial = False
        scanned_agents = 0
        total_agents = 0
        if agent_id:
            agent_results = await service.search_long_term(
                workspace_id=ws_id, query=query, agent_id=int(agent_id), limit=limit,
            )
            for m in agent_results:
                m["_tier"] = f"agent-{agent_id}"
            scanned_agents = 1
            total_agents = 1
        else:
            # Search top agents
            from core.models.core import Agent
            agents = (
                db.query(Agent.id)
                .filter(Agent.workspace_id == workspace_id)
                .limit(5)
                .all()
            )
            total_agents_query = (
                db.query(func.count(Agent.id))
                .filter(Agent.workspace_id == workspace_id)
                .scalar()
            ) or 0
            total_agents = int(total_agents_query)
            scanned_agents = len(agents)
            partial = total_agents > scanned_agents
            agent_tasks = [
                service.search_long_term(
                    workspace_id=ws_id, query=query, agent_id=aid, limit=5,
                )
                for (aid,) in agents
            ]
            agent_batches = await asyncio.gather(*agent_tasks) if agent_tasks else []
            for (aid,), res in zip(agents, agent_batches):
                for m in (res or []):
                    m["_tier"] = f"agent-{aid}"
                agent_results.extend(res or [])

        # Mark global
        for m in (global_results or []):
            m["_tier"] = "global"

        all_results = (global_results or []) + agent_results

        # Format
        lines = [f"Memory search for '{query}': {len(all_results)} result(s)\n"]
        for i, m in enumerate(all_results[:limit], 1):
            content = (m.get("memory") or m.get("content", "") or "")[:result_char_limit]
            tier = m.get("_tier", "unknown")
            created = m.get("created_at", "")
            lines.append(f"{i}. [{tier}] {content}")
            if created:
                lines.append(f"   Created: {created}")

        return {
            "success": True,
            "query": query,
            "total": len(all_results),
            "global_count": len(global_results or []),
            "agent_count": len(agent_results),
            "partial": partial,
            "scanned_agents": scanned_agents,
            "total_agents": total_agents,
            "results": [
                {
                    "memory": (m.get("memory") or m.get("content", "") or "")[:result_char_limit],
                    "tier": m.get("_tier", "unknown"),
                    "created_at": m.get("created_at"),
                }
                for m in all_results[:limit]
            ],
            "formatted": "\n".join(lines),
        }
    except Exception as e:
        logger.warning(f"[PlatformExecutor] Memory search failed: {e}", exc_info=True)
        return {"success": False, "error": f"Memory search error: {e}"}


async def browse_memories(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Browse/search memories via Mem0."""
    try:
        from modules.memory.unified_memory_service import get_unified_memory_service

        service = get_unified_memory_service()
        ws_id = str(workspace_id)
        limit = params.get("limit", 20)
        query = params.get("query")

        if query:
            results = await service.search_long_term(
                workspace_id=ws_id, query=query, limit=limit,
            )
        else:
            results = await service.get_all_memories(
                workspace_id=ws_id, limit=limit,
            )

        # Normalise to consistent format
        memories = []
        for m in results:
            if isinstance(m, dict):
                memories.append({
                    "id": m.get("id"),
                    "content": m.get("memory") or m.get("content", ""),
                    "score": m.get("score"),
                    "metadata": m.get("metadata") or m.get("metadata_"),
                    "created_at": m.get("created_at"),
                })

        return {
            "success": True,
            "memories": memories,
            "total": len(memories),
            "source": "mem0",
            "search_query": query,
        }
    except Exception as e:
        logger.error("[PlatformExecutor] browse_memories failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Memory service unavailable: {str(e)[:200]}"}


async def delete_memory(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Delete a memory by ID with workspace ownership check."""
    from modules.memory.unified_memory_service import get_unified_memory_service

    memory_id = params.get("memory_id")
    if not memory_id:
        return {"success": False, "error": "memory_id is required"}

    try:
        service = get_unified_memory_service()
        ws_id = str(workspace_id)

        # Ownership check -- verify memory belongs to this workspace
        all_mems = await service.get_all_memories(workspace_id=ws_id, limit=500)
        owned_ids = {str(m.get("id", "")) for m in (all_mems if isinstance(all_mems, list) else [])}
        if memory_id not in owned_ids:
            return {"success": False, "error": "Memory not found or not owned by this workspace"}

        deleted = await service.delete_memory(memory_id=memory_id)

        if deleted:
            return {"success": True, "message": f"Memory {memory_id} deleted"}
        return {"success": False, "error": f"Failed to delete memory {memory_id}"}
    except Exception as e:
        logger.error("[PlatformExecutor] delete_memory failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Memory service unavailable: {str(e)[:200]}"}

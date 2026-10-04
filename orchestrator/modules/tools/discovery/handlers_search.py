"""Search handlers for PlatformActionExecutor — chat history, memory search, browse/delete memories."""

import asyncio
import logging
from typing import Any, Dict, List, Tuple
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


# How much of each memory a search answer repeats; how many of the workspace's agents it reads.
MEMORY_CHARS = 150
AGENTS_SEARCHED = 5
AGENT_MEMORIES = 5


def _bounded(raw: Any, default: int, top: int) -> int:
    """A whole number up to ``top``; ``default`` for anything that isn't one."""
    try:
        return min(int(raw if raw is not None else default), top)
    except (TypeError, ValueError):
        return default


async def search_memory(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Search durable memories by query: the workspace's own, then its first agents'.

    F325 (night 9b): a memory that names an agent's report or output as where a fact
    came from is left out, and the answer says how many were
    (modules/memory/agent_sources)."""
    from modules.memory.agent_sources import agents_document_names, owners_memories
    from modules.memory.unified_memory_service import get_unified_memory_service

    query = params.get("query", "").strip()
    if not query:
        return {"success": False, "error": "query parameter is required"}
    limit = _bounded(params.get("limit"), 10, 50)
    try:
        service = get_unified_memory_service()
        if not service.is_durable_configured:
            return {"success": False, "error": "Memory service not configured (QDRANT_URL empty)"}
        found, scanned, total_agents = await _memories_in_tiers(db, service, workspace_id, query, limit)
        kept = owners_memories(found, agents_document_names(db, workspace_id))
        return _memory_search_answer(query, kept, limit, (scanned, total_agents), len(found) - len(kept))
    except Exception as e:
        logger.exception("[PlatformExecutor] Memory search failed")
        return {"success": False, "error": f"Memory search error: {e}"}


async def _memories_in_tiers(db: Session, service: Any, workspace_id: UUID, query: str,
                             limit: int) -> Tuple[List[Dict[str, Any]], int, int]:
    """The workspace's memories, then each of its first agents', each marked with its
    tier; and how many agents were read of how many there are. F189: no model-chosen
    agent_id; the search reads the workspace's own memories as it always could."""
    from core.models.core import Agent

    ws_id = str(workspace_id)
    global_results = await service.search_long_term(workspace_id=ws_id, query=query, limit=limit)
    agents = db.query(Agent.id).filter(Agent.workspace_id == workspace_id).limit(AGENTS_SEARCHED).all()
    total_agents = int(db.query(func.count(Agent.id)).filter(Agent.workspace_id == workspace_id).scalar() or 0)
    batches = await asyncio.gather(*[
        service.search_long_term(workspace_id=ws_id, query=query, agent_id=aid, limit=AGENT_MEMORIES)
        for (aid,) in agents
    ]) if agents else []
    found = [{**m, "_tier": "global"} for m in (global_results or []) if isinstance(m, dict)]
    for (aid,), batch in zip(agents, batches):
        found += [{**m, "_tier": f"agent-{aid}"} for m in (batch or []) if isinstance(m, dict)]
    return found, len(agents), total_agents


def _memory_search_answer(query: str, kept: List[Dict[str, Any]], limit: int, agents_read: Tuple[int, int],
                          left_out: int) -> Dict[str, Any]:
    """The search's answer: the memories kept, for the model and as text."""
    from modules.memory.agent_sources import left_out_note, memory_text

    shown = kept[:limit]
    lines = [f"Memory search for '{query}': {len(kept)} result(s)\n"]
    for i, m in enumerate(shown, 1):
        lines.append(f"{i}. [{m.get('_tier', 'unknown')}] {memory_text(m)[:MEMORY_CHARS]}")
        if m.get("created_at"):
            lines.append(f"   Created: {m.get('created_at')}")
    note = left_out_note(left_out)
    if note:
        lines.append(note["left_out_note"])
    scanned, total_agents = agents_read
    in_global = sum(1 for m in kept if m.get("_tier") == "global")
    return {
        "success": True, "query": query, "total": len(kept),
        "global_count": in_global, "agent_count": len(kept) - in_global,
        "partial": total_agents > scanned, "scanned_agents": scanned, "total_agents": total_agents,
        "results": [{"memory": memory_text(m)[:MEMORY_CHARS], "tier": m.get("_tier", "unknown"),
                     "created_at": m.get("created_at")} for m in shown],
        "formatted": "\n".join(lines),
        **note,
    }


async def browse_memories(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Browse/search durable memories. F325 (night 9b): one that names an agent's report
    or output as where a fact came from is left out, and the answer says how many were."""
    try:
        from modules.memory.agent_sources import agents_document_names, left_out_note, owners_memories
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
        found = [m for m in (results or []) if isinstance(m, dict)]
        kept = owners_memories(found, agents_document_names(db, workspace_id))

        # Normalise to consistent format
        memories = [{
            "id": m.get("id"),
            "content": m.get("memory") or m.get("content", ""),
            "score": m.get("score"),
            "metadata": m.get("metadata") or m.get("metadata_"),
            "created_at": m.get("created_at"),
        } for m in kept]

        return {
            "success": True,
            "memories": memories,
            "total": len(memories),
            "source": "durable_memory",
            "search_query": query,
            **left_out_note(len(found) - len(kept)),
        }
    except Exception as e:
        logger.exception("[PlatformExecutor] browse_memories failed")
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

        deleted = await service.delete_memory(memory_id=memory_id, workspace_id=ws_id)

        if deleted:
            return {"success": True, "message": f"Memory {memory_id} deleted"}
        return {"success": False, "error": f"Failed to delete memory {memory_id}"}
    except Exception as e:
        logger.error("[PlatformExecutor] delete_memory failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Memory service unavailable: {str(e)[:200]}"}

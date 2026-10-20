"""Search handlers for PlatformActionExecutor — memory search, browse/delete memories, and how a
chat history query becomes its words (the read is chat_history_read's)."""

import asyncio
import logging
from typing import Any, Dict, List, Tuple
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# P256-FIX-N15-4 (night 15, F425): platform_search_chat_history found nothing all night. It read the
# chats of the workspace's active members, joined through workspace_members, and the local edition's
# operator has no such row (core/seeds/seed_local_first_run.py writes none), so on local every search
# came back empty; and the chat's own workspace was never checked. It now reads the asking person's
# own chats in this workspace (the server's _driving_user_id, as GET /api/chat/search reads the
# signed-in user's), and finds a message holding every word of the query, in any order.
# P256-FIX-N17-11 (F443): the read itself, by words or by window, and its bounds are
# chat_history_read's; how a query becomes its words stays here, where the chat search began.
CHAT_SEARCH_TERMS = 8          # words of the query matched; the rest are ignored
_TERM_EDGES = " \"'“”‘’.,;:!?()[]{}"
_LIKE_SPECIALS = str.maketrans({"\\": "\\\\", "%": "\\%", "_": "\\_"})
NO_PERSON_CHATS = ("Chat history is searched for the person in the chat, and no person is behind this call: "
                   "nothing was searched.")


def chat_search_terms(query: str) -> List[str]:
    """The query's words (quotes and stops trimmed, LIKE wildcards escaped), at most ``CHAT_SEARCH_TERMS``."""
    words = [word.strip(_TERM_EDGES) for word in query.split()]
    return [word.translate(_LIKE_SPECIALS) for word in words if word][:CHAT_SEARCH_TERMS]


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


async def _asking_persons_rules(workspace_id: str, params: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """P256-FIX-N14-4: the standing rules of the person the call is made for (the server's
    ``_driving_user_id``), and {} or, when they could not be read, what the answer says of it."""
    from modules.context.sections.memory import owners_rule_rows
    from modules.memory.forget import subject_of

    try:
        return await owners_rule_rows(workspace_id, subject_of(params.get("_driving_user_id"))), {}
    except Exception:  # noqa: BLE001 — logged; the browse answers without them, and says so
        logger.exception("[N14-4] standing rules not read for platform_browse_memories")
        return [], {"standing_rules_unavailable": True}


async def browse_memories(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Browse/search durable memories. F325 (night 9b): one that names an agent's report
    or output as where a fact came from is left out, and the answer says how many were.
    P256-FIX-N14-4: with no query, the asking person's standing rules come first, each with
    the id platform_delete_memory takes (the same read the prompt's block makes)."""
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
        rules, unread = ([], {}) if query else await _asking_persons_rules(ws_id, params)
        rule_ids = {str(r.get("id")) for r in rules}

        # Normalise to consistent format
        memories = [{
            "id": m.get("id"),
            "content": m.get("memory") or m.get("content", ""),
            "score": m.get("score"),
            "metadata": m.get("metadata") or m.get("metadata_"),
            "created_at": m.get("created_at"),
            **({"standing_rule": True} if str(m.get("id")) in rule_ids else {}),
        } for m in [*rules, *(m for m in kept if str(m.get("id")) not in rule_ids)]]

        return {
            "success": True,
            "memories": memories,
            "total": len(memories),
            "source": "durable_memory",
            "search_query": query,
            **left_out_note(len(found) - len(kept)),
            **unread,
        }
    except Exception as e:
        logger.exception("[PlatformExecutor] browse_memories failed")
        return {"success": False, "error": f"Memory service unavailable: {str(e)[:200]}"}


async def delete_memory(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Delete a memory by the id platform_browse_memories (or the Memory page) shows, wherever in
    this workspace it was stored. P256-FIX-N14-4: the store reads the point's own namespace (the
    old check looked for the id among the first 500 rows and missed every newer one); a private
    memory is deleted only for its owner (the server's ``_driving_user_id``, never the model's)."""
    from modules.memory.forget import forget, subject_of

    memory_id = params.get("memory_id")
    if not memory_id:
        return {"success": False, "error": "memory_id is required"}
    try:
        return await forget(str(memory_id), workspace_id, subject_of(params.get("_driving_user_id")))
    except Exception as e:
        logger.error("[PlatformExecutor] delete_memory failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Memory service unavailable: {str(e)[:200]}"}

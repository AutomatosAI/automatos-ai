"""PRD-206 S3 — "Where did we leave off?"

One resume payload, consumed by these surfaces:
  - ``GET /api/memory/resume`` (workspace router, ``api/widget_memory.py``),
  - ``GET /api/memory/end-users/{id}/resume`` (PRD-274 W3, §5.7): an end user's own, for the product
    that serves them (``build_end_user_resume_payload``),
  - the ``platform_resume_context`` tool (so the question works in ANY chat),
  - the chat/widget resume buttons (which just ask the question).

Shape: ``{threads, recent_decisions, open_loops, suggested_next_steps,
projects}``. ``threads`` are the VIEWER'S recent chats (title + the S2
checkpoint summary); decisions/open loops come from the typed L3 memories
(S1 contract) with the Q7 private-scope rule applied; next steps are lifted
from the thread checkpoints. ``projects`` is present-but-empty until S4
(Phase 2) fills it — the payload shape is stable from day one.

Assembly is pure; the two loaders are thin I/O.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from modules.memory.injection_filter import visible_to_viewer

logger = logging.getLogger(__name__)

DEFAULT_THREAD_LIMIT = 5
DEFAULT_ITEM_LIMIT = 8
# How many typed memories to scan for decisions/open loops. get_all is scroll-
# ordered, so we over-fetch and sort by created_at ourselves.
MEMORY_SCAN_LIMIT = 200


# ---------------------------------------------------------------------------
# Pure assembly
# ---------------------------------------------------------------------------

def _memory_item(mem: Dict[str, Any]) -> Dict[str, Any]:
    meta = mem.get("metadata") if isinstance(mem.get("metadata"), dict) else {}
    return {
        "id": mem.get("id"),
        "text": mem.get("memory") or mem.get("content") or "",
        "created_at": mem.get("created_at"),
        "chat_id": meta.get("chat_id"),
        "importance": meta.get("importance"),
        "scope": meta.get("scope"),
    }


def _typed_items(
    memories: List[Dict[str, Any]],
    fact_type: str,
    viewer: Optional[str],
    limit: int,
) -> List[Dict[str, Any]]:
    matched = []
    for mem in memories:
        if not isinstance(mem, dict):
            continue
        meta = mem.get("metadata") if isinstance(mem.get("metadata"), dict) else {}
        if (meta.get("type") or meta.get("category")) != fact_type:
            continue
        if not visible_to_viewer(mem, viewer):
            continue
        matched.append(_memory_item(mem))
    matched.sort(key=lambda m: m.get("created_at") or "", reverse=True)
    return matched[:limit]


def assemble_resume_payload(
    threads: List[Dict[str, Any]],
    memories: List[Dict[str, Any]],
    *,
    viewer: Optional[str],
    limit_items: int = DEFAULT_ITEM_LIMIT,
) -> Dict[str, Any]:
    """The resume payload — pure, deterministic, viewer-scoped."""
    next_steps: List[str] = []
    for thread in threads:
        summary = thread.get("summary") or {}
        step = summary.get("next_step") if isinstance(summary, dict) else None
        if step and step not in next_steps:
            next_steps.append(step)

    return {
        "threads": list(threads),
        "recent_decisions": _typed_items(memories, "decision", viewer, limit_items),
        "open_loops": _typed_items(memories, "open_loop", viewer, limit_items),
        "suggested_next_steps": next_steps[:limit_items],
        # S4 (Phase 2) fills this from project_memories; shape stable now.
        "projects": [],
    }


def format_resume_for_llm(payload: Dict[str, Any]) -> str:
    """Compact text rendering for the tool result (the LLM answers from it)."""
    lines: List[str] = []
    threads = payload.get("threads") or []
    if threads:
        lines.append("Recent threads:")
        for t in threads:
            summary = t.get("summary") or {}
            topic = summary.get("topic") if isinstance(summary, dict) else None
            last = summary.get("last_summary") if isinstance(summary, dict) else None
            line = f"- {t.get('title') or 'Untitled'} (chat {t.get('chat_id')}"
            line += f", updated {t.get('updated_at')})"
            if topic or last:
                line += f" — {topic or ''}{': ' if topic and last else ''}{last or ''}"
            lines.append(line)
    decisions = payload.get("recent_decisions") or []
    if decisions:
        lines.append("Recent decisions:")
        lines.extend(f"- {d['text']}" for d in decisions)
    loops = payload.get("open_loops") or []
    if loops:
        lines.append("Open loops:")
        lines.extend(f"- {l['text']}" for l in loops)
    steps = payload.get("suggested_next_steps") or []
    if steps:
        lines.append("Suggested next steps:")
        lines.extend(f"- {s}" for s in steps)
    if not lines:
        lines.append(
            "No resume context yet — no checkpointed threads, decisions or "
            "open loops on record for this user/workspace."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_recent_threads(
    db, workspace_id: str, viewer_user_id: Optional[int], limit: int
) -> List[Dict[str, Any]]:
    """The viewer's recent chats, newest activity first. No viewer → no
    threads (another user's conversations are never listed; decisions and
    open loops still resume workspace-wide)."""
    if viewer_user_id is None:
        return []
    from core.models.core import Chat

    rows = (
        db.query(Chat)
        .filter(
            Chat.workspace_id == workspace_id,
            Chat.user_id == viewer_user_id,
            Chat.kind == "user",
        )
        .order_by(Chat.updated_at.desc())
        .limit(limit)
        .all()
    )
    return [_thread(r) for r in rows]


def _thread(row: Any) -> Dict[str, Any]:
    return {
        "chat_id": str(row.id),
        "title": row.title,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "summary": row.summary if isinstance(row.summary, dict) else None,
    }


def _load_end_user_threads(db, workspace_id: str, end_user_id: str, limit: int) -> List[Dict[str, Any]]:
    """The end user's own recent conversations (``chats.end_user_id``), newest activity first."""
    from core.models.core import Chat

    rows = (db.query(Chat).filter(Chat.workspace_id == workspace_id, Chat.end_user_id == end_user_id)
            .order_by(Chat.updated_at.desc()).limit(limit).all())
    return [_thread(r) for r in rows]


async def _load_typed_memories(workspace_id: str) -> List[Dict[str, Any]]:
    try:
        from modules.memory.unified_memory_service import get_unified_memory_service

        service = get_unified_memory_service()
        if not service.is_durable_configured:
            return []
        return await service.get_all_memories(
            workspace_id=workspace_id, limit=MEMORY_SCAN_LIMIT
        ) or []
    except Exception:
        logger.warning("[Resume] typed-memory load failed", exc_info=True)
        return []


async def _load_namespace_memories(workspace_id: str, namespace: str) -> List[Dict[str, Any]]:
    """Every memory of one namespace of the workspace (an end user's), up to the scan limit."""
    try:
        from modules.memory.unified_memory_service import get_unified_memory_service

        service = get_unified_memory_service()
        if not service.is_durable_configured:
            return []
        return await service.get_all_memories_scoped(
            user_id=namespace, limit=MEMORY_SCAN_LIMIT, workspace_id=workspace_id
        ) or []
    except Exception:
        logger.warning("[Resume] end-user memory load failed", exc_info=True)
        return []


async def build_end_user_resume_payload(
    db,
    *,
    workspace_id: str,
    end_user_id: str,
    readable: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]] = lambda rows: rows,
    limit_threads: int = DEFAULT_THREAD_LIMIT,
    limit_items: int = DEFAULT_ITEM_LIMIT,
) -> Dict[str, Any]:
    """Where an end user left off (PRD-274 W3, §5.7): their own recent conversations, and the
    decisions and open loops in their facts and the workspace's, read as them (``end_user:{id}``),
    never another end user's. ``readable`` narrows the workspace's (what the caller may read)."""
    from modules.memory.unified_memory_service import MemoryNamespace

    threads = await asyncio.to_thread(_load_end_user_threads, db, workspace_id, end_user_id, limit_threads)
    theirs = await _load_namespace_memories(workspace_id, MemoryNamespace(workspace_id).end_user(end_user_id))
    workspace = readable(await _load_typed_memories(workspace_id))
    return assemble_resume_payload(threads, theirs + workspace, viewer=f"end_user:{end_user_id}",
                                   limit_items=limit_items)


async def build_resume_payload(
    db,
    *,
    workspace_id: str,
    viewer_user_id: Optional[int] = None,
    limit_threads: int = DEFAULT_THREAD_LIMIT,
    limit_items: int = DEFAULT_ITEM_LIMIT,
) -> Dict[str, Any]:
    threads = _load_recent_threads(db, workspace_id, viewer_user_id, limit_threads)
    memories = await _load_typed_memories(workspace_id)
    viewer = f"user:{viewer_user_id}" if viewer_user_id is not None else None
    return assemble_resume_payload(
        threads, memories, viewer=viewer, limit_items=limit_items
    )

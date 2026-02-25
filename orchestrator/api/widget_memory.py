"""
Widget Memory API
=================

REST endpoints for the Widget-layer memory panel (US-013).
Provides simple CRUD + search for workspace-scoped memories backed by the
Mem0 integration (when available) with graceful in-memory fallback.

Prefix: /api/memory
"""

import logging
import uuid
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory", tags=["Widget Memory"])

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class MemoryCreate(BaseModel):
    """Body for POST /api/memory"""
    content: str = Field(..., min_length=1, max_length=10000, description="Memory text content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Arbitrary metadata")
    tags: Optional[List[str]] = Field(default=None, description="Tags for categorisation")


class MemoryItem(BaseModel):
    """Single memory record returned by the API"""
    id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    workspace_id: str
    created_at: str
    score: Optional[float] = None


class MemoryListResponse(BaseModel):
    """Response for GET /api/memory"""
    memories: List[MemoryItem]
    total: int


class MemorySearchResponse(BaseModel):
    """Response for GET /api/memory/search"""
    query: str
    results: List[MemoryItem]
    total: int


class MemoryDeleteResponse(BaseModel):
    """Response for DELETE /api/memory/:id"""
    id: str
    deleted: bool


# ---------------------------------------------------------------------------
# Mem0 client helper (lazy, optional)
# ---------------------------------------------------------------------------

_mem0_client: Any = None
_mem0_checked: bool = False


def _get_mem0_client() -> Any:
    """Return a Mem0Client or None if unavailable."""
    global _mem0_client, _mem0_checked
    if _mem0_checked:
        return _mem0_client
    _mem0_checked = True
    try:
        from modules.memory.integrations.mem0_client import Mem0Client
        _mem0_client = Mem0Client()
        logger.info("[widget_memory] Mem0Client initialised")
    except Exception as exc:
        logger.warning("[widget_memory] Mem0 unavailable, using in-memory fallback: %s", exc)
        _mem0_client = None
    return _mem0_client


# ---------------------------------------------------------------------------
# In-memory fallback store (keyed by workspace_id)
# ---------------------------------------------------------------------------
_fallback_store: Dict[str, List[Dict[str, Any]]] = {}


def _ws_key(workspace_id: Any) -> str:
    return str(workspace_id)


def _fallback_list(workspace_id: str) -> List[Dict[str, Any]]:
    return _fallback_store.get(workspace_id, [])


def _fallback_add(workspace_id: str, memory_id: str, content: str,
                  metadata: Optional[Dict[str, Any]], tags: Optional[List[str]]) -> Dict[str, Any]:
    record = {
        "id": memory_id,
        "content": content,
        "metadata": metadata or {},
        "tags": tags or [],
        "workspace_id": workspace_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _fallback_store.setdefault(workspace_id, []).append(record)
    return record


def _fallback_search(workspace_id: str, query: str) -> List[Dict[str, Any]]:
    """Naive substring search in the fallback store."""
    q_lower = query.lower()
    results = []
    for mem in _fallback_store.get(workspace_id, []):
        text = (mem.get("content") or "").lower()
        tag_text = " ".join(mem.get("tags") or []).lower()
        if q_lower in text or q_lower in tag_text:
            results.append({**mem, "score": 0.9})
    return results


def _fallback_delete(workspace_id: str, memory_id: str) -> bool:
    items = _fallback_store.get(workspace_id, [])
    for i, mem in enumerate(items):
        if mem["id"] == memory_id:
            items.pop(i)
            return True
    return False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=MemoryListResponse)
async def list_memories(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    limit: int = Query(50, ge=1, le=200, description="Max items to return"),
) -> MemoryListResponse:
    """List memories for the current workspace."""
    ws = _ws_key(ctx.workspace_id)
    client = _get_mem0_client()

    if client is not None:
        try:
            raw = client.get_all(user_id=ws, limit=limit)
            items = [
                MemoryItem(
                    id=m.get("id", str(uuid.uuid4())),
                    content=m.get("memory") or m.get("content") or "",
                    metadata=m.get("metadata"),
                    tags=m.get("metadata", {}).get("tags") if isinstance(m.get("metadata"), dict) else None,
                    workspace_id=ws,
                    created_at=m.get("created_at", datetime.now(timezone.utc).isoformat()),
                )
                for m in raw
            ]
            return MemoryListResponse(memories=items, total=len(items))
        except Exception as exc:
            logger.warning("[widget_memory] Mem0 list failed, falling back: %s", exc)

    # Fallback
    raw_items = _fallback_list(ws)[:limit]
    items = [MemoryItem(**m) for m in raw_items]
    return MemoryListResponse(memories=items, total=len(items))


@router.get("/search", response_model=MemorySearchResponse)
async def search_memories(
    q: str = Query(..., min_length=1, description="Search query"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
) -> MemorySearchResponse:
    """Search memories by query within the current workspace."""
    ws = _ws_key(ctx.workspace_id)
    client = _get_mem0_client()

    if client is not None:
        try:
            raw = client.search(query=q, user_id=ws, limit=limit)
            items = [
                MemoryItem(
                    id=m.get("id", str(uuid.uuid4())),
                    content=m.get("memory") or m.get("content") or "",
                    metadata=m.get("metadata"),
                    tags=m.get("metadata", {}).get("tags") if isinstance(m.get("metadata"), dict) else None,
                    workspace_id=ws,
                    created_at=m.get("created_at", datetime.now(timezone.utc).isoformat()),
                    score=m.get("score"),
                )
                for m in raw
            ]
            return MemorySearchResponse(query=q, results=items, total=len(items))
        except Exception as exc:
            logger.warning("[widget_memory] Mem0 search failed, falling back: %s", exc)

    # Fallback
    raw_items = _fallback_search(ws, q)[:limit]
    items = [MemoryItem(**m) for m in raw_items]
    return MemorySearchResponse(query=q, results=items, total=len(items))


@router.post("", response_model=MemoryItem, status_code=201)
async def store_memory(
    body: MemoryCreate,
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> MemoryItem:
    """Store a new memory in the current workspace."""
    ws = _ws_key(ctx.workspace_id)
    memory_id = str(uuid.uuid4())
    client = _get_mem0_client()

    if client is not None:
        try:
            messages = [{"role": "user", "content": body.content}]
            mem0_meta = body.metadata or {}
            if body.tags:
                mem0_meta["tags"] = body.tags
            result = client.add(messages=messages, user_id=ws, metadata=mem0_meta)
            # Attempt to pull the stored ID from Mem0 response
            if isinstance(result, dict):
                memory_id = result.get("id") or result.get("memory_id") or memory_id
            return MemoryItem(
                id=memory_id,
                content=body.content,
                metadata=body.metadata,
                tags=body.tags,
                workspace_id=ws,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as exc:
            logger.warning("[widget_memory] Mem0 store failed, falling back: %s", exc)

    # Fallback
    record = _fallback_add(ws, memory_id, body.content, body.metadata, body.tags)
    return MemoryItem(**record)


@router.delete("/{memory_id}", response_model=MemoryDeleteResponse)
async def delete_memory(
    memory_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> MemoryDeleteResponse:
    """Delete a memory by ID within the current workspace."""
    ws = _ws_key(ctx.workspace_id)
    client = _get_mem0_client()

    if client is not None:
        try:
            deleted = client.delete(memory_id=memory_id)
            return MemoryDeleteResponse(id=memory_id, deleted=deleted)
        except Exception as exc:
            logger.warning("[widget_memory] Mem0 delete failed, falling back: %s", exc)

    # Fallback
    deleted = _fallback_delete(ws, memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    return MemoryDeleteResponse(id=memory_id, deleted=True)

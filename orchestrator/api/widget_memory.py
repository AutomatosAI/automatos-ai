"""
Widget Memory API
=================

REST endpoints for the Widget-layer memory panel (US-013).
Provides simple CRUD + search for workspace-scoped memories backed by the
durable memory store (UnifiedMemoryService).

When the store is not configured or a call fails, the route says so with a 503
and a code (``memory_off``, ``memory_store_failed``, ``memory_unavailable``).
It never answers from an in-process copy: an earlier fallback kept "stored"
memories in a per-process dict, lost on restart and invisible to other workers,
and answered 201 for them (#1430, Wave 0).

Prefix: /api/memory
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from modules.memory.injection_filter import without_server_keys

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory", tags=["Widget Memory"])

# Type alias for forward reference
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modules.memory.unified_memory_service import UnifiedMemoryService

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


class ResumeThread(BaseModel):
    """One recent thread in the resume payload (PRD-206 S3)."""
    chat_id: str
    title: str
    updated_at: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None


class ResumeItem(BaseModel):
    """One typed memory (decision / open loop) in the resume payload."""
    id: Optional[str] = None
    text: str
    created_at: Optional[str] = None
    chat_id: Optional[str] = None
    importance: Optional[float] = None
    scope: Optional[str] = None


class ResumeResponse(BaseModel):
    """Response for GET /api/memory/resume — 'where did we leave off?'"""
    threads: List[ResumeThread]
    recent_decisions: List[ResumeItem]
    open_loops: List[ResumeItem]
    suggested_next_steps: List[str]
    projects: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# UnifiedMemoryService helper (lazy, optional)
# ---------------------------------------------------------------------------

_memory_service: Optional[Any] = None
_memory_service_checked: bool = False


def _get_memory_service() -> Optional["UnifiedMemoryService"]:
    """Return the UnifiedMemoryService singleton or None if unavailable."""
    global _memory_service, _memory_service_checked
    if _memory_service_checked:
        return _memory_service
    _memory_service_checked = True
    try:
        from modules.memory.unified_memory_service import get_unified_memory_service
        svc = get_unified_memory_service()
        if svc.is_durable_configured:
            _memory_service = svc
            logger.info("[widget_memory] Using UnifiedMemoryService")
        else:
            logger.warning("[widget_memory] Durable memory is not configured; /api/memory answers memory_off")
    except Exception:
        logger.exception("[widget_memory] UnifiedMemoryService unavailable; /api/memory answers memory_off")
        _memory_service = None
    return _memory_service


# ---------------------------------------------------------------------------
# Failures: a code the caller can act on, a message that leaks nothing
# ---------------------------------------------------------------------------

MEMORY_OFF = "memory_off"
MEMORY_STORE_FAILED = "memory_store_failed"
MEMORY_UNAVAILABLE = "memory_unavailable"

_MESSAGES = {
    MEMORY_OFF: "Long-term memory is not available on this install.",
    MEMORY_STORE_FAILED: "The memory was not stored. Try again in a moment.",
    MEMORY_UNAVAILABLE: "Memory could not be read just now. Try again in a moment.",
}


def _unavailable(code: str) -> HTTPException:
    """A 503 whose detail carries the code and a plain message, never the store's error text."""
    return HTTPException(status_code=503, detail={"code": code, "message": _MESSAGES[code]})


def _service_or_503() -> "UnifiedMemoryService":
    """The memory service, or ``memory_off`` when this install has no durable memory."""
    service = _get_memory_service()
    if service is None:
        raise _unavailable(MEMORY_OFF)
    return service


def _ws_key(workspace_id: Any) -> str:
    return str(workspace_id)


def _item(m: Dict[str, Any], ws: str, with_score: bool = False) -> MemoryItem:
    """One stored memory as the API returns it."""
    metadata = m.get("metadata")
    return MemoryItem(
        id=m.get("id", str(uuid.uuid4())),
        content=m.get("memory") or m.get("content") or "",
        metadata=metadata,
        tags=metadata.get("tags") if isinstance(metadata, dict) else None,
        workspace_id=ws,
        created_at=m.get("created_at", datetime.now(timezone.utc).isoformat()),
        score=m.get("score") if with_score else None,
    )


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
    service = _service_or_503()
    try:
        raw = await service.get_all_memories(workspace_id=ws, limit=limit)
    except Exception as exc:
        logger.exception("[widget_memory] Memory list failed for workspace %s", ws)
        raise _unavailable(MEMORY_UNAVAILABLE) from exc
    items = [_item(m, ws) for m in raw]
    return MemoryListResponse(memories=items, total=len(items))


def _internal_user_id(db: Session, ctx: RequestContext) -> Optional[int]:
    """Resolve the authenticated principal to the INTEGER ``users.id``.

    ``ctx.user.id`` on the Clerk path is the Clerk STRING — never usable as
    ``users.id`` (the #513 lesson). An already-integer id (local edition)
    passes through; otherwise resolve via ``users.clerk_user_id``. None when
    unresolvable — the resume payload then simply carries no threads.
    """
    user = getattr(ctx, "user", None)
    if user is None:
        return None
    uid = getattr(user, "id", None)
    if isinstance(uid, int):
        return uid
    clerk = getattr(user, "clerk_user_id", None) or (uid if isinstance(uid, str) else None)
    if not clerk:
        return None
    try:
        row = db.execute(
            text("SELECT id FROM users WHERE clerk_user_id = :cid LIMIT 1"),
            {"cid": clerk},
        ).fetchone()
    except Exception:
        logger.warning("[widget_memory] internal user resolution failed", exc_info=True)
        return None
    return int(row[0]) if row else None


@router.get("/resume", response_model=ResumeResponse)
async def resume_context(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
    limit_threads: int = Query(5, ge=1, le=20, description="Max recent threads"),
    limit_items: int = Query(8, ge=1, le=50, description="Max decisions/open loops"),
) -> ResumeResponse:
    """PRD-206 S3: 'Where did we leave off?' — the viewer's recent threads
    (with their S2 checkpoint summaries), recent decisions, open loops and
    suggested next steps. Private-scoped memories only surface for their
    owner (Q7)."""
    from modules.memory.resume_context import build_resume_payload

    payload = await build_resume_payload(
        db,
        workspace_id=_ws_key(ctx.workspace_id),
        viewer_user_id=_internal_user_id(db, ctx),
        limit_threads=limit_threads,
        limit_items=limit_items,
    )
    return ResumeResponse(**payload)


@router.get("/search", response_model=MemorySearchResponse)
async def search_memories(
    q: str = Query(..., min_length=1, description="Search query"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
) -> MemorySearchResponse:
    """Search memories by query within the current workspace."""
    ws = _ws_key(ctx.workspace_id)
    service = _service_or_503()
    try:
        raw = await service.search_long_term(workspace_id=ws, query=q, limit=limit)
    except Exception as exc:
        logger.exception("[widget_memory] Memory search failed for workspace %s", ws)
        raise _unavailable(MEMORY_UNAVAILABLE) from exc
    items = [_item(m, ws, with_score=True) for m in raw]
    return MemorySearchResponse(query=q, results=items, total=len(items))


@router.post("", response_model=MemoryItem, status_code=201, dependencies=[Depends(require_workspace_permission("knowledge:create"))])
async def store_memory(
    body: MemoryCreate,
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> MemoryItem:
    """Store a new memory in the current workspace.

    201 only when the store says it kept the memory, with the id it kept it under
    (a restated memory returns the existing id). A refused store is a 503:
    ``memory_off`` when durable memory is disabled, else ``memory_store_failed``.
    """
    ws = _ws_key(ctx.workspace_id)
    service = _service_or_503()
    metadata = without_server_keys(body.metadata)  # P256-FIX-N14-4: owner and stated_by are the server's
    if body.tags:
        metadata["tags"] = body.tags
    try:
        result = await service.store_long_term(workspace_id=ws, content=body.content, metadata=metadata or None)
    except Exception as exc:
        logger.exception("[widget_memory] Memory store raised for workspace %s", ws)
        raise _unavailable(MEMORY_STORE_FAILED) from exc
    stored = result if isinstance(result, dict) else {}
    if not (stored.get("success") and stored.get("id")):
        # The store's own reason goes to the log, never to the caller.
        logger.error("[widget_memory] Memory store refused for workspace %s: %s", ws, stored.get("error"))
        raise _unavailable(MEMORY_OFF if stored.get("skipped") else MEMORY_STORE_FAILED)
    return MemoryItem(
        id=str(stored["id"]),
        content=body.content,
        metadata=body.metadata,
        tags=body.tags,
        workspace_id=ws,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


@router.delete("/{memory_id}", response_model=MemoryDeleteResponse, dependencies=[Depends(require_workspace_permission("knowledge:delete"))])
async def delete_memory(
    memory_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> MemoryDeleteResponse:
    """Delete a memory by ID within the current workspace."""
    ws = _ws_key(ctx.workspace_id)
    service = _service_or_503()
    try:
        # PRD-156 S5: scope the delete to the caller's workspace so a memory
        # id from another workspace can't be deleted (the service enforces
        # ownership via workspace_id; previously omitted = cross-tenant delete).
        deleted = await service.delete_memory(memory_id=memory_id, workspace_id=ws)
    except Exception as exc:
        logger.exception("[widget_memory] Memory delete failed for workspace %s", ws)
        raise _unavailable(MEMORY_UNAVAILABLE) from exc
    return MemoryDeleteResponse(id=memory_id, deleted=deleted)

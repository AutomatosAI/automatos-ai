"""
Admin Workspaces API (PRD-131 follow-up)
=========================================

Admin console for workspace lifecycle management:
- List all workspaces (owner, plan, counts, storage)
- Workspace detail (with counts + breakdown)
- Pause a workspace (non-payment / abuse review)
- Resume a paused workspace
- Soft-delete a workspace (GDPR / demo cleanup)

Hard-delete with S3 cascade is deferred to a background worker — this module
only sets `deleted_at` and returns. Soft-deleted workspaces are excluded from
the normal list view (use `?include_deleted=true` to see them).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.core import Agent, Chat, Document, Message, User
from core.models.workspaces import Workspace
from services.workspace_purge import purge_workspace_sync

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/workspaces", tags=["Admin Workspaces"])


# ===================================================================
# Helpers
# ===================================================================

def _is_admin(ctx: RequestContext) -> bool:
    if not ctx.user:
        return False
    return getattr(ctx.user, "system_role", "user") == "admin"


def _assert_admin(ctx: RequestContext) -> None:
    if not _is_admin(ctx):
        raise HTTPException(status_code=403, detail="Admin access required")


# ===================================================================
# Pydantic Models
# ===================================================================

class WorkspaceListItem(BaseModel):
    id: str
    name: str
    slug: Optional[str] = None
    plan: Optional[str] = None
    is_personal: bool = False
    is_active: bool = True
    owner_email: Optional[str] = None
    owner_name: Optional[str] = None
    agents_count: int = 0
    documents_count: int = 0
    storage_bytes: int = 0
    chats_count: int = 0
    created_at: Optional[str] = None
    paused_at: Optional[str] = None
    paused_reason: Optional[str] = None
    deleted_at: Optional[str] = None


class WorkspaceDetail(WorkspaceListItem):
    clerk_org_id: Optional[str] = None
    messages_count: int = 0
    updated_at: Optional[str] = None


class PauseBody(BaseModel):
    reason: str = Field(..., min_length=1, max_length=500)


# ===================================================================
# Endpoints
# ===================================================================

@router.get("")
async def list_workspaces(
    include_deleted: bool = Query(False),
    include_paused: bool = Query(True),
    plan: Optional[str] = Query(None),
    search: Optional[str] = Query(None, description="Match against name or slug"),
    sort: str = Query("created_at", pattern="^(created_at|name|storage_bytes|agents_count)$"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List workspaces with counts + storage. Admin-only."""
    _assert_admin(ctx)

    try:
        q = db.query(Workspace)

        if not include_deleted:
            q = q.filter(Workspace.deleted_at.is_(None))
        if not include_paused:
            q = q.filter(Workspace.paused_at.is_(None))
        if plan:
            q = q.filter(Workspace.plan == plan)
        if search:
            needle = f"%{search.lower()}%"
            q = q.filter(
                (func.lower(Workspace.name).like(needle))
                | (func.lower(Workspace.slug).like(needle))
            )

        total = q.count()

        # Sort on DB columns only (counts are computed post-query)
        if sort in ("created_at", "name"):
            col = getattr(Workspace, sort)
            q = q.order_by(desc(col) if order == "desc" else col)
        else:
            q = q.order_by(desc(Workspace.created_at))

        workspaces = q.offset((page - 1) * limit).limit(limit).all()
        workspace_ids = [w.id for w in workspaces]

        # Batched count queries (one per metric, not N+1)
        agents_by_ws = dict(
            db.query(Agent.workspace_id, func.count(Agent.id))
            .filter(Agent.workspace_id.in_(workspace_ids))
            .group_by(Agent.workspace_id)
            .all()
        ) if workspace_ids else {}

        docs_agg = (
            db.query(
                Document.workspace_id,
                func.count(Document.id),
                func.coalesce(func.sum(Document.file_size), 0),
            )
            .filter(Document.workspace_id.in_(workspace_ids))
            .group_by(Document.workspace_id)
            .all()
        ) if workspace_ids else []
        docs_count_by_ws = {ws_id: cnt for ws_id, cnt, _ in docs_agg}
        storage_by_ws = {ws_id: int(size) for ws_id, _, size in docs_agg}

        chats_by_ws = dict(
            db.query(Chat.workspace_id, func.count(Chat.id))
            .filter(Chat.workspace_id.in_(workspace_ids))
            .group_by(Chat.workspace_id)
            .all()
        ) if workspace_ids else {}

        # Owner lookup (bulk)
        owner_ids = {w.owner_id for w in workspaces if w.owner_id is not None}
        owners = (
            db.query(User).filter(User.id.in_(owner_ids)).all() if owner_ids else []
        )
        owner_by_id = {u.id: u for u in owners}

        items: List[WorkspaceListItem] = []
        for w in workspaces:
            owner = owner_by_id.get(w.owner_id) if w.owner_id else None
            items.append(WorkspaceListItem(
                id=str(w.id),
                name=w.name,
                slug=w.slug,
                plan=w.plan,
                is_personal=bool(w.is_personal),
                is_active=bool(w.is_active),
                owner_email=owner.email if owner else None,
                owner_name=owner.name if owner else None,
                agents_count=agents_by_ws.get(w.id, 0),
                documents_count=docs_count_by_ws.get(w.id, 0),
                storage_bytes=storage_by_ws.get(w.id, 0),
                chats_count=chats_by_ws.get(w.id, 0),
                created_at=w.created_at.isoformat() if w.created_at else None,
                paused_at=w.paused_at.isoformat() if w.paused_at else None,
                paused_reason=w.paused_reason,
                deleted_at=w.deleted_at.isoformat() if w.deleted_at else None,
            ))

        # Optional client-side sorts (storage/agents) — applied post count join
        if sort in ("storage_bytes", "agents_count"):
            items.sort(
                key=lambda i: getattr(i, sort),
                reverse=(order == "desc"),
            )

        return {
            "items": [i.model_dump() for i in items],
            "total": total,
            "page": page,
            "limit": limit,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error listing workspaces: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list workspaces")


@router.get("/{workspace_id}", response_model=WorkspaceDetail)
async def get_workspace(
    workspace_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Workspace detail with full counts (incl. messages)."""
    _assert_admin(ctx)

    try:
        w = db.query(Workspace).filter(Workspace.id == workspace_id).first()
        if not w:
            raise HTTPException(status_code=404, detail="Workspace not found")

        owner = (
            db.query(User).filter(User.id == w.owner_id).first()
            if w.owner_id else None
        )

        agents_count = db.query(func.count(Agent.id)).filter(
            Agent.workspace_id == workspace_id
        ).scalar() or 0

        docs_count = db.query(func.count(Document.id)).filter(
            Document.workspace_id == workspace_id
        ).scalar() or 0

        storage_bytes = db.query(
            func.coalesce(func.sum(Document.file_size), 0)
        ).filter(Document.workspace_id == workspace_id).scalar() or 0

        chats_count = db.query(func.count(Chat.id)).filter(
            Chat.workspace_id == workspace_id
        ).scalar() or 0

        messages_count = db.query(func.count(Message.id)).filter(
            Message.workspace_id == workspace_id
        ).scalar() or 0

        return WorkspaceDetail(
            id=str(w.id),
            name=w.name,
            slug=w.slug,
            plan=w.plan,
            is_personal=bool(w.is_personal),
            is_active=bool(w.is_active),
            clerk_org_id=w.clerk_org_id,
            owner_email=owner.email if owner else None,
            owner_name=owner.name if owner else None,
            agents_count=agents_count,
            documents_count=docs_count,
            storage_bytes=int(storage_bytes),
            chats_count=chats_count,
            messages_count=messages_count,
            created_at=w.created_at.isoformat() if w.created_at else None,
            updated_at=w.updated_at.isoformat() if w.updated_at else None,
            paused_at=w.paused_at.isoformat() if w.paused_at else None,
            paused_reason=w.paused_reason,
            deleted_at=w.deleted_at.isoformat() if w.deleted_at else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error fetching workspace %s: %s", workspace_id, e)
        raise HTTPException(status_code=500, detail="Failed to fetch workspace")


@router.post("/{workspace_id}/pause")
async def pause_workspace(
    workspace_id: UUID,
    body: PauseBody,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Pause a workspace (non-payment, abuse review). Sets paused_at + paused_reason."""
    _assert_admin(ctx)

    w = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not w:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if w.deleted_at:
        raise HTTPException(status_code=400, detail="Workspace is deleted")
    if w.paused_at:
        raise HTTPException(status_code=400, detail="Workspace is already paused")

    w.paused_at = datetime.utcnow()
    w.paused_reason = body.reason
    w.is_active = False
    db.commit()

    logger.warning(
        "Workspace paused: id=%s name=%s by=%s reason=%s",
        workspace_id, w.name,
        getattr(ctx.user, "id", "?"), body.reason,
    )

    return {
        "success": True,
        "workspace_id": str(workspace_id),
        "paused_at": w.paused_at.isoformat(),
        "reason": w.paused_reason,
    }


@router.post("/{workspace_id}/resume")
async def resume_workspace(
    workspace_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Clear paused state and reactivate workspace."""
    _assert_admin(ctx)

    w = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not w:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if w.deleted_at:
        raise HTTPException(status_code=400, detail="Workspace is deleted")
    if not w.paused_at:
        raise HTTPException(status_code=400, detail="Workspace is not paused")

    w.paused_at = None
    w.paused_reason = None
    w.is_active = True
    db.commit()

    logger.info(
        "Workspace resumed: id=%s name=%s by=%s",
        workspace_id, w.name, getattr(ctx.user, "id", "?"),
    )

    return {"success": True, "workspace_id": str(workspace_id)}


@router.delete("/{workspace_id}")
async def delete_workspace(
    workspace_id: UUID,
    background_tasks: BackgroundTasks,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Delete a workspace.

    Marks `deleted_at` immediately so the workspace disappears from listings
    and access gates, then queues a background purge that wipes S3 objects,
    deletes the owner's Clerk user, cascades all workspace-scoped DB rows,
    and finally removes the `workspaces` row itself.

    The purge runs in `BackgroundTasks` (FastAPI thread) — fine for admin-
    triggered, low-volume deletions. See `services/workspace_purge.py`.
    """
    _assert_admin(ctx)

    w = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not w:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if w.deleted_at:
        raise HTTPException(status_code=400, detail="Workspace already deleted")

    w.deleted_at = datetime.utcnow()
    w.is_active = False
    db.commit()

    logger.warning(
        "Workspace soft-deleted (purge queued): id=%s name=%s by=%s",
        workspace_id, w.name, getattr(ctx.user, "id", "?"),
    )

    background_tasks.add_task(purge_workspace_sync, workspace_id)

    return {
        "success": True,
        "workspace_id": str(workspace_id),
        "deleted_at": w.deleted_at.isoformat(),
        "purge_queued": True,
    }


@router.post("/{workspace_id}/purge")
async def purge_workspace(
    workspace_id: UUID,
    background_tasks: BackgroundTasks,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Re-trigger hard-purge for an already-soft-deleted workspace.

    Used to clean up workspaces that were soft-deleted before the purge
    background task existed, or to retry a purge that previously errored.
    The workspace MUST already have `deleted_at` set — refuses otherwise.
    """
    _assert_admin(ctx)

    w = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not w:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if not w.deleted_at:
        raise HTTPException(
            status_code=400,
            detail="Workspace is not soft-deleted. Use DELETE first.",
        )

    logger.warning(
        "Workspace purge re-triggered: id=%s name=%s by=%s",
        workspace_id, w.name, getattr(ctx.user, "id", "?"),
    )

    background_tasks.add_task(purge_workspace_sync, workspace_id)
    return {"success": True, "workspace_id": str(workspace_id), "purge_queued": True}


@router.post("/{workspace_id}/restore")
async def restore_workspace(
    workspace_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Restore a soft-deleted workspace (undelete)."""
    _assert_admin(ctx)

    w = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not w:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if not w.deleted_at:
        raise HTTPException(status_code=400, detail="Workspace is not deleted")

    w.deleted_at = None
    w.is_active = True
    db.commit()

    logger.info(
        "Workspace restored: id=%s name=%s by=%s",
        workspace_id, w.name, getattr(ctx.user, "id", "?"),
    )

    return {"success": True, "workspace_id": str(workspace_id)}

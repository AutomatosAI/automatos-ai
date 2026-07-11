"""
Deliverables API (PRD-129: Workspace Outputs Hub)
==================================================

REST endpoints for the consumer-facing Gallery view of agent deliverables.

Routes
------
- GET    /api/deliverables                 List with filters + pagination
- GET    /api/deliverables/stats           Aggregate counts (by_type, by_agent)
- GET    /api/deliverables/{id}            Fetch one (optional ?include_content=)
- DELETE /api/deliverables/{id}            Soft delete

Auth / tenancy
--------------
Every request depends on ``get_request_context_hybrid`` which resolves the
caller's workspace (and validates header overrides via
``_user_has_workspace_access``). We then scope the service to
``ctx.workspace_id`` — never trust a client-supplied workspace id.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from services.deliverable_service import DeliverableService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/deliverables", tags=["deliverables"])


# ── List ─────────────────────────────────────────────────────────────
@router.get("")
async def list_deliverables(
    artifact_type: Optional[str] = Query(None, description="Filter by artifact_type (report, image, document, code, slide, spreadsheet, archive, audio, video)"),
    source_type: Optional[str] = Query(None, description="Filter by source (chat, task, mission, heartbeat, playbook, trigger, upload)"),
    source_type_exclude: Optional[str] = Query(None, description="Comma-separated source_types to exclude (e.g. 'heartbeat')"),
    source_id: Optional[str] = Query(None, description="Filter by originating mission/task/heartbeat id (PRD-164: mission deliverables tab)"),
    agent_id: Optional[int] = Query(None, description="Filter by agent id"),
    date_from: Optional[str] = Query(None, description="ISO timestamp — include rows created_at >= date_from"),
    date_to: Optional[str] = Query(None, description="ISO timestamp — include rows created_at <= date_to"),
    search: Optional[str] = Query(None, description="Case-insensitive search over title, summary, file_path"),
    limit: int = Query(24, ge=1, le=100),
    offset: int = Query(0, ge=0),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List deliverables for the current workspace."""
    svc = DeliverableService(db, ctx.workspace_id)
    return svc.list_deliverables(
        artifact_type=artifact_type,
        source_type=source_type,
        source_type_exclude=source_type_exclude,
        source_id=source_id,
        agent_id=agent_id,
        date_from=date_from,
        date_to=date_to,
        search=search,
        limit=limit,
        offset=offset,
    )


# ── Stats ────────────────────────────────────────────────────────────
# NOTE: declared BEFORE /{deliverable_id} so FastAPI matches the literal path
# first — otherwise `stats` would be captured as an id.
@router.get("/stats")
async def deliverable_stats(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Aggregate deliverable stats for the workspace."""
    svc = DeliverableService(db, ctx.workspace_id)
    return svc.get_stats()


# ── Retention ───────────────────────────────────────────────────────
@router.post("/retention", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def apply_retention(
    source_type: str = Body("heartbeat", embed=True),
    keep_per_agent: int = Body(50, embed=True),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Prune old deliverables — keeps the N most recent per agent for a given source_type."""
    svc = DeliverableService(db, ctx.workspace_id)
    return svc.apply_retention(source_type=source_type, keep_per_agent=keep_per_agent)


# ── Get One ──────────────────────────────────────────────────────────
@router.get("/{deliverable_id}")
async def get_deliverable(
    deliverable_id: str,
    include_content: bool = Query(False, description="When true, reads file content via WorkspaceClient"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Fetch a single deliverable (404 if missing / not in workspace)."""
    svc = DeliverableService(db, ctx.workspace_id)
    result = await svc.get_deliverable(deliverable_id, include_content=include_content)

    if not result.get("success"):
        raise HTTPException(
            status_code=404,
            detail=result.get("error", "Deliverable not found"),
        )
    return result


# ── Soft Delete ──────────────────────────────────────────────────────
@router.delete("/{deliverable_id}", dependencies=[Depends(require_workspace_permission("documents:delete"))])
async def delete_deliverable(
    deliverable_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Soft-delete a deliverable (sets deleted_at = NOW())."""
    svc = DeliverableService(db, ctx.workspace_id)
    result = svc.soft_delete(deliverable_id)

    if not result.get("success"):
        raise HTTPException(
            status_code=404,
            detail=result.get("error", "Deliverable not found"),
        )
    return result

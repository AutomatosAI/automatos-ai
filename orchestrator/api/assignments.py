"""
Assignments API
===============

Read-only endpoints for the Assignments page (Cluster 1 Part A).
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import select

from core.database.database import get_db
from core.models import WorkflowTemplate
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/assignments", tags=["assignments"])


@router.get("/recommended")
async def get_recommended(
    limit: int = Query(12, ge=1, le=24),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Marketplace discovery feed for the Assignments page.

    Returns approved marketplace playbooks the workspace has NOT yet installed,
    ranked by featured > install_count > recency. Excludes anything already
    cloned into this workspace (via cloned_from_id).
    """
    # Subquery: marketplace template IDs already installed in this workspace
    installed_subq = select(WorkflowTemplate.cloned_from_id).where(
        WorkflowTemplate.owner_type == "workspace",
        WorkflowTemplate.workspace_id == ctx.workspace_id,
        WorkflowTemplate.cloned_from_id.isnot(None),
    )

    rows = (
        db.query(WorkflowTemplate)
        .filter(
            WorkflowTemplate.owner_type == "marketplace",
            WorkflowTemplate.is_approved.is_(True),
            WorkflowTemplate.id.notin_(installed_subq),
        )
        .order_by(
            WorkflowTemplate.is_featured.desc(),
            WorkflowTemplate.install_count.desc(),
            WorkflowTemplate.created_at.desc(),
        )
        .limit(limit)
        .all()
    )

    items = [
        {
            "id": r.id,
            "type": "playbook",
            "name": r.name,
            "description": r.description or "",
            "icon": r.marketplace_icon or ((r.tags or [None])[0] if r.tags else None),
            "category": r.marketplace_category or "Marketplace",
            "install_count": r.install_count or 0,
            "is_featured": bool(r.is_featured),
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "source": "marketplace",
        }
        for r in rows
    ]

    return {
        "items": items,
        "workspace_count": 0,
        "marketplace_count": len(items),
    }

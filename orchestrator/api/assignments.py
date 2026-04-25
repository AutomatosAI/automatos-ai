"""
Assignments API
===============

Read-only endpoints for the Assignments page (Cluster 1 Part A).
"""

import logging
from typing import List, Dict, Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import case, func as sa_func

from core.database.database import get_db
from core.models import WorkflowTemplate as WorkflowRecipe
from core.models.orchestration import OrchestrationRun
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/assignments", tags=["assignments"])


@router.get("/recommended")
async def get_recommended(
    limit: int = Query(8, ge=1, le=20),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Return a mixed list of recommended items for the Assignments page.

    v1 ranking:
      - Workspace playbooks: (use_count * 0.6) + (recency_score * 0.4)
      - Marketplace items: ordered by global install_count
      - Missions: most recent completed runs

    Returns max `limit` items (default 8): up to half workspace, rest marketplace.
    If workspace has 0 items, all slots go to marketplace.
    """
    half = limit // 2

    # ── Workspace playbooks ranked by weighted score ──────────────
    # recency_score: days since last use, clamped to 30, inverted so recent = higher
    recency_expr = case(
        (
            WorkflowRecipe.last_used_at.isnot(None),
            30 - sa_func.least(
                sa_func.extract("epoch", sa_func.now() - WorkflowRecipe.last_used_at) / 86400,
                30,
            ),
        ),
        else_=0,
    )
    score_expr = (sa_func.coalesce(WorkflowRecipe.use_count, 0) * 0.6) + (recency_expr * 0.4)

    workspace_playbooks = (
        db.query(WorkflowRecipe)
        .filter(
            WorkflowRecipe.owner_type == "workspace",
            WorkflowRecipe.workspace_id == ctx.workspace_id,
        )
        .order_by(score_expr.desc())
        .limit(half)
        .all()
    )

    # ── Recent completed missions ────────────────────────────────
    recent_missions = (
        db.query(OrchestrationRun)
        .filter(
            OrchestrationRun.workspace_id == ctx.workspace_id,
            OrchestrationRun.state == "completed",
        )
        .order_by(OrchestrationRun.completed_at.desc().nullslast())
        .limit(half)
        .all()
    )

    # Combine workspace items
    workspace_items: List[Dict[str, Any]] = []
    for pb in workspace_playbooks:
        workspace_items.append({
            "id": pb.id,
            "type": "playbook",
            "name": pb.name,
            "description": pb.description or "",
            "icon": (pb.tags or [None])[0] if pb.tags else None,
            "category": pb.marketplace_category or "Playbook",
            "use_count": pb.use_count or 0,
            "source": "workspace",
        })
    for m in recent_missions:
        workspace_items.append({
            "id": str(m.id),
            "type": "mission",
            "name": m.goal[:120] if m.goal else "Mission",
            "description": f"Completed {m.completed_at.strftime('%b %d') if m.completed_at else ''}",
            "icon": None,
            "category": "Mission",
            "use_count": 0,
            "source": "workspace",
        })

    # Cap workspace items at half
    workspace_items = workspace_items[:half]

    # ── Marketplace items by install count ────────────────────────
    marketplace_limit = limit - len(workspace_items)
    marketplace_rows = (
        db.query(WorkflowRecipe)
        .filter(
            WorkflowRecipe.owner_type == "marketplace",
            WorkflowRecipe.is_approved.is_(True),
        )
        .order_by(WorkflowRecipe.install_count.desc(), WorkflowRecipe.created_at.desc())
        .limit(marketplace_limit)
        .all()
    )

    marketplace_items: List[Dict[str, Any]] = []
    for mp in marketplace_rows:
        marketplace_items.append({
            "id": mp.id,
            "type": "playbook",
            "name": mp.name,
            "description": mp.description or "",
            "icon": mp.marketplace_icon or ((mp.tags or [None])[0] if mp.tags else None),
            "category": mp.marketplace_category or "Marketplace",
            "install_count": mp.install_count or 0,
            "source": "marketplace",
        })

    return {
        "items": workspace_items + marketplace_items,
        "workspace_count": len(workspace_items),
        "marketplace_count": len(marketplace_items),
    }

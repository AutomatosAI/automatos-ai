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

from core.auth.actor import resolve_internal_user_id
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.core import Agent, Chat, Document, Message, User
from core.models.workspaces import Workspace
from core.workspaces.audit import AuditService
from core.utils.timestamps import utc_iso
from core.workspaces.models import WorkspaceMember
from services.plan_tiers import assign_plan, assignable_tiers, exposure_for_plan, get_tier
from services.workspace_purge import purge_workspace_sync

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/workspaces", tags=["Admin Workspaces"])


# ===================================================================
# Helpers
# ===================================================================

# The UTC-explicit serialiser now lives in core.utils.timestamps; the module-level
# name is kept so call sites and tests in this file read unchanged.
_utc_iso = utc_iso


def _is_admin(ctx: RequestContext) -> bool:
    if not ctx.user:
        return False
    # PRD-174 F043: shared admin check — super_admin ⊇ admin when the plane is on.
    from core.auth.roles import caller_is_admin
    return caller_is_admin(ctx.user)


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
    members_count: int = 0
    # Max ``users.last_sign_in`` across the workspace's active members and its
    # owner — "last authenticated request", stamped by core.auth.hybrid. None
    # means nobody on this workspace has been seen since the stamp shipped.
    last_active_at: Optional[str] = None
    plan_limits: Dict[str, Any] = Field(default_factory=dict)
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

        # Active members + last-seen, batched the same way (never N+1). The seat
        # cap in core/workspaces/invitations.py counts exactly these rows
        # (is_active members), so members_count is directly comparable to a
        # tier's seats — that is what the console's downgrade warning reads.
        member_agg = (
            db.query(
                WorkspaceMember.workspace_id,
                func.count(WorkspaceMember.id),
                func.max(User.last_sign_in),
            )
            .join(User, User.id == WorkspaceMember.user_id)
            .filter(WorkspaceMember.workspace_id.in_(workspace_ids))
            .filter(WorkspaceMember.is_active.is_(True))
            .group_by(WorkspaceMember.workspace_id)
            .all()
        ) if workspace_ids else []
        members_by_ws = {ws_id: cnt for ws_id, cnt, _ in member_agg}
        member_seen_by_ws = {ws_id: seen for ws_id, _, seen in member_agg}

        # Owner lookup (bulk)
        owner_ids = {w.owner_id for w in workspaces if w.owner_id is not None}
        owners = (
            db.query(User).filter(User.id.in_(owner_ids)).all() if owner_ids else []
        )
        owner_by_id = {u.id: u for u in owners}

        items: List[WorkspaceListItem] = []
        for w in workspaces:
            owner = owner_by_id.get(w.owner_id) if w.owner_id else None
            # The owner is counted even when no membership row exists for them —
            # a solo workspace is otherwise reported as never active.
            seen_candidates = [
                t for t in (member_seen_by_ws.get(w.id), getattr(owner, "last_sign_in", None))
                if t is not None
            ]
            last_active = max(seen_candidates) if seen_candidates else None
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
                members_count=members_by_ws.get(w.id, 0),
                last_active_at=_utc_iso(last_active),
                plan_limits=dict(w.plan_limits or {}),
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


# ── Plan tiers (operator console) ────────────────────────────────────────────
#
# Declared BEFORE the ``/{workspace_id}`` detail route on purpose: a literal
# segment registered after a path-parameter route is shadowed by it (the
# PRD-220 route-order trap). ``workspace_id`` is typed ``UUID`` so "plans" would
# 422 rather than fall through, but order is the thing that makes it correct.


class PlanTierInfo(BaseModel):
    name: str
    display_name: Optional[str] = None
    assignable: bool = False
    coming_soon: bool = False
    seats: Optional[int] = None
    max_agents: Optional[int] = None
    marketplace_depth: int = 1
    families: Dict[str, bool] = Field(default_factory=dict)
    nav: Dict[str, bool] = Field(default_factory=dict)


class PlanChangeBody(BaseModel):
    plan: str = Field(..., min_length=1, max_length=50)


@router.get("/plans")
async def list_plan_tiers(
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """The tier catalogue behind the console's plan dropdown. Admin-only.

    Every tier is returned, not just the assignable ones, so a workspace sitting
    on a non-assignable tier (``enterprise`` today) still renders its own plan
    as the current value instead of showing blank. ``assignable`` says which the
    dropdown may actually select.

    ``nav`` is the same exposure the client gets from
    ``GET /api/workspaces/current`` — it is what makes the consequence of a tier
    legible in the console ("Basic hides Team and Analytics") rather than
    something the operator has to infer from the families map.
    """
    _assert_admin(ctx)
    from config import PLAN_TIERS

    selectable = assignable_tiers()
    tiers = [
        PlanTierInfo(
            name=name,
            display_name=tier.get("display_name"),
            assignable=name in selectable,
            coming_soon=bool(tier.get("coming_soon")),
            seats=tier.get("seats"),
            max_agents=tier.get("max_agents"),
            marketplace_depth=int(tier.get("marketplace_depth", 1) or 1),
            families=dict(tier.get("families") or {}),
            nav=dict(exposure_for_plan(name).get("nav") or {}),
        ).model_dump()
        for name, tier in PLAN_TIERS.items()
        if isinstance(tier, dict)
    ]
    # No budget figure is advertised: the console never writes one (see below).
    return {"tiers": tiers}


@router.patch("/{workspace_id}/plan")
async def change_workspace_plan(
    workspace_id: UUID,
    body: PlanChangeBody,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Move a workspace onto another tier. Admin-only.

    Routed through ``services.plan_tiers.assign_plan`` — the ONE writer of
    ``workspaces.plan`` / ``plan_limits`` (FR-4 auditability) — so the tier's
    seats and agent cap apply exactly as they do on the onboarding path.

    ``with_budget=False`` (owner decision, 2026-09-11): an operator retiering a
    LIVE tenant must never hand it a spend ceiling it did not have a moment ago.
    ``modules/policy/budget.py`` enforces ``plan_limits.budget`` through
    ``check_budget``, and a workspace with no budget key is ceiling-less today —
    so minting one here would silently throttle a pilot. The same switch clears a
    stale TIER-owned ceiling a previous assignment left behind, while an admin
    custom budget (``source != "tier"``) is the customer's own and survives
    untouched.

    Downgrades are permitted but reported: a tier whose ``seats`` sit below the
    workspace's active member count comes back in ``warnings``. Nothing is
    removed — the cap binds the NEXT invitation
    (``core/workspaces/invitations.py``), never the members already in place.
    """
    _assert_admin(ctx)

    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    # Every sibling mutator in this router refuses on a soft-deleted workspace;
    # the console disables the control too, but that is client-side only and a
    # purge is already queued against these rows (services/workspace_purge.py).
    if workspace.deleted_at:
        raise HTTPException(
            status_code=400, detail="Cannot change the plan of a deleted workspace"
        )

    previous_plan = workspace.plan
    previous_limits = dict(workspace.plan_limits or {})

    try:
        new_limits = assign_plan(db, workspace, body.plan, with_budget=False)
    except ValueError as exc:
        # Unknown / non-assignable tier — the caller's input, not a server fault.
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Plan change failed for workspace %s: %s", workspace_id, exc)
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to change plan")

    limits_changed = {
        key: {"from": previous_limits.get(key), "to": new_limits.get(key)}
        for key in sorted(set(previous_limits) | set(new_limits))
        if previous_limits.get(key) != new_limits.get(key)
    }

    members_count = (
        db.query(func.count(WorkspaceMember.id))
        .filter(WorkspaceMember.workspace_id == workspace.id)
        .filter(WorkspaceMember.is_active.is_(True))
        .scalar()
    ) or 0

    warnings: List[str] = []
    seats = (get_tier(workspace.plan) or {}).get("seats")
    # Mirror the enforcement exactly (core/workspaces/invitations.py): -1 is the
    # only "unlimited" sentinel there, so a tier tuned to seats=0 via
    # AUTOMATOS_PLAN_TIERS_JSON blocks EVERY invite and must still warn. Do not
    # borrow the 0-means-unlimited convention that plan_tiers documents for
    # max_agents / watcher_limit — that is a different key with a different rule.
    if isinstance(seats, int) and seats != -1 and seats < members_count:
        warnings.append(
            f"{workspace.name} has {members_count} active members but "
            f"{workspace.plan} allows {seats}. No one was removed — the cap "
            f"applies to the next invitation."
        )

    AuditService(db).log(
        workspace_id=str(workspace.id),
        user_id=resolve_internal_user_id(db, ctx),
        action="workspace.plan_changed",
        resource_type="workspace",
        resource_id=str(workspace.id),
        resource_name=workspace.name,
        details={
            "from_plan": previous_plan,
            "to_plan": workspace.plan,
            "limits_changed": limits_changed,
            "budget_applied": False,
            "actor_email": getattr(ctx.user, "email", None),
            "warnings": warnings,
        },
    )
    logger.info(
        "[admin] workspace %s plan %s -> %s by %s",
        workspace.id, previous_plan, workspace.plan, getattr(ctx.user, "email", "?"),
    )

    return {
        "id": str(workspace.id),
        "plan": workspace.plan,
        "previous_plan": previous_plan,
        "plan_limits": new_limits,
        "limits_changed": limits_changed,
        "members_count": members_count,
        "warnings": warnings,
    }


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

        # Same shape as the list route — the detail view inherits these fields,
        # so leaving them at their defaults would report 0 members and no limits
        # for a workspace that has both.
        members_count, member_seen = (
            db.query(func.count(WorkspaceMember.id), func.max(User.last_sign_in))
            .join(User, User.id == WorkspaceMember.user_id)
            .filter(WorkspaceMember.workspace_id == workspace_id)
            .filter(WorkspaceMember.is_active.is_(True))
            .one()
        )
        seen_candidates = [
            t for t in (member_seen, getattr(owner, "last_sign_in", None)) if t is not None
        ]
        last_active = max(seen_candidates) if seen_candidates else None

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
            members_count=members_count or 0,
            last_active_at=_utc_iso(last_active),
            plan_limits=dict(w.plan_limits or {}),
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

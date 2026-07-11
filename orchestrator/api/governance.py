"""PRD-196 (P2-15) — the governance operator surface router.

One router for the whole governance pillar's read/write surface — audit-log
view + on/off status (S3), policy posture + budget editors (S4) — all gated by
the canonical ``require_workspace_admin`` (PRD-185 S12), Gerard's locked read
posture (196 Q2): the whole Governance tab is workspace-admin-only.

Grants stay on ``api.approval_grants`` and GDPR on ``api.gdpr`` — this router is
the audit/status/policy/budget half. Every read is ``ctx.workspace_id``-scoped,
fail-closed: the workspace filter is applied from the request context, NEVER
from a caller-supplied parameter, so no params can surface another tenant's rows.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.workspace_admin import require_workspace_admin
from core.database.database import get_db
from core.models.approval_grants import ApprovalGrant
from core.workspaces.audit import AuditLog
from modules.policy.flag import policy_plane_enabled

logger = logging.getLogger(__name__)

# The whole surface is workspace-admin-only (Gerard's 196 Q2 read posture).
router = APIRouter(
    prefix="/api/v1/governance",
    tags=["governance"],
    dependencies=[Depends(require_workspace_admin)],
)

# The policy-verdict look-back for the status tile's verdict counts.
_STATUS_WINDOW_DAYS = 30


def _parse_dt(value: Optional[str], field: str) -> Optional[datetime]:
    """Parse an ISO-8601 datetime filter, surfacing a bad value as 422."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=422, detail=f"invalid {field} datetime: {value!r}")


def _audit_row(row: AuditLog) -> Dict[str, Any]:
    return {
        "id": row.id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "actor_type": row.actor_type,
        "user_id": row.user_id,
        "action": row.action,
        "resource_type": row.resource_type,
        "resource_id": row.resource_id,
        "resource_name": row.resource_name,
        "details": row.details or {},
    }


def _retention_status() -> Dict[str, Any]:
    """Report the effective audit-retention window through the canonical S5
    reader (floor-enforced). Resilient before S5 lands: reports ``configured``
    false rather than inventing a number."""
    try:
        from services.audit_retention import (
            AUDIT_RETENTION_FLOOR_DAYS,
            effective_retention_days,
        )

        return {
            "retention_days": effective_retention_days(),
            "floor_days": AUDIT_RETENTION_FLOOR_DAYS,
            "configured": True,
        }
    except Exception:
        return {"retention_days": None, "floor_days": None, "configured": False}


@router.get("/audit-log")
async def get_audit_log(
    action_prefix: Optional[str] = None,
    actor_type: Optional[str] = None,
    resource_type: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    ctx: RequestContext = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Read this workspace's audit log, newest first, paginated + filterable.

    FAIL-CLOSED tenancy: the ``workspace_id`` filter comes from ``ctx``, never
    from a parameter — no combination of filters can return another workspace's
    rows. Filters: ``action_prefix`` (``policy:`` / ``gdpr:`` /
    ``approval_grant:``), ``actor_type``, ``resource_type``, ``since``/``until``.
    """
    q = db.query(AuditLog).filter(AuditLog.workspace_id == ctx.workspace_id)
    if action_prefix:
        q = q.filter(AuditLog.action.like(f"{action_prefix}%"))
    if actor_type:
        q = q.filter(AuditLog.actor_type == actor_type)
    if resource_type:
        q = q.filter(AuditLog.resource_type == resource_type)
    since_dt = _parse_dt(since, "since")
    if since_dt is not None:
        q = q.filter(AuditLog.created_at >= since_dt)
    until_dt = _parse_dt(until, "until")
    if until_dt is not None:
        q = q.filter(AuditLog.created_at <= until_dt)

    total = q.count()
    rows = q.order_by(AuditLog.created_at.desc()).offset(offset).limit(limit).all()
    return {
        "rows": [_audit_row(r) for r in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/status")
async def get_status(
    ctx: RequestContext = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """The honest governance status: is the policy plane enforcing, how many
    grants sit in each state, the recent policy-verdict counts, and the audit
    retention window. An OFF plane reports ``enforcing: false`` loudly — the
    operator is never falsely reassured (governance-policy §I.5)."""
    try:
        enforcing = bool(policy_plane_enabled())
    except Exception:
        enforcing = False

    grant_rows = (
        db.query(ApprovalGrant.status, func.count(ApprovalGrant.id))
        .filter(ApprovalGrant.workspace_id == ctx.workspace_id)
        .group_by(ApprovalGrant.status)
        .all()
    )
    by_status = {status: int(count) for status, count in grant_rows}

    since = datetime.now(timezone.utc) - timedelta(days=_STATUS_WINDOW_DAYS)
    verdict_rows = (
        db.query(AuditLog.action, func.count(AuditLog.id))
        .filter(
            AuditLog.workspace_id == ctx.workspace_id,
            AuditLog.action.like("policy:%"),
            AuditLog.created_at >= since,
        )
        .group_by(AuditLog.action)
        .all()
    )
    by_action = {action: int(count) for action, count in verdict_rows}

    return {
        "policy_plane": {"enforcing": enforcing},
        "grants": {"by_status": by_status, "total": sum(by_status.values())},
        "audit": {
            "policy_verdicts": {"total": sum(by_action.values()), "by_action": by_action},
            "window_days": _STATUS_WINDOW_DAYS,
        },
        "retention": _retention_status(),
    }

"""
Notifications API (PRD-128 US-004)
===================================

Endpoints backing the in-app notification bell dropdown.

All endpoints are workspace-scoped via ``RequestContext`` and enforce the
predicate::

    workspace_id = ctx.workspace_id
      AND (user_id = ctx.user_id OR user_id IS NULL)

so that rows targeted at the whole workspace (``user_id IS NULL``) and rows
targeted at the current user are both visible, but rows for *other* users in
the same workspace are not.

Notification rows are written by the ``NotificationDispatcher`` (US-003).
This module is purely a read/mutate surface for the frontend.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.core import User as UserModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/notifications", tags=["notifications"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class NotificationRow(BaseModel):
    id: str
    workspace_id: str
    user_id: Optional[int]
    event_type: str
    title: str
    message: Optional[str]
    link_type: Optional[str]
    link_id: Optional[str]
    agent_id: Optional[int]
    agent_name: Optional[str]
    status: str
    read_at: Optional[str]
    dismissed_at: Optional[str]
    created_at: str


class NotificationListResponse(BaseModel):
    success: bool
    notifications: List[NotificationRow]
    total: int
    limit: int
    offset: int


class UnreadCountResponse(BaseModel):
    success: bool
    count: int


class SimpleSuccessResponse(BaseModel):
    success: bool


class ReadAllResponse(BaseModel):
    success: bool
    marked_count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_user_id(db: Session, ctx: RequestContext) -> Optional[int]:
    """
    Resolve the current request's integer ``users.id`` (or ``None`` when the
    caller cannot be mapped to a DB user — e.g. API key auth).

    Matches first by ``clerk_user_id`` then by ``email``; mirrors the pattern
    used in ``api/workflow_recipes.py``.
    """
    if not ctx.user or not ctx.user.id:
        return None

    user = (
        db.query(UserModel)
        .filter(UserModel.clerk_user_id == ctx.user.id)
        .first()
    )
    if not user and ctx.user.email:
        user = (
            db.query(UserModel)
            .filter(UserModel.email == ctx.user.email)
            .first()
        )
    return user.id if user else None


def _row_to_model(row: Any) -> NotificationRow:
    """Convert a SQLAlchemy Row/mapping into a NotificationRow."""
    m = row._mapping if hasattr(row, "_mapping") else row
    return NotificationRow(
        id=str(m["id"]),
        workspace_id=str(m["workspace_id"]),
        user_id=m["user_id"],
        event_type=m["event_type"],
        title=m["title"],
        message=m["message"],
        link_type=m["link_type"],
        link_id=m["link_id"],
        agent_id=m["agent_id"],
        agent_name=m["agent_name"],
        status=m["status"],
        read_at=m["read_at"].isoformat() if m["read_at"] else None,
        dismissed_at=m["dismissed_at"].isoformat() if m["dismissed_at"] else None,
        created_at=m["created_at"].isoformat() if m["created_at"] else "",
    )


# The workspace+user visibility predicate, shared across all queries.
_VISIBILITY_PREDICATE = (
    "workspace_id = :workspace_id "
    "AND (user_id = :user_id OR user_id IS NULL)"
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=NotificationListResponse)
@router.get("/", response_model=NotificationListResponse)
async def list_notifications(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    unread_only: bool = Query(False),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> NotificationListResponse:
    """Paginated list of notifications for the current workspace + user."""
    user_id = _resolve_user_id(db, ctx)

    params: Dict[str, Any] = {
        "workspace_id": str(ctx.workspace_id),
        "user_id": user_id,
        "limit": limit,
        "offset": offset,
    }

    where_clauses = [_VISIBILITY_PREDICATE, "dismissed_at IS NULL"]
    if unread_only:
        where_clauses.append("read_at IS NULL")

    where_sql = " AND ".join(where_clauses)

    try:
        rows = db.execute(
            text(
                f"""
                SELECT id, workspace_id, user_id, event_type, title, message,
                       link_type, link_id, agent_id, agent_name, status,
                       read_at, dismissed_at, created_at
                  FROM notifications
                 WHERE {where_sql}
                 ORDER BY created_at DESC
                 LIMIT :limit OFFSET :offset
                """
            ),
            params,
        ).fetchall()

        total = db.execute(
            text(f"SELECT COUNT(*) FROM notifications WHERE {where_sql}"),
            params,
        ).scalar_one()
    except Exception:
        logger.exception("Failed to list notifications")
        raise HTTPException(status_code=500, detail="Failed to list notifications")

    return NotificationListResponse(
        success=True,
        notifications=[_row_to_model(r) for r in rows],
        total=int(total or 0),
        limit=limit,
        offset=offset,
    )


@router.get("/unread-count", response_model=UnreadCountResponse)
async def unread_count(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> UnreadCountResponse:
    """Count of unread, non-dismissed notifications for the current user."""
    user_id = _resolve_user_id(db, ctx)

    try:
        count = db.execute(
            text(
                f"""
                SELECT COUNT(*) FROM notifications
                 WHERE {_VISIBILITY_PREDICATE}
                   AND read_at IS NULL
                   AND dismissed_at IS NULL
                """
            ),
            {"workspace_id": str(ctx.workspace_id), "user_id": user_id},
        ).scalar_one()
    except Exception:
        logger.exception("Failed to get unread notification count")
        raise HTTPException(status_code=500, detail="Failed to get unread count")

    return UnreadCountResponse(success=True, count=int(count or 0))


@router.post("/{notification_id}/read", response_model=SimpleSuccessResponse)
async def mark_read(
    notification_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> SimpleSuccessResponse:
    """Mark a single notification as read (no-op if already read)."""
    user_id = _resolve_user_id(db, ctx)

    try:
        result = db.execute(
            text(
                f"""
                UPDATE notifications
                   SET read_at = NOW()
                 WHERE id = :id
                   AND {_VISIBILITY_PREDICATE}
                   AND read_at IS NULL
                """
            ),
            {
                "id": str(notification_id),
                "workspace_id": str(ctx.workspace_id),
                "user_id": user_id,
            },
        )
        # Verify row is at least visible (even if already read) before 404-ing.
        if result.rowcount == 0:
            exists = db.execute(
                text(
                    f"""
                    SELECT 1 FROM notifications
                     WHERE id = :id AND {_VISIBILITY_PREDICATE}
                    """
                ),
                {
                    "id": str(notification_id),
                    "workspace_id": str(ctx.workspace_id),
                    "user_id": user_id,
                },
            ).first()
            if not exists:
                db.rollback()
                raise HTTPException(status_code=404, detail="Notification not found")
        db.commit()
    except HTTPException:
        raise
    except Exception:
        db.rollback()
        logger.exception("Failed to mark notification read")
        raise HTTPException(status_code=500, detail="Failed to mark notification read")

    return SimpleSuccessResponse(success=True)


@router.post("/read-all", response_model=ReadAllResponse)
async def mark_all_read(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> ReadAllResponse:
    """Mark every unread notification for this workspace+user as read."""
    user_id = _resolve_user_id(db, ctx)

    try:
        result = db.execute(
            text(
                f"""
                UPDATE notifications
                   SET read_at = NOW()
                 WHERE {_VISIBILITY_PREDICATE}
                   AND read_at IS NULL
                   AND dismissed_at IS NULL
                """
            ),
            {"workspace_id": str(ctx.workspace_id), "user_id": user_id},
        )
        marked = result.rowcount or 0
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to mark all notifications read")
        raise HTTPException(status_code=500, detail="Failed to mark all read")

    return ReadAllResponse(success=True, marked_count=int(marked))


@router.post("/{notification_id}/dismiss", response_model=SimpleSuccessResponse)
async def dismiss(
    notification_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> SimpleSuccessResponse:
    """Soft-delete a notification by setting ``dismissed_at = NOW()``."""
    user_id = _resolve_user_id(db, ctx)

    try:
        result = db.execute(
            text(
                f"""
                UPDATE notifications
                   SET dismissed_at = NOW()
                 WHERE id = :id
                   AND {_VISIBILITY_PREDICATE}
                   AND dismissed_at IS NULL
                """
            ),
            {
                "id": str(notification_id),
                "workspace_id": str(ctx.workspace_id),
                "user_id": user_id,
            },
        )
        if result.rowcount == 0:
            exists = db.execute(
                text(
                    f"""
                    SELECT 1 FROM notifications
                     WHERE id = :id AND {_VISIBILITY_PREDICATE}
                    """
                ),
                {
                    "id": str(notification_id),
                    "workspace_id": str(ctx.workspace_id),
                    "user_id": user_id,
                },
            ).first()
            if not exists:
                db.rollback()
                raise HTTPException(status_code=404, detail="Notification not found")
        db.commit()
    except HTTPException:
        raise
    except Exception:
        db.rollback()
        logger.exception("Failed to dismiss notification")
        raise HTTPException(status_code=500, detail="Failed to dismiss notification")

    return SimpleSuccessResponse(success=True)

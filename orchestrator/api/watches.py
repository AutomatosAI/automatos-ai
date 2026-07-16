"""PRD-204 S11 -- the watchlist API.

The minimal read/cancel surface over the watch registry (PRD-204 S1/S2):
list this workspace's watches, drill into one (with its recent watch_events,
newest first), and cancel a live one. Everything else about a watch -- follow,
scoring, corrective actions -- belongs to the watcher tick and the S9 Auto
tools; this router deliberately stays thin over ``WatchService`` and reuses
the S9 serializer (``watch_to_dict``) so the HTTP and tool surfaces speak one
shape.

Auth mirrors the board router's plane (PRD-195 P2-14): the shared hybrid
dependency resolves the workspace, and ``require_workspace_permission`` gates
each route with the mission-plane strings -- ``missions:read`` for the reads
(every workspace role holds it) and ``missions:update`` for cancel (editor and
up), the same family the board's task mutations ride. Cross-workspace or
missing ids are a plain 404.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.watches import WatchEvent
from modules.tools.discovery.handlers_watches import watch_to_dict
from services.orchestration_state import InvalidTransitionError
from services.watch_service import WatchService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/watches", tags=["watches"])

# Recent-events cap for the detail view (a watch timeline is short-lived;
# 50 covers created -> terminal -> scored -> actions with headroom).
RECENT_EVENT_LIMIT = 50


def _valid_watch_id(watch_id: str) -> str:
    """Boundary validation: a malformed id reads as 'no such watch' (404),
    never as a DB-level cast error surfacing a 500."""
    try:
        return str(UUID(str(watch_id)))
    except (TypeError, ValueError):
        raise HTTPException(status_code=404, detail="Watch not found")


def _event_to_dict(event: WatchEvent) -> Dict[str, Any]:
    """The S9 recent-event shape (handlers_watches.get_watch), unchanged."""
    return {
        "event_type": event.event_type,
        "summary": event.summary,
        "score": event.score,
        "action_taken": event.action_taken,
        "requires_attention": event.requires_attention,
        "created_at": str(event.created_at) if event.created_at else None,
    }


@router.get(
    "",
    dependencies=[Depends(require_workspace_permission("missions:read"))],
)
async def list_watches(
    status: Optional[str] = Query(
        None, description="Filter by exact watch status (e.g. 'watching')"
    ),
    watch_type: Optional[str] = Query(
        None, description="Filter by watch type (mission / playbook_execution / scheduled_playbook)"
    ),
    include_closed: bool = Query(
        False, description="Without a status filter, include closed watches too"
    ),
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """This workspace's watches, newest first. Live-only unless asked."""
    watches = WatchService.list_watches(
        db,
        ctx.workspace_id,
        status=status,
        watch_type=watch_type,
        include_closed=include_closed,
        limit=limit,
        offset=offset,
    )
    return {
        "watches": [watch_to_dict(w) for w in watches],
        "total": len(watches),
    }


@router.get(
    "/{watch_id}",
    dependencies=[Depends(require_workspace_permission("missions:read"))],
)
async def get_watch(
    watch_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """One watch (with lineage) plus its recent events, newest first."""
    watch_id = _valid_watch_id(watch_id)
    watch = WatchService.get_watch(db, ctx.workspace_id, watch_id)
    if watch is None:
        raise HTTPException(status_code=404, detail="Watch not found")

    events = (
        db.query(WatchEvent)
        .filter(WatchEvent.watch_id == watch.id)
        .order_by(WatchEvent.created_at.desc())
        .limit(RECENT_EVENT_LIMIT)
        .all()
    )
    return {
        "watch": watch_to_dict(watch, include_lineage=True),
        "recent_events": [_event_to_dict(e) for e in events],
    }


@router.post(
    "/{watch_id}/cancel",
    dependencies=[Depends(require_workspace_permission("missions:update"))],
)
async def cancel_watch(
    watch_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Cancel a live watch. Closed watches refuse (422); unknown ids 404."""
    watch_id = _valid_watch_id(watch_id)
    try:
        watch = WatchService.cancel_watch(db, ctx.workspace_id, watch_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Watch not found")
    except InvalidTransitionError:
        raise HTTPException(
            status_code=422,
            detail="That watch is already closed and cannot be cancelled.",
        )
    db.commit()
    logger.info(
        "[Watches] %s cancelled via API in workspace %s", watch_id, ctx.workspace_id
    )
    return {"watch": watch_to_dict(watch)}

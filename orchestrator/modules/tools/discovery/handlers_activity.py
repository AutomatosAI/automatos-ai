"""Activity feed handler for PlatformActionExecutor."""

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def get_activity_feed(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Unified activity feed -- chats, recipe runs, routines."""
    from services.activity_service import ActivityService

    period = params.get("period", "7d")
    type_csv = params.get("type", "")
    types = [t.strip() for t in type_csv.split(",") if t.strip()] if type_csv else None
    limit = min(params.get("limit", 20), 50)

    service = ActivityService(db, workspace_id)
    feed = service.get_feed(types=types, period=period, limit=limit)

    return {
        "success": True,
        "period": period,
        "items": feed.get("items", []),
        "total": feed.get("total", 0),
        "message": f"Showing {len(feed.get('items', []))} activities from the last {period}.",
    }

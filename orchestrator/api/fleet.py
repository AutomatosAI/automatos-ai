"""PRD-228 US-002 — the fleet route.

One new endpoint, ``GET /api/v1/fleet``, exposing the read-model
(:func:`services.fleet_state.get_fleet_state`) for the caller's workspace.

Auth mirrors the board list exactly (``api/board_tasks.py`` ``list_tasks``):
the same ``require_task_context(TASKS_READ)`` guard resolves the workspace and
enforces the read scope, so a caller only ever sees their own workspace's floor
— the workspace is taken from the authenticated context, never a client
parameter. The guard is bound to a module-level name so tests can override it.

Read-only: the handler issues the read-model's bounded query set and returns
its deterministic shape; it performs no writes.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import require_task_context
from core.auth.scopes import TASKS_READ
from core.database.database import get_db
from services.fleet_state import get_fleet_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/fleet", tags=["fleet"])

# The board plane's read guard, mirrored (PRD-228 §7 / api/board_tasks.py).
_require_fleet_read = require_task_context(TASKS_READ)


@router.get("")
async def get_fleet(
    ctx: RequestContext = Depends(_require_fleet_read),
    db: Session = Depends(get_db),
):
    """Live fleet state for the caller's workspace (PRD-228 read-model)."""
    return get_fleet_state(db, ctx.workspace_id)

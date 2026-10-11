"""PRD-256 US-006: the pinned first-class tools, on the lanes that ship the dispatcher alone.

Eleven customer nights found the commonest false claim was a call that failed on its arguments
and was then reported as done (F108: ``platform_approve_mission`` with ``params: {}``; night
10b's ``data`` sent as JSON text). The review's join: on 53% of dispatcher calls the action used
sat outside the ranked top-15 of the enum, led by Auto's writes. Those writes are now promoted
and pinned (``TOOL_ROUTING_PROMOTION_PINS``): each is its own tool with a strict schema, and the
dispatcher's enum leaves them out.

The full path attaches them in ``tool_router`` (``_first_class_names`` and
``ActionRegistry.to_first_class_schemas``). Two lanes ship the dispatcher alone: Auto's short
ATOM lane (``consumers/chatbot/service.py``) and the heartbeat orchestrator's dispatcher-only
load (``modules/context/sections/tools.py``). Their dispatcher leaves promoted actions out of its
enum, so without this a promoted write would reach neither. Both call ``with_first_class``, which
attaches the same schemas through the same two functions, held to the same gates: never an
admin-only or super-admin action, never a category the workspace is not shown (the turn's
hidden scope), only what can run here, inside the plan tier's families.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TRACE = "first-class"


def pinned_first_class(workspace_id: Any = None, db: Any = None) -> List[Dict[str, Any]]:
    """The pinned promoted actions' own schemas for a caller that is no admin: the full path's
    first-class set with nothing ranked in (the pins alone)."""
    from modules.tools.discovery.action_registry import get_action_registry
    from modules.tools.tool_router import _apply_tier_exposure, _first_class_names

    registry = get_action_registry()
    promoted = {action.name for action in registry.get_all() if action.promoted}
    schemas = registry.to_first_class_schemas(exclude_admin=True, first_class_names=_first_class_names(None, promoted))
    if db is None or workspace_id is None:
        return schemas
    return _apply_tier_exposure(db, workspace_id, schemas, TRACE)


def with_first_class(tools: Optional[List[Dict[str, Any]]], workspace_id: Any = None,
                     db: Any = None) -> List[Dict[str, Any]]:
    """``tools`` with the pinned first-class schemas it does not already hold, after it.

    Returns a new list; no schema is mutated. A surface that ships no tools (a proactive
    opener) stays empty. A pin set that cannot be built is logged and the surface is
    returned as it was.
    """
    if not tools:
        return list(tools or [])
    held = {_name(tool) for tool in tools}
    try:
        added = [schema for schema in pinned_first_class(workspace_id, db) if _name(schema) not in held]
    except Exception:  # noqa: BLE001 — an unconfirmed pin set is left off; the surface stays as it was
        logger.exception("[%s] the pinned first-class tools could not be built", TRACE)
        return list(tools)
    return list(tools) + added


def _name(tool: Any) -> str:
    return str(((tool or {}).get("function") or {}).get("name", "")) if isinstance(tool, dict) else ""


__all__ = ["TRACE", "pinned_first_class", "with_first_class"]

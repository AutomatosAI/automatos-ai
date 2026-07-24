"""
Field handlers — PRD-108 Memory Field
======================================

Platform tool handlers for agents to interact with the shared mission field.
Agents call these during mission execution to share and retrieve knowledge.
"""

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _actor_agent_id(params: Dict[str, Any], kwargs: Dict[str, Any]) -> int:
    """Resolve the acting agent id.

    The platform executor invokes field handlers as
    ``handler(db, workspace_id, params)`` with no kwargs, and carries the
    actor in ``params["_agent_id"]`` (its actor convention — see
    platform_executor.py). Read it there first; fall back to a legacy
    ``agent_id`` kwarg, then to 0 (system). Coerce defensively so a JSON
    string id never raises downstream.
    """
    raw = params.get("_agent_id")
    if raw is None:
        raw = kwargs.get("agent_id", 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


async def field_query(
    db: Session,
    workspace_id: UUID,
    params: Dict[str, Any],
    **kwargs,
) -> Dict[str, Any]:
    """Query the field for relevant patterns.

    PRD-166 S1/Q20: prefer the active mission field (``field_id``); with no
    mission field, fall back to **workspace-persistent** recall across every
    mission that has run here. Only fails when neither has any memory yet.
    PRD-166 S2/D11: the result block is trimmed to a token budget and reports
    ``truncated`` — never a silent cap.
    """
    query_text = params.get("query", "")
    top_k = params.get("top_k") or 0  # 0 → config default (FIELD_QUERY_TOP_K)
    field_id = params.get("field_id") or kwargs.get("field_id")
    agent_id = _actor_agent_id(params, kwargs)

    if not query_text:
        return {"success": False, "error": "query is required"}

    try:
        from config import config
        from modules.context import field_scoring
        from modules.context.factory import get_shared_context

        field = get_shared_context()
        if not field:
            return {"success": False, "error": "Shared context backend unavailable"}
        inner = getattr(field, "_inner", field)

        scope = "mission"
        if field_id:
            results = await field.query(
                context_id=field_id, query=query_text, agent_id=agent_id, top_k=top_k,
            )
        elif hasattr(inner, "query_workspace"):
            # Q20: no active mission field → workspace-persistent recall (if any).
            scope = "workspace"
            results = await inner.query_workspace(
                str(workspace_id), query_text, agent_id, top_k,
            )
        else:
            results = []

        if not results:
            if not field_id and scope == "workspace":
                return {
                    "success": True, "results": [], "scope": scope, "truncated": False,
                    "message": "No field memory yet — patterns appear once missions run in this workspace.",
                }
            return {
                "success": True, "results": [], "scope": scope, "truncated": False,
                "message": "No relevant patterns found in the field.",
            }

        formatted = [{
            "key": r["key"],
            "value": r["value"],
            "relevance": round(r["score"], 4),
            "from_agent": r.get("agent_id", 0),
            "strength": round(r["decayed_strength"], 4),
            "mission_id": r.get("mission_id"),
        } for r in results]

        kept, truncated = field_scoring.budget_results(
            formatted, config.FIELD_QUERY_TOKEN_BUDGET,
        )
        return {
            "success": True,
            "results": kept,
            "count": len(kept),
            "truncated": truncated,
            "scope": scope,
        }

    except Exception as e:
        logger.error("[Field] Query failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Field query failed: {str(e)}"}


async def field_inject(
    db: Session,
    workspace_id: UUID,
    params: Dict[str, Any],
    **kwargs,
) -> Dict[str, Any]:
    """Inject a pattern into the shared mission field."""
    key = params.get("key", "")
    value = params.get("value", "")
    strength = params.get("strength", 1.0)
    field_id = params.get("field_id") or kwargs.get("field_id")
    agent_id = _actor_agent_id(params, kwargs)

    if not key or not value:
        return {"success": False, "error": "key and value are required"}

    if not field_id:
        return {
            "success": False,
            "error": "No shared field available — this tool only works during missions.",
        }

    try:
        from modules.context.factory import get_shared_context

        field = get_shared_context()
        if not field:
            return {"success": False, "error": "Shared context backend unavailable"}
        await field.inject(
            context_id=field_id,
            key=key,
            value=value[:4000],  # Cap to prevent embedding blow-up
            agent_id=agent_id,
            strength=min(max(strength, 0.0), 1.0),
        )

        return {
            "success": True,
            "message": f"Pattern '{key}' shared with the mission field.",
        }

    except Exception as e:
        logger.error("[Field] Inject failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Field inject failed: {str(e)}"}


async def field_stability(
    db: Session,
    workspace_id: UUID,
    params: Dict[str, Any],
    **kwargs,
) -> Dict[str, Any]:
    """Measure field convergence."""
    field_id = params.get("field_id") or kwargs.get("field_id")

    if not field_id:
        return {
            "success": False,
            "error": "No shared field available — this tool only works during missions.",
        }

    try:
        from modules.context.factory import get_shared_context

        field = get_shared_context()
        if not field:
            return {"success": False, "error": "Shared context backend unavailable"}
        stats = await field._inner.measure_stability(field_id) if hasattr(field._inner, "measure_stability") else {}

        return {
            "success": True,
            **stats,
        }

    except Exception as e:
        logger.error("[Field] Stability check failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Stability check failed: {str(e)}"}

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


async def field_query(
    db: Session,
    workspace_id: UUID,
    params: Dict[str, Any],
    **kwargs,
) -> Dict[str, Any]:
    """Query the shared mission field for relevant patterns."""
    query_text = params.get("query", "")
    top_k = params.get("top_k", 10)
    field_id = params.get("field_id") or kwargs.get("field_id")
    agent_id = kwargs.get("agent_id", 0)

    if not query_text:
        return {"success": False, "error": "query is required"}

    if not field_id:
        return {
            "success": False,
            "error": "No shared field available — this tool only works during missions.",
        }

    try:
        from modules.context.adapters.vector_field import VectorFieldSharedContext

        field = VectorFieldSharedContext()
        results = await field.query(
            context_id=field_id,
            query=query_text,
            agent_id=agent_id,
            top_k=top_k,
        )

        if not results:
            return {
                "success": True,
                "results": [],
                "message": "No relevant patterns found in the field.",
            }

        # Format for agent consumption
        formatted = []
        for r in results:
            formatted.append({
                "key": r["key"],
                "value": r["value"],
                "relevance": round(r["score"], 4),
                "from_agent": r["agent_id"],
                "strength": round(r["decayed_strength"], 4),
            })

        return {
            "success": True,
            "results": formatted,
            "count": len(formatted),
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
    agent_id = kwargs.get("agent_id", 0)

    if not key or not value:
        return {"success": False, "error": "key and value are required"}

    if not field_id:
        return {
            "success": False,
            "error": "No shared field available — this tool only works during missions.",
        }

    try:
        from modules.context.adapters.vector_field import VectorFieldSharedContext

        field = VectorFieldSharedContext()
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
        from modules.context.adapters.vector_field import VectorFieldSharedContext

        field = VectorFieldSharedContext()
        stats = await field.measure_stability(field_id)

        return {
            "success": True,
            **stats,
        }

    except Exception as e:
        logger.error("[Field] Stability check failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Stability check failed: {str(e)}"}

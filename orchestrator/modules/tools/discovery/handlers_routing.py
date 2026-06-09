"""Routing handler functions for PlatformActionExecutor (PRD-142 Wave 4, W4-S6).

Inserts a workspace-scoped ``routing_rules`` row (read by the UniversalRouter at
Tier 2a). The ``workspace_id`` comes from the executor context, never the params,
so a rule can only ever be created for the caller's own workspace.
"""

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def create_routing_rule(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a workspace-scoped routing rule.

    Requires a target (agent or playbook) and a matcher (source_pattern or
    source_channel) — a rule with neither would match nothing or everything, so
    both are validated before any write. Fail-closed on a bad/empty rule.
    """
    from core.models.routing import RoutingRule

    target_agent_id = params.get("target_agent_id")
    target_workflow_id = params.get("target_workflow_id")
    if target_agent_id is None and target_workflow_id is None:
        return {
            "success": False,
            "error": "A routing rule needs a target_agent_id or target_workflow_id",
        }

    source_pattern = params.get("source_pattern")
    source_channel = params.get("source_channel")
    if not source_pattern and not source_channel:
        return {
            "success": False,
            "error": "A routing rule needs a source_pattern or source_channel to match",
        }

    try:
        priority_raw = params.get("priority", 0)
        try:
            priority = int(priority_raw) if priority_raw is not None else 0
        except (TypeError, ValueError):
            priority = 0

        rule = RoutingRule(
            workspace_id=workspace_id,
            source_pattern=source_pattern,
            source_channel=source_channel,
            intent_keywords=params.get("intent_keywords") or [],
            target_agent_id=target_agent_id,
            target_workflow_id=target_workflow_id,
            priority=priority,
            is_active=True,
        )
        db.add(rule)
        db.commit()
        db.refresh(rule)

        return {
            "success": True,
            "data": {
                "id": rule.id,
                "workspace_id": str(workspace_id),
                "source_pattern": source_pattern,
                "source_channel": source_channel,
                "intent_keywords": rule.intent_keywords,
                "target_agent_id": target_agent_id,
                "target_workflow_id": target_workflow_id,
                "priority": priority,
            },
        }
    except Exception as exc:
        db.rollback()
        logger.error("[routing] create_routing_rule failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}

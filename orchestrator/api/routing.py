"""
Routing API endpoints (PRD-50: US-010, US-011).

Admin endpoints for managing routing rules, viewing decisions,
recording corrections, inspecting cache statistics, and setting up
trigger subscriptions.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.composio import TriggerSubscription
from core.models.routing import (
    ChannelSource,
    RoutingDecisionRecord,
    RoutingRule,
)
from core.routing.cache import get_routing_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/routing", tags=["Routing"])

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class CreateRuleRequest(BaseModel):
    source_pattern: Optional[str] = None
    intent_keywords: List[str] = Field(default_factory=list)
    target_agent_id: Optional[int] = None
    target_workflow_id: Optional[int] = None
    priority: int = 0


class RuleResponse(BaseModel):
    id: int
    workspace_id: UUID
    source_pattern: Optional[str]
    intent_keywords: List[str]
    target_agent_id: Optional[int]
    target_workflow_id: Optional[int]
    priority: int
    is_active: bool
    created_at: str

    class Config:
        from_attributes = True


class DecisionResponse(BaseModel):
    id: int
    request_id: UUID
    envelope_hash: str
    source: str
    route_type: str
    agent_id: Optional[int]
    workflow_id: Optional[int]
    confidence: float
    cached: bool
    was_corrected: bool
    corrected_agent_id: Optional[int]
    created_at: str

    class Config:
        from_attributes = True


class CorrectionRequest(BaseModel):
    request_id: UUID
    correct_agent_id: int


class TriggerSetupRequest(BaseModel):
    trigger_name: str = Field(
        default="JIRA_NEW_ISSUE_TRIGGER",
        description="Composio trigger name to subscribe to",
    )
    project_key: str = Field(
        default="PILOT",
        description="Jira project key to monitor",
    )
    agent_id: Optional[int] = Field(
        default=None,
        description="Agent ID to route triggered events to (optional)",
    )
    workflow_id: Optional[int] = Field(
        default=None,
        description="Workflow ID to route triggered events to (optional)",
    )


# ---------------------------------------------------------------------------
# GET /api/routing/decisions
# ---------------------------------------------------------------------------


@router.get("/decisions")
async def list_decisions(
    source: Optional[str] = Query(None, description="Filter by source channel"),
    agent_id: Optional[int] = Query(None, description="Filter by routed agent_id"),
    was_corrected: Optional[bool] = Query(None, description="Filter by correction status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=1000),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List recent routing decisions with optional filters. Paginated."""
    try:
        query = db.query(RoutingDecisionRecord).order_by(
            RoutingDecisionRecord.created_at.desc()
        )

        if source is not None:
            query = query.filter(RoutingDecisionRecord.source == source)
        if agent_id is not None:
            query = query.filter(RoutingDecisionRecord.agent_id == agent_id)
        if was_corrected is not None:
            query = query.filter(RoutingDecisionRecord.was_corrected == was_corrected)

        decisions = query.offset(skip).limit(limit).all()

        return [
            {
                "id": d.id,
                "request_id": str(d.request_id),
                "envelope_hash": d.envelope_hash,
                "source": d.source,
                "route_type": d.route_type,
                "agent_id": d.agent_id,
                "workflow_id": d.workflow_id,
                "confidence": d.confidence,
                "cached": d.cached,
                "was_corrected": d.was_corrected,
                "corrected_agent_id": d.corrected_agent_id,
                "created_at": str(d.created_at),
            }
            for d in decisions
        ]
    except Exception as e:
        logger.error("Error listing routing decisions: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# POST /api/routing/rules
# ---------------------------------------------------------------------------


@router.post("/rules")
async def create_rule(
    body: CreateRuleRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Create a routing rule for the workspace."""
    try:
        if body.target_agent_id is None and body.target_workflow_id is None:
            raise HTTPException(
                status_code=400,
                detail="Either target_agent_id or target_workflow_id is required",
            )

        rule = RoutingRule(
            workspace_id=ctx.workspace_id,
            source_pattern=body.source_pattern,
            intent_keywords=body.intent_keywords,
            target_agent_id=body.target_agent_id,
            target_workflow_id=body.target_workflow_id,
            priority=body.priority,
            is_active=True,
        )
        db.add(rule)
        db.commit()
        db.refresh(rule)

        return {
            "id": rule.id,
            "workspace_id": str(rule.workspace_id),
            "source_pattern": rule.source_pattern,
            "intent_keywords": rule.intent_keywords,
            "target_agent_id": rule.target_agent_id,
            "target_workflow_id": rule.target_workflow_id,
            "priority": rule.priority,
            "is_active": rule.is_active,
            "created_at": str(rule.created_at),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating routing rule: %s", e)
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# GET /api/routing/rules
# ---------------------------------------------------------------------------


@router.get("/rules")
async def list_rules(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List all active routing rules for the workspace."""
    try:
        rules = (
            db.query(RoutingRule)
            .filter(
                RoutingRule.workspace_id == ctx.workspace_id,
                RoutingRule.is_active.is_(True),
            )
            .order_by(RoutingRule.priority.desc())
            .all()
        )

        return [
            {
                "id": r.id,
                "workspace_id": str(r.workspace_id),
                "source_pattern": r.source_pattern,
                "intent_keywords": r.intent_keywords,
                "target_agent_id": r.target_agent_id,
                "target_workflow_id": r.target_workflow_id,
                "priority": r.priority,
                "is_active": r.is_active,
                "created_at": str(r.created_at),
            }
            for r in rules
        ]
    except Exception as e:
        logger.error("Error listing routing rules: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# DELETE /api/routing/rules/{rule_id}
# ---------------------------------------------------------------------------


@router.delete("/rules/{rule_id}")
async def delete_rule(
    rule_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Soft-delete a routing rule (sets is_active=False)."""
    try:
        rule = (
            db.query(RoutingRule)
            .filter(
                RoutingRule.id == rule_id,
                RoutingRule.workspace_id == ctx.workspace_id,
            )
            .first()
        )

        if rule is None:
            raise HTTPException(status_code=404, detail="Routing rule not found")

        rule.is_active = False
        db.commit()

        return {"status": "deleted", "rule_id": rule_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting routing rule: %s", e)
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# POST /api/routing/corrections
# ---------------------------------------------------------------------------


@router.post("/corrections")
async def record_correction(
    body: CorrectionRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Record a user correction for a routing decision.

    Updates the routing_decisions table and feeds back into the cache.
    """
    try:
        # Find the original decision
        decision = (
            db.query(RoutingDecisionRecord)
            .filter(RoutingDecisionRecord.request_id == body.request_id)
            .first()
        )

        if decision is None:
            raise HTTPException(
                status_code=404, detail="Routing decision not found"
            )

        # Mark as corrected in DB
        decision.was_corrected = True
        decision.corrected_agent_id = body.correct_agent_id
        db.commit()

        # Feed correction into cache so routing improves immediately.
        # The cache tracks repeated corrections and auto-updates after 2+.
        if decision.content and decision.workspace_id and decision.source:
            try:
                source = ChannelSource(decision.source)
                get_routing_cache().record_correction(
                    workspace_id=decision.workspace_id,
                    content=decision.content,
                    source=source,
                    correct_agent_id=body.correct_agent_id,
                )
            except (ValueError, Exception) as e:
                logger.warning("Cache correction failed (non-blocking): %s", e)

        return {
            "status": "corrected",
            "request_id": str(body.request_id),
            "correct_agent_id": body.correct_agent_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error recording routing correction: %s", e)
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# GET /api/routing/cache/stats
# ---------------------------------------------------------------------------


@router.get("/cache/stats")
async def get_cache_stats(
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Return routing cache statistics: size, hits, misses, hit rate, top routes."""
    try:
        return get_routing_cache().stats()
    except Exception as e:
        logger.error("Error getting cache stats: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# POST /api/routing/triggers/setup
# ---------------------------------------------------------------------------


@router.post("/triggers/setup")
async def setup_trigger(
    body: TriggerSetupRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Set up a Jira trigger subscription via Composio.

    This is the API-based alternative to the ``scripts/setup_jira_trigger.py``
    management script.  It:

    1. Calls Composio SDK to subscribe to the given trigger.
    2. Creates a ``TriggerSubscription`` record in the database linking the
       trigger to an optional ``agent_id`` or ``workflow_id``.
    3. Returns the subscription details.

    **Webhook URL configuration:**
    After setup, ensure the callback URL is registered in your Composio
    dashboard (https://app.composio.dev → Project Settings → Webhooks).
    """
    try:
        from core.composio.client import get_composio_client
        from core.composio.entity_manager import EntityManager

        # Resolve entity for the workspace
        entity_manager = EntityManager(db)
        entity = entity_manager.get_or_create_entity(ctx.workspace_id)
        entity_id = entity["composio_entity_id"]
        entity_pk = entity["id"]

        # Build callback URL
        backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        callback_url = f"{backend_url}/api/composio/webhook"

        # Subscribe via Composio SDK
        client = get_composio_client()
        result = client.subscribe_to_trigger(
            entity_id=entity_id,
            trigger_name=body.trigger_name,
            callback_url=callback_url,
        )

        composio_subscription_id = result.get("id", "")

        # Persist subscription in database
        subscription = TriggerSubscription(
            entity_id=entity_pk,
            trigger_name=body.trigger_name,
            callback_url=callback_url,
            composio_subscription_id=str(composio_subscription_id),
            agent_id=body.agent_id,
            workflow_id=body.workflow_id,
            is_active=True,
        )
        db.add(subscription)
        db.commit()
        db.refresh(subscription)

        return {
            "status": "subscribed",
            "subscription_id": subscription.id,
            "trigger_name": body.trigger_name,
            "project_key": body.project_key,
            "composio_subscription_id": composio_subscription_id,
            "callback_url": callback_url,
            "agent_id": body.agent_id,
            "workflow_id": body.workflow_id,
            "next_steps": (
                "Verify webhook URL in Composio dashboard: "
                "https://app.composio.dev → Project Settings → Webhooks. "
                f"Set webhook URL to: {callback_url}"
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error setting up trigger: %s", e)
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")

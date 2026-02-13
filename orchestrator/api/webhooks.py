"""
General Webhook Endpoints
=========================

Two webhook paths:
1. POST /api/webhooks/ws/{workspace_key}  — General workspace webhook
   Routes incoming requests through UniversalRouter to the right agent.
   No auth required — the workspace_key in the URL is the credential.

2. POST /api/webhooks/recipe/{webhook_id} — Recipe-specific webhook
   (Defined in workflow_recipes.py, registered separately.)
"""

import asyncio
import logging
from typing import Any, Dict, Set
from uuid import UUID, uuid4

from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.orm import Session

from core.database.database import get_db
from core.models.workspaces import Workspace
from core.routing.cache import get_routing_cache
from core.routing.engine import UniversalRouter
from core.routing.ingestors.webhook import WebhookIngestor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/webhooks", tags=["webhooks"])

# Track background tasks to prevent GC collection
_background_tasks: Set[asyncio.Task] = set()


@router.get("/ws/{workspace_key}")
async def workspace_webhook_verify(
    workspace_key: str,
    db: Session = Depends(get_db),
):
    """Verification endpoint — external services send GET to validate the URL exists."""
    workspace = db.query(Workspace).filter(
        Workspace.webhook_key == workspace_key,
        Workspace.is_active == True,
    ).first()

    if not workspace:
        raise HTTPException(status_code=404, detail="Unknown webhook")

    return {"status": "ok"}


@router.post("/ws/{workspace_key}")
async def general_workspace_webhook(
    workspace_key: str,
    body: Dict[str, Any] = Body(default={}),
    db: Session = Depends(get_db),
):
    """
    General workspace webhook — routes incoming requests to the right agent.

    The workspace_key in the URL is the credential (URL-as-secret pattern).
    No authentication required.

    Body (JSON):
    - message / text / content: The message to route
    - agent_id: Optional explicit agent override (Tier-0)
    - source / channel: Optional metadata for routing rules
    - Any other JSON fields are preserved in metadata
    """
    # 1. Look up workspace by webhook_key
    workspace = db.query(Workspace).filter(
        Workspace.webhook_key == workspace_key,
        Workspace.is_active == True,
    ).first()

    if not workspace:
        raise HTTPException(status_code=404, detail="Unknown webhook")

    # 2. Build RequestEnvelope via WebhookIngestor
    ingestor = WebhookIngestor()
    envelope = ingestor.ingest(
        body=body,
        workspace_id=workspace.id,
    )

    # 3. Route through UniversalRouter
    universal_router = UniversalRouter(db, cache=get_routing_cache())
    try:
        decision = await universal_router.route(envelope)
    except Exception:
        logger.exception("[webhook/ws] Router failed for workspace %s", workspace.id)
        decision = None

    if decision is None:
        return {
            "status": "received",
            "routed": False,
            "reason": "No route found — configure routing rules or add agents to your workspace.",
        }

    # 4. Dispatch based on routing decision
    if decision.route_type == "agent" and decision.agent_id is not None:
        # Execute agent synchronously for webhook callers who want a response
        try:
            result = await _execute_agent_sync(
                agent_id=decision.agent_id,
                content=envelope.content,
                metadata=envelope.metadata,
                workspace_id=workspace.id,
            )
            return {
                "status": "completed",
                "routed": True,
                "route_type": "agent",
                "agent_id": decision.agent_id,
                "confidence": decision.confidence,
                "result": result,
            }
        except Exception:
            logger.exception("[webhook/ws] Agent %d execution failed", decision.agent_id)
            return {
                "status": "error",
                "routed": True,
                "route_type": "agent",
                "agent_id": decision.agent_id,
                "error": "Agent execution failed",
            }

    elif decision.route_type == "workflow" and decision.workflow_id is not None:
        # Dispatch workflow/recipe async, return execution_id
        execution_id = await _dispatch_workflow_async(
            workflow_id=decision.workflow_id,
            envelope=envelope,
            db=db,
        )
        return {
            "status": "dispatched",
            "routed": True,
            "route_type": "workflow",
            "workflow_id": decision.workflow_id,
            "execution_id": execution_id,
            "confidence": decision.confidence,
        }

    # Orchestrate / unknown route_type
    return {
        "status": "received",
        "routed": True,
        "route_type": decision.route_type,
        "confidence": decision.confidence,
        "reasoning": decision.reasoning[:200] if decision.reasoning else "",
    }


# =============================================================================
# Dispatch Helpers
# =============================================================================

async def _execute_agent_sync(
    agent_id: int,
    content: str,
    metadata: Dict[str, Any],
    workspace_id: UUID,
) -> Dict[str, Any]:
    """Execute an agent synchronously and return the result."""
    from modules.agents.factory.agent_factory import AgentFactory

    db = next(get_db())
    try:
        factory = AgentFactory(db_session=db)
        result = await factory.execute_with_prompt(
            agent=agent_id,
            prompt=content,
            context=metadata,
        )
        return result
    finally:
        db.close()


async def _dispatch_workflow_async(
    workflow_id: int,
    envelope,
    db: Session,
) -> str:
    """Dispatch a workflow/recipe execution asynchronously, return execution_id."""
    from core.models.core import RecipeExecution
    from core.models import WorkflowTemplate as WorkflowRecipe
    from api.recipe_executor import execute_recipe_direct
    from datetime import datetime, timezone

    recipe = db.query(WorkflowRecipe).filter(
        WorkflowRecipe.id == workflow_id
    ).first()

    if not recipe or not recipe.steps:
        return "no_recipe_found"

    execution_id = f"ws-webhook-{uuid4().hex[:12]}"
    execution = RecipeExecution(
        execution_id=execution_id,
        recipe_id=recipe.id,
        workspace_id=envelope.workspace_id,
        status="pending",
        input_data={"content": envelope.content, "metadata": envelope.metadata},
        triggered_by="workspace_webhook",
        execution_metadata={
            "execution_type": "workspace_webhook",
            "total_steps": len(recipe.steps),
        },
    )
    db.add(execution)
    recipe.use_count += 1
    recipe.last_used_at = datetime.now(timezone.utc)
    db.commit()

    task = asyncio.create_task(
        execute_recipe_direct(
            recipe_execution_id=execution_id,
            recipe_id=recipe.id,
            workspace_id=envelope.workspace_id,
            input_data=execution.input_data,
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return execution_id

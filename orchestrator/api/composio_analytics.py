"""
Composio Analytics API (PRD-54 / US-003)
=========================================

Aggregated views of Composio app connections, action usage,
and per-agent tool mappings scoped to the current workspace.
"""

import logging
from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.core import Agent
from core.models.composio import ComposioEntity, ComposioConnection, AgentAppFeature

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analytics/composio", tags=["Composio Analytics"])


# ── Pydantic schemas ─────────────────────────────────────────────────

class ConnectedAppStats(BaseModel):
    app_name: str
    status: str
    total_actions_used: int
    agent_count: int
    documents_synced: int
    last_used_at: Optional[datetime] = None


class ActionLeaderboardEntry(BaseModel):
    action_name: str
    app_name: str
    total_usage_count: int
    agent_count: int
    last_used_at: Optional[datetime] = None


class AgentToolEntry(BaseModel):
    tool_name: str
    app_name: str
    usage_count: int
    enabled: bool


class AgentToolMapping(BaseModel):
    agent_id: int
    agent_name: str
    tools: List[AgentToolEntry]


# ── Helpers ───────────────────────────────────────────────────────────

DAYS_MAP = {7: 7, 30: 30, 90: 90}


def _since(days: int) -> datetime:
    d = DAYS_MAP.get(days, 30)
    return datetime.utcnow() - timedelta(days=d)


# ── Endpoints ─────────────────────────────────────────────────────────

@router.get("/apps", response_model=List[ConnectedAppStats])
async def get_connected_apps(
    days: int = Query(30, description="7|30|90"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Connected apps with action counts, agent count, and documents synced."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    since = _since(days)

    # Get the entity for this workspace
    entity = (
        db.query(ComposioEntity)
        .filter(ComposioEntity.workspace_id == ctx.workspace_id)
        .first()
    )
    if not entity:
        return []

    # Get all connections for this workspace's entity
    connections = (
        db.query(ComposioConnection)
        .filter(ComposioConnection.entity_id == entity.id)
        .all()
    )

    # Get workspace agent IDs for filtering AgentAppFeature
    agent_ids = [
        r[0]
        for r in db.query(Agent.id)
        .filter(Agent.workspace_id == ctx.workspace_id)
        .all()
    ]

    results: List[ConnectedAppStats] = []
    for conn in connections:
        # Aggregate action usage for this app across workspace agents
        if agent_ids:
            action_agg = (
                db.query(
                    func.sum(AgentAppFeature.usage_count).label("total_usage"),
                    func.count(func.distinct(AgentAppFeature.agent_id)).label("agent_count"),
                    func.max(AgentAppFeature.last_used_at).label("last_used"),
                )
                .filter(
                    AgentAppFeature.agent_id.in_(agent_ids),
                    AgentAppFeature.app_name == conn.app_name,
                    AgentAppFeature.created_at >= since,
                )
                .first()
            )
            total_usage = int(action_agg.total_usage or 0) if action_agg else 0
            agent_count = int(action_agg.agent_count or 0) if action_agg else 0
            last_used = action_agg.last_used if action_agg else None
        else:
            total_usage = 0
            agent_count = 0
            last_used = None

        results.append(
            ConnectedAppStats(
                app_name=conn.app_name,
                status=conn.status,
                total_actions_used=total_usage,
                agent_count=agent_count,
                documents_synced=conn.total_documents_synced or 0,
                last_used_at=last_used,
            )
        )

    return results


@router.get("/actions", response_model=List[ActionLeaderboardEntry])
async def get_action_leaderboard(
    days: int = Query(30, description="7|30|90"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Action leaderboard sorted by usage count descending."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    since = _since(days)

    # Get workspace agent IDs
    agent_ids = [
        r[0]
        for r in db.query(Agent.id)
        .filter(Agent.workspace_id == ctx.workspace_id)
        .all()
    ]
    if not agent_ids:
        return []

    rows = (
        db.query(
            AgentAppFeature.action_name,
            AgentAppFeature.app_name,
            func.sum(AgentAppFeature.usage_count).label("total_usage"),
            func.count(func.distinct(AgentAppFeature.agent_id)).label("agent_count"),
            func.max(AgentAppFeature.last_used_at).label("last_used"),
        )
        .filter(
            AgentAppFeature.agent_id.in_(agent_ids),
            AgentAppFeature.created_at >= since,
        )
        .group_by(AgentAppFeature.action_name, AgentAppFeature.app_name)
        .order_by(desc("total_usage"))
        .all()
    )

    return [
        ActionLeaderboardEntry(
            action_name=r.action_name,
            app_name=r.app_name,
            total_usage_count=int(r.total_usage or 0),
            agent_count=int(r.agent_count or 0),
            last_used_at=r.last_used,
        )
        for r in rows
    ]


@router.get("/agent-tools", response_model=List[AgentToolMapping])
async def get_agent_tools(
    days: int = Query(30, description="7|30|90"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Per-agent tool mapping with usage counts."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    since = _since(days)

    # Get workspace agents that have Composio features
    agents = (
        db.query(Agent)
        .filter(Agent.workspace_id == ctx.workspace_id)
        .all()
    )
    if not agents:
        return []

    agent_map = {a.id: a.name for a in agents}
    agent_ids = list(agent_map.keys())

    features = (
        db.query(AgentAppFeature)
        .filter(
            AgentAppFeature.agent_id.in_(agent_ids),
            AgentAppFeature.created_at >= since,
        )
        .order_by(AgentAppFeature.agent_id, desc(AgentAppFeature.usage_count))
        .all()
    )

    # Group by agent
    from collections import defaultdict

    by_agent: dict[int, list[AgentToolEntry]] = defaultdict(list)
    for f in features:
        by_agent[f.agent_id].append(
            AgentToolEntry(
                tool_name=f.action_name,
                app_name=f.app_name,
                usage_count=f.usage_count or 0,
                enabled=f.enabled,
            )
        )

    return [
        AgentToolMapping(
            agent_id=aid,
            agent_name=agent_map.get(aid, f"Agent {aid}"),
            tools=tools,
        )
        for aid, tools in by_agent.items()
        if tools  # Only include agents that have tools
    ]

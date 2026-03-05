"""
Heartbeat API Endpoints (PRD-55 US-008)
========================================

Manage and monitor heartbeat ticks for orchestrator and agents.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import desc, func, text
from sqlalchemy.orm import Session

from core.database.database import get_db
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/heartbeat", tags=["heartbeat"])


# ── Heartbeat Config Schema ───────────────────────────────────────

class HeartbeatConfigPayload(BaseModel):
    enabled: bool = False
    interval_minutes: int = Field(60, ge=5, le=1440)
    inherit_active_hours: bool = True
    active_hours_start: str = "08:00"
    active_hours_end: str = "20:00"
    prompt: str = ""
    auto_act: bool = False
    report_to: str = "orchestrator"  # orchestrator | direct | channel:<id> | webhook
    webhook_url: Optional[str] = None
    channel_id: Optional[str] = None


# ── Agent Heartbeat Config CRUD ───────────────────────────────────

def _verify_agent_ownership(agent_id: int, workspace_id: str, db: Session):
    """Verify agent belongs to workspace. Raises HTTPException on failure."""
    row = db.execute(
        text("SELECT workspace_id FROM agents WHERE id = :aid"),
        {"aid": agent_id},
    ).fetchone()
    if not row:
        raise HTTPException(404, f"Agent {agent_id} not found")
    if str(row.workspace_id) != str(workspace_id):
        raise HTTPException(403, "Agent does not belong to this workspace")


@router.get("/agents/{agent_id}/config")
async def get_agent_heartbeat_config(
    agent_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get heartbeat config for an agent."""
    _verify_agent_ownership(agent_id, str(ctx.workspace_id), db)

    row = db.execute(
        text("SELECT configuration FROM agents WHERE id = :aid"),
        {"aid": agent_id},
    ).fetchone()

    config = row.configuration if row and row.configuration else {}
    hb = config.get("heartbeat", {})
    return {
        "enabled": hb.get("enabled", False),
        "interval_minutes": hb.get("interval_minutes", 60),
        "inherit_active_hours": hb.get("inherit_active_hours", True),
        "active_hours_start": hb.get("active_hours_start", "08:00"),
        "active_hours_end": hb.get("active_hours_end", "20:00"),
        "prompt": hb.get("prompt", ""),
        "auto_act": hb.get("auto_act", False),
        "report_to": hb.get("report_to", "orchestrator"),
        "webhook_url": hb.get("webhook_url"),
        "channel_id": hb.get("channel_id"),
    }


@router.put("/agents/{agent_id}/config")
async def save_agent_heartbeat_config(
    agent_id: int,
    payload: HeartbeatConfigPayload,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Save heartbeat config for an agent, updating agent.configuration.heartbeat."""
    _verify_agent_ownership(agent_id, str(ctx.workspace_id), db)

    # Read current configuration (immutable pattern — build new dict)
    row = db.execute(
        text("SELECT configuration FROM agents WHERE id = :aid"),
        {"aid": agent_id},
    ).fetchone()

    current_config = dict(row.configuration) if row and row.configuration else {}
    new_config = {**current_config, "heartbeat": payload.dict()}

    db.execute(
        text("UPDATE agents SET configuration = :cfg WHERE id = :aid"),
        {"cfg": json.dumps(new_config), "aid": agent_id},
    )
    db.commit()

    # Reschedule or unschedule the heartbeat job
    try:
        from services.heartbeat_service import get_heartbeat_service
        service = get_heartbeat_service()
        if payload.enabled:
            service.schedule_agent_heartbeat(
                agent_id, str(ctx.workspace_id), payload.dict()
            )
        else:
            service.unschedule_heartbeat(f"agent_hb_{agent_id}")
    except Exception as e:
        logger.warning("Failed to update heartbeat schedule for agent %s: %s", agent_id, e)

    return {"ok": True}


@router.get("/agents/{agent_id}/last")
async def get_agent_last_heartbeat(
    agent_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get the most recent heartbeat result for an agent."""
    _verify_agent_ownership(agent_id, str(ctx.workspace_id), db)

    row = db.execute(
        text("""
            SELECT id, status, findings, actions_taken, tokens_used, cost, created_at
            FROM heartbeat_results
            WHERE workspace_id = :ws_id AND source_type = 'agent' AND source_id = :agent_id
            ORDER BY created_at DESC
            LIMIT 1
        """),
        {"ws_id": str(ctx.workspace_id), "agent_id": str(agent_id)},
    ).fetchone()

    if not row:
        return None

    return {
        "id": row.id,
        "status": row.status,
        "findings": row.findings,
        "actions_taken": row.actions_taken,
        "tokens_used": row.tokens_used,
        "cost": row.cost,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


# ── Orchestrator Heartbeat ─────────────────────────────────────────

@router.post("/orchestrator/run")
async def run_orchestrator_heartbeat(
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Trigger an immediate orchestrator heartbeat tick."""
    try:
        from services.heartbeat_service import get_heartbeat_service
        service = get_heartbeat_service()
        result = await service.run_orchestrator_heartbeat(str(ctx.workspace_id))
        return result
    except Exception as e:
        logger.error("Failed to run orchestrator heartbeat: %s", e)
        raise HTTPException(500, "Internal server error")


@router.get("/orchestrator/history")
async def get_orchestrator_heartbeat_history(
    limit: int = Query(20, ge=1, le=100),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List recent orchestrator heartbeat results."""
    rows = db.execute(
        text("""
            SELECT id, source_type, source_id, status, findings, actions_taken, tokens_used, cost, created_at
            FROM heartbeat_results
            WHERE workspace_id = :ws_id AND source_type = 'orchestrator'
            ORDER BY created_at DESC
            LIMIT :lim
        """),
        {"ws_id": str(ctx.workspace_id), "lim": limit}
    ).fetchall()

    return [
        {
            "id": r.id,
            "status": r.status,
            "findings": r.findings,
            "actions_taken": r.actions_taken,
            "tokens_used": r.tokens_used,
            "cost": r.cost,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


# ── Agent Heartbeat ────────────────────────────────────────────────

@router.post("/agents/{agent_id}/run")
async def run_agent_heartbeat(
    agent_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Trigger an immediate heartbeat tick for a specific agent."""
    _verify_agent_ownership(agent_id, str(ctx.workspace_id), db)

    try:
        from services.heartbeat_service import get_heartbeat_service
        service = get_heartbeat_service()
        result = await service.run_agent_heartbeat(agent_id)
        return result
    except Exception as e:
        logger.error("Failed to run agent heartbeat for %s: %s", agent_id, e)
        raise HTTPException(500, "Internal server error")


@router.get("/agents/{agent_id}/history")
async def get_agent_heartbeat_history(
    agent_id: int,
    limit: int = Query(20, ge=1, le=100),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List recent heartbeat results for a specific agent."""
    rows = db.execute(
        text("""
            SELECT id, status, findings, actions_taken, tokens_used, cost, created_at
            FROM heartbeat_results
            WHERE workspace_id = :ws_id AND source_type = 'agent' AND source_id = :agent_id
            ORDER BY created_at DESC
            LIMIT :lim
        """),
        {"ws_id": str(ctx.workspace_id), "agent_id": str(agent_id), "lim": limit}
    ).fetchall()

    return [
        {
            "id": r.id,
            "status": r.status,
            "findings": r.findings,
            "actions_taken": r.actions_taken,
            "tokens_used": r.tokens_used,
            "cost": r.cost,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


# ── Global Status ──────────────────────────────────────────────────

@router.get("/status")
async def get_heartbeat_status(
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Return status of all active heartbeat schedules."""
    try:
        from services.heartbeat_service import get_heartbeat_service
        service = get_heartbeat_service()
        return service.get_status()
    except Exception as e:
        logger.error("Failed to get heartbeat status: %s", e)
        return {"active": False, "jobs": [], "error": "Failed to retrieve heartbeat status"}


# ── Analytics ──────────────────────────────────────────────────────

@router.get("/analytics")
async def get_heartbeat_analytics(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Return heartbeat summary stats for analytics dashboard (US-010)."""
    from datetime import datetime, timedelta

    ws_id = str(ctx.workspace_id)
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)

    try:
        # Today's stats
        today_stats = db.execute(
            text("""
                SELECT
                    COUNT(*) as total,
                    SUM(tokens_used) as total_tokens,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successes,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors
                FROM heartbeat_results
                WHERE workspace_id = :ws_id AND created_at >= :start
            """),
            {"ws_id": ws_id, "start": today_start}
        ).fetchone()

        # Recent events (last 24h)
        recent = db.execute(
            text("""
                SELECT id, source_type, source_id, status, tokens_used, created_at
                FROM heartbeat_results
                WHERE workspace_id = :ws_id AND created_at >= :start
                ORDER BY created_at DESC
                LIMIT 20
            """),
            {"ws_id": ws_id, "start": yesterday_start}
        ).fetchall()

        return {
            "today": {
                "total_heartbeats": today_stats.total or 0,
                "total_tokens": today_stats.total_tokens or 0,
                "successes": today_stats.successes or 0,
                "errors": today_stats.errors or 0,
            },
            "recent_events": [
                {
                    "id": r.id,
                    "source_type": r.source_type,
                    "source_id": r.source_id,
                    "status": r.status,
                    "tokens_used": r.tokens_used,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in recent
            ],
        }
    except Exception as e:
        logger.error("Failed to get heartbeat analytics: %s", e)
        return {"today": {"total_heartbeats": 0, "total_tokens": 0}, "recent_events": []}

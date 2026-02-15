"""
Heartbeat API Endpoints (PRD-55 US-008)
========================================

Manage and monitor heartbeat ticks for orchestrator and agents.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, func, text
from sqlalchemy.orm import Session

from core.database.database import get_db
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/heartbeat", tags=["heartbeat"])


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
        raise HTTPException(500, f"Heartbeat failed: {str(e)}")


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
    # Verify agent belongs to the requesting workspace
    row = db.execute(
        text("SELECT workspace_id FROM agents WHERE id = :aid"),
        {"aid": agent_id},
    ).fetchone()
    if not row:
        raise HTTPException(404, f"Agent {agent_id} not found")
    if str(row.workspace_id) != str(ctx.workspace_id):
        logger.warning(
            "Unauthorized heartbeat attempt: agent %s belongs to %s, not %s",
            agent_id, row.workspace_id, ctx.workspace_id,
        )
        raise HTTPException(403, "Agent does not belong to this workspace")

    try:
        from services.heartbeat_service import get_heartbeat_service
        service = get_heartbeat_service()
        result = await service.run_agent_heartbeat(agent_id)
        return result
    except Exception as e:
        logger.error("Failed to run agent heartbeat for %s: %s", agent_id, e)
        raise HTTPException(500, f"Heartbeat failed: {str(e)}")


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
        return {"active": False, "jobs": [], "error": str(e)}


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

"""
Heartbeat API Endpoints (PRD-55 US-008, PRD-72 US-003)
=======================================================

Manage and monitor heartbeat ticks for orchestrator and agents.
PRD-72 adds workspace listing, toggle, and execution history endpoints.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import desc, func, text
from sqlalchemy.orm import Session

from core.database.database import get_db
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from core.auth.super_admin import require_super_admin
from core.models import Agent

logger = logging.getLogger(__name__)

# PRD-143 S6: observability tier — router-wide super-admin lock (fail-closed).
router = APIRouter(
    prefix="/api/heartbeat",
    tags=["heartbeat"],
    dependencies=[Depends(require_super_admin)],
)


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


# ── PRD-72: Activity Command Centre Endpoints ─────────────────────

@router.get("/workspace")
async def list_workspace_heartbeats(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List all heartbeat configurations for the current workspace.

    Returns agent heartbeats with their config, last run, and status.
    Used by the Activity Command Centre Routines tab (PRD-72).
    """
    ws_id = str(ctx.workspace_id)

    try:
        # Get all agents in this workspace
        agents = (
            db.query(Agent)
            .filter(Agent.workspace_id == ctx.workspace_id)
            .all()
        )

        heartbeats: List[Dict[str, Any]] = []
        for agent in agents:
            cfg = agent.configuration or {}
            hb = cfg.get("heartbeat", {})
            if not hb:
                continue

            # Get last execution for this agent heartbeat
            last_run_row = db.execute(
                text("""
                    SELECT id, status, findings, tokens_used, cost, created_at
                    FROM heartbeat_results
                    WHERE workspace_id = :ws_id
                      AND source_type = 'agent'
                      AND source_id = :agent_id
                    ORDER BY created_at DESC
                    LIMIT 1
                """),
                {"ws_id": ws_id, "agent_id": str(agent.id)},
            ).fetchone()

            # Get next run time from scheduler if available
            next_run_at = None
            try:
                from services.heartbeat_service import get_heartbeat_service
                service = get_heartbeat_service()
                if service._scheduler:
                    job = service._scheduler.get_job(f"agent_hb_{agent.id}")
                    if job and job.next_run_time:
                        next_run_at = job.next_run_time.isoformat()
            except Exception:
                pass  # Scheduler may not be running in all environments

            heartbeats.append({
                "id": agent.id,
                "agent_id": agent.id,
                "agent_name": agent.name,
                "agent_description": agent.description,
                "enabled": hb.get("enabled", False),
                "interval_minutes": hb.get("interval_minutes", 60),
                "prompt": hb.get("prompt", ""),
                "auto_act": hb.get("auto_act", False),
                "timezone": hb.get("timezone", "UTC"),
                "active_hours_start": hb.get("active_hours_start", "08:00"),
                "active_hours_end": hb.get("active_hours_end", "20:00"),
                "last_run": {
                    "status": last_run_row.status,
                    "created_at": last_run_row.created_at.isoformat() if last_run_row.created_at else None,
                    "tokens_used": last_run_row.tokens_used,
                } if last_run_row else None,
                "next_run_at": next_run_at,
            })

        return {"heartbeats": heartbeats, "total": len(heartbeats)}

    except Exception as e:
        logger.error("Failed to list workspace heartbeats: %s", e, exc_info=True)
        raise HTTPException(500, "Internal server error")


@router.patch("/{heartbeat_id}/toggle")
async def toggle_heartbeat(
    heartbeat_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Toggle a heartbeat's enabled state (pause/resume).

    heartbeat_id is the agent ID whose heartbeat config should be toggled.
    """
    agent = (
        db.query(Agent)
        .filter(Agent.id == heartbeat_id, Agent.workspace_id == ctx.workspace_id)
        .first()
    )
    if not agent:
        raise HTTPException(404, f"Agent {heartbeat_id} not found in this workspace")

    cfg = agent.configuration or {}
    hb = cfg.get("heartbeat")
    if not hb:
        raise HTTPException(404, f"Agent {heartbeat_id} has no heartbeat configuration")

    # Toggle the enabled flag — use immutable update
    new_enabled = not hb.get("enabled", False)
    updated_hb = {**hb, "enabled": new_enabled}
    updated_cfg = {**cfg, "heartbeat": updated_hb}

    # SQLAlchemy needs the column reassigned to detect JSON mutation
    agent.configuration = updated_cfg
    db.commit()
    db.refresh(agent)

    # Schedule or unschedule in the heartbeat service
    try:
        from services.heartbeat_service import get_heartbeat_service
        service = get_heartbeat_service()
        job_id = f"agent_hb_{agent.id}"
        if new_enabled:
            service.schedule_agent_heartbeat(
                agent.id, str(agent.workspace_id), updated_hb
            )
        else:
            service.unschedule_heartbeat(job_id)
    except Exception as sched_err:
        logger.warning("Failed to update heartbeat schedule for agent %s: %s", heartbeat_id, sched_err)

    return {
        "id": agent.id,
        "agent_id": agent.id,
        "agent_name": agent.name,
        "enabled": new_enabled,
        "interval_minutes": updated_hb.get("interval_minutes", 60),
        "message": f"Heartbeat {'enabled' if new_enabled else 'paused'} for {agent.name}",
    }


@router.get("/{heartbeat_id}/executions")
async def get_heartbeat_executions(
    heartbeat_id: int,
    limit: int = Query(10, ge=1, le=100),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get execution history for a specific agent's heartbeat.

    heartbeat_id is the agent ID. Returns the last N executions
    from heartbeat_results ordered by most recent first.
    """
    ws_id = str(ctx.workspace_id)

    # Verify agent belongs to this workspace
    agent = (
        db.query(Agent)
        .filter(Agent.id == heartbeat_id, Agent.workspace_id == ctx.workspace_id)
        .first()
    )
    if not agent:
        raise HTTPException(404, f"Agent {heartbeat_id} not found in this workspace")

    try:
        rows = db.execute(
            text("""
                SELECT id, status, findings, actions_taken, tokens_used, cost, created_at
                FROM heartbeat_results
                WHERE workspace_id = :ws_id
                  AND source_type = 'agent'
                  AND source_id = :agent_id
                ORDER BY created_at DESC
                LIMIT :lim
            """),
            {"ws_id": ws_id, "agent_id": str(heartbeat_id), "lim": limit},
        ).fetchall()

        executions = []
        for r in rows:
            findings = r.findings
            if isinstance(findings, str):
                findings = json.loads(findings)

            # Extract error message from findings if status is error
            error_message = None
            if r.status == "error" and findings:
                for f in findings:
                    if f.get("check") in ("error", "llm_error"):
                        error_message = f.get("detail", "")[:500]
                        break

            executions.append({
                "id": r.id,
                "status": r.status,
                "started_at": r.created_at.isoformat() if r.created_at else None,
                "completed_at": r.created_at.isoformat() if r.created_at else None,
                "duration_seconds": None,  # Not tracked separately; heartbeats are fast
                "tokens_used": r.tokens_used,
                "cost": r.cost,
                "error_message": error_message,
                "findings_count": len(findings) if findings else 0,
            })

        return {
            "agent_id": heartbeat_id,
            "agent_name": agent.name,
            "executions": executions,
            "total": len(executions),
        }

    except Exception as e:
        logger.error("Failed to get heartbeat executions for agent %s: %s", heartbeat_id, e, exc_info=True)
        raise HTTPException(500, "Internal server error")

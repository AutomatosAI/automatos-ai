"""
Heartbeat API endpoints (PRD-55: Autonomous Assistant Platform).

Provides manual triggers, history queries, and analytics for
orchestrator and agent heartbeat checks.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/heartbeat", tags=["Heartbeat"])


# ---------------------------------------------------------------------------
# Orchestrator endpoints
# ---------------------------------------------------------------------------


@router.post("/orchestrator/run")
async def run_orchestrator_heartbeat(
    workspace_id: str = Query(..., description="Workspace UUID"),
) -> Dict[str, Any]:
    """Trigger an immediate orchestrator heartbeat."""
    from services.heartbeat_service import get_heartbeat_service

    svc = get_heartbeat_service()
    try:
        result = await svc.run_orchestrator_heartbeat(workspace_id)
        return {"ok": True, "result": result}
    except Exception as exc:
        logger.error("Orchestrator heartbeat failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/orchestrator/history")
async def get_orchestrator_history(
    workspace_id: str = Query(..., description="Workspace UUID"),
    limit: int = Query(20, ge=1, le=100),
) -> Dict[str, Any]:
    """List recent orchestrator heartbeat results."""
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        rows = db.execute(
            text(
                "SELECT id, status, findings, actions_taken, tokens_used, cost, created_at "
                "FROM heartbeat_results "
                "WHERE source_type = 'orchestrator' AND workspace_id = :wid::uuid "
                "ORDER BY created_at DESC LIMIT :lim"
            ),
            {"wid": workspace_id, "lim": limit},
        ).fetchall()

        results = [
            {
                "id": r[0],
                "status": r[1],
                "findings": r[2],
                "actions_taken": r[3],
                "tokens_used": r[4],
                "cost": r[5],
                "created_at": r[6].isoformat() if r[6] else None,
            }
            for r in rows
        ]
        return {"ok": True, "results": results}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Agent endpoints
# ---------------------------------------------------------------------------


@router.post("/agents/{agent_id}/run")
async def run_agent_heartbeat(
    agent_id: int,
    workspace_id: str = Query(..., description="Workspace UUID"),
) -> Dict[str, Any]:
    """Trigger an immediate agent heartbeat."""
    from services.heartbeat_service import get_heartbeat_service

    svc = get_heartbeat_service()
    try:
        result = await svc.run_agent_heartbeat(str(agent_id), workspace_id)
        return {"ok": True, "result": result}
    except Exception as exc:
        logger.error("Agent heartbeat failed for agent %s: %s", agent_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/agents/{agent_id}/history")
async def get_agent_history(
    agent_id: int,
    workspace_id: str = Query(..., description="Workspace UUID"),
    limit: int = Query(20, ge=1, le=100),
) -> Dict[str, Any]:
    """List recent heartbeat results for a specific agent."""
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        rows = db.execute(
            text(
                "SELECT id, status, findings, actions_taken, tokens_used, cost, created_at "
                "FROM heartbeat_results "
                "WHERE source_type = 'agent' AND source_id = :aid "
                "AND workspace_id = :wid::uuid "
                "ORDER BY created_at DESC LIMIT :lim"
            ),
            {"aid": str(agent_id), "wid": workspace_id, "lim": limit},
        ).fetchall()

        results = [
            {
                "id": r[0],
                "status": r[1],
                "findings": r[2],
                "actions_taken": r[3],
                "tokens_used": r[4],
                "cost": r[5],
                "created_at": r[6].isoformat() if r[6] else None,
            }
            for r in rows
        ]
        return {"ok": True, "results": results}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Status & Analytics
# ---------------------------------------------------------------------------


@router.get("/status")
async def get_heartbeat_status(
    workspace_id: str = Query(..., description="Workspace UUID"),
) -> Dict[str, Any]:
    """All active heartbeats with next_run_at, last_run_at."""
    from services.heartbeat_service import get_heartbeat_service
    from core.database.database import SessionLocal

    svc = get_heartbeat_service()
    scheduler_status = svc.get_status()

    # Enrich with last_run_at from DB — scoped to workspace
    db = SessionLocal()
    try:
        filtered_jobs = []
        for job in scheduler_status.get("jobs", []):
            job_id = job.get("id", "")
            # Determine source_type and source_id from job_id
            if job_id.startswith("orchestrator_heartbeat_"):
                source_type = "orchestrator"
                source_id = job_id.replace("orchestrator_heartbeat_", "")
                # source_id is the workspace_id for orchestrator jobs
                if source_id != workspace_id:
                    continue
            elif job_id.startswith("agent_heartbeat_"):
                source_type = "agent"
                source_id = job_id.replace("agent_heartbeat_", "")
                # Check agent belongs to this workspace
                agent_row = db.execute(
                    text("SELECT workspace_id FROM agents WHERE id = :aid"),
                    {"aid": source_id},
                ).fetchone()
                if not agent_row or str(agent_row[0]) != workspace_id:
                    continue
            else:
                continue

            row = db.execute(
                text(
                    "SELECT created_at FROM heartbeat_results "
                    "WHERE source_type = :st AND source_id = :sid "
                    "AND workspace_id = :wid::uuid "
                    "ORDER BY created_at DESC LIMIT 1"
                ),
                {"st": source_type, "sid": source_id, "wid": workspace_id},
            ).fetchone()
            job["last_run_at"] = row[0].isoformat() if row else None
            filtered_jobs.append(job)

        scheduler_status["jobs"] = filtered_jobs
    finally:
        db.close()

    return {"ok": True, **scheduler_status}


@router.get("/analytics")
async def get_heartbeat_analytics(
    workspace_id: str = Query(..., description="Workspace UUID"),
) -> Dict[str, Any]:
    """Today's heartbeat stats: total runs, tokens, successes, errors, recent events."""
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        # Today's aggregates
        stats_row = db.execute(
            text(
                "SELECT "
                "  COUNT(*) as total, "
                "  COALESCE(SUM(tokens_used), 0) as tokens, "
                "  COUNT(*) FILTER (WHERE status = 'success') as successes, "
                "  COUNT(*) FILTER (WHERE status = 'error') as errors "
                "FROM heartbeat_results "
                "WHERE workspace_id = :wid::uuid "
                "AND created_at >= CURRENT_DATE"
            ),
            {"wid": workspace_id},
        ).fetchone()

        # Recent events (last 10)
        recent_rows = db.execute(
            text(
                "SELECT id, source_type, source_id, status, tokens_used, cost, created_at "
                "FROM heartbeat_results "
                "WHERE workspace_id = :wid::uuid "
                "ORDER BY created_at DESC LIMIT 10"
            ),
            {"wid": workspace_id},
        ).fetchall()

        recent = [
            {
                "id": r[0],
                "source_type": r[1],
                "source_id": r[2],
                "status": r[3],
                "tokens_used": r[4],
                "cost": r[5],
                "created_at": r[6].isoformat() if r[6] else None,
            }
            for r in recent_rows
        ]

        return {
            "ok": True,
            "today": {
                "total": stats_row[0] if stats_row else 0,
                "tokens": stats_row[1] if stats_row else 0,
                "successes": stats_row[2] if stats_row else 0,
                "errors": stats_row[3] if stats_row else 0,
            },
            "recent": recent,
        }
    finally:
        db.close()

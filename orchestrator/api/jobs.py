"""A workspace's dead jobs: list, retry, clear (PRD-276 §6.8).

A job becomes ``dead`` when its attempts are used up. Its workspace's managers (``workspace:manage``:
a retry can send to systems outside Automatos again) list them, re-queue one with its attempts reset,
or delete it. Only the caller's own workspace's jobs are reachable; anything else is a 404. Plain
``def`` routes: FastAPI runs them in its threadpool, off the event loop.
"""
from __future__ import annotations

from typing import Any, Dict
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.jobs.queue import NOTIFY_CHANNEL

router = APIRouter(prefix="/api/v1/jobs", tags=["Jobs"])

NO_SUCH_DEAD_JOB = "No dead job with that id in this workspace."
_RETRY = text("UPDATE jobs.jobs SET status = 'queued', attempts = 0, run_at = clock_timestamp(), last_error = NULL, "
              "finished_at = NULL, updated_at = clock_timestamp() "
              "WHERE id = :id AND workspace_id = CAST(:ws AS uuid) AND status = 'dead' RETURNING kind")
_CLEAR = text("DELETE FROM jobs.jobs WHERE id = :id AND workspace_id = CAST(:ws AS uuid) AND status = 'dead'")


_DEAD_PAGE = text("SELECT id, kind, attempts, last_error, created_at, finished_at FROM jobs.jobs "
                  "WHERE workspace_id = CAST(:ws AS uuid) AND status = 'dead' "
                  "ORDER BY finished_at DESC NULLS LAST, id LIMIT :limit OFFSET :offset")
_DEAD_COUNT = text("SELECT count(*) FROM jobs.jobs WHERE workspace_id = CAST(:ws AS uuid) AND status = 'dead'")


@router.get("/dead", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
def list_dead_jobs(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
                   ctx: RequestContext = Depends(get_request_context_hybrid),
                   db: Session = Depends(get_db)) -> Dict[str, Any]:
    """A page of the workspace's dead jobs, newest first: what failed and why, never their payloads.
    ``total`` counts all of them, so a manager can page through a backlog of any size."""
    params = {"ws": str(ctx.workspace_id), "limit": limit, "offset": offset}
    rows = db.execute(_DEAD_PAGE, params).all()
    jobs = [{"id": str(r[0]), "kind": r[1], "attempts": r[2], "last_error": r[3],
             "created_at": r[4], "finished_at": r[5]} for r in rows]
    return {"jobs": jobs, "total": int(db.execute(_DEAD_COUNT, params).scalar() or 0), "limit": limit, "offset": offset}


@router.post("/{job_id}/retry", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
def retry_dead_job(job_id: UUID, ctx: RequestContext = Depends(get_request_context_hybrid),
                   db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Re-queues a dead job now, its attempts reset."""
    row = db.execute(_RETRY, {"id": job_id, "ws": str(ctx.workspace_id)}).first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=NO_SUCH_DEAD_JOB)
    db.execute(text("SELECT pg_notify(:channel, :kind)"), {"channel": NOTIFY_CHANNEL, "kind": row[0]})
    db.commit()
    return {"id": str(job_id), "status": "queued"}


@router.delete("/{job_id}", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
def clear_dead_job(job_id: UUID, ctx: RequestContext = Depends(get_request_context_hybrid),
                   db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Deletes a dead job."""
    deleted = db.execute(_CLEAR, {"id": job_id, "ws": str(ctx.workspace_id)}).rowcount
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=NO_SUCH_DEAD_JOB)
    db.commit()
    return {"id": str(job_id), "deleted": True}

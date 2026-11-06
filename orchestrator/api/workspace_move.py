"""PRD-273 US-273-03 — Move a workspace: export it to one file.

* ``POST /api/workspace-move/export`` starts an export of the caller's workspace (the chosen
  categories, an optional passphrase) as a background job: 202 with the job, 409 while
  another export of the workspace is queued or running, 503 when this install has no object
  storage to hold the file.
* ``GET /api/workspace-move/export/{job_id}`` is the job's status, counts and file name.
* ``GET /api/workspace-move/export/{job_id}/file`` downloads the file, once: the first call
  streams it and deletes it afterwards; any later call, or one after the link expired
  (``WORKSPACE_EXPORT_LINK_TTL_HOURS``), is 410.

Owners and admins of the workspace only (``require_workspace_admin``), on both editions; every
export and download is audited (``services/workspace_move/jobs.py``). The file holds personal
data: memories, chats, documents. Plain ``def`` routes over the synchronous session, as AGENTS.md
asks; the download streams from object storage in chunks off the event loop.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from core.auth.dependencies import RequestContext
from core.auth.principal import resolve_user_pk
from core.auth.workspace_admin import require_workspace_admin
from core.database.database import get_db
from core.models.workspace_move import READY, WorkspaceMoveJob
from services.workspace_move import jobs, storage
from services.workspace_move.rules import CATEGORIES

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/workspace-move", tags=["workspace-move"])

MIN_PASSPHRASE_LENGTH = 12
MAX_PASSPHRASE_LENGTH = 1024
FILE_MEDIA_TYPE = "application/zip"
EXPIRED = "expired"
NO_STORAGE = "This install has no object storage to hold the export file."
BUSY = "This workspace already has an export in progress."
NOT_FOUND = "No such export."
GONE = "This download link has been used or has expired. Start a new export."
UNREADABLE = "The export file could not be read. Start a new export."
PERSONAL_DATA = "The file holds personal data: memories, chats and documents. Keep it safe."


class ExportRequest(BaseModel):
    """What to export: §5.2's categories (the defaults when absent; Auto always) and an
    optional passphrase that seals every part but the manifest (§8-Q5)."""

    model_config = ConfigDict(extra="forbid")

    categories: Optional[List[str]] = Field(default=None, max_length=len(CATEGORIES))
    passphrase: Optional[str] = Field(default=None, min_length=MIN_PASSPHRASE_LENGTH, max_length=MAX_PASSPHRASE_LENGTH)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _view(job: WorkspaceMoveJob, now: datetime) -> Dict[str, Any]:
    """The job as the page shows it; a ready job past its link's expiry reads as expired."""
    expired = job.status == READY and job.expires_at is not None and job.expires_at <= now
    return {
        "id": str(job.id),
        "status": EXPIRED if expired else job.status,
        "categories": list(job.categories or []),
        "encrypted": bool(job.encrypted),
        "file_name": job.file_name,
        "size_bytes": job.size_bytes,
        "summary": dict(job.summary or {}),
        "error": job.error,
        "created_at": _iso(job.created_at),
        "started_at": _iso(job.started_at),
        "finished_at": _iso(job.finished_at),
        "expires_at": _iso(job.expires_at),
        "downloaded_at": _iso(job.downloaded_at),
        "notice": PERSONAL_DATA,
    }


@router.post("/export", status_code=status.HTTP_202_ACCEPTED)
def start_export(
    body: ExportRequest,
    background: BackgroundTasks,
    ctx: RequestContext = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Start exporting this workspace to one file; poll the job, then download it once."""
    if not storage.configured():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=NO_STORAGE)
    categories = tuple(body.categories) if body.categories is not None else None
    try:
        job = jobs.start_export(
            db, ctx.workspace_id, categories=categories, encrypted=bool(body.passphrase),
            requested_by=resolve_user_pk(db, ctx),
        )
    except ValueError as unknown:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(unknown)) from unknown
    except jobs.ExportBusy as busy:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=BUSY) from busy
    view = _view(job, datetime.now(timezone.utc))
    background.add_task(jobs.run_export, job.id, body.passphrase)
    return view


@router.get("/export/{job_id}")
def export_status(
    job_id: UUID,
    ctx: RequestContext = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """The export's status, its counts and its file's name and size."""
    try:
        job = jobs.get_job(db, ctx.workspace_id, job_id)
    except jobs.ExportNotFound as missing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=NOT_FOUND) from missing
    return _view(job, datetime.now(timezone.utc))


@router.get("/export/{job_id}/file")
def download_export(
    job_id: UUID,
    ctx: RequestContext = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """The export file, once. It is deleted after it has been sent; a second call is 410."""
    try:
        key, name, size = jobs.claim_download(db, ctx.workspace_id, job_id, user_id=resolve_user_pk(db, ctx))
    except jobs.ExportNotFound as missing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=NOT_FOUND) from missing
    except jobs.ExportGone as gone:
        raise HTTPException(status_code=status.HTTP_410_GONE, detail=GONE) from gone
    try:
        body = storage.open_body(key)
    except (BotoCoreError, ClientError) as unreadable:
        logger.exception("[workspace-move] the file of export %s could not be read", job_id)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=UNREADABLE) from unreadable
    headers = {"Content-Disposition": f'attachment; filename="{name}"', "Cache-Control": "no-store"}
    if size is not None:
        headers["Content-Length"] = str(size)
    return StreamingResponse(
        storage.chunks(body), media_type=FILE_MEDIA_TYPE, headers=headers,
        background=BackgroundTask(jobs.discard_file, job_id),
    )

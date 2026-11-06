"""PRD-273 US-273-03 — the export job: queued, run in the background, downloaded once, expired.

* ``start_export`` records a ``queued`` job (``workspace_move_jobs``). One export per
  workspace at a time: the table's partial unique index refuses a second active one
  (``ExportBusy``); a job left ``queued`` or ``running`` past
  ``WORKSPACE_EXPORT_JOB_TIMEOUT_MINUTES`` (its server restarted) is failed first.
* ``run_export`` is the background task. It writes the file to a temporary file in one
  REPEATABLE READ snapshot (every table read at the same moment), uploads it to object
  storage (``storage.py``) and marks the job ``ready`` until ``WORKSPACE_EXPORT_LINK_TTL_HOURS``
  from now; any failure marks it ``failed`` with a message that leaks nothing, the detail in
  the log. It never raises. The passphrase lives only in its arguments: never stored, never
  logged.
* ``claim_download`` is the single use: one UPDATE moves a ready, unexpired job to
  ``downloaded``, so a second call finds nothing to claim (``ExportGone``). The route
  streams the file, then ``discard_file`` deletes it.
* ``sweep_exports`` (every ``SWEEP_MINUTES`` on the leader's scheduler) deletes the file of
  every expired job and of every downloaded one whose clean-up did not run, and fails dead
  jobs.

Every export and every download is audited, like the GDPR export.
"""
from __future__ import annotations

import logging
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional, Tuple
from uuid import UUID, uuid4

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from config import config
from core.models.workspace_move import (
    ACTIVE_STATUSES, DOWNLOADED, EXPIRED, EXPORT, FAILED, QUEUED, READY, RUNNING, WorkspaceMoveJob,
)
from services.workspace_move import storage
from services.workspace_move.auto_part import WorkspaceGone
from services.workspace_move.export import ExportResult, write_workspace_file
from services.workspace_move.manifest import file_name
from services.workspace_move.rules import resolve_categories
from services.workspace_move.writer import ExportTooLarge

logger = logging.getLogger(__name__)

SessionFactory = Callable[[], Session]
SWEEP_JOB_ID = "workspace_export_sweep"
SWEEP_MINUTES = 15
# A downloaded file whose clean-up did not run (the download broke off) is deleted after this.
DOWNLOAD_GRACE = timedelta(hours=6)
AUDIT_EXPORT = "workspace_move:export"
AUDIT_DOWNLOAD = "workspace_move:export_download"
FAILED_GENERIC = "The export failed. The server log has the details."
FAILED_GONE = "The workspace no longer exists."
FAILED_STORAGE = "Object storage is not configured on this install."
FAILED_STALE = "The export stopped before it finished (the server restarted). Start it again."
# The partial unique index that keeps one queued or running export per workspace.
ACTIVE_INDEX = "uq_workspace_move_jobs_active"
# The updates here are statements over rows, not edits of loaded objects; a session re-reads a job.
_NO_SYNC = {"synchronize_session": False}


class ExportBusy(RuntimeError):
    """The workspace already has an export queued or running."""


class ExportNotFound(LookupError):
    """No export of this workspace has that id."""


class ExportGone(RuntimeError):
    """The export's file is not there to download: already downloaded, expired, failed or not ready."""

    def __init__(self, status: str) -> None:
        super().__init__(f"The export is {status}.")
        self.status = status


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _open_session() -> Session:
    from core.database.database import SessionLocal

    return SessionLocal()


def _open_export_session() -> Session:
    """A session whose transaction reads one snapshot of the whole database."""
    db = _open_session()
    db.connection(execution_options={"isolation_level": "REPEATABLE READ"})
    return db


def _audit(db: Session, workspace_id: UUID | str, action: str, user_id: Optional[int], **details: Any) -> None:
    """An audit_logs row for an export or a download (on the caller's session, committed)."""
    from core.workspaces.audit import AuditService

    AuditService(db).log(
        workspace_id=str(workspace_id), user_id=user_id, action=action, resource_type="workspace",
        resource_id=str(workspace_id), details=details,
    )


def _fail_stale(db: Session, now: datetime, workspace_id: Optional[UUID | str] = None) -> int:
    """Fail queued and running exports older than the timeout; how many."""
    cutoff = now - timedelta(minutes=int(config.WORKSPACE_EXPORT_JOB_TIMEOUT_MINUTES))
    statement = update(WorkspaceMoveJob).where(
        WorkspaceMoveJob.kind == EXPORT, WorkspaceMoveJob.status.in_(ACTIVE_STATUSES),
        WorkspaceMoveJob.created_at < cutoff,
    )
    if workspace_id is not None:
        statement = statement.where(WorkspaceMoveJob.workspace_id == workspace_id)
    failed = db.execute(
        statement.values(status=FAILED, error=FAILED_STALE, finished_at=now), execution_options=_NO_SYNC,
    ).rowcount
    db.commit()
    return int(failed or 0)


def start_export(db: Session, workspace_id: UUID | str, *, categories: Optional[Tuple[str, ...]],
                 encrypted: bool, requested_by: Optional[int], now: Optional[datetime] = None) -> WorkspaceMoveJob:
    """Record a queued export of the workspace. Raises ValueError on an unknown category and
    ExportBusy when the workspace already has one queued or running."""
    chosen = resolve_categories(categories)
    _fail_stale(db, now or _now(), workspace_id)
    job = WorkspaceMoveJob(
        id=uuid4(), workspace_id=workspace_id, kind=EXPORT, status=QUEUED, categories=list(chosen),
        encrypted=encrypted, requested_by=requested_by, summary={},
    )
    db.add(job)
    try:
        db.commit()
    except IntegrityError as conflict:
        db.rollback()
        if getattr(getattr(conflict.orig, "diag", None), "constraint_name", None) != ACTIVE_INDEX:
            raise
        raise ExportBusy("This workspace already has an export in progress.") from conflict
    _audit(db, workspace_id, AUDIT_EXPORT, requested_by, job_id=str(job.id), categories=list(chosen),
           encrypted=encrypted)
    return job


def _claim_run(db: Session, job_id: UUID) -> Optional[WorkspaceMoveJob]:
    """Move a queued job to running; the job, or None when it is not queued any more."""
    claimed = db.execute(
        update(WorkspaceMoveJob).where(WorkspaceMoveJob.id == job_id, WorkspaceMoveJob.status == QUEUED)
        .values(status=RUNNING, started_at=_now()),
        execution_options=_NO_SYNC,
    ).rowcount
    db.commit()
    return db.get(WorkspaceMoveJob, job_id) if claimed else None


def _write_and_upload(job: WorkspaceMoveJob, passphrase: Optional[str]) -> Tuple[ExportResult, int, str]:
    """Write the file to a temporary file, upload it; (the result, its size, its object key)."""
    with tempfile.TemporaryFile(dir=config.WORKSPACE_EXPORT_TEMP_DIR or None) as out:
        db = _open_export_session()
        try:
            exported_by = f"user:{job.requested_by}" if job.requested_by is not None else None
            result = write_workspace_file(
                db, job.workspace_id, out, categories=tuple(job.categories or ()), passphrase=passphrase,
                exported_by=exported_by,
            )
        finally:
            db.rollback()
            db.close()
        size = out.seek(0, 2)
        out.seek(0)
        key = storage.export_key(job.workspace_id, job.id)
        storage.upload(out, key)
    return result, size, key


def _failure_message(failure: BaseException) -> str:
    from core.storage import StorageNotConfigured

    if isinstance(failure, ExportTooLarge):
        return str(failure)
    if isinstance(failure, WorkspaceGone):
        return FAILED_GONE
    if isinstance(failure, StorageNotConfigured):
        return FAILED_STORAGE
    return FAILED_GENERIC


def _finish(db: Session, job_id: UUID, **values: Any) -> bool:
    """Record a running job's outcome; False when it is no longer running (a sweep failed it)."""
    done = db.execute(
        update(WorkspaceMoveJob).where(WorkspaceMoveJob.id == job_id, WorkspaceMoveJob.status == RUNNING)
        .values(finished_at=_now(), **values),
        execution_options=_NO_SYNC,
    ).rowcount
    db.commit()
    return bool(done)


def _record_ready(db: Session, job: WorkspaceMoveJob, result: ExportResult, size: int, key: str) -> None:
    now = _now()
    name = file_name(result.manifest["source"]["workspace_name"], now)
    expires = now + timedelta(hours=int(config.WORKSPACE_EXPORT_LINK_TTL_HOURS))
    if not _finish(db, job.id, status=READY, storage_key=key, file_name=name, size_bytes=size,
                   summary=result.summary, expires_at=expires):
        storage.delete(key)


def _delete_uploaded(key: Optional[str]) -> None:
    """Delete a file uploaded for a job that did not reach ready; a failure is logged (the file
    has no recorded key, so nothing else would delete it)."""
    if key is None:
        return
    try:
        storage.delete(key)
    except Exception:
        logger.exception("[workspace-move] the file %s of a failed export could not be deleted", key)


def _run_claimed(db: Session, job: WorkspaceMoveJob, passphrase: Optional[str]) -> None:
    """Write, upload and record a running job; on any failure, record it failed and delete what
    was uploaded."""
    job_id, workspace_id, key = job.id, job.workspace_id, None
    try:
        result, size, key = _write_and_upload(job, passphrase)
        _record_ready(db, job, result, size, key)
    except Exception as failure:
        logger.exception("[workspace-move] export %s of workspace %s failed", job_id, workspace_id)
        db.rollback()
        _delete_uploaded(key)
        _finish(db, job_id, status=FAILED, error=_failure_message(failure))
        return
    logger.info("[workspace-move] export %s ready: %d bytes", job_id, size)


def run_export(job_id: UUID, passphrase: Optional[str] = None) -> None:
    """The background task: write, upload and record one export. Never raises."""
    db = _open_session()
    try:
        job = _claim_run(db, job_id)
        if job is not None:
            _run_claimed(db, job, passphrase)
    except Exception:
        logger.exception("[workspace-move] export %s could not be recorded", job_id)
    finally:
        db.close()


def get_job(db: Session, workspace_id: UUID | str, job_id: UUID) -> WorkspaceMoveJob:
    """The workspace's export with that id; ExportNotFound otherwise."""
    job = db.get(WorkspaceMoveJob, job_id, populate_existing=True)
    if job is None or str(job.workspace_id) != str(workspace_id) or job.kind != EXPORT:
        raise ExportNotFound("No such export.")
    return job


def claim_download(db: Session, workspace_id: UUID | str, job_id: UUID, *, user_id: Optional[int],
                   now: Optional[datetime] = None) -> Tuple[str, str, Optional[int]]:
    """Use the download link: (object key, file name, size) once; ExportGone after that."""
    job = get_job(db, workspace_id, job_id)
    moment = now or _now()
    row = db.execute(
        update(WorkspaceMoveJob)
        .where(WorkspaceMoveJob.id == job.id, WorkspaceMoveJob.status == READY,
               WorkspaceMoveJob.expires_at > moment, WorkspaceMoveJob.storage_key.isnot(None))
        .values(status=DOWNLOADED, downloaded_at=moment)
        .returning(WorkspaceMoveJob.storage_key, WorkspaceMoveJob.file_name, WorkspaceMoveJob.size_bytes),
        execution_options=_NO_SYNC,
    ).first()
    db.commit()
    if row is None:
        db.refresh(job)
        raise ExportGone(EXPIRED if job.status == READY else job.status)
    _audit(db, workspace_id, AUDIT_DOWNLOAD, user_id, job_id=str(job.id))
    return row.storage_key, row.file_name, row.size_bytes


def release_download(db: Session, workspace_id: UUID | str, job_id: UUID) -> None:
    """Give a claimed download back (its file could not be opened), so the link works again."""
    db.execute(
        update(WorkspaceMoveJob)
        .where(WorkspaceMoveJob.id == job_id, WorkspaceMoveJob.workspace_id == workspace_id,
               WorkspaceMoveJob.status == DOWNLOADED, WorkspaceMoveJob.storage_key.isnot(None))
        .values(status=READY, downloaded_at=None),
        execution_options=_NO_SYNC,
    )
    db.commit()


def discard_file(job_id: UUID, *, status: Optional[str] = None,
                 session_factory: Optional[SessionFactory] = None) -> None:
    """Delete a job's file and forget its key (and set ``status`` when given). Never raises."""
    db = (session_factory or _open_session)()
    try:
        job = db.get(WorkspaceMoveJob, job_id)
        if job is None or job.storage_key is None:
            return
        storage.delete(job.storage_key)
        values: Dict[str, Any] = {"storage_key": None, **({"status": status} if status else {})}
        db.execute(update(WorkspaceMoveJob).where(WorkspaceMoveJob.id == job_id).values(**values),
                   execution_options=_NO_SYNC)
        db.commit()
    except Exception:
        logger.exception("[workspace-move] the file of export %s could not be deleted; the sweep retries", job_id)
        db.rollback()
    finally:
        db.close()


def _due_files(db: Session, now: datetime):
    expired = db.query(WorkspaceMoveJob.id).filter(
        WorkspaceMoveJob.status == READY, WorkspaceMoveJob.expires_at <= now,
    ).all()
    leftovers = db.query(WorkspaceMoveJob.id).filter(
        WorkspaceMoveJob.status == DOWNLOADED, WorkspaceMoveJob.storage_key.isnot(None),
        WorkspaceMoveJob.downloaded_at <= now - DOWNLOAD_GRACE,
    ).all()
    return [row.id for row in expired], [row.id for row in leftovers]


def sweep_exports(*, now: Optional[datetime] = None,
                  session_factory: Optional[SessionFactory] = None) -> Dict[str, int]:
    """Delete expired and left-over files, fail dead jobs; the counts. Never raises."""
    factory = session_factory or _open_session
    moment = now or _now()
    db = factory()
    try:
        stale = _fail_stale(db, moment)
        expired, leftovers = _due_files(db, moment)
    except Exception:
        logger.exception("[workspace-move] the export sweep could not read its jobs")
        db.rollback()
        return {"expired": 0, "leftovers": 0, "stale": 0}
    finally:
        db.close()
    for job_id in expired:
        discard_file(job_id, status=EXPIRED, session_factory=factory)
    for job_id in leftovers:
        discard_file(job_id, session_factory=factory)
    counts = {"expired": len(expired), "leftovers": len(leftovers), "stale": stale}
    if any(counts.values()):
        logger.info("[workspace-move] export sweep: %s", counts)
    return counts


def register(scheduler: Any) -> bool:
    """The sweep on the leader's scheduler, every ``SWEEP_MINUTES``; False when this worker hosts none."""
    if scheduler is None or not getattr(scheduler, "running", False):
        return False
    from apscheduler.triggers.interval import IntervalTrigger

    scheduler.add_job(
        sweep_exports, IntervalTrigger(minutes=SWEEP_MINUTES), id=SWEEP_JOB_ID,
        replace_existing=True, max_instances=1, coalesce=True,
    )
    logger.info("[workspace-move] export sweep every %d minutes", SWEEP_MINUTES)
    return True


__all__ = [
    "ExportBusy", "ExportGone", "ExportNotFound", "claim_download", "discard_file", "get_job", "register",
    "release_download", "run_export", "start_export", "sweep_exports",
]

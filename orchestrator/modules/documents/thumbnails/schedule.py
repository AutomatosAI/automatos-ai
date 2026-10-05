"""F353 (issue #947): a document Deliverable's picture is drawn after it is made, off the request.

Two decorators hand the new Deliverable to one background thread:

* ``thumbnail_after_register`` on ``DeliverableService.register`` (every PDF,
  Word document and sheet, from DocGen or written by an agent);
* ``thumbnail_after_report`` on ``ReportService.create_report`` (reports).

The decorated call returns exactly what it did before, as soon as it did before:
queueing is a ``put_nowait`` on a bounded queue. One daemon thread draws the
queued Deliverables one at a time (``job.render_for_output``, which renders in a
child process), so a burst of documents queues instead of competing for CPU, and
a full queue drops the extra ones with a log line (the backfill draws them later).

``DOCUMENT_THUMBNAILS_ENABLED`` turns it off (the test suite runs with it off;
a test that wants it on says so).
"""
from __future__ import annotations

import functools
import logging
import queue
import threading
from typing import Any, Callable, Dict, Optional, Tuple
from uuid import UUID

from config import config

logger = logging.getLogger(__name__)

QUEUE_MAX = 200
WORKER_NAME = "document-thumbnails"

_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue(maxsize=QUEUE_MAX)
_worker_lock = threading.Lock()
_worker: Optional[threading.Thread] = None


def _draw_next() -> None:
    """Draw the next queued Deliverable; a failure is logged and the thread goes on."""
    from core.database.database import SessionLocal
    from modules.documents.thumbnails.job import render_for_output

    workspace_id, output_id = _queue.get()
    try:
        render_for_output(SessionLocal, workspace_id, output_id)
    except Exception:
        logger.exception("[F353] thumbnail job failed for %s", output_id)
    finally:
        _queue.task_done()


def _run() -> None:
    while True:
        _draw_next()


def _ensure_worker() -> None:
    global _worker
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_run, name=WORKER_NAME, daemon=True)
            _worker.start()


def schedule_thumbnail(workspace_id: str | UUID, output_id: str | UUID) -> bool:
    """Queue ``output_id`` for drawing; False when off or when the queue is full."""
    if not config.DOCUMENT_THUMBNAILS_ENABLED or not workspace_id or not output_id:
        return False
    try:
        _queue.put_nowait((str(workspace_id), str(output_id)))
    except queue.Full:
        logger.warning("[F353] thumbnail queue full; %s waits for the backfill", output_id)
        return False
    _ensure_worker()
    return True


def thumbnail_after_register(register: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    """Queue a registered PDF, Word document or sheet for its picture."""
    from modules.documents.thumbnails.eligibility import wants_thumbnail

    @functools.wraps(register)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        result = register(self, *args, **kwargs)
        if result.get("success") and wants_thumbnail(result.get("artifact_type"), kwargs.get("file_path")):
            schedule_thumbnail(self.workspace_id, result.get("deliverable_id"))
        return result

    return wrapper


def thumbnail_after_report(create_report: Callable[..., Any]) -> Callable[..., Any]:
    """Queue a new report for its picture (the job skips heartbeat reports)."""

    @functools.wraps(create_report)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        result = await create_report(self, *args, **kwargs)
        if result.get("success"):
            schedule_thumbnail(self.workspace_id, result.get("report_id"))
        return result

    return wrapper

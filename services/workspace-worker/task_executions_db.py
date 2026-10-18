"""The worker's one database write: a task's lifecycle in ``task_executions``.

PRD-262 US-044 took SQLAlchemy (27.7 MB installed) out of the worker image for
this single UPDATE. It runs through psycopg2, which the worker already installs,
and main.py calls it with ``asyncio.to_thread``: the synchronous write used to run
on the event loop that also serves the health, canvas and file routes.

The statement is fixed text with bound ``%(name)s`` parameters. A column the
status change does not touch keeps its value, as the SET list built per call did
before: ``started_at`` moves only when the task starts running, ``completed_at``
only when it finishes, and ``error_message`` only when there is an error.

A failed write never fails the task. Redis carries the status the platform polls;
this row is the durable record, and a failure is logged.
"""

from __future__ import annotations

import logging
import re
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import psycopg2

logger = logging.getLogger("workspace-worker")

# The statuses a task ends in: each stamps completed_at.
FINISHED_STATUSES = frozenset({"completed", "failed", "timed_out", "cancelled"})
# The write runs in a thread, but an unreachable database must not hold that
# thread for libpq's default, which is to wait forever.
CONNECT_TIMEOUT_SECONDS = 10
# SQLAlchemy's dialect+driver scheme (``postgresql+psycopg2://``), which libpq rejects.
_DRIVER_SCHEME = re.compile(r"^postgres(?:ql)?\+[A-Za-z0-9_]+://")

UPDATE_TASK_EXECUTION = """
UPDATE task_executions
   SET status = %(status)s,
       worker_id = %(worker_id)s,
       started_at = CASE WHEN %(started)s THEN %(now)s ELSE started_at END,
       completed_at = CASE WHEN %(finished)s THEN %(now)s ELSE completed_at END,
       error_message = COALESCE(%(error)s, error_message)
 WHERE id = %(task_id)s
"""


def libpq_dsn(database_url: str) -> str:
    """``database_url`` as libpq takes it: ``postgresql+psycopg2://…`` becomes ``postgresql://…``.

    Everything after the scheme (credentials, host, database, query) is unchanged.
    """
    return _DRIVER_SCHEME.sub("postgresql://", database_url.strip(), count=1)


def update_parameters(
    task_id: str, status: str, worker_id: str, error: Optional[str], now: datetime
) -> Dict[str, Any]:
    """The bound parameters of ``UPDATE_TASK_EXECUTION`` for one status change."""
    return {
        "task_id": task_id,
        "status": status,
        "worker_id": worker_id,
        "started": status == "running",
        "finished": status in FINISHED_STATUSES,
        "now": now,
        "error": error or None,
    }


def record_task_status(
    database_url: str, task_id: str, status: str, worker_id: str, error: Optional[str] = None
) -> bool:
    """Write one status change to ``task_executions``. False, logged, when the write fails.

    Synchronous and blocking: main.py runs it in a thread. The connection lasts
    for this one statement, so no connection is shared between threads.
    """
    params = update_parameters(task_id, status, worker_id, error, datetime.now(timezone.utc))
    dsn = libpq_dsn(database_url)
    try:
        with closing(psycopg2.connect(dsn, connect_timeout=CONNECT_TIMEOUT_SECONDS)) as conn:
            with conn, conn.cursor() as cursor:  # commits on success, rolls back on error
                cursor.execute(UPDATE_TASK_EXECUTION, params)
    except psycopg2.Error:
        logger.exception("DB update failed for task %s (%s)", task_id[:8], status)
        return False
    return True

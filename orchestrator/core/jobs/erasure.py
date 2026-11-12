"""The purge's barrier (PRD-276 §6.8): a workspace's jobs go before its data.

``services.workspace_purge`` erases a workspace only after it was soft-deleted, so its jobs are no
longer claimed. Its first data step (``_s3_purged``) is wrapped to delete the workspace's ``jobs``
rows first, on a session of its own. That ``DELETE`` waits for any row lock a handler holds while it
sends under its fence, so once it commits no handler of the workspace is mid-send, and any later
fence matches no row. Only then does the purge erase data. If the barrier fails, the purge stops.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable

from sqlalchemy import text

logger = logging.getLogger(__name__)


def jobs_stop_first(first_data_step: Callable[..., bool]) -> Callable[..., bool]:
    """Wraps the purge's first data step: the workspace's jobs are deleted before it runs."""

    @functools.wraps(first_data_step)
    def barrier_then_step(workspace_id: Any, result: Any) -> bool:
        from core.database.database import SessionLocal
        from services.workspace_purge import PURGE_RAISED

        db = SessionLocal()
        try:
            deleted = db.execute(text("DELETE FROM jobs.jobs WHERE workspace_id = CAST(:ws AS uuid)"),
                                 {"ws": str(workspace_id)}).rowcount
            db.commit()
        except Exception:  # noqa: BLE001 — logged; the purge holds rather than erase under a running send
            db.rollback()
            logger.exception("[jobs] could not stop workspace %s's jobs; the purge holds", workspace_id)
            result.skipped_reason = PURGE_RAISED
            return False
        finally:
            db.close()
        if deleted:
            logger.info("[jobs] stopped %d job(s) of workspace %s before its purge", deleted, workspace_id)
        return first_data_step(workspace_id, result)

    return barrier_then_step

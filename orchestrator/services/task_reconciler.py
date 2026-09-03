"""
TaskReconciler — Stall Detection + Auto-Retry
==============================================
Symphony-inspired tick loop that runs every TASK_RECONCILE_INTERVAL_SECONDS
on the UnifiedScheduler.

Each tick:
1. Detects running executions that exceeded TASK_STALL_TIMEOUT_SECONDS
2. Detects pending executions stuck longer than TASK_PENDING_TIMEOUT_SECONDS
3. Marks stalled executions as failed
4. Retries eligible executions with exponential backoff
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from apscheduler.schedulers.asyncio import AsyncIOScheduler

logger = logging.getLogger(__name__)


ORPHANED_BOARD_SQL = """
    SELECT bt.id, bt.source_type, bt.source_id, bt.started_at
    FROM board_tasks bt
    WHERE bt.status = 'in_progress'
      AND bt.started_at < :cutoff
      AND (bt.lease_until IS NULL OR bt.lease_until < :now)
      AND (
          bt.source_type = 'user'
          OR (bt.source_type = 'recipe' AND EXISTS (
              SELECT 1 FROM recipe_executions re
              WHERE re.execution_id = bt.source_id
                AND re.status IN ('completed', 'failed', 'cancelled')
          ))
      )
"""


class TaskReconciler:

    def __init__(self):
        self._scheduler: Optional[AsyncIOScheduler] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, scheduler: AsyncIOScheduler):
        """Register reconciliation tick on the shared scheduler."""
        from config import config as app_config

        self._scheduler = scheduler
        interval = app_config.TASK_RECONCILE_INTERVAL_SECONDS

        self._scheduler.add_job(
            self._tick,
            "interval",
            seconds=interval,
            id="task_reconciler_tick",
            replace_existing=True,
            max_instances=1,
        )
        logger.info(
            "[TaskReconciler] Started — tick every %ds, stall=%ds, pending=%ds, max_retries=%d",
            interval,
            app_config.TASK_STALL_TIMEOUT_SECONDS,
            app_config.TASK_PENDING_TIMEOUT_SECONDS,
            app_config.TASK_MAX_RETRIES,
        )

    async def stop(self):
        """Remove the tick job from the scheduler."""
        if self._scheduler and self._scheduler.get_job("task_reconciler_tick"):
            self._scheduler.remove_job("task_reconciler_tick")
            logger.info("[TaskReconciler] Stopped")

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    async def _tick(self):
        """Single reconciliation pass — detect stalls, mark failed, retry."""
        from config import config as app_config
        from core.database.database import SessionLocal
        from sqlalchemy import text

        db = SessionLocal()
        try:
            now = datetime.now(timezone.utc)

            # 1. Stalled running executions
            stalled_running = db.execute(
                text("""
                    SELECT execution_id, recipe_id, workspace_id, input_data,
                           COALESCE(attempt_count, 1) AS attempt_count,
                           execution_metadata, started_at
                    FROM recipe_executions
                    WHERE status = 'running'
                      AND started_at < :cutoff
                """),
                {"cutoff": now - _timedelta_seconds(app_config.TASK_STALL_TIMEOUT_SECONDS)},
            ).fetchall()

            # 2. Stuck pending executions (never started)
            stuck_pending = db.execute(
                text("""
                    SELECT execution_id, recipe_id, workspace_id, input_data,
                           COALESCE(attempt_count, 1) AS attempt_count,
                           execution_metadata, started_at
                    FROM recipe_executions
                    WHERE status = 'pending'
                      AND started_at < :cutoff
                """),
                {"cutoff": now - _timedelta_seconds(app_config.TASK_PENDING_TIMEOUT_SECONDS)},
            ).fetchall()

            # 3. Orphaned board tasks — in_progress with no backing async task
            #    (standalone user tasks executed via fire-and-forget, or recipe
            #    tasks whose execution already finished/failed but bridge missed).
            #    PRD-234: a ticket with a LIVE lease is not orphaned — a CLI host
            #    renews the lease for every session it runs (heartbeat + events),
            #    and the board dispatcher's lease sweep already re-queues a run
            #    whose lease lapsed. Without this guard every Claude Code session
            #    longer than TASK_STALL_TIMEOUT_SECONDS was force-closed as done
            #    (tickets 79/80, 2026-09-03) and the host's real result refused.
            orphaned_board = db.execute(
                text(ORPHANED_BOARD_SQL),
                {"cutoff": now - _timedelta_seconds(app_config.TASK_STALL_TIMEOUT_SECONDS), "now": now},
            ).fetchall()

            total = len(stalled_running) + len(stuck_pending) + len(orphaned_board)
            if total == 0:
                return

            logger.warning(
                "[TaskReconciler] Found %d stalled (running=%d, pending=%d, orphan_board=%d)",
                total, len(stalled_running), len(stuck_pending), len(orphaned_board),
            )

            for row in stalled_running:
                await self._handle_stalled(row, db, reason="running", timeout=app_config.TASK_STALL_TIMEOUT_SECONDS)

            for row in stuck_pending:
                await self._handle_stalled(row, db, reason="pending", timeout=app_config.TASK_PENDING_TIMEOUT_SECONDS)

            for row in orphaned_board:
                error_msg = (
                    f"Stalled: in_progress for >{app_config.TASK_STALL_TIMEOUT_SECONDS}s "
                    f"with no active execution (source={row.source_type})"
                )
                db.execute(
                    text("""
                        UPDATE board_tasks
                        SET status = 'done',
                            error_message = :error,
                            completed_at = :now
                        WHERE id = :id AND status = 'in_progress'
                    """),
                    {"error": error_msg, "now": now, "id": row.id},
                )
                logger.warning("[TaskReconciler] Closed orphaned board task %d — %s", row.id, error_msg)

            db.commit()

        except Exception as e:
            logger.error("[TaskReconciler] Tick failed: %s", e, exc_info=True)
            db.rollback()
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Handle a single stalled execution
    # ------------------------------------------------------------------

    async def _handle_stalled(self, row, db, *, reason: str, timeout: int):
        """Mark execution as failed and optionally schedule a retry."""
        from config import config as app_config
        from sqlalchemy import text

        execution_id = row.execution_id
        recipe_id = row.recipe_id
        workspace_id = row.workspace_id
        input_data = row.input_data or {}
        attempt_count = row.attempt_count
        metadata = row.execution_metadata or {}
        max_retries = _get_max_retries(metadata, app_config.TASK_MAX_RETRIES)

        error_msg = f"Stalled: no progress for {timeout}s (status was '{reason}')"

        # Mark failed
        db.execute(
            text("""
                UPDATE recipe_executions
                SET status = 'failed',
                    error_message = :error,
                    completed_at = :now
                WHERE execution_id = :eid
            """),
            {"error": error_msg, "now": datetime.now(timezone.utc), "eid": execution_id},
        )
        logger.warning(
            "[TaskReconciler] Marked execution %s as failed — %s", execution_id, error_msg,
        )

        # Sync linked board task so it doesn't stay stuck in_progress
        try:
            from services.board_task_bridge import complete_recipe_board_task
            complete_recipe_board_task(db, execution_id, success=False, error_message=error_msg)
        except Exception as bt_err:
            logger.warning("[TaskReconciler] Board task sync failed for %s: %s", execution_id, bt_err)

        # Check retry eligibility
        if attempt_count >= max_retries:
            logger.info(
                "[TaskReconciler] Execution %s exhausted retries (%d/%d)",
                execution_id, attempt_count, max_retries,
            )
            return

        # Schedule retry with backoff
        next_attempt = attempt_count + 1
        backoff_ms = self._calculate_backoff_ms(next_attempt)
        backoff_seconds = backoff_ms / 1000.0

        retry_execution_id = f"retry-{uuid4().hex[:12]}"
        retry_metadata = {
            **metadata,
            "retry_of": execution_id,
            "backoff_ms": backoff_ms,
            "attempt": next_attempt,
        }

        # Insert new execution record
        db.execute(
            text("""
                INSERT INTO recipe_executions
                    (execution_id, recipe_id, workspace_id, status, input_data,
                     triggered_by, execution_metadata, attempt_count, retry_of, started_at)
                VALUES
                    (:eid, :rid, :wid, 'pending', CAST(:input AS jsonb),
                     'auto_retry', CAST(:meta AS jsonb), :attempt, :retry_of, :now)
            """),
            {
                "eid": retry_execution_id,
                "rid": recipe_id,
                "wid": str(workspace_id),
                "input": _json_dumps(input_data),
                "meta": _json_dumps(retry_metadata),
                "attempt": next_attempt,
                "retry_of": execution_id,
                "now": datetime.now(timezone.utc),
            },
        )

        logger.info(
            "[TaskReconciler] Scheduling retry %s for recipe %d (attempt %d/%d, backoff %.1fs)",
            retry_execution_id, recipe_id, next_attempt, max_retries, backoff_seconds,
        )

        # Fire after backoff delay
        loop = asyncio.get_event_loop()
        loop.call_later(
            backoff_seconds,
            lambda: asyncio.ensure_future(
                self._launch_retry(retry_execution_id, recipe_id, workspace_id, input_data)
            ),
        )

    async def _launch_retry(
        self,
        execution_id: str,
        recipe_id: int,
        workspace_id,
        input_data: dict,
    ):
        """Actually fire the retried execution."""
        try:
            # PRD-142 W3-S12: retried playbooks launch via the engine seam.
            from services.playbook_engine import get_playbook_engine

            get_playbook_engine().launch(
                recipe_execution_id=execution_id,
                recipe_id=recipe_id,
                workspace_id=UUID(str(workspace_id)),
                input_data=input_data,
            )
            logger.info("[TaskReconciler] Launched retry execution %s", execution_id)
        except Exception as e:
            logger.error(
                "[TaskReconciler] Failed to launch retry %s: %s", execution_id, e, exc_info=True,
            )

    # ------------------------------------------------------------------
    # Backoff calculation
    # ------------------------------------------------------------------

    def _calculate_backoff_ms(self, attempt: int) -> int:
        """Exponential backoff: min(10_000 * 2^(attempt-1), MAX_RETRY_BACKOFF_MS)."""
        from config import config as app_config

        backoff = 10_000 * (2 ** (attempt - 1))
        return min(backoff, app_config.TASK_MAX_RETRY_BACKOFF_MS)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return reconciler status for health checks."""
        has_job = (
            self._scheduler is not None
            and self._scheduler.get_job("task_reconciler_tick") is not None
        )
        return {"active": has_job}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _timedelta_seconds(seconds: int):
    """Return a timedelta for the given seconds."""
    from datetime import timedelta
    return timedelta(seconds=seconds)


def _get_max_retries(metadata: dict, default: int) -> int:
    """Extract max_retries from execution_config in metadata, or use default."""
    exec_config = metadata.get("execution_config", {})
    if isinstance(exec_config, dict):
        val = exec_config.get("max_retries")
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
    return default


def _json_dumps(obj) -> str:
    """Serialize to JSON string for raw SQL params."""
    import json
    return json.dumps(obj, default=str)


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_reconciler: Optional[TaskReconciler] = None


def get_task_reconciler() -> TaskReconciler:
    global _reconciler
    if _reconciler is None:
        _reconciler = TaskReconciler()
    return _reconciler

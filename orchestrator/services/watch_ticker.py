"""
WatchTicker -- PRD-204 S5
=========================

The watcher heartbeat sweep on the UnifiedScheduler (TaskReconciler
pattern). The S3 event hooks are the FAST path -- a terminal state normally
reaches the watch in the producer's own transaction. The tick is the
fallback and the trend brain:

- claims due watches (``next_check_at <= now``) via FOR UPDATE SKIP LOCKED
  (single-writer; the claim's reschedule doubles as the lease),
- refreshes the target's state with a cheap status read; a terminal state
  the hooks missed is ingested here (idempotent -- the event_key dedupe
  makes "hook then sweep" write exactly one event),
- handles ``deadline_at`` -> expired,
- for scheduled-playbook watches: croniter expected-fire vs the latest
  ``recipe_executions`` row -> missed-run event; ``breaker_is_open`` ->
  benched event,
- writes a watch_event only on MEANINGFUL change (running -> running
  writes nothing),
- reschedules ``next_check_at`` (done inside the claim).

Notification rules (PRD-204 S5/S10): terminal-state verdicts are owned by
the DECISION STEP (services/watch_decider.py) -- the tick records the
terminal (idempotent) and hands the live watch to
``WatchDecider.decide_terminal`` which scores (S6), acts (S7/S8) or closes,
and dispatches ``watch_verdict``/``watch_action``. The tick itself
dispatches only what no other owner covers: missed run and expiry
(``watch_escalation``). Benched watch_events are recorded here, but the
``playbook_benched`` NOTIFICATION is owned by the scheduler skip path (S4,
once per breaker-open period) -- dispatching it from the tick too would
double-notify every bench.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy.orm import Session

from core.models.watches import Watch
from core.models.watch_enums import (
    CLAIMABLE_WATCH_STATUSES,
    WatchEventType,
    WatchPolicy,
    WatchStatus,
    WatchTargetType,
)
from services.watch_service import WatchService

logger = logging.getLogger(__name__)

# Terminal vocabularies per target table (cheap status reads).
_RUN_TERMINAL = frozenset({"completed", "failed", "cancelled"})
_EXECUTION_TERMINAL = frozenset({"completed", "failed", "cancelled"})
_BOARD_TERMINAL = {"done": "completed", "failed": "failed"}

# A cron fire only counts as missed once it is this many seconds overdue --
# absorbs scheduler jitter and slow launches without false alarms.
MISSED_RUN_GRACE_SECONDS = 120


def _aware(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalise DB datetimes (some legacy columns are tz-naive UTC)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class WatchTicker:
    """Background sweep loop over the watch registry."""

    def __init__(self):
        self._scheduler: Optional[AsyncIOScheduler] = None

    # ------------------------------------------------------------------
    # Lifecycle (TaskReconciler pattern)
    # ------------------------------------------------------------------

    async def start(self, scheduler: AsyncIOScheduler):
        """Register the watcher tick on the shared scheduler."""
        from config import config as app_config

        self._scheduler = scheduler
        interval = app_config.WATCHER_TICK_SECONDS
        self._scheduler.add_job(
            self._tick,
            "interval",
            seconds=interval,
            id="watch_ticker_tick",
            replace_existing=True,
            max_instances=1,
        )
        logger.info("[WatchTicker] Started -- tick every %ds", interval)

    async def stop(self):
        """Remove the tick job from the scheduler."""
        if self._scheduler and self._scheduler.get_job("watch_ticker_tick"):
            self._scheduler.remove_job("watch_ticker_tick")
            logger.info("[WatchTicker] Stopped")

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    async def _tick(self):
        """Scheduled entry point -- opens its own session."""
        from core.database.database import SessionLocal

        db = SessionLocal()
        try:
            await self.tick_once(db, now=datetime.now(timezone.utc))
        except Exception:
            logger.error("[WatchTicker] tick failed", exc_info=True)
        finally:
            db.close()

    async def tick_once(self, db: Session, now: datetime) -> int:
        """One sweep pass at a fixed ``now`` (injected for frozen-clock
        tests). Returns the number of watches processed. Per-watch errors
        roll back that watch's work only -- the sweep keeps going.
        """
        claimed = WatchService.claim_due_watches(db, now=now)  # commits
        for watch in claimed:
            try:
                await self._check_watch(db, watch, now)
                db.commit()
            except Exception:
                db.rollback()
                logger.error(
                    "[WatchTicker] check failed for watch %s",
                    getattr(watch, "id", "?"),
                    exc_info=True,
                )
        return len(claimed)

    # ------------------------------------------------------------------
    # Per-watch checks
    # ------------------------------------------------------------------

    async def _check_watch(self, db: Session, watch: Watch, now: datetime) -> None:
        if watch.target_type == WatchTargetType.SCHEDULED_PLAYBOOK.value:
            await self._check_scheduled_playbook(db, watch, now)
        else:
            await self._check_run_target(db, watch, now)

        # Deadline -> expired (only if the checks above left the watch live).
        if (
            WatchStatus(watch.status) in CLAIMABLE_WATCH_STATUSES
            and watch.deadline_at is not None
            and _aware(watch.deadline_at) <= now
        ):
            event = WatchService.ingest(
                db,
                watch,
                event_type=WatchEventType.EXPIRED.value,
                event_key="expired",
                summary=(
                    f"Deadline {_aware(watch.deadline_at).isoformat()} passed "
                    f"without a verdict"
                ),
                requires_attention=True,
            )
            WatchService.transition(
                db, watch, WatchStatus.EXPIRED, reason="deadline passed"
            )
            if event is not None:
                await self._dispatch_watch_event(
                    db,
                    watch,
                    event_type="watch_escalation",
                    title=f"Watch expired: {watch.title[:110]}",
                    message=(
                        "The deadline passed before the watched work reached "
                        "a verdict."
                    ),
                    status="warning",
                )

    # -------------------------------------------------- run-shaped targets

    async def _check_run_target(self, db: Session, watch: Watch, now: datetime) -> None:
        """Cheap status read for mission / playbook-execution / board-task
        targets. Writes NOTHING while the target is still in flight."""
        state = self._read_target_state(db, watch)

        if state is None:
            event = WatchService.ingest(
                db,
                watch,
                event_type=WatchEventType.TARGET_MISSING.value,
                event_key=f"missing:{watch.target_type}:{watch.target_id}",
                summary=(
                    f"Target {watch.target_type}:{watch.target_id} no longer "
                    f"exists (deleted or archived)"
                ),
                requires_attention=True,
            )
            if event is not None and WatchStatus(watch.status) != WatchStatus.NEEDS_ATTENTION:
                WatchService.transition(
                    db, watch, WatchStatus.NEEDS_ATTENTION, reason="target missing"
                )
            return

        if state not in _RUN_TERMINAL:
            return  # no-noise: running -> running writes nothing

        # Record the terminal (idempotent; the producer hook usually beat
        # us here). PRD-204 S10: ingest no longer closes scorable terminals
        # -- the DECISION STEP owns score -> act/close -> notify.
        WatchService.ingest_terminal(
            db,
            workspace_id=watch.workspace_id,
            target_type=watch.target_type,
            target_id=watch.target_id,
            terminal_state=state,
            summary="Detected by watcher sweep",
            now=now,
        )
        if WatchStatus(watch.status) in CLAIMABLE_WATCH_STATUSES:
            from services.watch_decider import get_watch_decider

            decision = await get_watch_decider().decide_terminal(
                db, watch, state, now
            )
            logger.info(
                "[WatchTicker] decision for watch %s (terminal=%s): %s",
                watch.id,
                state,
                decision,
            )

    def _read_target_state(self, db: Session, watch: Watch) -> Optional[str]:
        """Return the target's current status string, or None if the target
        row is gone / unreadable-by-type."""
        target_type = watch.target_type
        if target_type == WatchTargetType.MISSION.value:
            from core.models.orchestration import OrchestrationRun

            row = (
                db.query(OrchestrationRun.state)
                .filter(OrchestrationRun.id == watch.target_id)
                .first()
            )
            return row[0] if row else None

        if target_type == WatchTargetType.PLAYBOOK_EXECUTION.value:
            from core.models.core import RecipeExecution

            row = (
                db.query(RecipeExecution.status)
                .filter(RecipeExecution.execution_id == watch.target_id)
                .first()
            )
            return row[0] if row else None

        if target_type == WatchTargetType.BOARD_TASK.value:
            from core.models.core import BoardTask

            try:
                task_id = int(watch.target_id)
            except (TypeError, ValueError):
                return None
            row = db.query(BoardTask.status).filter(BoardTask.id == task_id).first()
            if row is None:
                return None
            return _BOARD_TERMINAL.get(row[0], "running")

        logger.warning(
            "[WatchTicker] unknown target_type %r on watch %s",
            target_type,
            watch.id,
        )
        return None

    # ------------------------------------------- scheduled-playbook targets

    async def _check_scheduled_playbook(
        self, db: Session, watch: Watch, now: datetime
    ) -> None:
        """Missed-run + benched checks for a cron-scheduled playbook."""
        from core.models import WorkflowTemplate as WorkflowPlaybook
        from core.models.core import RecipeExecution

        try:
            recipe_id = int(watch.target_id)
        except (TypeError, ValueError):
            recipe_id = None
        playbook = (
            db.query(WorkflowPlaybook).filter(WorkflowPlaybook.id == recipe_id).first()
            if recipe_id is not None
            else None
        )
        if playbook is None:
            event = WatchService.ingest(
                db,
                watch,
                event_type=WatchEventType.TARGET_MISSING.value,
                event_key=f"missing:scheduled_playbook:{watch.target_id}",
                summary=f"Scheduled playbook {watch.target_id} no longer exists",
                requires_attention=True,
            )
            if event is not None and WatchStatus(watch.status) != WatchStatus.NEEDS_ATTENTION:
                WatchService.transition(
                    db, watch, WatchStatus.NEEDS_ATTENTION, reason="target missing"
                )
            return

        sc = playbook.schedule_config or {}
        cron_expression = sc.get("cron_expression")
        if sc.get("type") != "cron" or not cron_expression:
            return  # nothing schedule-shaped to supervise

        # --- Benched: breaker open -> record on the watch timeline. The
        # latest terminal execution id keys the event: while the breaker is
        # open the scheduler creates no new rows, so the key is stable for
        # the whole open period -> exactly one benched event per period.
        # The playbook_benched NOTIFICATION is owned by the scheduler skip
        # path (S4) -- see module docstring.
        from services.playbook_breaker import breaker_is_open

        if breaker_is_open(db, playbook.id):
            latest_terminal = (
                db.query(RecipeExecution.id)
                .filter(
                    RecipeExecution.recipe_id == playbook.id,
                    RecipeExecution.status.in_(("completed", "failed")),
                )
                .order_by(RecipeExecution.started_at.desc())
                .first()
            )
            bench_key = (
                f"benched:{latest_terminal[0]}" if latest_terminal else "benched:none"
            )
            WatchService.ingest(
                db,
                watch,
                event_type=WatchEventType.BENCHED.value,
                event_key=bench_key,
                summary=(
                    f"Playbook '{playbook.name}' is benched: the repeated-"
                    f"failure breaker is open, scheduled runs are paused"
                ),
                requires_attention=True,
            )

        # --- Missed run: expected fire (croniter, via the single source of
        # truth schedule_util.next_run) vs the latest execution row.
        from services.schedule_util import next_run

        latest = (
            db.query(RecipeExecution.started_at)
            .filter(RecipeExecution.recipe_id == playbook.id)
            .order_by(RecipeExecution.started_at.desc())
            .first()
        )
        baseline = _aware(latest[0]) if latest else _aware(watch.created_at)
        if baseline is None:
            return

        expected = next_run(cron_expression, now=baseline)
        if expected is None:
            return
        overdue_by = (now - expected).total_seconds()
        if overdue_by < MISSED_RUN_GRACE_SECONDS:
            return  # not due yet, or within launch-jitter grace

        # Did anything actually run at/after the expected fire?
        ran = (
            db.query(RecipeExecution.id)
            .filter(
                RecipeExecution.recipe_id == playbook.id,
                RecipeExecution.started_at >= expected.replace(tzinfo=None),
            )
            .first()
        )
        if ran is not None:
            return

        event = WatchService.ingest(
            db,
            watch,
            event_type=WatchEventType.MISSED_RUN.value,
            event_key=f"missed:{expected.isoformat()}",
            summary=(
                f"Playbook '{playbook.name}' missed its expected fire at "
                f"{expected.isoformat()} (no execution row)"
            ),
            snapshot={"expected_fire": expected.isoformat()},
            requires_attention=True,
        )
        if event is not None:
            await self._dispatch_watch_event(
                db,
                watch,
                event_type="watch_escalation",
                title=f"Missed scheduled run: {playbook.name}",
                message=(
                    f"Expected a run at {expected.isoformat()} but none "
                    f"started. The schedule may be stalled."
                ),
                status="warning",
            )

        # PRD-204 S10: persistent watches also report OUTCOME FLIPS between
        # consecutive runs (completed <-> failed) -- notify on meaningful
        # change, never close on a run terminal.
        if watch.policy == WatchPolicy.PERSISTENT.value:
            from services.watch_decider import get_watch_decider

            await get_watch_decider().observe_scheduled(db, watch, playbook, now)

    # ------------------------------------------------------------------
    # Notification seam (mirrors coordinator_service._dispatch_mission_event)
    # ------------------------------------------------------------------

    async def _dispatch_watch_event(
        self,
        db: Session,
        watch: Watch,
        *,
        event_type: str,
        title: str,
        message: Optional[str],
        status: str = "ok",
    ) -> None:
        """Fire a watch-related event through the shared notification seam.

        Kept as a ticker method (test seam); the body is the S6 shared
        helper so the verdict/action/decision paths and the sweep all
        dispatch identically. Never raises into the sweep.
        """
        from services.watch_notifications import dispatch_watch_notification

        await dispatch_watch_notification(
            db,
            watch,
            event_type=event_type,
            title=title,
            message=message,
            status=status,
        )


# ---------------------------------------------------------------------------
# Singleton (house pattern)
# ---------------------------------------------------------------------------

_watch_ticker: Optional[WatchTicker] = None


def get_watch_ticker() -> WatchTicker:
    global _watch_ticker
    if _watch_ticker is None:
        _watch_ticker = WatchTicker()
    return _watch_ticker

"""
WatchService -- PRD-204 S2
==========================

Single owner of watch-registry mutations: create/get/list/cancel, the guarded
status ``transition`` (ALLOWED_WATCH_TRANSITIONS, house style of
``orchestration_state``), idempotent ``ingest`` (duplicate event_key swallowed
via the UNIQUE constraint under a savepoint), the terminal-ingest seam the
S3 producer hooks call, ``record_action`` with the action-budget hard stop,
the ``follow`` lineage helper, and the tick's ``claim_due_watches``
(FOR UPDATE SKIP LOCKED, board_dispatcher idiom).

Transaction contract:
- Every method except ``claim_due_watches`` joins the CALLER's transaction
  (flush only, never commit) -- same principle as orchestration_state.
- ``claim_due_watches`` COMMITS: the claim is its own transaction so
  SKIP LOCKED serialises concurrent tickers (claim_tasks precedent).

Notifications are deliberately NOT dispatched here: producers own their
terminal notifications (mission_complete / mission_failed / playbook_*), and
the watcher tick (S5) owns watcher-only notifications (sweep-caught terminal,
missed run, expiry). Keeping this module sync + DB-only lets the fail-soft
hooks call it from sync producers (transition_run) and async ones alike.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from core.models.watches import (
    DEFAULT_ACTION_BUDGET,
    DEFAULT_CHECK_INTERVAL_SECONDS,
    DEFAULT_QUALITY_THRESHOLD,
    Watch,
    WatchEvent,
)
from core.models.watch_enums import (
    ALLOWED_WATCH_TRANSITIONS,
    CLAIMABLE_WATCH_STATUSES,
    LIVE_WATCH_STATUSES,
    TERMINAL_WATCH_STATUSES,
    WATCH_STATUS_FOR_TERMINAL_TARGET,
    WatchEventType,
    WatchPolicy,
    WatchStatus,
)
from services.orchestration_state import ConflictError, InvalidTransitionError

logger = logging.getLogger(__name__)

# Default batch size for the tick's claim.
DEFAULT_CLAIM_LIMIT = 50


class WatchAlreadyExistsError(Exception):
    """A non-terminal watch already supervises this target."""

    def __init__(self, workspace_id: Any, target_type: str, target_id: str):
        self.workspace_id = workspace_id
        self.target_type = target_type
        self.target_id = target_id
        super().__init__(
            f"A live watch already exists for {target_type}:{target_id} "
            f"in workspace {workspace_id}"
        )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def watch_auto_create_enabled(db: Session, workspace_id: UUID | str) -> bool:
    """PRD-204 S9 (Section 8 Q1): the ``watch_auto_create`` workspace setting.

    Default ON. Reads ``workspace.settings`` (the approval_policy pattern).
    Only an explicit boolean False turns it off; a missing workspace returns
    False (nothing to attach a watch to), any other read problem defaults ON.
    """
    try:
        from core.models.workspaces import Workspace

        ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
        if ws is None:
            return False
        value = (ws.settings or {}).get("watch_auto_create")
        if isinstance(value, bool):
            return value
        return True
    except Exception:
        logger.warning(
            "[Watch] watch_auto_create read failed for %s -- defaulting ON",
            workspace_id,
            exc_info=True,
        )
        return True


class WatchService:
    """Stateless service over the watch registry (caller manages sessions)."""

    # ------------------------------------------------------------------
    # Create / read / cancel
    # ------------------------------------------------------------------

    @staticmethod
    def create_watch(
        db: Session,
        *,
        workspace_id: UUID | str,
        watch_type: str,
        target_type: str,
        target_id: str,
        title: str,
        created_by: Optional[str] = None,
        owner_agent_id: Optional[int] = None,
        description: Optional[str] = None,
        success_criteria: Optional[str] = None,
        failure_criteria: Optional[str] = None,
        quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
        check_interval_seconds: int = DEFAULT_CHECK_INTERVAL_SECONDS,
        deadline_at: Optional[datetime] = None,
        policy: str = WatchPolicy.RUN_AND_REPORT.value,
        allowed_actions: Optional[List[str]] = None,
        action_budget: int = DEFAULT_ACTION_BUDGET,
        now: Optional[datetime] = None,
    ) -> Watch:
        """Create a watch on a target. One live watch per target.

        ``lineage`` invariant: the ordered target chain INCLUDES the original
        target, so ``lineage[-1]`` always mirrors the live target columns.

        Raises WatchAlreadyExistsError when a non-terminal watch already
        supervises the target (the partial unique index is the race-safe
        backstop -- a concurrent create surfaces as IntegrityError).
        """
        moment = now or _utcnow()
        target_id = str(target_id)

        existing = WatchService.find_live_watch(
            db, workspace_id=workspace_id, target_type=target_type, target_id=target_id
        )
        if existing is not None:
            raise WatchAlreadyExistsError(workspace_id, target_type, target_id)

        watch = Watch(
            workspace_id=str(workspace_id),
            created_by=created_by,
            owner_agent_id=owner_agent_id,
            watch_type=watch_type,
            target_type=target_type,
            target_id=target_id,
            title=title,
            description=description,
            status=WatchStatus.WATCHING.value,
            success_criteria=success_criteria,
            failure_criteria=failure_criteria,
            quality_threshold=quality_threshold,
            check_interval_seconds=check_interval_seconds,
            next_check_at=moment + timedelta(seconds=check_interval_seconds),
            deadline_at=deadline_at,
            policy=policy,
            allowed_actions=allowed_actions,
            action_budget=action_budget,
            actions_taken=0,
            lineage=[
                {
                    "target_type": target_type,
                    "target_id": target_id,
                    "since": moment.isoformat(),
                    "reason": "created",
                }
            ],
        )
        db.add(watch)
        db.flush()

        WatchService.ingest(
            db,
            watch,
            event_type=WatchEventType.CREATED.value,
            event_key="created",
            summary=f"Watch created on {target_type}:{target_id}",
        )
        return watch

    @staticmethod
    def get_watch(
        db: Session, workspace_id: UUID | str, watch_id: UUID | str
    ) -> Optional[Watch]:
        """Workspace-scoped point read."""
        return (
            db.query(Watch)
            .filter(
                Watch.id == str(watch_id),
                Watch.workspace_id == str(workspace_id),
            )
            .first()
        )

    @staticmethod
    def list_watches(
        db: Session,
        workspace_id: UUID | str,
        *,
        status: Optional[str] = None,
        watch_type: Optional[str] = None,
        include_closed: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Watch]:
        """Workspace-scoped listing, newest first."""
        q = db.query(Watch).filter(Watch.workspace_id == str(workspace_id))
        if status is not None:
            q = q.filter(Watch.status == status)
        elif not include_closed:
            q = q.filter(
                Watch.status.in_([s.value for s in LIVE_WATCH_STATUSES])
            )
        if watch_type is not None:
            q = q.filter(Watch.watch_type == watch_type)
        return (
            q.order_by(Watch.created_at.desc()).offset(offset).limit(limit).all()
        )

    @staticmethod
    def find_live_watch(
        db: Session,
        *,
        workspace_id: UUID | str,
        target_type: str,
        target_id: str,
    ) -> Optional[Watch]:
        """The (at most one) non-terminal watch supervising a target."""
        return (
            db.query(Watch)
            .filter(
                Watch.workspace_id == str(workspace_id),
                Watch.target_type == target_type,
                Watch.target_id == str(target_id),
                Watch.status.in_([s.value for s in LIVE_WATCH_STATUSES]),
            )
            .first()
        )

    @staticmethod
    def cancel_watch(
        db: Session,
        workspace_id: UUID | str,
        watch_id: UUID | str,
        *,
        reason: Optional[str] = None,
    ) -> Watch:
        """Cancel a live watch (guarded -- cancelling a closed watch raises)."""
        watch = WatchService.get_watch(db, workspace_id, watch_id)
        if watch is None:
            raise ValueError(f"Watch {watch_id} not found in workspace {workspace_id}")
        WatchService.transition(
            db, watch, WatchStatus.CANCELLED, reason=reason or "Cancelled"
        )
        WatchService.ingest(
            db,
            watch,
            event_type=WatchEventType.CANCELLED.value,
            event_key="cancelled",
            summary=reason or "Watch cancelled",
        )
        return watch

    # ------------------------------------------------------------------
    # Guarded transition (house style of transition_run)
    # ------------------------------------------------------------------

    @staticmethod
    def transition(
        db: Session,
        watch: Watch,
        new_status: WatchStatus,
        *,
        reason: Optional[str] = None,
    ) -> Watch:
        """Transition a watch with the allowed-transition guard.

        Sets ``closed_at`` on terminal statuses. Flushes so the optimistic
        version check fires (StaleDataError -> ConflictError, house pattern).
        """
        current = WatchStatus(watch.status)
        allowed = ALLOWED_WATCH_TRANSITIONS.get(current, frozenset())
        if new_status not in allowed:
            raise InvalidTransitionError(
                entity_type="watch",
                entity_id=watch.id,
                current_state=current.value,
                target_state=new_status.value,
            )

        watch.status = new_status.value
        if new_status in TERMINAL_WATCH_STATUSES:
            watch.closed_at = _utcnow()
        elif current in TERMINAL_WATCH_STATUSES:
            # escalated -> watching renewal re-opens the watch
            watch.closed_at = None

        try:
            db.flush()
        except StaleDataError:
            raise ConflictError(entity_type="watch", entity_id=watch.id)

        logger.info(
            "[Watch] %s transitioned: %s -> %s%s",
            watch.id,
            current.value,
            new_status.value,
            f" ({reason})" if reason else "",
        )
        return watch

    # ------------------------------------------------------------------
    # Idempotent ingest
    # ------------------------------------------------------------------

    @staticmethod
    def ingest(
        db: Session,
        watch: Watch,
        *,
        event_type: str,
        event_key: str,
        summary: Optional[str] = None,
        snapshot: Optional[Dict[str, Any]] = None,
        score: Optional[float] = None,
        action_taken: Optional[str] = None,
        requires_attention: bool = False,
    ) -> Optional[WatchEvent]:
        """Record an observation exactly once.

        A duplicate (watch_id, event_key) is swallowed via a SAVEPOINT so the
        caller's outer transaction survives, and ``None`` is returned -- the
        caller can use "was this the first writer?" to decide follow-up work
        (the S5 tick notifies only when the sweep beat the producer hook).
        """
        event = WatchEvent(
            watch_id=watch.id,
            event_type=event_type,
            summary=summary,
            snapshot=snapshot,
            score=score,
            action_taken=action_taken,
            requires_attention=requires_attention,
            event_key=event_key,
        )
        try:
            with db.begin_nested():
                db.add(event)
                db.flush()
        except IntegrityError:
            logger.debug(
                "[Watch] duplicate event swallowed (watch=%s key=%r)",
                watch.id,
                event_key,
            )
            return None
        return event

    # ------------------------------------------------------------------
    # Terminal ingest -- the seam the S3 hooks and the S5 sweep both call
    # ------------------------------------------------------------------

    @staticmethod
    def ingest_terminal(
        db: Session,
        *,
        workspace_id: UUID | str,
        target_type: str,
        target_id: str,
        terminal_state: str,
        summary: Optional[str] = None,
        cost_snapshot: Optional[Dict[str, Any]] = None,
        output_pointer: Optional[str] = None,
    ) -> Optional[WatchEvent]:
        """Ingest a target's terminal state into its live watch, if any.

        Idempotent: exactly one terminal event per (watch, target). On first
        ingest the watch closes by outcome (WATCH_STATUS_FOR_TERMINAL_TARGET)
        with an unscored v1 verdict -- S6 inserts run-level scoring between
        "terminal observed" and "watch closed" without changing this seam.

        Returns the NEW WatchEvent when this call was the first writer,
        else ``None`` (already ingested, or no live watch on the target).
        """
        watch = WatchService.find_live_watch(
            db,
            workspace_id=workspace_id,
            target_type=target_type,
            target_id=str(target_id),
        )
        if watch is None:
            return None

        snapshot: Dict[str, Any] = {"terminal_state": terminal_state}
        if cost_snapshot:
            snapshot["cost"] = cost_snapshot
        if output_pointer:
            snapshot["output_pointer"] = output_pointer

        event = WatchService.ingest(
            db,
            watch,
            event_type=WatchEventType.TERMINAL.value,
            event_key=f"terminal:{target_type}:{target_id}",
            summary=summary or f"Target reached terminal state '{terminal_state}'",
            snapshot=snapshot,
        )

        # Close the watch even when the event was a duplicate: a prior
        # attempt may have committed the event but lost the close on a
        # version conflict (e.g. the tick's claim bumped version_id under
        # the producer hook). find_live_watch guarantees the watch is live
        # here, so the close is the self-healing half of idempotency.
        # Unknown terminal vocabularies park the watch for a human instead
        # of guessing a verdict.
        new_status = WATCH_STATUS_FOR_TERMINAL_TARGET.get(
            terminal_state, WatchStatus.NEEDS_ATTENTION
        )
        verdict = (
            f"Target {target_type}:{target_id} reached terminal state "
            f"'{terminal_state}'."
        )
        if summary:
            verdict = f"{verdict} {summary}"
        watch.final_verdict = verdict
        WatchService.transition(
            db, watch, new_status, reason=f"target terminal: {terminal_state}"
        )
        return event

    # ------------------------------------------------------------------
    # Bounded corrective actions
    # ------------------------------------------------------------------

    @staticmethod
    def record_action(
        db: Session,
        watch: Watch,
        *,
        action: str,
        summary: Optional[str] = None,
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Watch, bool]:
        """Record a corrective action against the watch's action budget.

        Returns ``(watch, True)`` when the action was recorded (and the watch
        moved WATCHING -> ACTING). Returns ``(watch, False)`` on the hard
        stop: the budget is exhausted, a ``budget_exhausted`` event lands
        (requires_attention) and the watch parks in NEEDS_ATTENTION. The
        caller (S8/S10 decision step) owns any escalation notification --
        this module stays sync + DB-only.
        """
        taken = watch.actions_taken or 0
        budget = watch.action_budget or 0

        if taken >= budget:
            WatchService.ingest(
                db,
                watch,
                event_type=WatchEventType.BUDGET_EXHAUSTED.value,
                event_key=f"budget_exhausted:{taken}",
                summary=(
                    f"Action budget exhausted ({taken}/{budget}) -- "
                    f"refused action '{action}'"
                ),
                requires_attention=True,
            )
            if WatchStatus(watch.status) != WatchStatus.NEEDS_ATTENTION:
                WatchService.transition(
                    db,
                    watch,
                    WatchStatus.NEEDS_ATTENTION,
                    reason="action budget exhausted",
                )
            return watch, False

        watch.actions_taken = taken + 1
        WatchService.ingest(
            db,
            watch,
            event_type=WatchEventType.ACTION.value,
            event_key=f"action:{watch.actions_taken}:{action}",
            summary=summary or f"Action taken: {action}",
            snapshot=snapshot,
            action_taken=action,
        )
        if WatchStatus(watch.status) == WatchStatus.WATCHING:
            WatchService.transition(
                db, watch, WatchStatus.ACTING, reason=f"action: {action}"
            )
        else:
            db.flush()
        return watch, True

    # ------------------------------------------------------------------
    # Lineage -- the watch follows the work (PRD-204 Section 8 Q9)
    # ------------------------------------------------------------------

    @staticmethod
    def follow(
        db: Session,
        watch: Watch,
        *,
        new_target_type: str,
        new_target_id: str,
        reason: Optional[str] = None,
        now: Optional[datetime] = None,
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> Watch:
        """Repoint a live watch at a new target (rerun/replan stays the SAME
        watch). Appends to ``lineage`` immutably and pulls the next check
        forward so the tick picks the new target up promptly. ``snapshot``
        (e.g. the S7 step_overrides) lands on the FOLLOW event for
        before/after comparison.
        """
        if WatchStatus(watch.status) in TERMINAL_WATCH_STATUSES:
            raise InvalidTransitionError(
                entity_type="watch",
                entity_id=watch.id,
                current_state=watch.status,
                target_state="follow",
            )

        moment = now or _utcnow()
        new_target_id = str(new_target_id)
        entry = {
            "target_type": new_target_type,
            "target_id": new_target_id,
            "since": moment.isoformat(),
            "reason": reason or "follow",
        }
        # Immutable append -- never mutate the JSONB list in place.
        watch.lineage = [*(watch.lineage or []), entry]
        watch.target_type = new_target_type
        watch.target_id = new_target_id
        watch.next_check_at = moment

        WatchService.ingest(
            db,
            watch,
            event_type=WatchEventType.FOLLOW.value,
            event_key=f"follow:{new_target_type}:{new_target_id}",
            summary=reason or f"Watch now follows {new_target_type}:{new_target_id}",
            snapshot=snapshot,
        )
        db.flush()
        return watch

    # ------------------------------------------------------------------
    # Tick claim -- FOR UPDATE SKIP LOCKED (board_dispatcher idiom)
    # ------------------------------------------------------------------

    @staticmethod
    def claim_due_watches(
        db: Session,
        *,
        limit: int = DEFAULT_CLAIM_LIMIT,
        now: Optional[datetime] = None,
    ) -> List[Watch]:
        """Atomically claim up to ``limit`` due watches for this ticker.

        The locked SELECT grabs only rows no other transaction holds
        (SKIP LOCKED), and the surrounding UPDATE reschedules
        ``next_check_at`` in the same statement -- the reschedule IS the
        lease, so a crashed ticker simply re-claims on a later tick.
        ``version_id`` is bumped so optimistic ORM writers see the claim.

        NOTE: COMMITS the claim (its own transaction), then returns freshly
        loaded rows -- same contract as board_dispatcher.claim_tasks.
        """
        moment = now or _utcnow()
        claimable = ", ".join(
            repr(s.value) for s in sorted(CLAIMABLE_WATCH_STATUSES)
        )
        rows = db.execute(
            text(
                f"""
                UPDATE watches AS w
                   SET last_checked_at = :now,
                       next_check_at   = :now
                           + make_interval(secs => w.check_interval_seconds),
                       updated_at      = :now,
                       version_id      = w.version_id + 1
                 WHERE w.id IN (
                       SELECT id
                         FROM watches
                        WHERE status IN ({claimable})
                          AND next_check_at IS NOT NULL
                          AND next_check_at <= :now
                        ORDER BY next_check_at
                        FOR UPDATE SKIP LOCKED
                        LIMIT :limit
                 )
             RETURNING w.id
                """
            ),
            {"now": moment, "limit": limit},
        ).fetchall()
        db.commit()

        ids = [r[0] for r in rows]
        if not ids:
            return []

        # populate_existing is REQUIRED: the raw UPDATE above bumped
        # version_id outside the ORM, so an identity-mapped instance from
        # earlier in this session still carries the stale version. Without
        # the refresh, the next optimistic-locked write on a claimed watch
        # would StaleDataError against its own claim.
        claimed = (
            db.query(Watch)
            .filter(Watch.id.in_(ids))
            .order_by(Watch.next_check_at.asc())
            .execution_options(populate_existing=True)
            .all()
        )
        logger.info("[Watch] claimed %d due watch(es)", len(claimed))
        return claimed

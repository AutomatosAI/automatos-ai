"""PRD-161 — Postgres-native board-task dispatch spine (claim / lease / requeue).

One dispatch path so an assigned ``BoardTask`` ALWAYS executes exactly once,
fails honestly, and retries deliberately. No new table (``board_tasks`` carries
``lease_until`` + ``attempts``) and no new service — the same Postgres provides:

* **Exactly-once claim** — ``FOR UPDATE SKIP LOCKED`` lets each of N concurrent
  workers lock a disjoint set of rows; a row locked by one worker is skipped by
  the others rather than double-claimed.
* **Honest requeue** — a claimed task carries a lease; if the worker crashes or
  hangs, the lease expires and the sweeper requeues it (``attempts`` already
  incremented) instead of silently closing crashed work as ``done``.
* **Low-latency wakeup** — assign/create fire ``pg_notify`` so a listening
  claimant picks the task up sub-second; the poll loop is the fallback.

This module is the primitive layer (claim/notify/sweep). The background loop that
consumes it and runs each claimed task individually lives alongside it.
"""
from __future__ import annotations

import asyncio
import logging
import select
import threading
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.orm import Session

from core.models.core import BoardTask
from services.board_events import notify_board_event

logger = logging.getLogger(__name__)

# Claimants LISTEN on this channel to skip the poll wait; assign/create NOTIFY it.
NOTIFY_CHANNEL = "board_task_available"

# Priority ordering for claim selection — highest urgency, then oldest first.
# Inlined as data (no hardcoded behaviour elsewhere); mirrors the board's
# urgent > high > medium > low taxonomy.
_PRIORITY_ORDER_SQL = (
    "CASE priority "
    "WHEN 'urgent' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END"
)


def notify_task_available(db: Session, *, workspace_id, task_id: int) -> None:
    """Fire ``pg_notify`` so a listening claimant wakes immediately.

    Best-effort: a failed NOTIFY only costs the poll-loop's latency, never
    correctness, so it never raises into the caller's request path.
    """
    try:
        db.execute(
            text("SELECT pg_notify(:chan, :payload)"),
            {"chan": NOTIFY_CHANNEL, "payload": f"{workspace_id}:{task_id}"},
        )
    except Exception:  # noqa: BLE001 — NOTIFY is an optimisation, not a guarantee
        logger.debug("[dispatch] pg_notify failed for task %s", task_id, exc_info=True)


def claim_tasks(
    db: Session,
    *,
    worker_id: str,
    limit: int,
    lease_seconds: int,
    max_slots_per_agent: Optional[int] = None,
) -> List[BoardTask]:
    """Atomically claim up to ``limit`` assigned tasks for this worker.

    ``FOR UPDATE SKIP LOCKED`` is the exactly-once guarantee: the locked SELECT
    grabs only rows no other transaction holds, and the surrounding UPDATE flips
    them to ``in_progress`` with a fresh lease and ``attempts + 1`` in the same
    transaction. Concurrent workers therefore claim disjoint sets — never the
    same task twice. Recipe-mirror tasks are excluded (the recipe executor drives
    those).

    When ``max_slots_per_agent`` is set (PRD-161 S4), an agent never runs more
    than that many tasks at once: the claim only picks ``slots − (in_progress for
    that agent)`` per agent, so extra tasks stay ``assigned`` — the DB is the
    queue (double-texting is queued, never dropped). ``None`` = no cap.

    Returns the freshly claimed rows (already committed), highest priority first.
    """
    now = datetime.now(timezone.utc)
    lease_until = now + timedelta(seconds=lease_seconds)

    if max_slots_per_agent is None:
        # No cap: single-statement exactly-once claim.
        locked = f"""
            SELECT id
              FROM board_tasks
             WHERE status = 'assigned'
               AND assigned_agent_id IS NOT NULL
               AND source_type <> 'recipe'
             ORDER BY {_PRIORITY_ORDER_SQL}, created_at
             FOR UPDATE SKIP LOCKED
             LIMIT :limit
        """
        params = {"lease_until": lease_until, "now": now, "limit": limit}
    else:
        # Slot-aware: rank each agent's assigned tasks and keep only as many as
        # the agent has free slots (slots − currently in_progress). Two steps,
        # because FOR UPDATE can't co-exist with the window/aggregate it needs.
        candidate_rows = db.execute(
            text(
                f"""
                WITH running AS (
                    SELECT assigned_agent_id, COUNT(*) AS n
                      FROM board_tasks
                     WHERE status = 'in_progress' AND assigned_agent_id IS NOT NULL
                     GROUP BY assigned_agent_id
                ),
                ranked AS (
                    SELECT t.id,
                           ROW_NUMBER() OVER (
                               PARTITION BY t.assigned_agent_id
                               ORDER BY {_PRIORITY_ORDER_SQL}, t.created_at
                           ) AS rn,
                           :slots - COALESCE(r.n, 0) AS free
                      FROM board_tasks t
                      LEFT JOIN running r ON r.assigned_agent_id = t.assigned_agent_id
                     WHERE t.status = 'assigned'
                       AND t.assigned_agent_id IS NOT NULL
                       AND t.source_type <> 'recipe'
                )
                SELECT id FROM ranked WHERE rn <= free ORDER BY rn, id LIMIT :limit
                """
            ),
            {"slots": max_slots_per_agent, "limit": limit},
        ).fetchall()
        candidate_ids = [r[0] for r in candidate_rows]
        if not candidate_ids:
            return []
        locked = """
            SELECT id
              FROM board_tasks
             WHERE id = ANY(:ids) AND status = 'assigned'
             FOR UPDATE SKIP LOCKED
        """
        params = {"lease_until": lease_until, "now": now, "ids": candidate_ids}

    rows = db.execute(
        text(
            f"""
            UPDATE board_tasks AS t
               SET status      = 'in_progress',
                   attempts    = t.attempts + 1,
                   lease_until = :lease_until,
                   started_at  = COALESCE(t.started_at, :now),
                   updated_at  = :now
             WHERE t.id IN ({locked})
         RETURNING t.id
            """
        ),
        params,
    ).fetchall()
    db.commit()

    ids = [r[0] for r in rows]
    if not ids:
        return []

    claimed = (
        db.query(BoardTask)
        .filter(BoardTask.id.in_(ids))
        .order_by(BoardTask.lease_until.asc())
        .all()
    )
    # PRD-180 S1 (F090): a claim flips assigned → in_progress; push that to the
    # Command Centre so the human watches the card start, not on a poll tick.
    for t in claimed:
        notify_board_event(
            db, workspace_id=t.workspace_id, task_id=t.id, status="in_progress",
            event="task_claimed",
        )
    logger.info("[dispatch] worker=%s claimed %d task(s): %s", worker_id, len(ids), ids)
    return claimed


def requeue_expired_leases(db: Session, *, max_attempts: int) -> dict:
    """Sweeper — reclaim work whose lease expired (worker crashed or hung).

    An ``in_progress`` task past its ``lease_until`` is presumed abandoned. While
    it still has attempts left it returns to ``assigned`` for another claim
    (``attempts`` was already incremented at claim time — this is the honest
    requeue, NOT a silent close-as-done). Once attempts reach ``max_attempts``
    (Q41: 2) it becomes terminal ``failed`` with a reason, so dead work fails
    loudly instead of looping forever.

    Returns ``{"requeued": [ids...], "failed": [ids...]}``.
    """
    now = datetime.now(timezone.utc)

    requeued = db.execute(
        text(
            """
            UPDATE board_tasks
               SET status      = 'assigned',
                   lease_until = NULL,
                   updated_at  = :now
             WHERE status = 'in_progress'
               AND lease_until IS NOT NULL
               AND lease_until < :now
               AND attempts < :max_attempts
         RETURNING id
            """
        ),
        {"now": now, "max_attempts": max_attempts},
    ).fetchall()

    failed = db.execute(
        text(
            """
            UPDATE board_tasks
               SET status        = 'failed',
                   lease_until   = NULL,
                   completed_at  = :now,
                   updated_at    = :now,
                   error_message = COALESCE(error_message, 'Lease expired after max attempts — no worker completed the task')
             WHERE status = 'in_progress'
               AND lease_until IS NOT NULL
               AND lease_until < :now
               AND attempts >= :max_attempts
         RETURNING id
            """
        ),
        {"now": now, "max_attempts": max_attempts},
    ).fetchall()

    db.commit()

    result = {"requeued": [r[0] for r in requeued], "failed": [r[0] for r in failed]}
    if result["requeued"] or result["failed"]:
        # PRD-180 S1 (F090): a crashed task returning to the queue or dying is a
        # real state change the human should see immediately, not on a poll tick.
        _notify_swept(db, result["requeued"], "assigned", "task_requeued")
        _notify_swept(db, result["failed"], "failed", "task_failed")
        logger.info(
            "[dispatch] sweeper requeued=%s failed=%s",
            result["requeued"],
            result["failed"],
        )
    return result


def _notify_swept(db: Session, task_ids: List[int], status: str, event: str) -> None:
    """Fire a board-event NOTIFY for each swept task, looking up its workspace.

    Best-effort: the sweeper's correctness never depends on the UI ping, so a
    failed lookup/notify is logged and skipped rather than raised.
    """
    if not task_ids:
        return
    try:
        rows = db.execute(
            text("SELECT id, workspace_id FROM board_tasks WHERE id = ANY(:ids)"),
            {"ids": task_ids},
        ).fetchall()
        for task_id, workspace_id in rows:
            notify_board_event(
                db, workspace_id=workspace_id, task_id=task_id, status=status,
                event=event,
            )
    except Exception:  # noqa: BLE001 — UI ping is best-effort, never breaks the sweep
        logger.debug("[dispatch] sweep notify failed for %s", task_ids, exc_info=True)


def renew_lease(db: Session, task_id: int, *, lease_seconds: int) -> bool:
    """PRD-171 F024: extend a still-running task's lease (a live heartbeat).

    The lease (``BOARD_DISPATCH_LEASE_SECONDS``, default 600s) is the crash
    deadline: ``requeue_expired_leases`` presumes any ``in_progress`` row past it
    is abandoned and requeues it for another claim. A *legitimately* long run
    (an agent working > 600s) would be swept back to ``assigned`` and re-claimed
    — double execution, breaking exactly-once under lease expiry. The running
    worker therefore heartbeats: while its execution is alive it pushes
    ``lease_until`` forward, so the sweep only ever catches a genuinely dead run
    (the process is gone, so nothing renews and the lease truly lapses).

    Renews ONLY while ``status = 'in_progress'`` — a task that already reached a
    terminal state is left untouched (never resurrects a finished/failed row).
    Returns ``True`` if a row was renewed. Best-effort: the caller must not fail
    the run because a heartbeat write failed.
    """
    now = datetime.now(timezone.utc)
    new_lease = now + timedelta(seconds=lease_seconds)
    rows = db.execute(
        text(
            """
            UPDATE board_tasks
               SET lease_until = :new_lease,
                   updated_at  = :now
             WHERE id = :task_id
               AND status = 'in_progress'
         RETURNING id
            """
        ),
        {"new_lease": new_lease, "now": now, "task_id": task_id},
    ).fetchall()
    db.commit()
    renewed = bool(rows)
    if renewed:
        logger.debug("[dispatch] renewed lease for task %s → %s", task_id, new_lease)
    return renewed


def scan_sla_breaches(db: Session) -> List[dict]:
    """Flag tasks past their SLA deadline that haven't reached a terminal state.

    Each overdue task is marked ``sla_breach_notified`` exactly once (so the
    sweeper doesn't re-fire every tick) and returned so the caller can dispatch a
    breach notification. Wires the previously-dead ``sla_deadline`` column.
    """
    now = datetime.now(timezone.utc)
    rows = db.execute(
        text(
            """
            UPDATE board_tasks
               SET sla_breach_notified = true,
                   updated_at = :now
             WHERE sla_deadline IS NOT NULL
               AND sla_deadline < :now
               AND sla_breach_notified = false
               AND status NOT IN ('done', 'failed')
         RETURNING id, workspace_id, assigned_agent_id, title
            """
        ),
        {"now": now},
    ).fetchall()
    db.commit()
    breached = [
        {"task_id": r[0], "workspace_id": str(r[1]), "agent_id": r[2], "title": r[3]}
        for r in rows
    ]
    if breached:
        logger.info("[dispatch] SLA breach: %s", [b["task_id"] for b in breached])
    return breached


async def _notify_sla_breach(task_info: dict) -> None:
    """Fire a ``task_sla_breach`` notification for one overdue task."""
    from core.database.database import SessionLocal
    from core.services.notification_dispatcher import NotificationDispatcher

    db = SessionLocal()
    try:
        disp = NotificationDispatcher(db, task_info["workspace_id"])
        await disp.dispatch(
            event_type="task_sla_breach",
            title=f"SLA breached: {task_info['title']}",
            message="This task is past its SLA deadline and still unfinished.",
            link_type="task",
            link_id=str(task_info["task_id"]),
            agent_id=task_info.get("agent_id"),
            status="warn",
        )
        db.commit()
    except Exception:  # noqa: BLE001
        logger.error(
            "[dispatch] sla-breach notify failed for task %s",
            task_info.get("task_id"), exc_info=True,
        )
    finally:
        db.close()


# ── Background dispatch loop ────────────────────────────────────────────────


class _NotifyListener(threading.Thread):
    """Holds a dedicated LISTEN connection and wakes the async loop on NOTIFY.

    psycopg2 ``LISTEN`` is blocking, so it lives in its own daemon thread and
    signals the event loop via ``call_soon_threadsafe``. If it can't start (or
    dies), the dispatcher still drains on the poll interval — NOTIFY buys
    latency, never correctness.
    """

    def __init__(self, wake: "asyncio.Event", loop: "asyncio.AbstractEventLoop"):
        super().__init__(daemon=True, name="board-dispatch-listen")
        self._wake = wake
        self._loop = loop
        self._stop = threading.Event()
        self._raw = None

    def run(self) -> None:
        try:
            from core.database.database import engine

            self._raw = engine.raw_connection()
            self._raw.connection.autocommit = True
            cur = self._raw.cursor()
            cur.execute(f"LISTEN {NOTIFY_CHANNEL}")
            logger.info("[dispatch] LISTEN %s active", NOTIFY_CHANNEL)
            while not self._stop.is_set():
                if select.select([self._raw.connection], [], [], 1.0)[0]:
                    self._raw.connection.poll()
                    if self._raw.connection.notifies:
                        self._raw.connection.notifies.clear()
                        self._loop.call_soon_threadsafe(self._wake.set)
        except Exception:  # noqa: BLE001 — listener is best-effort; poll covers us
            logger.warning(
                "[dispatch] NOTIFY listener stopped — poll fallback still active",
                exc_info=True,
            )
        finally:
            try:
                if self._raw is not None:
                    self._raw.close()
            except Exception:  # noqa: BLE001
                pass

    def stop(self) -> None:
        self._stop.set()


def _claim_and_sweep(session_factory, cfg, worker_id: str) -> List[dict]:
    """One blocking DB unit: sweep expired leases, then claim a batch.

    Returns plain dicts (detached from the session) describing what to run, so
    the caller never touches a closed-session ORM object.
    """
    db = session_factory()
    try:
        requeue_expired_leases(db, max_attempts=cfg.BOARD_DISPATCH_MAX_ATTEMPTS)
        breached = scan_sla_breaches(db)
        claimed = claim_tasks(
            db,
            worker_id=worker_id,
            limit=cfg.BOARD_DISPATCH_CLAIM_BATCH,
            lease_seconds=cfg.BOARD_DISPATCH_LEASE_SECONDS,
            max_slots_per_agent=cfg.BOARD_DISPATCH_AGENT_SLOTS,
        )
        out = []
        for t in claimed:
            prompt = t.raw_prompt or t.description or t.title
            if t.review_feedback:
                # Q44: a rejected task redoes the work with reviewer feedback in
                # context. Consume it so the correction applies to this run only.
                prompt = (
                    f"{prompt}\n\n## Reviewer feedback on your previous attempt\n"
                    f"{t.review_feedback}\n\n"
                    "Address this feedback in your redo."
                )
                t.review_feedback = None
            out.append(
                {
                    "task_id": t.id,
                    "agent_id": t.assigned_agent_id,
                    "workspace_id": str(t.workspace_id),
                    "prompt": prompt,
                    "review_mode": t.review_mode or "auto",
                    "attachment_ids": t.attachment_ids or [],
                }
            )
        db.commit()  # persist consumed review_feedback
        return {"claimed": out, "breached": breached}
    finally:
        db.close()


def _launch_one(task: dict) -> None:
    """Hand one claimed task to the existing per-task execution path.

    Imported lazily to avoid a circular import (api.board_tasks imports this
    module for ``notify_task_available``).
    """
    from api.board_tasks import _launch_task_execution

    _launch_task_execution(
        task_id=task["task_id"],
        agent_id=task["agent_id"],
        workspace_id=task["workspace_id"],
        prompt=task["prompt"],
        review_mode=task["review_mode"],
        attachment_ids=task["attachment_ids"],
    )


async def run_dispatch_loop(*, stop_event: Optional["asyncio.Event"] = None) -> None:
    """The single board dispatch spine (PRD-161 S2).

    Replaces the heartbeat 3-tasks-into-1-prompt fold-in: every tick sweeps
    expired leases, claims a batch of assigned tasks (``FOR UPDATE SKIP LOCKED``
    → exactly-once), and launches EACH claimed task through the existing
    per-task execution path INDIVIDUALLY — never batched. Wakes sub-second on
    ``pg_notify``; the poll interval is the fallback. The blocking DB unit is
    offloaded so the loop never stalls other work (S4 deepens this).
    """
    from config import config
    from core.database.database import SessionLocal

    if not config.BOARD_DISPATCH_ENABLED:
        logger.info("[dispatch] BOARD_DISPATCH_ENABLED is false — loop not started")
        return

    loop = asyncio.get_running_loop()
    wake = asyncio.Event()
    worker_id = f"dispatch-{uuid4().hex[:8]}"

    listener = _NotifyListener(wake, loop)
    listener.start()
    logger.info(
        "[dispatch] loop up (worker=%s, poll=%ss, lease=%ss)",
        worker_id,
        config.BOARD_DISPATCH_POLL_SECONDS,
        config.BOARD_DISPATCH_LEASE_SECONDS,
    )

    try:
        while not (stop_event is not None and stop_event.is_set()):
            try:
                tick = await asyncio.to_thread(
                    _claim_and_sweep, SessionLocal, config, worker_id
                )
                for task in tick["claimed"]:
                    _launch_one(task)
                for breach in tick["breached"]:
                    await _notify_sla_breach(breach)
            except Exception:  # noqa: BLE001 — a bad tick must never kill the loop
                logger.exception("[dispatch] tick failed — continuing")

            wake.clear()
            try:
                await asyncio.wait_for(
                    wake.wait(), timeout=config.BOARD_DISPATCH_POLL_SECONDS
                )
            except asyncio.TimeoutError:
                pass
    except asyncio.CancelledError:
        logger.info("[dispatch] loop cancelled")
        raise
    finally:
        listener.stop()

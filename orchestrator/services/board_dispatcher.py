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
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.orm import Session

from core.cli_runtime import PROVIDER_CLAUDE, RUNTIME_API, RUNTIME_CLI
from core.models.core import BoardTask
from services.board_events import notify_board_event
from services.pasted_data import pasted_data_rule
from services.ticket_owner_ask import ticket_answers_block
from services.ticket_redo import redo_block

logger = logging.getLogger(__name__)

# Claimants LISTEN on this channel to skip the poll wait; assign/create NOTIFY it.
NOTIFY_CHANNEL = "board_task_available"
MAX_ADOPTED_FILES = 20

# F209: a ticket's current run, in ``runtime_ref``. Every start stamps a fresh one
# (this claim, a PATCH into in_progress, the status tool) and a redispatch clears
# it, so a run that no longer holds the ticket can never finalize it. The claim
# SQL below spells the same key.
RUN_ID_KEY = "run_id"

# Priority ordering for claim selection — highest urgency, then oldest first.
# Inlined as data (no hardcoded behaviour elsewhere); mirrors the board's
# urgent > high > medium > low taxonomy.
_PRIORITY_ORDER_SQL = (
    "CASE priority "
    "WHEN 'urgent' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END"
)


def runtime_predicate_sql(runtime: str, alias: str = "board_tasks") -> str:
    """PRD-234 S1a: which runtime's tickets a claimant may take.

    The dispatch loop runs API agents; a paired CLI host claims the tickets of
    ``runtime: cli`` agents. The rule reads the agent's JSON configuration (no
    agents column) and treats a missing/NULL configuration as ``api`` — every
    agent that exists today.
    """
    kind = (
        f"COALESCE((SELECT a.configuration->>'runtime' FROM agents a "
        f"WHERE a.id = {alias}.assigned_agent_id), '{RUNTIME_API}')"
    )
    if runtime == RUNTIME_CLI:
        return f"{kind} = '{RUNTIME_CLI}'"
    return f"{kind} <> '{RUNTIME_CLI}'"


def recipe_exclusion_sql(runtime: str, alias: str = "board_tasks") -> str:
    """Recipe-mirror tickets of API agents are driven by the recipe executor, never
    the board. A session agent's playbook step IS a ticket its CLI host must claim
    (PRD-239 S3), so the exclusion applies to the API runtime only."""
    if runtime == RUNTIME_CLI:
        return ""
    return f"AND {alias}.source_type <> 'recipe'"


# The mission engine runs its own steps (PRD-171 F025): the dispatch loop never
# claims a mission's mirror tickets, whoever set one 'assigned' (a PATCH, a tool,
# a grant's re-queue). A CLI agent's step is run by its host for the mission, and
# its card may be the step's mirror itself (F094); the mission's own ticket never.
MISSION_MIRROR_TYPES = ("orchestration", "orchestration_task")
CLI_CLAIMABLE_MIRRORS = ("orchestration_task",)


def mission_mirror_exclusion_sql(runtime: str, alias: str = "board_tasks") -> str:
    barred = [kind for kind in MISSION_MIRROR_TYPES
              if not (runtime == RUNTIME_CLI and kind in CLI_CLAIMABLE_MIRRORS)]
    kinds = ", ".join(f"'{kind}'" for kind in barred)
    return f"AND {alias}.source_type NOT IN ({kinds})"


def provider_predicate_sql(providers: Optional[Sequence[str]], alias: str = "board_tasks") -> str:
    """CLI adapter design §8.2: a host claims only the tickets of agents whose CLI
    it serves (``capabilities.providers``). ``None`` = no filter (the API runtime,
    or a host that announced nothing); an empty list claims nothing — a host with
    no CLI installed must never take work it cannot run, and it must never be
    refused *after* the claim (that is a claim/release loop across two hosts of
    different CLIs). The provider is read from the agent's configuration; a
    ``cli`` agent without one is ``claude`` (what every session agent ran on
    before the field existed)."""
    if providers is None:
        return ""
    provider = (
        f"COALESCE((SELECT a.configuration->>'provider' FROM agents a "
        f"WHERE a.id = {alias}.assigned_agent_id), '{PROVIDER_CLAUDE}')"
    )
    return f"AND {provider} = ANY(CAST(:providers AS text[]))"


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
    runtime: str = RUNTIME_API,
    workspace_id=None,
    providers: Optional[Sequence[str]] = None,
) -> List[BoardTask]:
    """Atomically claim up to ``limit`` assigned tasks for this worker.

    PRD-234 S1a: ``runtime`` selects whose tickets are claimable — ``api`` (the
    dispatch loop, default: behaviour unchanged) or ``cli`` (a paired CLI host);
    ``workspace_id`` confines a host's claim to its own workspace; ``providers``
    confines it to the tickets of agents whose CLI that host serves (CLI adapter
    design §8.2 — ``None`` = no filter, ``[]`` = nothing).

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
    runtime_sql = runtime_predicate_sql(runtime, alias="board_tasks")
    runtime_sql_t = runtime_predicate_sql(runtime, alias="t")
    # Recipe-mirror tickets of API agents are driven by the recipe executor, never
    # the board — but a session agent's playbook step IS a ticket its host must
    # claim (PRD-239 S3): the exclusion applies to the API runtime only.
    recipe_sql = recipe_exclusion_sql(runtime, alias="board_tasks") + " " + mission_mirror_exclusion_sql(runtime, "board_tasks")
    recipe_sql_t = recipe_exclusion_sql(runtime, alias="t") + " " + mission_mirror_exclusion_sql(runtime, "t")
    ws_sql = "AND workspace_id = CAST(:ws AS uuid)" if workspace_id is not None else ""
    ws_sql_t = "AND t.workspace_id = CAST(:ws AS uuid)" if workspace_id is not None else ""
    ws_params = {"ws": str(workspace_id)} if workspace_id is not None else {}
    provider_sql = provider_predicate_sql(providers, alias="board_tasks")
    provider_sql_t = provider_predicate_sql(providers, alias="t")
    if providers is not None:
        ws_params = {**ws_params, "providers": list(providers)}

    if max_slots_per_agent is None:
        # No cap: single-statement exactly-once claim.
        locked = f"""
            SELECT id
              FROM board_tasks
             WHERE status = 'assigned'
               AND assigned_agent_id IS NOT NULL
               {recipe_sql}
               AND {runtime_sql}
               {ws_sql}
               {provider_sql}
             ORDER BY {_PRIORITY_ORDER_SQL}, created_at
             FOR UPDATE SKIP LOCKED
             LIMIT :limit
        """
        params = {"lease_until": lease_until, "now": now, "limit": limit, **ws_params}
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
                       {recipe_sql_t}
                       AND {runtime_sql_t}
                       {ws_sql_t}
                       {provider_sql_t}
                )
                SELECT id FROM ranked WHERE rn <= free ORDER BY rn, id LIMIT :limit
                """
            ),
            {"slots": max_slots_per_agent, "limit": limit, **ws_params},
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
                   updated_at  = :now,
                   -- F209: each claim is its own run; finalize writes only for it.
                   runtime_ref = COALESCE(t.runtime_ref, '{{}}'::jsonb)
                                 || jsonb_build_object('run_id', gen_random_uuid()::text)
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


def adopt_session_files(db: Session, *, now: datetime, max_attempts: int) -> Dict[int, List[str]]:
    """F014 (night 3, #612 and #614): a ticket about to be failed on lease
    expiry whose session left files in its own folder, ``sessions/<ticket>``,
    did the work — the worker just never reported, so nothing was registered.
    Register those files as the ticket's deliverables (the delivered path then
    sends it to ``review``) and name them in the review note. Night 3 marked
    both tickets failed while their answers sat in that folder. Returns
    ``{ticket id: [files]}``. Fail-soft per ticket.

    "Nothing was registered" means by THIS run (``runtime_ref.deliverables``,
    rebuilt by every claim), not ever: a mission step's card spans its runs
    (F094), so an earlier run's files do not mean this one reported.
    """
    from config import config
    from services.cli_host_service import DEFAULT_FOLDER_SESSIONS, _register_session_deliverables

    rows = db.execute(
        text(
            """
            SELECT bt.id FROM board_tasks bt
             WHERE bt.status = 'in_progress'
               AND bt.lease_until IS NOT NULL
               AND bt.lease_until < :now
               AND bt.attempts >= :max_attempts
               AND (bt.runtime_ref -> 'deliverables') IS NULL
            """
        ),
        {"now": now, "max_attempts": max_attempts},
    ).fetchall()
    adopted: Dict[int, List[str]] = {}
    for (task_id,) in rows:
        try:
            task = db.get(BoardTask, task_id)
            folder = Path(config.WORKSPACE_VOLUME_PATH) / str(task.workspace_id) / DEFAULT_FOLDER_SESSIONS / str(task_id)
            if not folder.is_dir():
                continue
            files = sorted(str(f) for f in folder.rglob("*") if f.is_file() and not f.name.startswith("."))
            registered = _register_session_deliverables(
                db, task, files[:MAX_ADOPTED_FILES], agent_id=task.assigned_agent_id, agent_name=None,
                session_id=(task.runtime_ref or {}).get("session_id"),
            )
            if not registered:
                continue
            names = [r["file_path"] for r in registered]
            task.review_feedback = task.review_feedback or (
                "Finished, worker never reported — its session left " + ", ".join(names)
                + " (now on the ticket). Sent to review instead of failed."
            )
            # This run's files, as a result that reported would record them (a
            # later mission step reads them from here, F161).
            task.runtime_ref = {**(task.runtime_ref or {}), "deliverables": registered}
            db.flush()
            adopted[task_id] = names
        except Exception:  # noqa: BLE001 — a ticket we cannot adopt still fails as before
            logger.warning("[dispatch] could not adopt session files for ticket %s", task_id, exc_info=True)
    if adopted:
        logger.info("[dispatch] adopted session files for %s", adopted)
    return adopted


def requeue_expired_leases(db: Session, *, max_attempts: int) -> dict:
    """Sweeper — reclaim work whose lease expired (worker crashed or hung).

    An ``in_progress`` task past its ``lease_until`` is presumed abandoned. While
    it still has attempts left it returns to ``assigned`` for another claim
    (``attempts`` was already incremented at claim time — this is the honest
    requeue, NOT a silent close-as-done). Once attempts reach ``max_attempts``
    (Q41: 2) it becomes terminal ``failed`` with a reason, so dead work fails
    loudly instead of looping forever.

    Returns ``{"requeued": [...], "failed": [...], "delivered": [...]}`` —
    ``delivered`` are the ones that produced files and went to ``review``.
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

    # Night 3 (#612, #614): the session wrote its answer into sessions/<ticket>
    # and never reported, so nothing was registered — adopt those files first.
    adopt_session_files(db, now=now, max_attempts=max_attempts)

    # A ticket whose worker never reported but which LEFT FILES did the work —
    # night 1 wrote "no worker completed the task" over six delivered files
    # (F013). Check the deliverables the run registered before naming it, and
    # send those to ``review`` for a human verdict instead of ``failed``. THIS
    # run's (runtime_ref.deliverables): a mission step's card spans runs (F094).
    delivered = db.execute(
        text(
            """
            UPDATE board_tasks bt
               SET status        = 'review',
                   lease_until   = NULL,
                   completed_at  = :now,
                   updated_at    = :now,
                   error_message = NULL,
                   review_feedback = COALESCE(
                       bt.review_feedback,
                       'Finished, worker never reported — its deliverables are on the ticket.'
                   )
             WHERE bt.status = 'in_progress'
               AND bt.lease_until IS NOT NULL
               AND bt.lease_until < :now
               AND bt.attempts >= :max_attempts
               -- this run registered at least one file: a list holding an object
               AND COALESCE(bt.runtime_ref -> 'deliverables', CAST('[]' AS jsonb)) @> CAST('[{}]' AS jsonb)
         RETURNING bt.id
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

    result = {
        "requeued": [r[0] for r in requeued],
        "failed": [r[0] for r in failed],
        "delivered": [r[0] for r in delivered],
    }
    if result["delivered"]:
        _notify_swept(db, result["delivered"], "review", "task_updated")
    if result["requeued"] or result["failed"] or result["delivered"]:
        # PRD-180 S1 (F090): a crashed task returning to the queue or dying is a
        # real state change the human should see immediately, not on a poll tick.
        _notify_swept(db, result["requeued"], "assigned", "task_requeued")
        _notify_swept(db, result["failed"], "failed", "task_failed")
        logger.info(
            "[dispatch] sweeper requeued=%s failed=%s delivered=%s",
            result["requeued"],
            result["failed"],
            result["delivered"],
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


def renew_lease(db: Session, task_id: int, *, lease_seconds: int, run_id: Optional[str] = None) -> bool:
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
               -- F209: a run renews only the claim it holds, never a newer run's
               AND (CAST(:run_id AS text) IS NULL OR runtime_ref->>'run_id' = :run_id)
         RETURNING id
            """
        ),
        {"new_lease": new_lease, "now": now, "task_id": task_id, "run_id": run_id},
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
            # F199: data pasted into the brief is counted and totalled with code.
            pasted = pasted_data_rule(t.description or prompt)
            if pasted:
                prompt = f"{prompt}\n\n{pasted}"
            # F183: a ticket the owner's answer re-queued runs with the answer.
            answers = ticket_answers_block(getattr(t, "planning_data", None))
            if answers:
                prompt = f"{prompt}\n\n{answers}"
            # Q44 + F198: a sent-back task corrects its last draft with every
            # correction on the ticket; the waiting feedback is consumed here.
            redo = redo_block(t)
            if redo:
                prompt = f"{prompt}\n\n{redo}"
                t.review_feedback = None
            out.append(
                {
                    "task_id": t.id,
                    "agent_id": t.assigned_agent_id,
                    "workspace_id": str(t.workspace_id),
                    "prompt": prompt,
                    "review_mode": t.review_mode or "auto",
                    "attachment_ids": t.attachment_ids or [],
                    "run_id": (getattr(t, "runtime_ref", None) or {}).get(RUN_ID_KEY),
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
        run_id=task.get("run_id"),
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

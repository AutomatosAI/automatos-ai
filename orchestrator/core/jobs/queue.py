"""The queue's database operations (PRD-276 §6.2-6.3): enqueue, claim, renew, fence, finish, sweep.

Every operation is one parameterised statement or a short transaction on the caller's session, so
it runs off the event loop (the worker calls these through ``asyncio.to_thread``). Every write a
worker makes about a job it claimed is fenced by that claim: ``WHERE id = :id AND status = 'running'
AND claim_id = :claim_id``. A worker whose update matches no row has lost the job, and writes
nothing more. Times come from ``clock_timestamp()``, never ``now()`` (the transaction's start).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import text

from core.jobs.registry import JobKind, kind_of

logger = logging.getLogger(__name__)

NOTIFY_CHANNEL = "jobs_available"
ERROR_MAX = 2000
LEASE_EXPIRED = "lease expired"


@dataclass(frozen=True)
class Job:
    """A claimed job, as its handler receives it."""

    id: UUID
    kind: str
    workspace_id: Optional[UUID]
    payload: Dict[str, Any]
    attempts: int
    max_attempts: int
    claim_id: UUID


_ENQUEUE = text("""
    INSERT INTO jobs (id, kind, workspace_id, payload, run_at, max_attempts, idempotency_key)
    VALUES (:id, :kind, :ws, CAST(:payload AS jsonb), COALESCE(:run_at, clock_timestamp()), :max_attempts, :key)
    ON CONFLICT DO NOTHING
    RETURNING id
""")


def enqueue(db: Any, kind: str, payload: Dict[str, Any], *, workspace_id: Any = None, key: Optional[str] = None,
            run_at: Optional[datetime] = None) -> Optional[UUID]:
    """Queues ``kind`` in the caller's transaction (it exists only if that commits). Answers its id,
    or None when a job with the same (kind, workspace, key) already exists. Does not commit."""
    spec = kind_of(kind)
    row = db.execute(_ENQUEUE, {"id": uuid4(), "kind": kind, "ws": str(workspace_id) if workspace_id else None,
                                "payload": json.dumps(payload or {}), "run_at": run_at,
                                "max_attempts": spec.max_attempts, "key": key}).first()
    if row is None:
        return None
    db.execute(text("SELECT pg_notify(:channel, :kind)"), {"channel": NOTIFY_CHANNEL, "kind": kind})
    return row[0]


def enqueue_now(kind: str, payload: Dict[str, Any], **options: Any) -> Optional[UUID]:
    """``enqueue`` on a session of its own, committed at once: for a caller with no session and no
    business write before it (§6.2). It does not share anyone's transaction."""
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        queued = enqueue(db, kind, payload, **options)
        db.commit()
        return queued
    finally:
        db.close()


# A kind's claims take turns (a transaction-scoped advisory lock), so each sees the last one's
# committed claims and the per-workspace limit holds across workers, not just within one claim.
_CLAIM_TURN = text("SELECT pg_advisory_xact_lock(hashtext('jobs:claim:' || :kind))")
CLAIM_SCAN = 10  # candidates looked at per job wanted, so a busy workspace can't crowd out the rest

_CLAIM = text("""
    WITH running AS (
        SELECT workspace_id, count(*) AS n FROM jobs WHERE kind = :kind AND status = 'running' GROUP BY workspace_id
    ), candidates AS (
        SELECT j.id, j.workspace_id, j.run_at FROM jobs j
        LEFT JOIN workspaces w ON w.id = j.workspace_id
        WHERE j.kind = :kind AND j.status = 'queued' AND j.run_at <= clock_timestamp()
          AND (j.workspace_id IS NULL OR w.deleted_at IS NULL)
        ORDER BY j.run_at
        LIMIT :scan
        FOR UPDATE OF j SKIP LOCKED
    ), picked AS (
        SELECT id FROM (
            SELECT c.id, c.run_at, COALESCE(r.n, 0)
                   + row_number() OVER (PARTITION BY c.workspace_id ORDER BY c.run_at) AS nth
            FROM candidates c LEFT JOIN running r ON r.workspace_id IS NOT DISTINCT FROM c.workspace_id
        ) ranked
        WHERE nth <= :per_workspace
        ORDER BY run_at
        LIMIT :limit
    )
    UPDATE jobs SET status = 'running', attempts = jobs.attempts + 1, claim_id = gen_random_uuid(),
                    claimed_by = :worker, lease_until = clock_timestamp() + make_interval(secs => :lease),
                    updated_at = clock_timestamp()
    FROM picked WHERE jobs.id = picked.id
    RETURNING jobs.id, jobs.kind, jobs.workspace_id, jobs.payload, jobs.attempts, jobs.max_attempts, jobs.claim_id
""")


def claim(db: Any, spec: JobKind, worker_id: str, limit: int) -> List[Job]:
    """Claims up to ``limit`` due jobs of ``spec.kind`` (skipping a closed workspace's, and any workspace
    already at the kind's per-workspace limit), each with a new claim id and a lease. Commits."""
    if limit < 1:
        return []
    db.execute(_CLAIM_TURN, {"kind": spec.kind})
    rows = db.execute(_CLAIM, {"kind": spec.kind, "worker": worker_id, "limit": limit, "scan": limit * CLAIM_SCAN,
                               "lease": spec.lease_seconds, "per_workspace": spec.workspace_concurrency}).all()
    db.commit()
    return [Job(id=r[0], kind=r[1], workspace_id=r[2], payload=r[3] or {}, attempts=r[4], max_attempts=r[5],
                claim_id=r[6]) for r in rows]


# Every write about a claimed job is fenced by its claim: "WHERE id AND status = 'running' AND claim_id".
_RENEW = text("UPDATE jobs SET lease_until = clock_timestamp() + make_interval(secs => :lease), "
              "updated_at = clock_timestamp() WHERE id = :id AND status = 'running' AND claim_id = :claim_id RETURNING id")
_HOLD = text("SELECT id FROM jobs WHERE id = :id AND status = 'running' AND claim_id = :claim_id FOR UPDATE")
_COMPLETE = text("UPDATE jobs SET status = 'done', finished_at = clock_timestamp(), lease_until = NULL, "
                 "updated_at = clock_timestamp() WHERE id = :id AND status = 'running' AND claim_id = :claim_id RETURNING id")


def renew(db: Any, job: Job, lease_seconds: int) -> bool:
    """Extends the lease while the handler runs; False when the claim was lost. Commits."""
    row = db.execute(_RENEW, {"id": job.id, "claim_id": job.claim_id, "lease": lease_seconds}).first()
    db.commit()
    return row is not None


def fence(db: Any, job: Job, extend_seconds: int) -> bool:
    """Takes the fence before a send or a final write, in the caller's transaction (§6.3, §6.8): locks
    the job row while the claim holds, checks its workspace is live, and extends the lease from the
    wall clock to cover what follows. False means stop and roll back. Does not commit."""
    held = db.execute(_HOLD, {"id": job.id, "claim_id": job.claim_id}).first()
    if held is None:
        return False
    if job.workspace_id is not None and not db.execute(
            text("SELECT 1 FROM workspaces WHERE id = :ws AND deleted_at IS NULL"), {"ws": job.workspace_id}).first():
        return False
    db.execute(text("UPDATE jobs SET lease_until = greatest(lease_until, clock_timestamp() + "
                    "make_interval(secs => :extend)) WHERE id = :id"), {"id": job.id, "extend": extend_seconds})
    return True


def complete(db: Any, job: Job, *, commit: bool = True) -> bool:
    """Marks the job done; False when the claim was lost (nothing written). A handler finishing under
    its own fence passes ``commit=False`` and commits with its own writes."""
    row = db.execute(_COMPLETE, {"id": job.id, "claim_id": job.claim_id}).first()
    if commit:
        db.commit()
    return row is not None


def fail(db: Any, job: Job, error: str) -> bool:
    """A failed attempt: back in the queue at the kind's next delay, or ``dead`` once its attempts are
    used up. False when the claim was lost (nothing written). Commits."""
    spec = kind_of(job.kind)
    params = {"id": job.id, "claim_id": job.claim_id, "error": (error or "")[:ERROR_MAX],
              "delay": spec.delay_after(job.attempts)}
    row = db.execute(_FAIL, params).first()
    db.commit()
    return row is not None


_FAIL = text("""
    UPDATE jobs SET
        status = CASE WHEN attempts >= max_attempts THEN 'dead' ELSE 'queued' END,
        finished_at = CASE WHEN attempts >= max_attempts THEN clock_timestamp() END,
        run_at = CASE WHEN attempts >= max_attempts THEN run_at
                      ELSE clock_timestamp() + make_interval(secs => :delay) END,
        claim_id = NULL, lease_until = NULL, last_error = :error, updated_at = clock_timestamp()
    WHERE id = :id AND status = 'running' AND claim_id = :claim_id RETURNING id
""")

# The sweep: the same next state, and claim_id cleared with it, so the old holder's token matches nothing.
_EXPIRE = text("""
    UPDATE jobs SET
        status = CASE WHEN attempts >= max_attempts THEN 'dead' ELSE 'queued' END,
        finished_at = CASE WHEN attempts >= max_attempts THEN clock_timestamp() END,
        run_at = CASE WHEN attempts >= max_attempts THEN run_at
                      ELSE clock_timestamp() + make_interval(secs => :delay) END,
        claim_id = NULL, lease_until = NULL, last_error = :error, updated_at = clock_timestamp()
    WHERE id = :id AND status = 'running' RETURNING id
""")


def sweep_expired(db: Any, limit: int = 100) -> int:
    """An expired lease is a failed attempt (§6.3): the job's ``claim_id`` is cleared in the same update,
    so its old holder's token matches nothing from that moment, and it goes back in the queue or, with
    no attempts left, becomes dead. Commits; answers how many it swept."""
    rows = db.execute(text("SELECT id, kind, attempts FROM jobs WHERE status = 'running' AND lease_until < "
                           "clock_timestamp() ORDER BY lease_until LIMIT :limit FOR UPDATE SKIP LOCKED"),
                      {"limit": limit}).all()
    for job_id, kind, attempts in rows:
        delay = _delay_or_default(kind, attempts)
        db.execute(_EXPIRE, {"id": job_id, "error": LEASE_EXPIRED, "delay": delay})
    db.commit()
    if rows:
        logger.warning("[jobs] swept %d job(s) whose lease ran out", len(rows))
    return len(rows)


def _delay_or_default(kind: str, attempts: int) -> int:
    try:
        return kind_of(kind).delay_after(attempts)
    except KeyError:  # a kind no longer registered: retried at the default pace
        from core.jobs.registry import DEFAULT_BACKOFF

        return DEFAULT_BACKOFF[min(max(attempts, 1), len(DEFAULT_BACKOFF)) - 1]


def prune(db: Any, retention_days: int) -> int:
    """Deletes ``done`` jobs finished more than ``retention_days`` ago (§6.3). Commits."""
    deleted = db.execute(text("DELETE FROM jobs WHERE status = 'done' AND finished_at < clock_timestamp() - "
                              "make_interval(days => :days)"), {"days": retention_days}).rowcount
    db.commit()
    return int(deleted or 0)

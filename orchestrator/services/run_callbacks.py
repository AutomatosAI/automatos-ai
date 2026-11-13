"""Signed completion callbacks for runs an API caller starts (PRD-274 Wave 1, §8-8).

``POST /api/agents/{id}/execute`` may name a ``callback_url``. When the run's ticket ends (done,
review, failed or cancelled), its ending is POSTed there once, with the run id:

* **Signed.** ``X-Automatos-Signature: t=<unix seconds>,v1=<hex>``, where ``v1`` is
  HMAC-SHA256 over ``"<t>.<raw body>"`` with the workspace's signing secret
  (``GET /api/v1/callbacks/signing-secret``). A receiver recomputes it and refuses a stale ``t``
  (``verify_signature``). ``X-Automatos-Delivery`` names the delivery; retries keep it.
* **Safe to send.** The URL must be https and is resolved through the platform's outbound seam
  (``core.security.web_access``) when the run is requested and again before every attempt: a
  private, loopback, link-local or metadata address is refused, the POST goes to the address that
  was checked, and a redirect is never followed.
* **Durable, then recorded.** Each callback is a ``run_callback.deliver`` job (PRD-276 §6.5): its
  id is the delivery id, kept by every attempt; a 5xx, a 429 or no answer is retried at
  ``DELIVERY_BACKOFF``, up to ``DELIVERY_ATTEMPTS``; a 2xx delivers; any other 4xx is the receiver's
  answer and is not retried. Each send happens under the job's fence, and the outcome stays on the
  ``run_callbacks`` row. A restart re-runs the job; it can't lose it. The board's completion writer
  queues the job; a sweep queues it for a ticket ended any other way (``schedules_the_callback_sweep``).
* **At least once.** A worker can die after the receiver accepted and before the outcome is written;
  the retry sends the same ``X-Automatos-Delivery``. A receiver that drops a delivery id it has
  already seen gets each callback once in effect.
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from uuid import UUID

from core.jobs.registry import CATCH_UP_LATEST, job_handler

logger = logging.getLogger(__name__)

SECRET_PREFIX = "whsec_"
SECRET_BYTES = 32
SIGNATURE_HEADER = "X-Automatos-Signature"
EVENT_HEADER = "X-Automatos-Event"
DELIVERY_HEADER = "X-Automatos-Delivery"
EVENT = "agent_run.finished"
SIGNATURE_VERSION = "v1"
TOLERANCE_SECONDS = 300
TIMEOUT_SECONDS = 10.0
URL_MAX = 2048
RESULT_MAX = 50_000
ERROR_MAX = 500
REQUIRED_SCHEME = "https"
TOO_MANY_REQUESTS = 429


class CallbackUrlRefused(ValueError):
    """A ``callback_url`` the platform will not POST to; the message says why."""


@dataclass(frozen=True)
class Delivery:
    """One callback, ready to send."""

    url: str
    secret: str
    body: bytes
    delivery_id: str


@dataclass(frozen=True)
class Outcome:
    status: str
    attempts: int
    status_code: Optional[int] = None
    error: Optional[str] = None


# ── The signing secret ─────────────────────────────────────────────────────────────────────

def _uuid(value: Any) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


def _new_secret() -> str:
    return SECRET_PREFIX + secrets.token_urlsafe(SECRET_BYTES)


def signing_secret(db: Any, workspace_id: Any) -> str:
    """The workspace's signing secret, made on its first use."""
    from sqlalchemy.exc import IntegrityError

    from core.credentials.encryption import get_encryption_service
    from core.models.run_callbacks import WorkspaceSigningSecret

    crypto = get_encryption_service()
    row = db.get(WorkspaceSigningSecret, _uuid(workspace_id))
    if row is not None:
        return crypto.decrypt(row.secret_encrypted)
    secret = _new_secret()
    db.add(WorkspaceSigningSecret(workspace_id=_uuid(workspace_id), secret_encrypted=crypto.encrypt(secret)))
    try:
        db.commit()
    except IntegrityError:  # another request made it first: theirs is the secret
        db.rollback()
        return crypto.decrypt(db.get(WorkspaceSigningSecret, _uuid(workspace_id)).secret_encrypted)
    return secret


def rotate_signing_secret(db: Any, workspace_id: Any) -> str:
    """A new signing secret for the workspace; callbacks sent from now on are signed with it."""
    from core.credentials.encryption import get_encryption_service
    from core.models.run_callbacks import WorkspaceSigningSecret

    secret = _new_secret()
    encrypted = get_encryption_service().encrypt(secret)
    row = db.get(WorkspaceSigningSecret, _uuid(workspace_id), with_for_update=True)
    if row is None:
        db.add(WorkspaceSigningSecret(workspace_id=_uuid(workspace_id), secret_encrypted=encrypted))
    else:
        row.secret_encrypted, row.rotated_at = encrypted, datetime.now(timezone.utc)
    db.commit()
    return secret


def signature(secret: str, timestamp: int, body: bytes) -> str:
    """The ``X-Automatos-Signature`` value for ``body`` sent at ``timestamp``."""
    mac = hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={timestamp},{SIGNATURE_VERSION}={mac}"


def verify_signature(secret: str, header: str, body: bytes, *, now: Optional[float] = None,
                     tolerance: int = TOLERANCE_SECONDS) -> bool:
    """What a receiver checks: the signature matches and was made within ``tolerance`` seconds."""
    parts = dict(item.split("=", 1) for item in header.split(",") if "=" in item)
    try:
        timestamp = int(parts.get("t", ""))
    except ValueError:
        return False
    if abs((time.time() if now is None else now) - timestamp) > tolerance:
        return False
    expected = signature(secret, timestamp, body).rsplit(f"{SIGNATURE_VERSION}=", 1)[1]
    return hmac.compare_digest(expected, parts.get(SIGNATURE_VERSION, "").strip())


# ── The URL ────────────────────────────────────────────────────────────────────────────────

def check_callback_url(url: str) -> None:
    """Refuses a URL the platform won't POST to: not https, or resolving to a blocked address."""
    from core.security.web_access import resolve_outbound

    target = resolve_outbound(url, enforce_switch=False)  # a caller's destination, not agent web access
    if not target.ok:
        raise CallbackUrlRefused(f"callback_url refused: {target.reason}")
    if target.scheme != REQUIRED_SCHEME:
        raise CallbackUrlRefused("callback_url must be https")


def new_callback(workspace_id: Any, url: str) -> Callable[[Any, Any], None]:
    """What ``file_ticket`` adds beside the new ticket, before it is dispatched."""
    from core.models.run_callbacks import RunCallback

    def attach(db: Any, task: Any) -> None:
        db.add(RunCallback(workspace_id=_uuid(workspace_id), task_id=task.id, url=url))

    return attach


# ── Delivery, as a durable job (PRD-276 §6.5) ──────────────────────────────────────────────

DELIVER_KIND = "run_callback.deliver"
DELIVERY_BACKOFF = (5, 30, 120, 600)  # seconds after attempts 1-4: long enough to ride out a receiver's restart
DELIVERY_ATTEMPTS = 5
SEND_MARGIN_SECONDS = 5  # the fence's lease covers the send's timeout plus this


class NotYetDelivered(RuntimeError):
    """A 5xx, a 429 or no answer: the job goes back in the queue at its next delay."""


def job_key(task_id: int) -> str:
    """One delivery job per ticket: the finalize path and the sweep can't both queue it."""
    return f"task:{task_id}"


def sends_run_callback(finalize: Callable[..., Awaitable[Optional[str]]]) -> Callable[..., Awaitable[Optional[str]]]:
    """Wraps the board's one completion writer: once it writes a ticket's ending, a pending
    callback for that ticket is queued as a ``run_callback.deliver`` job."""

    @functools.wraps(finalize)
    async def finalize_then_call_back(db: Any, **kwargs: Any) -> Optional[str]:
        ended = await finalize(db, **kwargs)
        if ended is not None:
            await _queue_after_ending(kwargs["task_id"], kwargs.get("run_id"))
        return ended

    return finalize_then_call_back


async def _queue_after_ending(task_id: int, run_id: Optional[str]) -> None:
    """On its own session, off the event loop: the writer's session is left as it was. A failure is
    logged and the callback stays pending, for the sweep to queue; the ending is never held up."""
    try:
        await asyncio.to_thread(_on_own_session, queue_if_pending, task_id, run_id)
    except Exception:  # noqa: BLE001 — logged; the sweep queues it later
        logger.exception("[run_callbacks] could not queue ticket %s's callback; the sweep will", task_id)


def queue_if_pending(db: Any, task_id: int, run_id: Optional[str] = None) -> bool:
    """Queues the ticket's callback when one is pending; True when a job went in. Commits."""
    from core.jobs import enqueue
    from core.models.run_callbacks import PENDING, RunCallback

    row = db.query(RunCallback.workspace_id).filter(RunCallback.task_id == task_id,
                                                    RunCallback.status == PENDING).first()
    if row is None:
        return False
    queued = enqueue(db, DELIVER_KIND, {"task_id": task_id, "run_id": run_id}, workspace_id=row[0],
                     key=job_key(task_id))
    db.commit()
    return queued is not None


@job_handler(DELIVER_KIND, backoff=DELIVERY_BACKOFF, max_attempts=DELIVERY_ATTEMPTS)
async def deliver_run_callback(job: Any) -> Optional[str]:
    """Sends the ticket's ending once per attempt and records it; the status written. The job's id
    is the delivery id, the same on every attempt, so a receiver that drops ids it has seen gets
    each callback once in effect. A retryable answer raises, and the worker re-queues the job."""
    from core.database.database import SessionLocal

    if not await asyncio.to_thread(_on_own_session, _secret_ready, job.payload["task_id"]):
        return None  # no callback left to send
    db = SessionLocal()
    try:
        status = await _send_under_fence(db, job)
    finally:
        db.close()
    from core.models.run_callbacks import PENDING

    if status == PENDING:
        raise NotYetDelivered(f"callback for ticket {job.payload['task_id']} not yet delivered")
    return status


def _secret_ready(db: Any, task_id: int) -> bool:
    """Makes the workspace's signing secret now, if it has none, so nothing commits under the fence."""
    from core.models.run_callbacks import RunCallback

    row = db.query(RunCallback.workspace_id).filter(RunCallback.task_id == task_id).first()
    if row is None:
        return False
    signing_secret(db, row[0])
    return True


async def _send_under_fence(db: Any, job: Any) -> Optional[str]:
    """§6.8: the fence (the job row locked while the claim holds, the workspace live, the lease
    stretched past the send), then a fresh read, the send, and the outcome, in one transaction."""
    from core.jobs.queue import fence

    if not await asyncio.to_thread(fence, db, job, int(TIMEOUT_SECONDS) + SEND_MARGIN_SECONDS):
        await asyncio.to_thread(db.rollback)
        logger.info("[run_callbacks] job %s lost its claim or its workspace closed; nothing sent", job.id)
        return None
    delivery = await asyncio.to_thread(_prepare, db, job.payload["task_id"], job.payload.get("run_id"), str(job.id))
    if delivery is None:
        await asyncio.to_thread(db.rollback)
        return None
    code, error = await _post_once(delivery)
    outcome = _outcome(code, error, job)
    await asyncio.to_thread(_record, db, job, outcome)
    return outcome.status


def _on_own_session(work: Callable[..., Any], *args: Any) -> Any:
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        return work(db, *args)
    finally:
        db.close()


def _prepare(db: Any, task_id: int, run_id: Optional[str], delivery_id: str) -> Optional[Delivery]:
    from api.board_tasks import ticket_cost
    from core.models.core import BoardTask
    from core.models.run_callbacks import DELIVERED, RunCallback
    from services.board_dispatcher import RUN_ID_KEY

    callback = db.query(RunCallback).filter(RunCallback.task_id == task_id, RunCallback.status != DELIVERED).first()
    task = db.get(BoardTask, task_id)
    if callback is None or task is None:
        return None
    payload = run_payload(task, run_id or (task.runtime_ref or {}).get(RUN_ID_KEY), ticket_cost(db, task_id),
                          delivery_id)
    body = json.dumps(payload, separators=(",", ":"), default=str).encode()
    return Delivery(url=callback.url, secret=signing_secret(db, task.workspace_id), body=body, delivery_id=delivery_id)


def run_payload(task: Any, run_id: Optional[str], cost: Dict[str, Any], delivery_id: str) -> Dict[str, Any]:
    """What the callback says about the run."""
    def iso(moment: Optional[datetime]) -> Optional[str]:
        return moment.isoformat() if moment else None

    return {"event": EVENT, "delivery_id": delivery_id, "task_id": task.id, "run_id": run_id,
            "agent_id": task.assigned_agent_id, "workspace_id": str(task.workspace_id), "status": task.status,
            "result": (task.result or "")[:RESULT_MAX] or None, "error": task.error_message,
            "started_at": iso(task.started_at), "completed_at": iso(task.completed_at), "cost": cost}


def _outcome(code: Optional[int], error: Optional[str], job: Any) -> Outcome:
    """A 2xx delivers; a 4xx other than 429 is the receiver's answer; anything else is retried
    until the job's last attempt."""
    from core.models.run_callbacks import DELIVERED, FAILED, PENDING

    if code is not None and 200 <= code < 300:
        return Outcome(DELIVERED, 1, code)
    final = (code is not None and code < 500 and code != TOO_MANY_REQUESTS) or job.attempts >= job.max_attempts
    return Outcome(FAILED if final else PENDING, 1, code, error)


async def _post_once(delivery: Delivery) -> Tuple[Optional[int], Optional[str]]:
    """One signed POST to the address checked just now; (status code, error)."""
    from core.security.web_access import build_pinned_request, resolve_outbound_async

    target = await resolve_outbound_async(delivery.url, enforce_switch=False)
    if not target.ok or target.scheme != REQUIRED_SCHEME:
        return None, f"refused: {target.reason or 'not https'}"
    headers = {"Content-Type": "application/json", EVENT_HEADER: EVENT, DELIVERY_HEADER: delivery.delivery_id,
               SIGNATURE_HEADER: signature(delivery.secret, int(time.time()), delivery.body)}
    try:
        async with _client() as client:
            response = await client.send(build_pinned_request(client, "POST", delivery.url, target,
                                                              content=delivery.body, headers=headers))
        return response.status_code, None if response.status_code < 300 else f"HTTP {response.status_code}"
    except Exception as exc:  # noqa: BLE001 — logged and recorded on the callback; the ticket is untouched
        logger.warning("[run_callbacks] POST to %s failed: %s", target.host, exc)
        return None, type(exc).__name__


def _client() -> Any:
    """The client callbacks go through (tests route it into an ``httpx.MockTransport``)."""
    import httpx

    return httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=False)


def _record(db: Any, job: Any, outcome: Outcome) -> None:
    """The attempt on the callback row and, when it was the last, the job done: one commit, under the fence."""
    from core.jobs.queue import complete
    from core.models.run_callbacks import DELIVERED, PENDING, RunCallback

    callback = db.query(RunCallback).filter(RunCallback.task_id == job.payload["task_id"]).first()
    if callback is not None:
        callback.status, callback.attempts = outcome.status, callback.attempts + outcome.attempts
        callback.last_status_code = outcome.status_code
        callback.last_error = (outcome.error or None) and outcome.error[:ERROR_MAX]
        if outcome.status == DELIVERED:
            callback.delivered_at = datetime.now(timezone.utc)
    if outcome.status != PENDING:
        complete(db, job, commit=False)
    db.commit()
    logger.info("[run_callbacks] ticket %s: callback %s on attempt %d", job.payload["task_id"], outcome.status,
                job.attempts)


# ── The sweep: callbacks no completion writer queued ───────────────────────────────────────

SWEEP_KIND = "run_callback.sweep"
SWEEP_EVERY_SECONDS = 600
SWEEP_AFTER_SECONDS = 300  # a ticket ended this long ago with its callback still pending, and no job
SWEEP_BATCH = 50
ENDED = ("done", "review", "failed", "cancelled")


def schedules_the_callback_sweep(start: Callable[..., Any]) -> Callable[..., Any]:
    """Wraps ``UnifiedScheduler.start``: once the shared scheduler runs, a platform schedule queues a
    ``run_callback.sweep`` every ``SWEEP_EVERY_SECONDS`` (catch-up ``latest``: one sweep finds every
    callback a missed one would have). The sweep queues a delivery job for each callback still pending
    well after its ticket ended. Tickets end in many places (a cancel, the dispatcher's sweep, the
    reconciler), and only the board's completion writer queues the job itself; the sweep also queued,
    once, what was pending at cutover."""

    @functools.wraps(start)
    def started(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = start(self, *args, **kwargs)
        scheduler = getattr(self, "_scheduler", None)
        if scheduler is not None:
            from core.jobs.schedules import PLATFORM, interval, schedule

            schedule(scheduler, SWEEP_KIND, PLATFORM, interval(seconds=SWEEP_EVERY_SECONDS))
        return result

    return started


@job_handler(SWEEP_KIND, catch_up=CATCH_UP_LATEST, concurrency=1, workspace_concurrency=1)
async def run_sweep(job: Any) -> None:
    """A sweep slot (PRD-276 Wave 2c)."""
    await queue_stale_callbacks()


async def queue_stale_callbacks() -> int:
    """Queues the stale pending callbacks that have no job yet; how many it queued."""
    return await asyncio.to_thread(_on_own_session, _queue_stale)


def _queue_stale(db: Any) -> int:
    from core.jobs import enqueue

    rows = _stale_pending(db)
    for task_id, workspace_id in rows:
        enqueue(db, DELIVER_KIND, {"task_id": task_id, "run_id": None}, workspace_id=workspace_id,
                key=job_key(task_id))
    db.commit()
    if rows:
        logger.info("[run_callbacks] queued %d callback(s) no completion writer queued", len(rows))
    return len(rows)


def _stale_pending(db: Any) -> List[Tuple[int, Any]]:
    from datetime import timedelta

    from sqlalchemy import exists, func

    from core.models.core import BoardTask
    from core.models.jobs import Job as JobRow
    from core.models.run_callbacks import PENDING, RunCallback

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=SWEEP_AFTER_SECONDS)
    has_job = exists().where(JobRow.kind == DELIVER_KIND,
                             JobRow.idempotency_key == func.concat("task:", RunCallback.task_id))
    rows = (db.query(RunCallback.task_id, RunCallback.workspace_id).join(BoardTask, BoardTask.id == RunCallback.task_id)
            .filter(RunCallback.status == PENDING, BoardTask.status.in_(ENDED), BoardTask.completed_at < cutoff,
                    ~has_job)
            .order_by(BoardTask.completed_at).limit(SWEEP_BATCH).all())
    return [(row[0], row[1]) for row in rows]

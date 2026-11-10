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
* **Retried, then recorded.** Up to three attempts (``BACKOFF_SECONDS``); a 2xx delivers; a 4xx
  other than 429 is the receiver's answer and is not retried. The outcome stays on the
  ``run_callbacks`` row. Attempts are made in this process: a restart mid-retry leaves the row
  ``pending``.
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
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

SECRET_PREFIX = "whsec_"
SECRET_BYTES = 32
SIGNATURE_HEADER = "X-Automatos-Signature"
EVENT_HEADER = "X-Automatos-Event"
DELIVERY_HEADER = "X-Automatos-Delivery"
EVENT = "agent_run.finished"
SIGNATURE_VERSION = "v1"
TOLERANCE_SECONDS = 300
BACKOFF_SECONDS = (0, 5, 15)  # the wait before each attempt
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


# ── Delivery ───────────────────────────────────────────────────────────────────────────────

def sends_run_callback(finalize: Callable[..., Awaitable[Optional[str]]]) -> Callable[..., Awaitable[Optional[str]]]:
    """Wraps the board's one completion writer: once it writes a ticket's ending, a pending
    callback for that ticket is delivered in the background."""

    @functools.wraps(finalize)
    async def finalize_then_call_back(db: Any, **kwargs: Any) -> Optional[str]:
        ended = await finalize(db, **kwargs)
        if ended is not None and await _has_pending_callback(kwargs["task_id"]):
            from core.utils.background_tasks import launch_guarded

            launch_guarded(deliver_run_callback(kwargs["task_id"], kwargs.get("run_id")),
                           subsystem="run_callbacks", operation="deliver", agent_id=kwargs.get("agent_id"))
        return ended

    return finalize_then_call_back


async def _has_pending_callback(task_id: int) -> bool:
    """On its own session, off the event loop: the writer's session is left as it was. A read
    that fails is logged and the callback stays pending; the ticket's ending is never held up."""
    try:
        return await asyncio.to_thread(_on_own_session, _pending, task_id)
    except Exception:  # noqa: BLE001 — logged; the row stays pending for a later send
        logger.exception("[run_callbacks] could not read ticket %s's callback; it stays pending", task_id)
        return False


def _pending(db: Any, task_id: int) -> bool:
    from core.models.run_callbacks import PENDING, RunCallback

    return db.query(RunCallback.id).filter(RunCallback.task_id == task_id, RunCallback.status == PENDING).first() \
        is not None


async def deliver_run_callback(task_id: int, run_id: Optional[str] = None) -> Optional[str]:
    """Sends the ticket's ending to its callback URL and records how it went; the status written."""
    delivery = await asyncio.to_thread(_on_own_session, _prepare, task_id, run_id)
    if delivery is None:
        return None
    outcome = await _send_with_retries(delivery)
    await asyncio.to_thread(_on_own_session, _record, task_id, outcome)
    return outcome.status


def _on_own_session(work: Callable[..., Any], *args: Any) -> Any:
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        return work(db, *args)
    finally:
        db.close()


def _prepare(db: Any, task_id: int, run_id: Optional[str]) -> Optional[Delivery]:
    from api.board_tasks import ticket_cost
    from core.models.core import BoardTask
    from core.models.run_callbacks import PENDING, RunCallback
    from services.board_dispatcher import RUN_ID_KEY

    callback = db.query(RunCallback).filter(RunCallback.task_id == task_id, RunCallback.status == PENDING).first()
    task = db.get(BoardTask, task_id)
    if callback is None or task is None:
        return None
    payload = run_payload(task, run_id or (task.runtime_ref or {}).get(RUN_ID_KEY), ticket_cost(db, task_id))
    body = json.dumps(payload, separators=(",", ":"), default=str).encode()
    return Delivery(url=callback.url, secret=signing_secret(db, task.workspace_id), body=body,
                    delivery_id=payload["delivery_id"])


def run_payload(task: Any, run_id: Optional[str], cost: Dict[str, Any]) -> Dict[str, Any]:
    """What the callback says about the run."""
    def iso(moment: Optional[datetime]) -> Optional[str]:
        return moment.isoformat() if moment else None

    return {"event": EVENT, "delivery_id": str(uuid4()), "task_id": task.id, "run_id": run_id,
            "agent_id": task.assigned_agent_id, "workspace_id": str(task.workspace_id), "status": task.status,
            "result": (task.result or "")[:RESULT_MAX] or None, "error": task.error_message,
            "started_at": iso(task.started_at), "completed_at": iso(task.completed_at), "cost": cost}


async def _send_with_retries(delivery: Delivery) -> Outcome:
    attempts, code, error = 0, None, None
    for wait in BACKOFF_SECONDS:
        if wait:
            await asyncio.sleep(wait)
        attempts += 1
        code, error = await _post_once(delivery)
        if code is not None and 200 <= code < 300:
            from core.models.run_callbacks import DELIVERED

            return Outcome(DELIVERED, attempts, code)
        if code is not None and code < 500 and code != TOO_MANY_REQUESTS:
            break  # the receiver's answer: sending it again won't change it
    from core.models.run_callbacks import FAILED

    return Outcome(FAILED, attempts, code, error)


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


def _record(db: Any, task_id: int, outcome: Outcome) -> None:
    from core.models.run_callbacks import DELIVERED, RunCallback

    callback = db.query(RunCallback).filter(RunCallback.task_id == task_id).first()
    if callback is None:
        return
    callback.status, callback.attempts = outcome.status, callback.attempts + outcome.attempts
    callback.last_status_code, callback.last_error = outcome.status_code, (outcome.error or None) and outcome.error[:ERROR_MAX]
    if outcome.status == DELIVERED:
        callback.delivered_at = datetime.now(timezone.utc)
    db.commit()
    logger.info("[run_callbacks] ticket %s: callback %s after %d attempt(s)", task_id, outcome.status, outcome.attempts)
